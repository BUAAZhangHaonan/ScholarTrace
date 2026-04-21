"""Authenticated MCP SSE call simulation for ScholarTrace.

This script demonstrates MCP JSON-RPC tool calls with bearer authentication,
then executes the real ScholarTrace tools `query` and `read`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client


def _unwrap_result_wrapper(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap MCP result wrapper when payload is {'result': ...}."""
    if "result" not in payload:
        return payload

    result_value = payload.get("result")
    if isinstance(result_value, dict):
        return result_value
    if isinstance(result_value, str):
        try:
            decoded = json.loads(result_value)
        except json.JSONDecodeError:
            return payload
        if isinstance(decoded, dict):
            return decoded
    return payload


def _parse_result_payload(call_result: Any) -> dict[str, Any]:
    """Extract JSON payload from MCP CallToolResult."""
    if getattr(call_result, "structuredContent", None):
        structured = call_result.structuredContent
        if isinstance(structured, dict):
            return _unwrap_result_wrapper(structured)
        return {"raw": str(structured)}

    text_chunks: list[str] = []
    for item in getattr(call_result, "content", []):
        item_type = getattr(item, "type", "")
        if item_type == "text":
            text_chunks.append(getattr(item, "text", ""))

    if not text_chunks:
        return {"raw": str(call_result)}

    merged_text = "\n".join(text_chunks).strip()
    try:
        payload = json.loads(merged_text)
    except json.JSONDecodeError:
        return {"text": merged_text}

    if isinstance(payload, dict):
        return _unwrap_result_wrapper(payload)
    return {"raw": str(payload)}


def _build_query_theme_text(theme_text: str) -> str:
    """Append explicit retrieval instruction for top-10 target profile."""
    request_suffix = (
        "\n\nSelection target:\n"
        "- return exactly 10 papers\n"
        "- prioritize newest publications\n"
        "- prioritize high influence\n"
        "- prioritize highest relevance to the theme\n"
    )
    return theme_text.strip() + request_suffix


def _extract_query_error(payload: dict[str, Any]) -> tuple[str, bool, str, int]:
    """Return (code, retryable, message, retry_after_seconds) for query errors."""
    error = payload.get("error")
    if not isinstance(error, dict):
        return "", False, "", 0
    code = str(error.get("code", ""))
    retryable = bool(error.get("retryable", False))
    message = str(error.get("message", ""))

    retry_after_seconds = 0
    marker = "retry in "
    if marker in message:
        tail = message.split(marker, 1)[1].strip()
        number_text = ""
        for ch in tail:
            if ch.isdigit():
                number_text += ch
            else:
                break
        if number_text:
            retry_after_seconds = int(number_text)

    return code, retryable, message, retry_after_seconds


async def run_mcp_simulation(
    *,
    sse_url: str,
    access_token: str,
    theme_file: Path,
    final_limit: int,
    agent_candidate_limit: int,
    coarse_pool_limit: int,
    include_rationale: bool,
    max_query_retries: int,
    retry_backoff_seconds: float,
    min_agent_candidate_limit: int,
    query_read_timeout_seconds: int,
    output_json: Path,
) -> dict[str, Any]:
    theme_text = theme_file.read_text(encoding="utf-8")
    query_theme_document = _build_query_theme_text(theme_text)

    headers = {"Authorization": f"Bearer {access_token}"}

    query_args_base = {
        "theme_document": query_theme_document,
        "final_limit": final_limit,
        "include_rationale": include_rationale,
    }

    read_args: dict[str, Any] | None = None

    query_rpc = {
        "jsonrpc": "2.0",
        "id": "query-1",
        "method": "tools/call",
        "params": {
            "name": "query",
                "arguments": {
                    **query_args_base,
                    "agent_candidate_limit": agent_candidate_limit,
                    "coarse_pool_limit": coarse_pool_limit,
                },
        },
    }

    result_bundle: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sse_url": sse_url,
        "theme_file": str(theme_file),
        "auth_header_preview": "Authorization: Bearer ***",
        "query_attempts": [],
        "jsonrpc_examples": {
            "query": query_rpc,
        },
    }

    async with sse_client(sse_url, headers=headers, timeout=10, sse_read_timeout=300) as streams:
        read_stream, write_stream = streams
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_resp = await session.list_tools()
            tool_names = [tool.name for tool in tools_resp.tools]
            result_bundle["tools"] = tool_names

            query_payload: dict[str, Any] = {}
            effective_query_args: dict[str, Any] = {}
            candidate_limit = max(agent_candidate_limit, min_agent_candidate_limit)
            coarse_limit = max(coarse_pool_limit, candidate_limit)

            for attempt_idx in range(max_query_retries + 1):
                effective_query_args = {
                    **query_args_base,
                    "agent_candidate_limit": candidate_limit,
                    "coarse_pool_limit": coarse_limit,
                }
                result_bundle["query_attempts"].append(
                    {
                        "attempt": attempt_idx + 1,
                        "agent_candidate_limit": candidate_limit,
                        "coarse_pool_limit": coarse_limit,
                    }
                )
                try:
                    query_result = await session.call_tool(
                        "query",
                        effective_query_args,
                        read_timeout_seconds=timedelta(seconds=query_read_timeout_seconds),
                    )
                    query_payload = _parse_result_payload(query_result)
                except Exception as exc:
                    query_payload = {
                        "error": {
                            "code": "client_timeout_or_transport_error",
                            "message": str(exc),
                            "retryable": attempt_idx < max_query_retries,
                        }
                    }

                papers = query_payload.get("papers") if isinstance(query_payload, dict) else None
                if isinstance(papers, list) and papers:
                    break

                code, retryable, message, retry_after_seconds = _extract_query_error(query_payload)
                can_retry = attempt_idx < max_query_retries and retryable
                if not can_retry:
                    break

                wait_seconds = max(
                    retry_backoff_seconds * (attempt_idx + 1),
                    float(retry_after_seconds),
                )
                await asyncio.sleep(wait_seconds)

                if code == "query_failed":
                    # Reduce second-stage load gradually under provider pressure.
                    next_candidate = max(min_agent_candidate_limit, candidate_limit - 20)
                    candidate_limit = next_candidate
                    coarse_limit = max(next_candidate, min(coarse_limit, next_candidate * 4))

            result_bundle["effective_query_arguments"] = effective_query_args
            result_bundle["query_result"] = query_payload

            papers = query_payload.get("papers") if isinstance(query_payload, dict) else None
            if isinstance(papers, list) and papers:
                first_paper_id = papers[0].get("paper_id")
                if isinstance(first_paper_id, str) and first_paper_id:
                    read_args = {
                        "paper_id": first_paper_id,
                        "depth": "summary",
                        "allow_acquire": False,
                    }
                    read_rpc = {
                        "jsonrpc": "2.0",
                        "id": "read-1",
                        "method": "tools/call",
                        "params": {
                            "name": "read",
                            "arguments": read_args,
                        },
                    }
                    result_bundle["jsonrpc_examples"]["read"] = read_rpc

                    read_result = await session.call_tool(
                        "read",
                        read_args,
                        read_timeout_seconds=timedelta(seconds=query_read_timeout_seconds),
                    )
                    result_bundle["read_result"] = _parse_result_payload(read_result)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run authenticated ScholarTrace MCP simulation.")
    parser.add_argument(
        "--sse-url",
        default="http://172.17.194.210:8001/sse",
        help="ScholarTrace MCP SSE endpoint URL",
    )
    parser.add_argument(
        "--access-token",
        default="g203-mcp",
        help="MCP bearer token",
    )
    parser.add_argument(
        "--theme-file",
        default="docs/examples/sycophancy_affective_hallucination_research_brief.md",
        help="Theme document path",
    )
    parser.add_argument("--final-limit", type=int, default=10)
    parser.add_argument("--agent-candidate-limit", type=int, default=120)
    parser.add_argument("--coarse-pool-limit", type=int, default=500)
    parser.add_argument("--max-query-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-seconds", type=float, default=8.0)
    parser.add_argument("--min-agent-candidate-limit", type=int, default=40)
    parser.add_argument("--query-read-timeout-seconds", type=int, default=240)
    parser.add_argument(
        "--include-rationale",
        action="store_true",
        default=True,
        help="Include agent rationale in query response",
    )
    parser.add_argument(
        "--output-json",
        default="docs/mcp_query_simulation_result.json",
        help="Output JSON path for full simulation result",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(
        run_mcp_simulation(
            sse_url=args.sse_url,
            access_token=args.access_token,
            theme_file=Path(args.theme_file),
            final_limit=args.final_limit,
            agent_candidate_limit=args.agent_candidate_limit,
            coarse_pool_limit=args.coarse_pool_limit,
            include_rationale=args.include_rationale,
            max_query_retries=args.max_query_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            min_agent_candidate_limit=args.min_agent_candidate_limit,
            query_read_timeout_seconds=args.query_read_timeout_seconds,
            output_json=Path(args.output_json),
        )
    )

    query_result = result.get("query_result", {})
    papers = query_result.get("papers", []) if isinstance(query_result, dict) else []
    print(f"Tools: {result.get('tools', [])}")
    print(f"Total final papers: {len(papers)}")
    for idx, paper in enumerate(papers[:10], start=1):
        title = paper.get("title", "")
        year = paper.get("year", "N/A")
        venue = paper.get("venue", "N/A")
        score = paper.get("agent_score", "N/A")
        print(f"{idx:02d}. [{year}] {title} | {venue} | agent_score={score}")
    print(f"Saved full result JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
