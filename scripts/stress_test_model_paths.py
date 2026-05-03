#!/usr/bin/env python3
"""Stress test for ScholarTrace model paths.

Sends N concurrent MCP query requests to a debug server on port 8002,
measures per-query timing, success rate, and aggregate stats.
Tests both deepseek_flash and glm_extended model paths.

Usage:
    SCHOLARTRACE_MODEL_PATH=deepseek_flash python scripts/stress_test_model_paths.py
    SCHOLARTRACE_MODEL_PATH=glm_extended python scripts/stress_test_model_paths.py
    python scripts/stress_test_model_paths.py  # default path
"""

from __future__ import annotations

import asyncio
import json
import time
import sys
import os

import httpx


class SSEClient:
    """Manages a single MCP SSE session: GET /sse kept open, POSTs for messages."""

    def __init__(self, base_url: str, auth_token: str, timeout: float):
        self.base_url = base_url
        self.auth_token = auth_token
        self.timeout = timeout
        self._sse_client: httpx.AsyncClient | None = None  # for SSE GET (long-lived)
        self._post_client: httpx.AsyncClient | None = None  # for POSTs (separate connections)
        self._sse_stream_ctx = None
        self._sse_resp: httpx.Response | None = None
        self._session_id: str | None = None
        self._post_url: str | None = None
        self._line_iter = None

    async def connect(self):
        """Open SSE stream, extract session_id from endpoint event."""
        # Separate clients so POSTs don't interfere with the SSE stream
        self._sse_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout + 30),
        )
        self._post_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout + 30),
        )
        self._sse_stream_ctx = self._sse_client.stream(
            "GET",
            f"{self.base_url}/sse",
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "Accept": "text/event-stream",
            },
        )
        self._sse_resp = await self._sse_stream_ctx.__aenter__()
        if self._sse_resp.status_code != 200:
            raise RuntimeError(f"SSE connect returned {self._sse_resp.status_code}")

        self._line_iter = self._sse_resp.aiter_lines()

        # Read the first "endpoint" event
        event_type, data = await self._read_next_event()
        if event_type != "endpoint":
            raise RuntimeError(f"Expected endpoint event, got {event_type}: {data}")

        # Parse session_id
        if "session_id=" not in data:
            raise RuntimeError(f"No session_id in endpoint: {data}")
        self._session_id = data.split("session_id=")[1].split("&")[0]
        self._post_url = f"{self.base_url}{data}"

    async def _read_next_event(self) -> tuple[str | None, str | None]:
        """Read the next complete SSE event from the stream."""
        event_type = None
        data_parts: list[str] = []
        async for line in self._line_iter:
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_parts.append(line[len("data:"):].strip())
            elif line == "" and event_type is not None:
                return event_type, "\n".join(data_parts)
        return None, None

    async def _wait_for_message(self, msg_id: int | None, timeout: float) -> dict | None:
        """Wait for a specific JSON-RPC message by id from SSE stream."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            event_type, data = await self._read_next_event()
            if event_type is None:
                return None  # stream ended
            if event_type != "message" or data is None:
                continue
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            if msg_id is None or msg.get("id") == msg_id:
                return msg
            # Not the one we want, keep reading

    async def _post(self, payload: dict | None, raw_jsonrpc: str | None = None) -> int:
        """POST a message to the server. Returns HTTP status code."""
        if raw_jsonrpc:
            content = raw_jsonrpc.encode()
        else:
            self._req_counter += 1
            payload["id"] = self._req_counter
            content = json.dumps(payload).encode()

        resp = await self._client.post(
            self._post_url,
            content=content,
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            },
        )
        return resp.status_code, self._req_counter if payload else None

    async def initialize(self):
        """Send MCP initialize + initialized notification."""
        init_id = 1
        init_payload = json.dumps({
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "stress-test", "version": "1.0.0"},
            },
        })
        resp = await self._post_client.post(
            self._post_url,
            content=init_payload.encode(),
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            },
        )
        if resp.status_code != 202:
            raise RuntimeError(f"Initialize POST returned {resp.status_code}: {resp.text[:200]}")

        # Wait for initialize response
        result = await self._wait_for_message(init_id, timeout=30)
        if result is None:
            raise RuntimeError("No initialize response from SSE")
        if "error" in result:
            raise RuntimeError(f"Initialize error: {result['error']}")

        # Send initialized notification (no id, no response expected)
        notif = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        await self._post_client.post(
            self._post_url,
            content=notif.encode(),
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            },
        )

    async def call_tool(self, tool_name: str, arguments: dict, req_id: int) -> dict:
        """Call an MCP tool and wait for the response."""
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        })
        resp = await self._post_client.post(
            self._post_url,
            content=payload.encode(),
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            },
        )
        if resp.status_code != 202:
            raise RuntimeError(f"Tool call POST returned {resp.status_code}")

        return await self._wait_for_message(req_id, timeout=self.timeout)

    async def close(self):
        """Clean up the SSE connection."""
        if self._sse_stream_ctx is not None:
            try:
                await self._sse_stream_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        if self._sse_client is not None:
            await self._sse_client.aclose()
        if self._post_client is not None:
            await self._post_client.aclose()

SSE_PORT = int(os.environ.get("SCHOLARTRACE_STRESS_PORT", "8002"))
SSE_HOST = os.environ.get("SCHOLARTRACE_STRESS_HOST", "127.0.0.1")
ACCESS_TOKEN = os.environ.get("SCHOLARTRACE_ACCESS_TOKEN")
if not ACCESS_TOKEN:
    print("ERROR: SCHOLARTRACE_ACCESS_TOKEN not set", file=sys.stderr)
    sys.exit(1)
CONCURRENT_QUERIES = int(os.environ.get("SCHOLARTRACE_STRESS_CONCURRENT", "5"))
TARGET_SECONDS = 300  # 5 min target per query
BASE_URL = f"http://{SSE_HOST}:{SSE_PORT}"

THEME_DOCUMENTS = [
    "Transformer-based models for code generation and program synthesis, focusing on recent advances in large language models for code understanding, completion, and generation tasks.",
    "Graph neural networks for molecular property prediction and drug discovery, including message passing networks and equivariant architectures.",
    "Reinforcement learning from human feedback (RLHF) for aligning large language models with human preferences and values.",
    "Diffusion models for image and video generation, including latent diffusion, classifier-free guidance, and accelerated sampling methods.",
    "Retrieval-augmented generation (RAG) combining knowledge retrieval with language model generation for factually grounded text.",
]


async def send_query(
    query_id: int,
    theme_doc: str,
) -> dict:
    """Send a single MCP query via proper SSE protocol and return timing info."""
    start = time.monotonic()

    sse = SSEClient(BASE_URL, ACCESS_TOKEN, TARGET_SECONDS)
    try:
        # Connect and get session
        await sse.connect()

        # Initialize MCP session
        await sse.initialize()

        # Send the query
        req_id = query_id * 1000 + 10
        result = await sse.call_tool(
            "query",
            {
                "theme_document": theme_doc,
                "final_limit": 20,
                "agent_candidate_limit": 100,
            },
            req_id,
        )

        elapsed = time.monotonic() - start

        if result is None:
            return {
                "query_id": query_id,
                "elapsed": elapsed,
                "papers": 0,
                "status": "error: no query response from SSE",
            }

        if "error" in result:
            err = result["error"]
            return {
                "query_id": query_id,
                "elapsed": elapsed,
                "papers": 0,
                "status": f"error: MCP error {err.get('code')}: {err.get('message', '')[:100]}",
            }

        # Parse papers count from result
        papers = 0
        result_content = result.get("result", {})
        if isinstance(result_content, dict):
            content = result_content.get("content", [])
            for item in content:
                text = item.get("text", "")
                if "total_final" in text:
                    try:
                        data = json.loads(text)
                        papers = data.get("total_final", 0)
                    except (json.JSONDecodeError, TypeError):
                        pass

        return {
            "query_id": query_id,
            "elapsed": elapsed,
            "papers": papers,
            "status": "ok",
        }

    except Exception as exc:
        elapsed = time.monotonic() - start
        import traceback
        tb = traceback.format_exc()
        return {
            "query_id": query_id,
            "elapsed": elapsed,
            "papers": 0,
            "status": f"error: {exc}\n{tb}",
        }
    finally:
        await sse.close()


async def run_stress_test():
    model_path = os.environ.get("SCHOLARTRACE_MODEL_PATH", "default")
    print(f"=== ScholarTrace Stress Test ===")
    print(f"Model path: {model_path}")
    print(f"Target: {SSE_HOST}:{SSE_PORT}")
    print(f"Concurrent queries: {CONCURRENT_QUERIES}")
    print(f"Target per-query: <{TARGET_SECONDS}s")
    print()

    # Build query list
    queries = []
    for i in range(CONCURRENT_QUERIES):
        theme = THEME_DOCUMENTS[i % len(THEME_DOCUMENTS)]
        queries.append((i + 1, theme))

    print(f"Launching {len(queries)} concurrent queries...")
    overall_start = time.monotonic()

    tasks = [send_query(qid, theme) for qid, theme in queries]
    results = await asyncio.gather(*tasks)

    overall_elapsed = time.monotonic() - overall_start

    # Report
    print(f"\n=== Results (model_path={model_path}) ===")
    print(f"Total wall time: {overall_elapsed:.2f}s")
    print()

    ok_count = 0
    total_papers = 0
    times = []
    for r in sorted(results, key=lambda x: x["query_id"]):
        status_marker = "OK" if r["status"] == "ok" else "FAIL"
        extra = "" if r["status"] == "ok" else f" ({r['status']})"
        print(
            f"  query {r['query_id']:2d}: {r['elapsed']:6.1f}s  "
            f"papers={r['papers']:3d}  [{status_marker}]{extra}"
        )
        if r["status"] == "ok":
            ok_count += 1
            total_papers += r["papers"]
            times.append(r["elapsed"])

    print()
    print(f"Success rate: {ok_count}/{len(results)} ({100*ok_count/len(results):.0f}%)")
    if times:
        print(f"Avg time (ok):  {sum(times)/len(times):.1f}s")
        print(f"Min time (ok):  {min(times):.1f}s")
        print(f"Max time (ok):  {max(times):.1f}s")
        print(f"Total papers:   {total_papers}")
        under_target = sum(1 for t in times if t < TARGET_SECONDS)
        print(f"Under {TARGET_SECONDS}s target: {under_target}/{len(times)}")

    # Exit code: 0 if all succeeded and under target
    if ok_count == len(results) and all(t < TARGET_SECONDS for t in times):
        print("\nPASS: All queries succeeded within target time.")
        return 0
    else:
        print(f"\nFAIL: {len(results) - ok_count} failures, {sum(1 for t in times if t >= TARGET_SECONDS)} over target.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_stress_test()))
