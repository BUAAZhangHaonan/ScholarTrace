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
import urllib.request
import urllib.error
import sys
import os

SSE_PORT = int(os.environ.get("SCHOLARTRACE_STRESS_PORT", "8002"))
SSE_HOST = os.environ.get("SCHOLARTRACE_STRESS_HOST", "127.0.0.1")
ACCESS_TOKEN = os.environ.get("SCHOLARTRACE_ACCESS_TOKEN", "g203-mcp")
CONCURRENT_QUERIES = int(os.environ.get("SCHOLARTRACE_STRESS_CONCURRENT", "5"))
TARGET_SECONDS = 300  # 5 min target per query

THEME_DOCUMENTS = [
    "Transformer-based models for code generation and program synthesis, focusing on recent advances in large language models for code understanding, completion, and generation tasks.",
    "Graph neural networks for molecular property prediction and drug discovery, including message passing networks and equivariant architectures.",
    "Reinforcement learning from human feedback (RLHF) for aligning large language models with human preferences and values.",
    "Diffusion models for image and video generation, including latent diffusion, classifier-free guidance, and accelerated sampling methods.",
    "Retrieval-augmented generation (RAG) combining knowledge retrieval with language model generation for factually grounded text.",
]


async def send_query(
    session_id: str,
    theme_doc: str,
    query_id: int,
) -> dict:
    """Send a single MCP query via SSE and return timing info."""
    start = time.monotonic()
    url = f"http://{SSE_HOST}:{SSE_PORT}/messages"

    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": query_id,
        "method": "tools/call",
        "params": {
            "name": "query",
            "arguments": {
                "theme_document": theme_doc,
                "final_limit": 20,
                "agent_candidate_limit": 100,
            },
        },
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=TARGET_SECONDS) as resp:
            body = resp.read().decode()
        elapsed = time.monotonic() - start
        result = json.loads(body) if body else {}
        papers = 0
        if "result" in result and isinstance(result["result"], dict):
            content = result["result"].get("content", [])
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
        return {
            "query_id": query_id,
            "elapsed": elapsed,
            "papers": 0,
            "status": f"error: {exc}",
        }


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
        queries.append((f"q{i+1}", theme, i + 1))

    print(f"Launching {len(queries)} concurrent queries...")
    overall_start = time.monotonic()

    tasks = [send_query(sid, theme, qid) for sid, theme, qid in queries]
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
