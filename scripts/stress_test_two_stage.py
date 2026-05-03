#!/usr/bin/env python3
"""
Stress test: 10 concurrent MCP query requests via SSE.

Each request targets:
  - agent_candidate_limit=200 (coarse pool)
  - final_limit=20
  - two_stage_enabled=true

Expected load profile per active request:
  Stage 1: 200/10 = 20 batches × 20 concurrent glm-4.6 calls
  Stage 2: 1 glm-5-turbo call with 128K context

With Semaphore(3), at most 3 requests run concurrently:
  Max concurrent glm-4.6 calls: 3 × 20 = 60
  Max concurrent glm-5-turbo calls: 3
"""

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# --- Configuration ---
MCP_URL = "http://127.0.0.1:8001/sse"
AUTH_TOKEN = os.environ.get("SCHOLARTRACE_ACCESS_TOKEN")
if not AUTH_TOKEN:
    print("ERROR: SCHOLARTRACE_ACCESS_TOKEN not set", file=sys.stderr)
    sys.exit(1)
NUM_CONCURRENT = 10
AGENT_CANDIDATE_LIMIT = 200
FINAL_LIMIT = 20
TIMEOUT_PER_QUERY = 600  # 10 min per query max

# Diverse research topics to avoid cache hits
TOPICS = [
    "transformer attention mechanisms for efficient inference in large language models",
    "graph neural networks for molecular property prediction and drug discovery",
    "reinforcement learning from human feedback alignment techniques",
    "diffusion models for high-resolution image generation and editing",
    "neural architecture search and automated machine learning",
    "federated learning privacy preserving distributed training",
    "multimodal learning vision language models contrastive alignment",
    "causal inference and treatment effect estimation from observational data",
    "knowledge distillation and model compression for edge deployment",
    "prompt engineering and in-context learning capabilities of large language models",
]


@dataclass
class QueryResult:
    query_id: int
    topic: str
    success: bool = False
    error: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    theme_id: str = ""
    total_final: int = 0
    paper_count: int = 0
    first_paper_title: str = ""
    first_paper_score: float = 0.0
    first_paper_agent_score: float = 0.0
    papers_with_agent_score: int = 0
    stage1_time_s: float = 0.0
    stage2_time_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time


async def run_single_query(query_id: int, topic: str) -> QueryResult:
    """Run a single MCP query and return results."""
    result = QueryResult(
        query_id=query_id,
        topic=topic,
        start_time=time.monotonic(),
    )

    try:
        async with asyncio.timeout(TIMEOUT_PER_QUERY):
            async with sse_client(MCP_URL, headers={"Authorization": f"Bearer {AUTH_TOKEN}"}) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    resp = await session.call_tool("query", {
                        "theme_document": topic,
                        "final_limit": FINAL_LIMIT,
                        "agent_candidate_limit": AGENT_CANDIDATE_LIMIT,
                    })

                    result.end_time = time.monotonic()
                    result.success = True

                    for content in resp.content:
                        if not hasattr(content, "text"):
                            continue
                        data = json.loads(content.text)

                        result.theme_id = data.get("theme_id", "")
                        result.total_final = data.get("total_final", 0)
                        result.paper_count = len(data.get("papers", []))

                        papers = data.get("papers", [])
                        if papers:
                            p = papers[0]
                            result.first_paper_title = p.get("title", "")[:100]
                            result.first_paper_score = p.get("composite_score", 0)
                            result.first_paper_agent_score = p.get("agent_score", 0)
                            result.papers_with_agent_score = sum(
                                1 for pp in papers if pp.get("agent_score") is not None
                            )

    except asyncio.TimeoutError:
        result.end_time = time.monotonic()
        result.error = "TIMEOUT"
    except Exception as e:
        result.end_time = time.monotonic()
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


async def main():
    print("=" * 80)
    print("TWO-STAGE PIPELINE STRESS TEST")
    print("=" * 80)
    print(f"  Concurrent requests: {NUM_CONCURRENT}")
    print(f"  Candidates per request: {AGENT_CANDIDATE_LIMIT}")
    print(f"  Final limit per request: {FINAL_LIMIT}")
    print(f"  Timeout per request: {TIMEOUT_PER_QUERY}s")
    print(f"  MCP URL: {MCP_URL}")
    print()

    wall_start = time.monotonic()
    print(f"[{time.strftime('%H:%M:%S')}] Launching {NUM_CONCURRENT} concurrent queries...")

    # Launch all queries concurrently
    tasks = [
        run_single_query(i, TOPICS[i % len(TOPICS)])
        for i in range(NUM_CONCURRENT)
    ]
    results = await asyncio.gather(*tasks)

    wall_end = time.monotonic()
    wall_duration = wall_end - wall_start

    # --- Analysis ---
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    print(f"\n  Total: {len(results)}  |  Success: {len(successes)}  |  Failed: {len(failures)}")
    print(f"  Wall time: {wall_duration:.1f}s")
    print()

    # Per-query details
    print("-" * 80)
    print(f"{'ID':>3}  {'Status':<8}  {'Time':>7}  {'Papers':>6}  {'AgentSc':>8}  {'Theme ID':<38}  Error")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x.start_time):
        status = "OK" if r.success else "FAIL"
        print(
            f"{r.query_id:>3}  {status:<8}  {r.duration_s:>6.1f}s  {r.paper_count:>6}  "
            f"{r.papers_with_agent_score:>8}  {r.theme_id:<38}  {r.error[:40] if r.error else ''}"
        )

    # Aggregate stats
    if successes:
        durations = [r.duration_s for r in successes]
        paper_counts = [r.paper_count for r in successes]
        agent_scores = [r.papers_with_agent_score for r in successes]

        print()
        print("-" * 80)
        print("AGGREGATE STATISTICS (successful queries)")
        print("-" * 80)
        print(f"  Duration:   min={min(durations):.1f}s  max={max(durations):.1f}s  "
              f"avg={sum(durations)/len(durations):.1f}s  median={sorted(durations)[len(durations)//2]:.1f}s")
        print(f"  Papers:     min={min(paper_counts)}  max={max(paper_counts)}  "
              f"avg={sum(paper_counts)/len(paper_counts):.1f}  target={FINAL_LIMIT}")
        print(f"  AgentScore: min={min(agent_scores)}  max={max(agent_scores)}  "
              f"avg={sum(agent_scores)/len(agent_scores):.1f}")
        print(f"  Throughput: {len(successes) / wall_duration * 60:.2f} queries/min")

    if failures:
        print()
        print("-" * 80)
        print("FAILED QUERIES")
        print("-" * 80)
        for r in failures:
            print(f"  Query {r.query_id}: {r.error}")

    # Pass/fail verdict
    print()
    print("=" * 80)
    if len(successes) == NUM_CONCURRENT and all(r.paper_count > 0 for r in successes):
        print("VERDICT: PASS — all queries returned papers")
    elif len(successes) == NUM_CONCURRENT:
        print("VERDICT: PARTIAL PASS — all queries succeeded but some returned 0 papers")
    else:
        print(f"VERDICT: FAIL — {len(failures)}/{NUM_CONCURRENT} queries failed")
    print("=" * 80)

    # Save raw results
    output_path = "docs/examples/results/stress_test_two_stage_result.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": {
                    "num_concurrent": NUM_CONCURRENT,
                    "agent_candidate_limit": AGENT_CANDIDATE_LIMIT,
                    "final_limit": FINAL_LIMIT,
                },
                "wall_duration_s": round(wall_duration, 2),
                "total": len(results),
                "success": len(successes),
                "failed": len(failures),
                "results": [
                    {
                        "query_id": r.query_id,
                        "topic": r.topic,
                        "success": r.success,
                        "duration_s": round(r.duration_s, 2),
                        "paper_count": r.paper_count,
                        "papers_with_agent_score": r.papers_with_agent_score,
                        "total_final": r.total_final,
                        "theme_id": r.theme_id,
                        "error": r.error,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
