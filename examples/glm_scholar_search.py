#!/usr/bin/env python3
"""
ScholarTrace + BigModel GLM: Intelligent Literature Search

Uses ScholarTrace to retrieve papers and BigModel GLM to analyze them.

Usage:
    # Default: search with sycophancy research brief
    python examples/glm_scholar_search.py

    # Custom query
    python examples/glm_scholar_search.py --query "your research topic"

    # Specify paper count
    python examples/glm_scholar_search.py --limit 100

Environment:
    BIGMODEL_API_KEY — required, BigModel GLM API key
    SCHOLARTRACE_API_URL — optional, defaults to http://localhost:8000
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx


SCHOLARTRACE_URL = os.environ.get(
    "SCHOLARTRACE_API_URL", "http://localhost:8000")
BIGMODEL_API_KEY = os.environ.get(
    "BIGMODEL_API_KEY",
    "d177a2b9dd11494089bfaaae1d313c86.nen8jEX2bgDa6ViW",
)
BIGMODEL_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
DEFAULT_THEME_PATH = "docs/examples/sycophancy_affective_hallucination_research_brief.md"
MAX_FULLTEXT_READS = 20
CONCURRENT_REQUESTS = 5


def load_theme(path: str | None = None) -> str:
    """Load theme document from file or return default."""
    if path and Path(path).exists():
        return Path(path).read_text()
    if Path(DEFAULT_THEME_PATH).exists():
        return Path(DEFAULT_THEME_PATH).read_text()
    return path or "machine learning"


def create_theme(client: httpx.Client, text: str) -> dict:
    """Create a theme on ScholarTrace and return it."""
    resp = client.post(f"{SCHOLARTRACE_URL}/themes", data={"text": text})
    resp.raise_for_status()
    return resp.json()


def launch_retrieval(client: httpx.Client, theme_id: str) -> dict:
    """Launch retrieval job and return job info."""
    resp = client.post(
        f"{SCHOLARTRACE_URL}/retrieval/jobs", data={"theme_id": theme_id}
    )
    resp.raise_for_status()
    return resp.json()


def wait_for_job(client: httpx.Client, job_id: str, timeout: int = 600) -> dict:
    """Poll until job completes."""
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get(f"{SCHOLARTRACE_URL}/retrieval/jobs/{job_id}")
        resp.raise_for_status()
        job = resp.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(2)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def get_papers(client: httpx.Client, theme_id: str, limit: int = 50) -> list[dict]:
    """Get ranked papers for a theme."""
    resp = client.get(
        f"{SCHOLARTRACE_URL}/themes/{theme_id}/papers", params={"limit": limit}
    )
    resp.raise_for_status()
    return resp.json()


def get_paper_detail(client: httpx.Client, paper_id: str) -> dict:
    """Get full paper metadata."""
    resp = client.get(f"{SCHOLARTRACE_URL}/papers/{paper_id}")
    resp.raise_for_status()
    return resp.json()


def get_paper_fulltext(client: httpx.Client, paper_id: str) -> dict:
    """Get full text for a paper."""
    resp = client.get(f"{SCHOLARTRACE_URL}/papers/{paper_id}/fulltext")
    resp.raise_for_status()
    return resp.json()


def fetch_fulltexts_concurrent(
    client: httpx.Client,
    papers: list[dict],
    max_reads: int = MAX_FULLTEXT_READS,
    concurrency: int = CONCURRENT_REQUESTS,
) -> list[dict]:
    """Fetch full texts for top papers with concurrency control.

    Args:
        client: httpx client to use.
        papers: List of paper dicts (must have 'id' key).
        max_reads: Maximum number of full texts to fetch (default 20).
        concurrency: Max concurrent requests (default 5).

    Returns:
        List of (paper_index, fulltext_data) tuples for successful fetches.
    """
    import concurrent.futures

    targets = papers[:max_reads]
    results: list[dict] = []

    def _fetch_one(idx_paper: tuple[int, dict]) -> dict | None:
        idx, paper = idx_paper
        try:
            ft = get_paper_fulltext(client, paper["id"])
            return {"index": idx, "paper_id": paper["id"], "title": paper.get("title", ""), **ft}
        except Exception as e:
            print(f"  [warn] fulltext failed for {paper.get('title', '?')[:40]}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_fetch_one, (i, p)): i
            for i, p in enumerate(targets)
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda r: r["index"])
    return results


def summarize_with_glm(papers: list[dict], theme_text: str) -> str:
    """Use BigModel GLM to summarize and analyze papers."""
    # Prepare paper summaries for the model
    paper_info = []
    for i, p in enumerate(papers[:20], 1):  # Top 20 for summary
        info = f"{i}. {p.get('title', 'Untitled')}\n"
        info += f"   Year: {p.get('year', '?')}, Venue: {p.get('venue', '?')}\n"
        info += f"   Score: {p.get('composite_score', 0):.3f}\n"
        if p.get("abstract"):
            info += f"   Abstract: {p['abstract'][:300]}...\n"
        paper_info.append(info)

    prompt = (
        f"You are an expert research assistant. Analyze the following "
        f"{len(papers)} papers retrieved for this research theme:\n\n"
        f"THEME: {theme_text[:500]}\n\n"
        f"TOP PAPERS:\n{chr(10).join(paper_info)}\n\n"
        f"Please provide:\n"
        f"1. A brief overview of the research landscape (2-3 sentences)\n"
        f"2. The top 10 most important papers with one-line summaries\n"
        f"3. Key research trends you observe\n"
        f"4. Research gaps or opportunities\n\n"
        f"Keep the response concise and actionable."
    )

    resp = httpx.post(
        BIGMODEL_URL,
        headers={
            "Authorization": f"Bearer {BIGMODEL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "glm-5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def interactive_mode(client: httpx.Client, papers: list[dict], theme_id: str):
    """Interactive mode: chat about papers, fetch full text on demand."""
    print("\n" + "=" * 60)
    print(
        "Interactive Mode \u2014 ask about papers, type 'fulltext N' for paper N details"
    )
    print("Type 'quit' to exit")
    print("=" * 60)

    paper_context = "\n".join(
        f"[{i}] {p['title']} "
        f"({p.get('year', '?')}, {p.get('venue', '?')}, "
        f"score={p.get('composite_score', 0):.3f})"
        for i, p in enumerate(papers[:50])
    )

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a research assistant. Here are {len(papers)} papers:\n"
                f"{paper_context}\n\n"
                f"Answer questions about these papers. If you need full text, "
                f"say [FETCH_FULLTEXT N] where N is the paper number."
            ),
        }
    ]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Check if user wants fulltext
        if user_input.lower().startswith("fulltext "):
            try:
                idx = int(user_input.split()[1])
                if 0 <= idx < len(papers):
                    ft = get_paper_fulltext(client, papers[idx]["id"])
                    print(
                        f"\n--- Full text status: {ft.get('access_status', 'unknown')} ---"
                    )
                    for sec in ft.get("sections", []):
                        print(f"\n## {sec.get('section_title', 'Section')}")
                        print(sec.get("text_content", "")[:500])
                else:
                    print(f"Index out of range (0-{len(papers)-1})")
            except (ValueError, IndexError):
                print("Usage: fulltext N")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            resp = httpx.post(
                BIGMODEL_URL,
                headers={
                    "Authorization": f"Bearer {BIGMODEL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "glm-5-turbo",
                    "messages": messages,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": reply})
            print(f"\nGLM: {reply}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="ScholarTrace + BigModel GLM Literature Search"
    )
    parser.add_argument(
        "--query", "-q", help="Custom research query (default: sycophancy brief)"
    )
    parser.add_argument("--theme-file", "-t",
                        help="Path to theme document file")
    parser.add_argument(
        "--limit", "-n", type=int, default=50, help="Number of papers (default: 50)"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive chat mode"
    )
    parser.add_argument(
        "--no-glm", action="store_true", help="Skip GLM summary, just retrieve"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ScholarTrace + BigModel GLM Literature Search")
    print("=" * 60)

    with httpx.Client(timeout=30) as client:
        # Load theme
        if args.query:
            theme_text = args.query
        else:
            theme_text = load_theme(args.theme_file)
        print(f"\nTheme: {theme_text[:100]}...")

        # Create theme
        print("\n[1/4] Creating theme...")
        theme = create_theme(client, theme_text)
        theme_id = theme["id"]
        print(f"  Theme ID: {theme_id}")
        print(f"  Queries generated: {len(theme.get('parsed_queries', []))}")

        # Launch retrieval
        print("\n[2/4] Launching retrieval...")
        job = launch_retrieval(client, theme_id)
        job_id = job["id"]
        print(f"  Job ID: {job_id}")

        # Wait for completion
        print("\n[3/4] Waiting for retrieval to complete...")
        job = wait_for_job(client, job_id)
        if job["status"] == "failed":
            print(f"  ERROR: {job.get('error_message', 'Unknown error')}")
            sys.exit(1)
        print(f"  Found {job.get('result_count', 0)} papers")

        # Get papers
        print(f"\n[4/4] Fetching top {args.limit} papers...")
        papers = get_papers(client, theme_id, args.limit)
        print(f"  Retrieved {len(papers)} papers")

        # Display top papers
        print(f"\n{'=' * 60}")
        print(f"TOP {min(10, len(papers))} PAPERS")
        print(f"{'=' * 60}")
        for i, p in enumerate(papers[:10], 1):
            year = p.get("year", "?")
            venue = (p.get("venue") or "unknown")[:30]
            score = p.get("composite_score", 0)
            title = p.get("title", "Untitled")[:70]
            print(f"  {i:2d}. [{score:.3f}] {title}... ({year}, {venue})")

        # Concurrent full-text fetching (max 20 papers, 5 concurrent)
        print(f"\n{'=' * 60}")
        print(f"FETCHING FULL TEXTS (up to {MAX_FULLTEXT_READS}, {CONCURRENT_REQUESTS} concurrent)")
        print(f"{'=' * 60}")
        fulltexts = fetch_fulltexts_concurrent(client, papers)
        available = sum(1 for ft in fulltexts if ft.get("fulltext_available"))
        print(f"  Retrieved {len(fulltexts)} full texts ({available} available)")

        # GLM summary
        if not args.no_glm and BIGMODEL_API_KEY:
            print(f"\n{'=' * 60}")
            print("GLM ANALYSIS")
            print(f"{'=' * 60}")
            try:
                summary = summarize_with_glm(papers, theme_text)
                print(summary)
            except Exception as e:
                print(f"GLM call failed: {e}")

        # Interactive mode
        if args.interactive:
            interactive_mode(client, papers, theme_id)

        # Export
        export_resp = client.get(
            f"{SCHOLARTRACE_URL}/themes/{theme_id}/export",
            params={"format": "json"},
        )
        if export_resp.status_code == 200:
            export_path = Path("scholartrace_results.json")
            export_path.write_text(
                json.dumps(export_resp.json(), indent=2, ensure_ascii=False)
            )
            print(f"\nResults exported to: {export_path}")


if __name__ == "__main__":
    main()
