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
    SCHOLARTRACE_BIGMODEL_API_KEY — required, BigModel GLM API key
    SCHOLARTRACE_API_URL — optional, defaults to http://localhost:9000
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

from scholartrace.services.prompt_budget import DEFAULT_PROMPT_BUDGET, PromptBudget


SCHOLARTRACE_URL = os.environ.get(
    "SCHOLARTRACE_API_URL", "http://localhost:9000")
DEFAULT_THEME_PATH = "docs/examples/sycophancy_affective_hallucination_research_brief.md"
MAX_FULLTEXT_READS = 20
CONCURRENT_REQUESTS = 5


def get_bigmodel_api_key() -> str:
    """Return the configured BigModel API key or fail closed."""
    api_key = os.environ.get("SCHOLARTRACE_BIGMODEL_API_KEY", "").strip()
    if api_key:
        return api_key
    raise RuntimeError(
        "SCHOLARTRACE_BIGMODEL_API_KEY must be set to use GLM analysis",
    )


def get_bigmodel_url() -> str:
    return os.environ.get(
        "SCHOLARTRACE_BIGMODEL_BASE_URL",
        "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
    )


def get_bigmodel_model() -> str:
    return os.environ.get("SCHOLARTRACE_BIGMODEL_MODEL", "glm-5-turbo")


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
    """Read cached full-text state for a paper."""
    resp = client.get(f"{SCHOLARTRACE_URL}/papers/{paper_id}/fulltext")
    resp.raise_for_status()
    return resp.json()


def acquire_paper_fulltext(client: httpx.Client, paper_id: str) -> dict:
    """Explicitly acquire full text for a paper."""
    resp = client.post(f"{SCHOLARTRACE_URL}/papers/{paper_id}/fulltext/acquire")
    resp.raise_for_status()
    return resp.json()


def ensure_paper_fulltext(client: httpx.Client, paper: dict) -> dict:
    """Read cache, acquire on miss, then re-read cached state."""
    cached = get_paper_fulltext(client, paper["id"])
    if not cached.get("needs_acquisition"):
        return {"title": paper.get("title", ""), **cached}

    acquire_paper_fulltext(client, paper["id"])
    refreshed = get_paper_fulltext(client, paper["id"])
    return {"title": paper.get("title", ""), **refreshed}


def ensure_fulltexts_concurrent(
    client: httpx.Client,
    papers: list[dict],
    max_reads: int = MAX_FULLTEXT_READS,
    concurrency: int = CONCURRENT_REQUESTS,
) -> list[dict]:
    """Ensure cached full text for top papers with concurrency control.

    Args:
        client: httpx client to use.
        papers: List of paper dicts (must have 'id' key).
        max_reads: Maximum number of full texts to fetch (default 20).
        concurrency: Max concurrent requests (default 5).

    Returns:
        List of cached-state payloads after optional explicit acquire.
    """
    import concurrent.futures

    targets = papers[:max_reads]
    results: list[dict] = []

    def _fetch_one(idx_paper: tuple[int, dict]) -> dict | None:
        idx, paper = idx_paper
        try:
            ft = ensure_paper_fulltext(client, paper)
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


def fetch_fulltexts_concurrent(
    client: httpx.Client,
    papers: list[dict],
    max_reads: int = MAX_FULLTEXT_READS,
    concurrency: int = CONCURRENT_REQUESTS,
) -> list[dict]:
    """Backward-compatible alias for the explicit acquire flow."""
    return ensure_fulltexts_concurrent(client, papers, max_reads, concurrency)


def _paper_summary_line(paper: dict, index: int, budget: PromptBudget) -> str:
    abstract = budget.truncate_text(paper.get("abstract") or "", 700)
    return (
        f"{index}. {paper.get('title', 'Untitled')}\n"
        f"   Year: {paper.get('year', '?')}, Venue: {paper.get('venue', '?')}\n"
        f"   Score: {paper.get('composite_score', 0):.3f}\n"
        f"   Abstract: {abstract}\n"
    )


def build_summary_messages(
    papers: list[dict],
    theme_text: str,
    budget: PromptBudget = DEFAULT_PROMPT_BUDGET,
) -> list[dict]:
    """Build one bounded GLM summary request from the first packed batch."""
    batches = build_summary_request_batches(papers, theme_text, budget=budget)
    return batches[0] if batches else [{"role": "user", "content": ""}]


def build_summary_request_batches(
    papers: list[dict],
    theme_text: str,
    budget: PromptBudget = DEFAULT_PROMPT_BUDGET,
) -> list[list[dict]]:
    """Build bounded GLM summary requests that cover all packed paper batches."""
    theme_text = budget.truncate_text(theme_text, 4_000)
    intro = (
        f"You are an expert research assistant. Analyze the following "
        f"{len(papers)} papers retrieved for this research theme:\n\n"
        f"THEME: {theme_text}\n\n"
        f"TOP PAPERS:\n"
    )
    outro = (
        "\n\nPlease provide:\n"
        "1. A brief overview of the research landscape (2-3 sentences)\n"
        "2. The top 10 most important papers with one-line summaries\n"
        "3. Key research trends you observe\n"
        "4. Research gaps or opportunities\n\n"
        "Keep the response concise and actionable."
    )
    paper_lines = [
        _paper_summary_line(paper, index, budget)
        for index, paper in enumerate(papers, 1)
    ]
    batches = budget.pack_items(
        paper_lines,
        fixed_messages=[],
        prefix=intro,
        suffix=outro,
    )
    return [
        [{"role": "user", "content": intro + "\n".join(batch) + outro}]
        for batch in batches
    ] or [[{"role": "user", "content": intro + outro}]]


def build_interactive_system_prompt(
    papers: list[dict],
    budget: PromptBudget = DEFAULT_PROMPT_BUDGET,
) -> str:
    intro = (
        f"You are a research assistant. Here are {len(papers)} papers:\n"
    )
    outro = (
        "\n\nAnswer questions about these papers. "
        "Use cached full-text status when available. "
        "If the user wants network retrieval, tell them to run 'acquire N'."
    )
    paper_lines = [
        (
            f"[{i}] {p['title']} "
            f"({p.get('year', '?')}, {p.get('venue', '?')}, "
            f"score={p.get('composite_score', 0):.3f})"
        )
        for i, p in enumerate(papers[:50])
    ]
    batches = budget.pack_items(
        paper_lines,
        prefix=intro,
        suffix=outro,
    )
    return intro + "\n".join(batches[0]) + outro if batches else intro + outro


def prepare_chat_messages(
    messages: list[dict],
    budget: PromptBudget = DEFAULT_PROMPT_BUDGET,
) -> list[dict]:
    return budget.trim_messages(messages, preserve=1)


def summarize_with_glm(
    papers: list[dict],
    theme_text: str,
    budget: PromptBudget = DEFAULT_PROMPT_BUDGET,
) -> str:
    """Use BigModel GLM to summarize and analyze papers."""
    api_key = get_bigmodel_api_key()
    batch_messages = build_summary_request_batches(papers, theme_text, budget=budget)
    batch_findings: list[str] = []

    for messages in batch_messages:
        resp = httpx.post(
            get_bigmodel_url(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": get_bigmodel_model(),
                "messages": messages,
                "temperature": 0.7,
            },
            timeout=60,
        )
        resp.raise_for_status()
        batch_findings.append(resp.json()["choices"][0]["message"]["content"])

    if len(batch_findings) == 1:
        return batch_findings[0]

    synthesis_messages = [
        {
            "role": "user",
            "content": budget.truncate_text(
                "You are an expert research assistant. Merge the batch findings below into one final answer.\n\n"
                f"Batch Findings:\n{chr(10).join(f'- {finding}' for finding in batch_findings)}",
                budget.max_input_tokens - 512,
            ),
        }
    ]
    resp = httpx.post(
        get_bigmodel_url(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": get_bigmodel_model(),
            "messages": synthesis_messages,
            "temperature": 0.7,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def interactive_mode(client: httpx.Client, papers: list[dict], theme_id: str):
    """Interactive mode: chat about papers and manage explicit full-text acquire."""
    api_key = get_bigmodel_api_key()

    print("\n" + "=" * 60)
    print("Interactive Mode - ask about papers, 'fulltext N' reads cache, 'acquire N' fetches")
    print("Type 'quit' to exit")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": build_interactive_system_prompt(papers),
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

        if user_input.lower().startswith("acquire "):
            try:
                idx = int(user_input.split()[1])
                if 0 <= idx < len(papers):
                    ft = ensure_paper_fulltext(client, papers[idx])
                    print(
                        f"\n--- Full text status: {ft.get('access_status', 'unknown')} ---"
                    )
                    for sec in ft.get("sections", []):
                        print(f"\n## {sec.get('section_title', 'Section')}")
                        print(sec.get("text_content", "")[:500])
                else:
                    print(f"Index out of range (0-{len(papers)-1})")
            except (ValueError, IndexError):
                print("Usage: acquire N")
            continue

        messages.append({"role": "user", "content": user_input})
        messages = prepare_chat_messages(messages)

        try:
            resp = httpx.post(
                get_bigmodel_url(),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": get_bigmodel_model(),
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
        print(
            f"CHECKING CACHE AND ACQUIRING FULL TEXTS "
            f"(up to {MAX_FULLTEXT_READS}, {CONCURRENT_REQUESTS} concurrent)"
        )
        print(f"{'=' * 60}")
        fulltexts = ensure_fulltexts_concurrent(client, papers)
        available = sum(1 for ft in fulltexts if ft.get("fulltext_available"))
        print(f"  Checked {len(fulltexts)} papers ({available} cached full texts available)")

        # GLM summary
        if not args.no_glm:
            print(f"\n{'=' * 60}")
            print("GLM ANALYSIS")
            print(f"{'=' * 60}")
            try:
                summary = summarize_with_glm(papers, theme_text)
                print(summary)
            except Exception as e:
                print(f"GLM call failed: {e}")
                sys.exit(1)

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
