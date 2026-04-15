#!/usr/bin/env python3
"""ScholarTrace Verification Script.

Runs the full pipeline with the sycophancy research brief and verifies
that 100+ unique papers are retrieved from multiple sources.

Usage:
    python scripts/verify_scholartrace.py [--theme FILE] [--target N]

Environment variables:
    SCHOLARTRACE_SEMANTIC_SCHOLAR_API_KEY -- optional, for higher rate limits
    SCHOLARTRACE_OPENALEX_MAILTO -- optional, for polite pool
    SCHOLARTRACE_CROSSREF_MAILTO -- optional, for polite pool
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify ScholarTrace retrieves 100+ papers"
    )
    parser.add_argument(
        "--theme",
        default="docs/examples/sycophancy_affective_hallucination_research_brief.md",
        help="Path to theme document",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100,
        help="Minimum number of papers to retrieve (default: 100)",
    )
    args = parser.parse_args()

    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    from scholartrace.services.retrieval import run_retrieval_for_document
    from scholartrace.services.storage import StorageService

    print("=" * 60)
    print("ScholarTrace Verification Script")
    print("=" * 60)

    # Read theme document
    theme_path = Path(args.theme)
    if not theme_path.exists():
        print(f"ERROR: Theme document not found: {theme_path}")
        sys.exit(1)

    doc_text = theme_path.read_text()
    print(f"\nTheme document: {theme_path.name} ({len(doc_text)} chars)")

    # Setup storage
    tmpdir = tempfile.mkdtemp(prefix="scholartrace_verify_")
    db_path = os.path.join(tmpdir, "verify.db")
    storage = StorageService(db_path=db_path)
    storage.init_db()

    # Run retrieval
    print(f"\nRunning retrieval (target: {args.target} papers)...")
    start = time.time()

    try:
        theme, works = await run_retrieval_for_document(doc_text, storage)
    except Exception as e:
        print(f"\nERROR: Retrieval failed: {e}")
        import traceback

        traceback.print_exc()
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(1)

    elapsed = time.time() - start

    # Report results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Theme ID: {theme.id}")
    print(f"Topics extracted: {len(theme.parsed_topics)}")
    print(f"Queries generated: {len(theme.parsed_queries)}")
    print(f"  Queries: {theme.parsed_queries[:5]}...")
    print(f"Unique papers: {len(works)}")
    print(f"Time: {elapsed:.1f}s")

    if len(works) >= args.target:
        print(f"\nPASS: Retrieved {len(works)} papers (target: {args.target})")
    else:
        print(f"\nFAIL: Retrieved {len(works)} papers (target: {args.target})")

    # Source breakdown
    source_counts: dict[str, int] = {}
    for w in works:
        for s in w.source_provenance:
            source_counts[s] = source_counts.get(s, 0) + 1
    print("\nSource breakdown:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")

    # Top 10 papers
    print("\nTop 10 papers:")
    for i, w in enumerate(works[:10], 1):
        year_str = str(w.year) if w.year else "?"
        venue_str = w.venue or "unknown"
        title_display = w.title[:80] if len(w.title) > 80 else w.title
        print(
            f"  {i}. [{w.composite_score:.3f}] {title_display}... "
            f"({year_str}, {venue_str[:20]})"
        )

    # Year distribution
    year_dist: dict[int, int] = {}
    for w in works:
        if w.year:
            year_dist[w.year] = year_dist.get(w.year, 0) + 1
    print("\nYear distribution:")
    for year in sorted(year_dist.keys()):
        bar = "#" * year_dist[year]
        print(f"  {year}: {bar} ({year_dist[year]})")

    # Fulltext availability
    fulltext_count = sum(1 for w in works if w.fulltext_available)
    print(f"\nFulltext available: {fulltext_count}/{len(works)}")

    # Save results
    results_path = os.path.join(tmpdir, "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "theme_id": theme.id,
                "total_papers": len(works),
                "target": args.target,
                "passed": len(works) >= args.target,
                "elapsed_seconds": elapsed,
                "top_papers": [
                    {
                        "title": w.title,
                        "year": w.year,
                        "score": w.composite_score,
                        "doi": w.doi,
                        "arxiv_id": w.arxiv_id,
                    }
                    for w in works[:20]
                ],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_path}")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Exit code
    sys.exit(0 if len(works) >= args.target else 1)


if __name__ == "__main__":
    asyncio.run(main())
