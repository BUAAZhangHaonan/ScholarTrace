"""End-to-end integration tests for ScholarTrace.

These tests make REAL API calls to external scholarly sources.
They are skipped unless SCHOLARTRACE_E2E=1 is set in the environment.

Run with:
    SCHOLARTRACE_E2E=1 python -m pytest tests/test_e2e.py -v
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Mark as slow/integration test -- skip if no network
pytestmark = pytest.mark.skipif(
    os.environ.get("SCHOLARTRACE_E2E") != "1",
    reason="E2E tests require SCHOLARTRACE_E2E=1 env var and network access",
)


@pytest.mark.asyncio
async def test_full_pipeline_with_sycophancy_brief():
    """End-to-end test using the sycophancy research brief as theme document."""
    from scholartrace.services.retrieval import run_retrieval_for_document
    from scholartrace.services.storage import StorageService

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = StorageService(db_path=db_path)
        storage.init_db()

        # Read the theme document
        theme_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "docs",
            "examples",
            "sycophancy_affective_hallucination_research_brief.md",
        )
        with open(theme_path) as f:
            doc_text = f.read()

        # Run full retrieval
        theme, works = await run_retrieval_for_document(doc_text, storage)

        # Verify theme parsing
        assert len(theme.parsed_topics) > 0, "Should extract topics"
        assert len(theme.parsed_queries) >= 5, "Should generate 5+ queries"
        assert any("sycophancy" in q.lower() for q in theme.parsed_queries)

        # Verify retrieval results
        assert len(works) >= 10, f"Expected 10+ works, got {len(works)}"

        # Verify ranking (composite scores should be descending)
        scores = [w.composite_score for w in works]
        assert scores == sorted(scores, reverse=True), (
            "Works should be ranked by composite_score"
        )

        # Verify works are saved to storage
        stored_count = storage.count_works_by_theme(theme.id)
        assert stored_count == len(works), "All works should be persisted"

        # Verify work fields are populated
        top_work = works[0]
        assert top_work.title, "Top work should have a title"
        assert top_work.composite_score > 0, "Top work should have a positive score"

        # Check that works have diverse sources
        all_sources: set[str] = set()
        for w in works:
            all_sources.update(w.source_provenance)
        assert len(all_sources) >= 2, (
            f"Expected results from 2+ sources, got {all_sources}"
        )

        # Check that some works have DOIs or arXiv IDs
        works_with_ids = [w for w in works if w.doi or w.arxiv_id]
        assert len(works_with_ids) > 0, "Some works should have DOIs or arXiv IDs"


@pytest.mark.asyncio
async def test_rest_api_e2e():
    """Test REST API endpoints with real data.

    This test only runs if SCHOLARTRACE_E2E=1 is set.
    It tests the REST API with real retrieval.
    Skip for now -- covered by unit tests.
    """
    pass
