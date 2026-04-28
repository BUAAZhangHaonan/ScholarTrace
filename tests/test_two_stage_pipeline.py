"""Stress tests for the two-stage pipeline.

Covers:
1. Unit tests: Stage 1 scoring, Stage 2 selection, full two-stage pipeline
2. Regression: single-stage pipeline still works
3. Concurrency: 3 simultaneous queries
4. Edge cases: empty results, all-LLM-fail fallback, large candidate pools
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from scholartrace.config import Settings
from scholartrace.deepxiv.agent import DeepXivAgentError
from scholartrace.models.schemas import (
    RawCandidate,
    SourceName,
    Theme,
    Work,
)
from scholartrace.services.retrieval import (
    _compute_stage1_scores,
    _estimate_papers_tokens,
    _run_stage1_scoring,
    _run_stage2_selection,
    _run_two_stage_pipeline,
    run_query_pipeline,
)
from scholartrace.services.storage import StorageService

import scholartrace.services.retrieval as retrieval_mod


@pytest.fixture(autouse=True)
def _reset_llm_semaphore():
    """Reset the global LLM semaphore before each test to avoid state leakage."""
    retrieval_mod._llm_semaphore = None
    yield
    retrieval_mod._llm_semaphore = None


# ---------------------------------------------------------------------------
# Base mock agent — provides static methods that retrieval.py calls on the class
# ---------------------------------------------------------------------------
class _FakeAgentBase:
    """Base class for mock agents. Provides static prompt/snippet builders."""

    @staticmethod
    def build_stage1_prompt() -> str:
        return "FAKE STAGE1 PROMPT"

    @staticmethod
    def build_stage2_prompt(max_select: int = 20) -> str:
        return f"FAKE STAGE2 PROMPT (max_select={max_select})"

    @staticmethod
    def paper_snippet(paper: dict) -> str:
        return f"{paper.get('title', '')} {paper.get('abstract', '')}"

    async def close(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_candidate(
    title: str = "Test Paper",
    doi: str | None = None,
    arxiv_id: str | None = None,
    s2_id: str | None = None,
    openalex_id: str | None = None,
    year: int | None = 2024,
    authors: list[str] | None = None,
    abstract: str | None = None,
    citation_count: int = 0,
    venue: str | None = None,
) -> RawCandidate:
    return RawCandidate(
        title=title,
        doi=doi,
        arxiv_id=arxiv_id,
        s2_id=s2_id,
        openalex_id=openalex_id,
        source=SourceName.OPENALEX,
        year=year,
        authors=authors or ["Author One"],
        abstract=abstract,
        citation_count=citation_count,
        venue=venue,
    )


def _make_mock_connector(candidates: list[RawCandidate]) -> AsyncMock:
    connector = AsyncMock()
    connector.source_name = "openalex"
    connector.search = AsyncMock(return_value=candidates)
    connector.close = AsyncMock(return_value=None)
    return connector


def _tmp_storage(tmp_path: Path) -> StorageService:
    db_path = tmp_path / "test.db"
    storage = StorageService(db_path)
    storage.init_db()
    return storage


def _run(coro):
    return asyncio.run(coro)


def _many_candidates(n: int) -> list[RawCandidate]:
    """Generate n candidates with unique titles that survive fuzzy dedup.

    Each title includes a unique SHA256 prefix so that token_sort_ratio
    stays well below the 0.85 dedup threshold even for same-year papers.
    """
    return [
        _make_candidate(
            title=(
                f"{hashlib.sha256(str(i).encode()).hexdigest()[:12]}: "
                f"Research Paper {i}"
            ),
            doi=f"10.1/{i}",
            openalex_id=f"W{i}",
            year=2020 + (i % 5),
            citation_count=i * 10,
            abstract=f"Abstract for paper {i}. " * 10,  # ~250 chars
            venue="NeurIPS" if i % 3 == 0 else "ICML" if i % 3 == 1 else "AAAI",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Unit tests for Stage 1 helpers
# ---------------------------------------------------------------------------
class TestComputeStage1Scores:
    def test_basic_scoring(self):
        results = [
            {"index": 0, "relevance": 9, "recency": 8, "novelty": 7, "quality": 6},
            {"index": 1, "relevance": 5, "recency": 4, "novelty": 3, "quality": 2},
        ]
        scored = _compute_stage1_scores(results)
        assert len(scored) == 2
        # score = relevance + recency*0.6 + novelty*0.3 + quality*0.2
        expected_0 = 9 + 8 * 0.6 + 7 * 0.3 + 6 * 0.2
        assert abs(scored[0]["agent_score"] - expected_0) < 0.01
        assert scored[0]["index"] == 0

    def test_missing_fields_default_to_zero(self):
        results = [{"index": 0}, {"index": 1, "relevance": 10}]
        scored = _compute_stage1_scores(results)
        assert scored[0]["agent_score"] == 0.0
        assert scored[1]["agent_score"] == 10.0

    def test_non_int_index_skipped(self):
        results = [
            {"index": "bad", "relevance": 9},
            {"index": None, "relevance": 8},
            {"index": 2, "relevance": 7},
        ]
        scored = _compute_stage1_scores(results)
        assert len(scored) == 1
        assert scored[0]["index"] == 2


class TestEstimateTokens:
    def test_returns_positive_for_nonempty(self):
        papers = [
            {"title": "Test", "abstract": "Abstract here", "authors": [], "year": 2024, "citation_count": 0}
        ]
        tokens = _estimate_papers_tokens(papers, "test question", "system prompt")
        assert tokens > 0

    def test_empty_papers_still_has_fixed_cost(self):
        tokens = _estimate_papers_tokens([], "question", "system prompt")
        assert tokens > 0

    def test_more_papers_more_tokens(self):
        papers_1 = [{"title": "A", "abstract": "B", "authors": [], "year": 2024, "citation_count": 0}]
        papers_3 = papers_1 * 3
        t1 = _estimate_papers_tokens(papers_1, "q", "s")
        t3 = _estimate_papers_tokens(papers_3, "q", "s")
        assert t3 > t1


# ---------------------------------------------------------------------------
# Two-stage pipeline integration tests
# ---------------------------------------------------------------------------
class TestTwoStagePipeline:
    def test_two_stage_happy_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Full two-stage pipeline with 30 candidates, returns final_limit papers."""
        storage = _tmp_storage(tmp_path)
        n_papers = 30
        theme = Theme(
            id="theme-twostage",
            document_text="two-stage test topic",
            parsed_queries=["two-stage test topic"],
        )
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=True,
            stage1_model="glm-4.6",
            stage1_batch_size=10,
            llm_global_concurrency=20,
            stage1_max_retries=1,
            stage2_model="glm-5-turbo",
            stage2_max_context_tokens=100_000,
            agent_candidate_limit=30,
            final_limit=5,
        )

        cands = _many_candidates(n_papers)
        mock_connectors = [
            _make_mock_connector(cands),
            *[
                _make_mock_connector([])
                for _ in range(5)
            ],  # arxiv, s2, dblp, openreview, crossref
        ]

        call_log = {"stage1_calls": 0, "stage2_calls": 0}

        class _FakeAgent(_FakeAgentBase):
            def __init__(self, *args, **kwargs):
                self._backend = kwargs.get("backend", "glm")
                self._model = kwargs.get("model", "unknown")

            async def score_papers_batch(self, papers, question, *, system_prompt):
                call_log["stage1_calls"] += 1
                return [
                    {
                        "index": i,
                        "relevance": 8.0 - i * 0.2,
                        "recency": 7.0,
                        "novelty": 6.0,
                        "quality": 5.0,
                        "reason": f"Scored paper {i}",
                    }
                    for i in range(len(papers))
                ]

            async def select_papers(self, papers, question, *, max_select, system_prompt):
                call_log["stage2_calls"] += 1
                return [
                    {
                        "index": i,
                        "selected": i < max_select,
                        "final_rank": i + 1,
                        "relevance": 9.0 - i * 0.5,
                        "reason": f"Selected paper {i}",
                    }
                    for i in range(len(papers))
                ]

            async def close(self):
                pass

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        result = _run(
            run_query_pipeline("two-stage test topic", storage, settings=settings)
        )

        assert result.total_retrieved == n_papers
        assert result.total_final == 5
        assert len(result.works) == 5
        # Stage 1 should have been called (30 papers / 10 batch_size = 3 batches)
        assert call_log["stage1_calls"] == 3
        # Stage 2 should have been called at least once
        assert call_log["stage2_calls"] >= 1

    def test_two_stage_stage2_fails_falls_back_to_stage1_scores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Stage 2 LLM call fails -> use Stage 1 scores directly."""
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-s2fail",
            document_text="stage2 fail test",
            parsed_queries=["stage2 fail test"],
        )
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=True,
            stage1_batch_size=5,
            agent_candidate_limit=10,
            final_limit=3,
        )

        cands = _many_candidates(10)
        mock_connectors = [_make_mock_connector(cands)] + [_make_mock_connector([]) for _ in range(5)]

        class _FakeAgent(_FakeAgentBase):
            def __init__(self, *args, model="glm-4.6", **kwargs):
                self._model = model

            async def score_papers_batch(self, papers, question, *, system_prompt):
                return [
                    {"index": i, "relevance": float(len(papers) - i), "recency": 5.0, "novelty": 5.0, "quality": 5.0}
                    for i in range(len(papers))
                ]

            async def select_papers(self, papers, question, *, max_select, system_prompt):
                raise DeepXivAgentError("Stage 2 API is down")

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        result = _run(
            run_query_pipeline("stage2 fail test", storage, settings=settings)
        )

        # Should still return papers via Stage 1 fallback
        assert result.total_final == 3
        assert len(result.works) == 3

    def test_two_stage_both_stages_fail_falls_back_to_deterministic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Both Stage 1 and Stage 2 LLM calls fail -> deterministic fallback."""
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-bothfail",
            document_text="both fail test",
            parsed_queries=["both fail test"],
        )
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=True,
            stage1_batch_size=5,
            stage1_max_retries=0,
            agent_candidate_limit=10,
            final_limit=3,
        )

        cands = _many_candidates(10)
        mock_connectors = [_make_mock_connector(cands)] + [_make_mock_connector([]) for _ in range(5)]

        class _FailingAgent(_FakeAgentBase):
            def __init__(self, *args, **kwargs):
                pass

            async def score_papers_batch(self, papers, question, *, system_prompt):
                raise DeepXivAgentError("Stage 1 API is down")

            async def select_papers(self, papers, question, *, max_select, system_prompt):
                raise DeepXivAgentError("Stage 2 API is down")

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FailingAgent)

        result = _run(
            run_query_pipeline("both fail test", storage, settings=settings)
        )

        # Should fall back to deterministic scoring and still return papers
        assert result.total_final >= 1
        assert len(result.works) >= 1

    def test_two_stage_empty_candidates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """No candidates -> empty result, no crash."""
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-empty",
            document_text="empty test",
            parsed_queries=["empty test"],
        )
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=True,
            agent_candidate_limit=10,
            final_limit=5,
        )

        mock_connectors = [_make_mock_connector([])] * 6

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)

        result = _run(
            run_query_pipeline("empty test", storage, settings=settings)
        )

        assert result.total_final == 0
        assert len(result.works) == 0

    def test_two_stage_large_pool_200_candidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """200 candidates (realistic production size), batch_size=10 -> 20 batches."""
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-large",
            document_text="large pool test",
            parsed_queries=["large pool test"],
        )
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=True,
            stage1_batch_size=10,
            llm_global_concurrency=20,
            agent_candidate_limit=200,
            final_limit=20,
        )

        cands = _many_candidates(200)
        mock_connectors = [_make_mock_connector(cands)] + [_make_mock_connector([]) for _ in range(5)]

        batch_counts = {"stage1": 0}

        class _FakeAgent(_FakeAgentBase):
            def __init__(self, *args, **kwargs):
                pass

            async def score_papers_batch(self, papers, question, *, system_prompt):
                batch_counts["stage1"] += 1
                return [
                    {"index": i, "relevance": 8.0 - (i % 5), "recency": 7.0, "novelty": 6.0, "quality": 5.0}
                    for i in range(len(papers))
                ]

            async def select_papers(self, papers, question, *, max_select, system_prompt):
                return [
                    {
                        "index": i,
                        "selected": i < max_select,
                        "final_rank": i + 1,
                        "relevance": 9.0 - i * 0.2,
                        "reason": "ok",
                    }
                    for i in range(min(len(papers), max_select))
                ]

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        start = time.monotonic()
        result = _run(
            run_query_pipeline("large pool test", storage, settings=settings)
        )

        elapsed = time.monotonic() - start
        assert result.total_retrieved == 200
        assert result.total_final == 20
        assert len(result.works) == 20
        # 200 papers / batch_size=10 = 20 batches
        assert batch_counts["stage1"] == 20
        # Each work should have agent metadata
        for w in result.works:
            assert w.agent_score is not None and w.agent_score > 0


# ---------------------------------------------------------------------------
# Regression: single-stage still works
# ---------------------------------------------------------------------------
class TestSingleStageRegression:
    def test_single_stage_still_works(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """With two_stage_enabled=False, the original pipeline still works."""
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-single",
            document_text="single stage test",
            parsed_queries=["single stage test"],
        )
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=False,
            agent_candidate_limit=5,
            final_limit=2,
        )

        cands = _many_candidates(5)
        mock_connectors = [_make_mock_connector(cands)] + [_make_mock_connector([]) for _ in range(5)]

        class _FakeAgent(_FakeAgentBase):
            def __init__(self, *args, **kwargs):
                pass

            async def rerank_papers(self, papers, question, **kwargs):
                return [
                    {
                        "index": i,
                        "selected": i < 2,
                        "agent_score": float(len(papers) - i),
                        "agent_rank": i + 1,
                        "agent_rationale": f"Rank {i+1}",
                    }
                    for i in range(len(papers))
                ]

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        result = _run(
            run_query_pipeline("single stage test", storage, settings=settings)
        )

        assert result.total_final == 2
        assert len(result.works) == 2


# ---------------------------------------------------------------------------
# Concurrency stress test
# ---------------------------------------------------------------------------
class TestConcurrency:
    def test_three_concurrent_queries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """3 simultaneous queries should all succeed without errors."""
        storage = _tmp_storage(tmp_path)
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=True,
            stage1_batch_size=10,
            agent_candidate_limit=20,
            final_limit=5,
        )

        themes = [
            Theme(
                id=f"theme-concurrent-{i}",
                document_text=f"concurrent query {i}",
                parsed_queries=[f"concurrent query {i}"],
            )
            for i in range(3)
        ]

        call_counts = {"total": 0}

        class _FakeAgent(_FakeAgentBase):
            def __init__(self, *args, **kwargs):
                pass

            async def score_papers_batch(self, papers, question, *, system_prompt):
                call_counts["total"] += 1
                # Simulate slight processing delay
                await asyncio.sleep(0.01)
                return [
                    {"index": i, "relevance": 7.0, "recency": 6.0, "novelty": 5.0, "quality": 4.0}
                    for i in range(len(papers))
                ]

            async def select_papers(self, papers, question, *, max_select, system_prompt):
                call_counts["total"] += 1
                await asyncio.sleep(0.01)
                return [
                    {
                        "index": i,
                        "selected": i < max_select,
                        "final_rank": i + 1,
                        "relevance": 8.0,
                        "reason": "ok",
                    }
                    for i in range(min(len(papers), max_select))
                ]

        # Build mock connectors per theme
        def _make_connectors_for_theme(n_cands):
            cands = _many_candidates(n_cands)
            return [_make_mock_connector(cands)] + [_make_mock_connector([]) for _ in range(5)]

        theme_idx = [0]

        original_parse_theme = retrieval_mod.parse_theme

        def _next_theme(doc):
            idx = theme_idx[0]
            theme_idx[0] += 1
            return themes[min(idx, len(themes) - 1)]

        monkeypatch.setattr(retrieval_mod, "parse_theme", _next_theme)

        connector_idx = [0]
        connectors_list = [_make_connectors_for_theme(20) for _ in range(3)]

        def _rotate_connectors(_s):
            idx = connector_idx[0]
            connector_idx[0] += 1
            return connectors_list[min(idx, len(connectors_list) - 1)]

        monkeypatch.setattr(retrieval_mod, "_build_connectors", _rotate_connectors)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        async def _run_query(i):
            return await run_query_pipeline(
                f"concurrent query {i}",
                storage,
                settings=settings,
            )

        async def _run_all():
            return await asyncio.gather(*[_run_query(i) for i in range(3)])

        results = _run(_run_all())

        for i, result in enumerate(results):
            assert result.total_final == 5, f"Query {i} returned {result.total_final} papers"
            assert len(result.works) == 5, f"Query {i} has {len(result.works)} works"

    def test_rapid_sequential_queries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """10 sequential queries back-to-back should all succeed."""
        storage = _tmp_storage(tmp_path)
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=True,
            stage1_batch_size=10,
            agent_candidate_limit=15,
            final_limit=3,
        )

        class _FakeAgent(_FakeAgentBase):
            def __init__(self, *args, **kwargs):
                pass

            async def score_papers_batch(self, papers, question, *, system_prompt):
                return [
                    {"index": i, "relevance": 7.0, "recency": 6.0, "novelty": 5.0, "quality": 4.0}
                    for i in range(len(papers))
                ]

            async def select_papers(self, papers, question, *, max_select, system_prompt):
                return [
                    {
                        "index": i,
                        "selected": i < max_select,
                        "final_rank": i + 1,
                        "relevance": 8.0,
                        "reason": "ok",
                    }
                    for i in range(min(len(papers), max_select))
                ]

        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        for i in range(10):
            theme = Theme(
                id=f"theme-rapid-{i}",
                document_text=f"rapid query {i}",
                parsed_queries=[f"rapid query {i}"],
            )
            cands = _many_candidates(15)
            mock_connectors = [_make_mock_connector(cands)] + [_make_mock_connector([]) for _ in range(5)]
            monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s, c=mock_connectors: c)
            monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc, t=theme: t)

            result = _run(
                run_query_pipeline(f"rapid query {i}", storage, settings=settings)
            )
            assert result.total_final == 3, f"Query {i} failed"
            assert len(result.works) == 3, f"Query {i} returned wrong count"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_single_paper_candidate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Only 1 candidate -> pipeline should still work."""
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-single-paper",
            document_text="single paper",
            parsed_queries=["single paper"],
        )
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=True,
            agent_candidate_limit=1,
            final_limit=1,
        )

        cands = [_make_candidate("Only Paper", doi="10.1/only", openalex_id="W-only")]
        mock_connectors = [_make_mock_connector(cands)] + [_make_mock_connector([]) for _ in range(5)]

        class _FakeAgent(_FakeAgentBase):
            def __init__(self, *args, **kwargs):
                pass

            async def score_papers_batch(self, papers, question, *, system_prompt):
                return [{"index": 0, "relevance": 9.0, "recency": 8.0, "novelty": 7.0, "quality": 6.0}]

            async def select_papers(self, papers, question, *, max_select, system_prompt):
                return [{"index": 0, "selected": True, "final_rank": 1, "relevance": 9.0, "reason": "only one"}]

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        result = _run(run_query_pipeline("single paper", storage, settings=settings))
        assert result.total_final == 1
        assert result.works[0].title == "Only Paper"

    def test_final_limit_larger_than_candidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """final_limit=50 but only 10 candidates -> return all 10."""
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-limits",
            document_text="limits test",
            parsed_queries=["limits test"],
        )
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=True,
            agent_candidate_limit=10,
            final_limit=50,  # more than candidates
        )

        cands = _many_candidates(10)
        mock_connectors = [_make_mock_connector(cands)] + [_make_mock_connector([]) for _ in range(5)]

        class _FakeAgent(_FakeAgentBase):
            def __init__(self, *args, **kwargs):
                pass

            async def score_papers_batch(self, papers, question, *, system_prompt):
                return [
                    {"index": i, "relevance": 7.0, "recency": 6.0, "novelty": 5.0, "quality": 4.0}
                    for i in range(len(papers))
                ]

            async def select_papers(self, papers, question, *, max_select, system_prompt):
                return [
                    {"index": i, "selected": True, "final_rank": i + 1, "relevance": 8.0, "reason": "ok"}
                    for i in range(len(papers))
                ]

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        result = _run(run_query_pipeline("limits test", storage, settings=settings))
        assert result.total_final == 10
        assert len(result.works) == 10

    def test_stage1_partial_batch_failures(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Some Stage 1 batches succeed, some fail -> mix of LLM and deterministic scores."""
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-partial",
            document_text="partial failure",
            parsed_queries=["partial failure"],
        )
        settings = Settings(
            bigmodel_api_key="glm-key",
            two_stage_enabled=True,
            stage1_batch_size=5,
            stage1_max_retries=0,
            agent_candidate_limit=20,
            final_limit=5,
        )

        cands = _many_candidates(20)
        mock_connectors = [_make_mock_connector(cands)] + [_make_mock_connector([]) for _ in range(5)]

        batch_counter = [0]

        class _PartialFailAgent(_FakeAgentBase):
            def __init__(self, *args, **kwargs):
                pass

            async def score_papers_batch(self, papers, question, *, system_prompt):
                batch_counter[0] += 1
                # Fail every other batch
                if batch_counter[0] % 2 == 0:
                    raise DeepXivAgentError("Batch failed")
                return [
                    {"index": i, "relevance": 8.0, "recency": 7.0, "novelty": 6.0, "quality": 5.0}
                    for i in range(len(papers))
                ]

            async def select_papers(self, papers, question, *, max_select, system_prompt):
                return [
                    {"index": i, "selected": i < max_select, "final_rank": i + 1, "relevance": 8.0, "reason": "ok"}
                    for i in range(min(len(papers), max_select))
                ]

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _PartialFailAgent)

        result = _run(run_query_pipeline("partial failure", storage, settings=settings))
        assert result.total_final == 5
        assert len(result.works) == 5
