"""Tests for retrieval orchestration and job manager."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from scholartrace.models.schemas import (
    JobStatus,
    RawCandidate,
    SourceName,
    Theme,
)
from scholartrace.services.retrieval import (
    run_retrieval,
    run_retrieval_for_document,
)
from scholartrace.services.storage import StorageService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_candidate(
    title: str = "Test Paper",
    doi: str | None = None,
    arxiv_id: str | None = None,
    s2_id: str | None = None,
    openalex_id: str | None = None,
    dblp_key: str | None = None,
    openreview_id: str | None = None,
    source: SourceName = SourceName.OPENALEX,
    year: int | None = 2024,
    authors: list[str] | None = None,
    abstract: str | None = None,
    citation_count: int = 0,
    reference_count: int = 0,
    venue: str | None = None,
) -> RawCandidate:
    return RawCandidate(
        title=title,
        doi=doi,
        arxiv_id=arxiv_id,
        s2_id=s2_id,
        openalex_id=openalex_id,
        dblp_key=dblp_key,
        openreview_id=openreview_id,
        source=source,
        year=year,
        authors=authors or ["Author One"],
        abstract=abstract,
        citation_count=citation_count,
        reference_count=reference_count,
        venue=venue,
    )


def _make_mock_connector(
    source_name: str, candidates: list[RawCandidate]
) -> AsyncMock:
    """Create a mock connector that returns the given candidates."""
    connector = AsyncMock()
    connector.source_name = source_name
    connector.search = AsyncMock(return_value=candidates)
    connector.close = AsyncMock(return_value=None)
    return connector


def _make_failing_connector(source_name: str, error: Exception) -> AsyncMock:
    connector = AsyncMock()
    connector.source_name = source_name
    connector.search = AsyncMock(side_effect=error)
    connector.close = AsyncMock(return_value=None)
    return connector


def _tmp_storage(tmp_path: Path) -> StorageService:
    db_path = tmp_path / "test.db"
    storage = StorageService(db_path)
    storage.init_db()
    return storage


def _sample_theme() -> Theme:
    return Theme(
        id="theme-1",
        document_text="reinforcement learning from human feedback",
        parsed_topics=["reinforcement learning", "human feedback"],
        parsed_methods=["RLHF"],
        parsed_datasets=[],
        parsed_queries=[
            "reinforcement learning human feedback", "RLHF reward model"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestJobLifecycle:
    """Test PENDING -> RUNNING -> COMPLETED / FAILED transitions."""

    def test_job_pending_to_completed(self, tmp_path: Path) -> None:
        from scholartrace.jobs.manager import JobManager

        storage = _tmp_storage(tmp_path)
        storage.save_theme(Theme(id="theme-1", document_text="job theme"))
        mgr = JobManager(storage)

        job = mgr.create_job("theme-1")
        assert job.status == JobStatus.PENDING
        assert job.theme_id == "theme-1"

        mgr.start_job(job.id)
        fetched = mgr.get_job(job.id)
        assert fetched is not None
        assert fetched.status == JobStatus.RUNNING

        mgr.complete_job(job.id, result_count=42)
        fetched = mgr.get_job(job.id)
        assert fetched is not None
        assert fetched.status == JobStatus.COMPLETED
        assert fetched.result_count == 42
        assert fetched.completed_at is not None

    def test_job_failure(self, tmp_path: Path) -> None:
        from scholartrace.jobs.manager import JobManager

        storage = _tmp_storage(tmp_path)
        storage.save_theme(Theme(id="theme-2", document_text="job theme"))
        mgr = JobManager(storage)

        job = mgr.create_job("theme-2")
        mgr.start_job(job.id)
        mgr.fail_job(job.id, error_message="boom")

        fetched = mgr.get_job(job.id)
        assert fetched is not None
        assert fetched.status == JobStatus.FAILED
        assert fetched.error_message == "boom"

    def test_get_nonexistent_job(self, tmp_path: Path) -> None:
        from scholartrace.jobs.manager import JobManager

        storage = _tmp_storage(tmp_path)
        mgr = JobManager(storage)
        assert mgr.get_job("no-such-id") is None

    def test_create_or_get_active_job_reuses_pending_job(self, tmp_path: Path) -> None:
        from scholartrace.jobs.manager import JobManager

        storage = _tmp_storage(tmp_path)
        storage.save_theme(Theme(id="theme-3", document_text="job theme"))
        mgr = JobManager(storage)

        first = mgr.create_or_get_active_job("theme-3")
        second = mgr.create_or_get_active_job("theme-3")

        assert first.id == second.id


class TestRetrievalPipeline:
    """Full pipeline tests using mocked connectors."""

    def test_full_pipeline(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple queries, multiple sources -> dedup, ranking, storage."""
        storage = _tmp_storage(tmp_path)
        theme = _sample_theme()

        oa_cands = [
            _make_candidate("RLHF Paper", doi="10.1/a", openalex_id="W1",
                            source=SourceName.OPENALEX, citation_count=50, year=2024),
            _make_candidate("Reward Modeling", doi="10.1/b", openalex_id="W2",
                            source=SourceName.OPENALEX, citation_count=10, year=2023),
        ]
        arxiv_cands = [
            _make_candidate("PPO for RLHF", arxiv_id="2301.001",
                            source=SourceName.ARXIV, year=2023),
        ]
        s2_cands = [
            _make_candidate("Alignment Survey", s2_id="s2-1",
                            source=SourceName.SEMANTIC_SCHOLAR, citation_count=100, year=2024),
        ]
        dblp_cands: list[RawCandidate] = []
        or_cands: list[RawCandidate] = []
        crossref_cands: list[RawCandidate] = []

        mock_connectors = [
            _make_mock_connector("openalex", oa_cands),
            _make_mock_connector("arxiv", arxiv_cands),
            _make_mock_connector("semantic_scholar", s2_cands),
            _make_mock_connector("dblp", dblp_cands),
            _make_mock_connector("openreview", or_cands),
            _make_mock_connector("crossref", crossref_cands),
        ]

        import scholartrace.services.retrieval as retrieval_mod

        monkeypatch.setattr(retrieval_mod, "_build_connectors",
                            lambda _s: mock_connectors)

        works = asyncio.get_event_loop().run_until_complete(
            run_retrieval(theme, storage)
        )

        # 4 distinct papers
        assert len(works) == 4

        # Works are sorted by composite_score descending
        scores = [w.composite_score for w in works]
        assert scores == sorted(scores, reverse=True)

        # Each work has scores computed
        for w in works:
            assert w.composite_score > 0.0

        # Works are persisted and linked to theme
        stored = storage.list_works_by_theme(theme.id, limit=100)
        assert len(stored) == 4
        assert storage.count_works_by_theme(theme.id) == 4

        conn = storage._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE theme_id = ?",
            (theme.id,),
        ).fetchone()
        assert row["c"] == 0

    def test_one_source_fails_others_succeed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """One connector raises; others still produce results."""
        storage = _tmp_storage(tmp_path)
        theme = _sample_theme()

        good_cands = [_make_candidate(
            "Good Paper", doi="10.1/g", openalex_id="W10", citation_count=5)]
        mock_connectors = [
            _make_mock_connector("openalex", good_cands),
            _make_failing_connector("arxiv", RuntimeError("arXiv is down")),
            _make_mock_connector("semantic_scholar", []),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
        ]

        import scholartrace.services.retrieval as retrieval_mod
        monkeypatch.setattr(retrieval_mod, "_build_connectors",
                            lambda _s: mock_connectors)

        works = asyncio.get_event_loop().run_until_complete(
            run_retrieval(theme, storage)
        )
        assert len(works) == 1
        assert works[0].title == "Good Paper"

    def test_dedup_across_sources(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same paper from OpenAlex and Semantic Scholar -> one Work."""
        storage = _tmp_storage(tmp_path)
        theme = _sample_theme()

        # Same paper via DOI overlap
        oa_cands = [
            _make_candidate("Unified RLHF Method", doi="10.1/rlhf", openalex_id="W20",
                            source=SourceName.OPENALEX, citation_count=30, year=2024),
        ]
        s2_cands = [
            _make_candidate("Unified RLHF Method", doi="10.1/rlhf", s2_id="s2-20",
                            source=SourceName.SEMANTIC_SCHOLAR, citation_count=30, year=2024),
        ]

        mock_connectors = [
            _make_mock_connector("openalex", oa_cands),
            _make_mock_connector("arxiv", []),
            _make_mock_connector("semantic_scholar", s2_cands),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
        ]

        import scholartrace.services.retrieval as retrieval_mod
        monkeypatch.setattr(retrieval_mod, "_build_connectors",
                            lambda _s: mock_connectors)

        works = asyncio.get_event_loop().run_until_complete(
            run_retrieval(theme, storage)
        )
        assert len(works) == 1
        assert works[0].source_provenance == ["openalex", "semantic_scholar"]

    def test_ranking_produces_ordered_results(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """composite_score is strictly descending for papers with different scores."""
        storage = _tmp_storage(tmp_path)
        theme = _sample_theme()

        oa_cands = [
            _make_candidate("Highly Cited RLHF", doi="10.1/h",
                            openalex_id="W30", citation_count=500, year=2024),
            _make_candidate("Medium Cited RLHF", doi="10.1/m",
                            openalex_id="W31", citation_count=50, year=2024),
            _make_candidate("Low Cited RLHF", doi="10.1/l",
                            openalex_id="W32", citation_count=1, year=2020),
        ]

        mock_connectors = [
            _make_mock_connector("openalex", oa_cands),
            _make_mock_connector("arxiv", []),
            _make_mock_connector("semantic_scholar", []),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
        ]

        import scholartrace.services.retrieval as retrieval_mod
        monkeypatch.setattr(retrieval_mod, "_build_connectors",
                            lambda _s: mock_connectors)

        works = asyncio.get_event_loop().run_until_complete(
            run_retrieval(theme, storage)
        )
        assert len(works) == 3
        for i in range(len(works) - 1):
            assert works[i].composite_score >= works[i + 1].composite_score

    def test_works_saved_to_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """After run_retrieval, works can be fetched individually from storage."""
        storage = _tmp_storage(tmp_path)
        theme = _sample_theme()

        cands = [
            _make_candidate("Saved Paper 1", doi="10.1/s1", openalex_id="W40"),
            _make_candidate("Saved Paper 2", doi="10.1/s2", openalex_id="W41"),
        ]
        mock_connectors = [
            _make_mock_connector("openalex", cands),
            _make_mock_connector("arxiv", []),
            _make_mock_connector("semantic_scholar", []),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
        ]

        import scholartrace.services.retrieval as retrieval_mod
        monkeypatch.setattr(retrieval_mod, "_build_connectors",
                            lambda _s: mock_connectors)

        works = asyncio.get_event_loop().run_until_complete(
            run_retrieval(theme, storage)
        )

        for w in works:
            fetched = storage.get_work(w.id)
            assert fetched is not None
            assert fetched.title == w.title
            assert fetched.doi == w.doi

    def test_catastrophic_failure_marks_job_failed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If ranking raises, run_retrieval does not persist a job row."""
        storage = _tmp_storage(tmp_path)
        theme = _sample_theme()

        cands = [_make_candidate("Paper", doi="10.1/f", openalex_id="W50")]
        mock_connectors = [
            _make_mock_connector("openalex", cands),
            _make_mock_connector("arxiv", []),
            _make_mock_connector("semantic_scholar", []),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
        ]

        import scholartrace.services.retrieval as retrieval_mod
        monkeypatch.setattr(retrieval_mod, "_build_connectors",
                            lambda _s: mock_connectors)

        # Make ranking blow up
        def _boom(*args, **kwargs):
            raise ValueError("ranking exploded")

        monkeypatch.setattr(retrieval_mod, "rank_papers", _boom)

        with pytest.raises(ValueError, match="ranking exploded"):
            asyncio.get_event_loop().run_until_complete(
                run_retrieval(theme, storage)
            )

        conn = storage._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE theme_id = ?",
            (theme.id,),
        ).fetchone()
        assert row["c"] == 0

    def test_persist_failure_rolls_back_all_work_state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        storage = _tmp_storage(tmp_path)
        theme = _sample_theme()

        cands = [
            _make_candidate("Paper 1", doi="10.1/p1", openalex_id="W70"),
            _make_candidate("Paper 2", doi="10.1/p2", openalex_id="W71"),
        ]
        mock_connectors = [
            _make_mock_connector("openalex", cands),
            _make_mock_connector("arxiv", []),
            _make_mock_connector("semantic_scholar", []),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
        ]

        import scholartrace.services.retrieval as retrieval_mod

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)

        original_save_work = storage.save_work
        calls = {"count": 0}

        def _flaky_save_work(work, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise sqlite3.OperationalError("simulated write failure")
            return original_save_work(work, *args, **kwargs)

        monkeypatch.setattr(storage, "save_work", _flaky_save_work)

        with pytest.raises(sqlite3.OperationalError, match="simulated write failure"):
            asyncio.get_event_loop().run_until_complete(
                run_retrieval(theme, storage)
            )

        conn = storage._get_conn()
        works_count = conn.execute("SELECT COUNT(*) AS c FROM works").fetchone()["c"]
        links_count = conn.execute(
            "SELECT COUNT(*) AS c FROM theme_works WHERE theme_id = ?",
            (theme.id,),
        ).fetchone()["c"]
        assert works_count == 0
        assert links_count == 0


class TestRetrievalForDocument:
    """Test the convenience wrapper that parses a document first."""

    def test_run_retrieval_for_document(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        storage = _tmp_storage(tmp_path)

        cands = [
            _make_candidate("Doc Paper", doi="10.1/d", openalex_id="W60"),
        ]
        mock_connectors = [
            _make_mock_connector("openalex", cands),
            _make_mock_connector("arxiv", []),
            _make_mock_connector("semantic_scholar", []),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
        ]

        import scholartrace.services.retrieval as retrieval_mod
        monkeypatch.setattr(retrieval_mod, "_build_connectors",
                            lambda _s: mock_connectors)

        doc = "This paper discusses reinforcement learning from human feedback and reward modeling."
        theme, works = asyncio.get_event_loop().run_until_complete(
            run_retrieval_for_document(doc, storage)
        )

        assert theme.id
        assert theme.parsed_queries
        assert len(works) == 1
        assert works[0].title == "Doc Paper"

        # Theme is persisted
        fetched_theme = storage.get_theme(theme.id)
        assert fetched_theme is not None
