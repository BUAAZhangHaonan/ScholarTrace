"""Tests for retrieval orchestration and job manager."""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from scholartrace.config import Settings
from scholartrace.models.schemas import (
    JobStatus,
    RawCandidate,
    SourceName,
    Theme,
)
from scholartrace.services.retrieval import (
    QueryPipelineConfigurationError,
    run_retrieval,
    run_retrieval_for_document,
    run_query_pipeline,
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


def _run(coro):
    return asyncio.run(coro)


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

    def test_build_connectors_includes_deepxiv_when_tokens_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scholartrace.services.retrieval as retrieval_mod

        def _factory(name: str):
            return lambda settings=None: type("Connector", (), {"source_name": name})()

        monkeypatch.setattr(retrieval_mod, "OpenAlexConnector", _factory("openalex"))
        monkeypatch.setattr(retrieval_mod, "ArxivConnector", _factory("arxiv"))
        monkeypatch.setattr(
            retrieval_mod, "SemanticScholarConnector", _factory("semantic_scholar")
        )
        monkeypatch.setattr(retrieval_mod, "DblpConnector", _factory("dblp"))
        monkeypatch.setattr(retrieval_mod, "OpenReviewConnector", _factory("openreview"))
        monkeypatch.setattr(retrieval_mod, "CrossrefConnector", _factory("crossref"))
        monkeypatch.setattr(retrieval_mod, "DeepXivConnector", _factory("deepxiv"))

        connectors = retrieval_mod._build_connectors(Settings(deepxiv_tokens="token-a"))

        assert [connector.source_name for connector in connectors] == [
            "openalex",
            "arxiv",
            "semantic_scholar",
            "dblp",
            "openreview",
            "crossref",
            "deepxiv",
        ]

    def test_build_connectors_includes_deepxiv_when_auto_register_is_explicit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scholartrace.services.retrieval as retrieval_mod

        def _factory(name: str):
            return lambda settings=None: type("Connector", (), {"source_name": name})()

        monkeypatch.setattr(retrieval_mod, "OpenAlexConnector", _factory("openalex"))
        monkeypatch.setattr(retrieval_mod, "ArxivConnector", _factory("arxiv"))
        monkeypatch.setattr(
            retrieval_mod, "SemanticScholarConnector", _factory("semantic_scholar")
        )
        monkeypatch.setattr(retrieval_mod, "DblpConnector", _factory("dblp"))
        monkeypatch.setattr(retrieval_mod, "OpenReviewConnector", _factory("openreview"))
        monkeypatch.setattr(retrieval_mod, "CrossrefConnector", _factory("crossref"))
        monkeypatch.setattr(retrieval_mod, "DeepXivConnector", _factory("deepxiv"))

        connectors = retrieval_mod._build_connectors(
            Settings(
                deepxiv_auto_register=True,
                deepxiv_register_sdk_secret="sdk-secret",
            )
        )

        assert connectors[-1].source_name == "deepxiv"

    def test_build_connectors_omits_deepxiv_when_not_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import scholartrace.services.retrieval as retrieval_mod

        def _factory(name: str):
            return lambda settings=None: type("Connector", (), {"source_name": name})()

        monkeypatch.setattr(retrieval_mod, "OpenAlexConnector", _factory("openalex"))
        monkeypatch.setattr(retrieval_mod, "ArxivConnector", _factory("arxiv"))
        monkeypatch.setattr(
            retrieval_mod, "SemanticScholarConnector", _factory("semantic_scholar")
        )
        monkeypatch.setattr(retrieval_mod, "DblpConnector", _factory("dblp"))
        monkeypatch.setattr(retrieval_mod, "OpenReviewConnector", _factory("openreview"))
        monkeypatch.setattr(retrieval_mod, "CrossrefConnector", _factory("crossref"))
        monkeypatch.setattr(retrieval_mod, "DeepXivConnector", _factory("deepxiv"))

        with caplog.at_level("INFO"):
            connectors = retrieval_mod._build_connectors(Settings())

        assert [connector.source_name for connector in connectors] == [
            "openalex",
            "arxiv",
            "semantic_scholar",
            "dblp",
            "openreview",
            "crossref",
        ]
        assert "DeepXiv is not configured for unified retrieval" in caplog.text

    def test_build_connectors_omits_deepxiv_for_delimiter_only_tokens(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import scholartrace.services.retrieval as retrieval_mod

        def _factory(name: str):
            return lambda settings=None: type("Connector", (), {"source_name": name})()

        monkeypatch.setattr(retrieval_mod, "OpenAlexConnector", _factory("openalex"))
        monkeypatch.setattr(retrieval_mod, "ArxivConnector", _factory("arxiv"))
        monkeypatch.setattr(
            retrieval_mod, "SemanticScholarConnector", _factory("semantic_scholar")
        )
        monkeypatch.setattr(retrieval_mod, "DblpConnector", _factory("dblp"))
        monkeypatch.setattr(retrieval_mod, "OpenReviewConnector", _factory("openreview"))
        monkeypatch.setattr(retrieval_mod, "CrossrefConnector", _factory("crossref"))
        monkeypatch.setattr(retrieval_mod, "DeepXivConnector", _factory("deepxiv"))

        with caplog.at_level("INFO"):
            connectors = retrieval_mod._build_connectors(Settings(deepxiv_tokens=" , "))

        assert [connector.source_name for connector in connectors] == [
            "openalex",
            "arxiv",
            "semantic_scholar",
            "dblp",
            "openreview",
            "crossref",
        ]
        assert "DeepXiv is not configured for unified retrieval" in caplog.text

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

        works = _run(run_retrieval(theme, storage))

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

        works = _run(run_retrieval(theme, storage))
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

        works = _run(run_retrieval(theme, storage))
        assert len(works) == 1
        assert works[0].source_provenance == ["openalex", "semantic_scholar"]

    def test_dedup_across_openalex_and_deepxiv_preserves_provenance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        storage = _tmp_storage(tmp_path)
        theme = _sample_theme()

        oa_cands = [
            _make_candidate(
                "Unified RLHF Method",
                doi="10.1/deepxiv",
                openalex_id="W-deepxiv",
                source=SourceName.OPENALEX,
            ),
        ]
        deepxiv_cands = [
            _make_candidate(
                "Unified RLHF Method",
                doi="10.1/deepxiv",
                arxiv_id="2401.00001",
                source=SourceName.DEEPXIV,
            ),
        ]

        mock_connectors = [
            _make_mock_connector("openalex", oa_cands),
            _make_mock_connector("arxiv", []),
            _make_mock_connector("semantic_scholar", []),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
            _make_mock_connector("deepxiv", deepxiv_cands),
        ]

        import scholartrace.services.retrieval as retrieval_mod

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)

        works = _run(run_retrieval(theme, storage))

        assert len(works) == 1
        assert works[0].source_provenance == ["openalex", "deepxiv"]

    def test_retrieval_preserves_work_identity_and_provenance_across_deepxiv_reruns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        storage = _tmp_storage(tmp_path)
        theme = _sample_theme()

        openalex_only = [
            _make_mock_connector(
                "openalex",
                [
                    _make_candidate(
                        "Stable Identity Paper",
                        doi="10.1/stable",
                        openalex_id="W-stable",
                        source=SourceName.OPENALEX,
                    )
                ],
            ),
            _make_mock_connector("arxiv", []),
            _make_mock_connector("semantic_scholar", []),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
            _make_mock_connector("deepxiv", []),
        ]

        import scholartrace.services.retrieval as retrieval_mod

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: openalex_only)

        first = _run(run_retrieval(theme, storage))
        assert len(first) == 1

        deepxiv_only = [
            _make_mock_connector("openalex", []),
            _make_mock_connector("arxiv", []),
            _make_mock_connector("semantic_scholar", []),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
            _make_mock_connector(
                "deepxiv",
                [
                    _make_candidate(
                        "Stable Identity Paper",
                        doi="10.1/stable",
                        arxiv_id="2401.00002",
                        source=SourceName.DEEPXIV,
                    )
                ],
            ),
        ]
        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: deepxiv_only)

        second = _run(run_retrieval(theme, storage))

        assert len(second) == 1
        assert second[0].id == first[0].id
        assert second[0].source_provenance == ["openalex", "deepxiv"]

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

        works = _run(run_retrieval(theme, storage))
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

        works = _run(run_retrieval(theme, storage))

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
            _run(run_retrieval(theme, storage))

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
            _run(run_retrieval(theme, storage))

        conn = storage._get_conn()
        works_count = conn.execute("SELECT COUNT(*) AS c FROM works").fetchone()["c"]
        links_count = conn.execute(
            "SELECT COUNT(*) AS c FROM theme_works WHERE theme_id = ?",
            (theme.id,),
        ).fetchone()["c"]
        assert works_count == 0
        assert links_count == 0

    def test_deepxiv_runtime_failure_does_not_break_unified_retrieval(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        storage = _tmp_storage(tmp_path)
        theme = _sample_theme()

        mock_connectors = [
            _make_mock_connector(
                "openalex",
                [
                    _make_candidate(
                        "OpenAlex Survives",
                        doi="10.1/survive",
                        openalex_id="W-survive",
                        source=SourceName.OPENALEX,
                    )
                ],
            ),
            _make_mock_connector("arxiv", []),
            _make_mock_connector("semantic_scholar", []),
            _make_mock_connector("dblp", []),
            _make_mock_connector("openreview", []),
            _make_mock_connector("crossref", []),
            _make_failing_connector("deepxiv", RuntimeError("DeepXiv unavailable")),
        ]

        import scholartrace.services.retrieval as retrieval_mod

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)

        works = _run(run_retrieval(theme, storage))

        assert len(works) == 1
        assert works[0].title == "OpenAlex Survives"


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
        theme, works = _run(run_retrieval_for_document(doc, storage))

        assert theme.id
        assert theme.parsed_queries
        assert len(works) == 1
        assert works[0].title == "Doc Paper"

        # Theme is persisted
        fetched_theme = storage.get_theme(theme.id)
        assert fetched_theme is not None


class TestQueryPipeline:
    def test_run_query_pipeline_applies_agent_rerank_and_persists_final_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-query",
            document_text="query theme",
            parsed_topics=["alignment"],
            parsed_methods=["rerank"],
            parsed_datasets=[],
            parsed_queries=["alignment rerank"],
        )
        settings = Settings(
            bigmodel_api_key="glm-key",
            agent_candidate_limit=3,
            final_limit=2,
        )

        cands = [
            _make_candidate("Alignment Paper Alpha", doi="10.1/a", openalex_id="W1"),
            _make_candidate("Reward Modeling Beta", doi="10.1/b", openalex_id="W2"),
            _make_candidate("Preference Tuning Gamma", doi="10.1/c", openalex_id="W3"),
            _make_candidate("Safety Evaluation Delta", doi="10.1/d", openalex_id="W4"),
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

        class _FakeAgent:
            def __init__(self, *args, **kwargs):
                self.closed = False

            async def rerank_papers(self, papers, question):
                assert len(papers) == 3
                assert question == "query theme"
                return [
                    {
                        "index": 1,
                        "selected": True,
                        "agent_score": 9.5,
                        "agent_rank": 1,
                        "agent_rationale": "Best fit.",
                    },
                    {
                        "index": 0,
                        "selected": True,
                        "agent_score": 8.5,
                        "agent_rank": 2,
                        "agent_rationale": "Good fit.",
                    },
                    {
                        "index": 2,
                        "selected": False,
                        "agent_score": 4.0,
                        "agent_rank": 3,
                        "agent_rationale": "Weak fit.",
                    },
                ]

            async def close(self):
                self.closed = True

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        result = _run(
            run_query_pipeline(
                "query theme",
                storage,
                settings=settings,
            )
        )

        assert result.theme.id == "theme-query"
        assert result.total_retrieved == 4
        assert result.total_after_dedup == 4
        assert result.total_after_first_stage == 4
        assert result.total_agent_candidates == 3
        assert result.total_final == 2
        assert [work.title for work in result.works] == [
            "Reward Modeling Beta",
            "Alignment Paper Alpha",
        ]
        assert result.works[0].agent_rank == 1
        assert result.works[0].agent_score == 9.5
        assert result.works[0].agent_rationale == "Best fit."

        stored = storage.list_works_by_theme(theme.id, limit=10)
        assert [work.title for work in stored] == [
            "Reward Modeling Beta",
            "Alignment Paper Alpha",
        ]

    def test_run_query_pipeline_uses_explicit_limits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-query-explicit",
            document_text="explicit query theme",
            parsed_queries=["explicit query theme"],
        )
        settings = Settings(bigmodel_api_key="glm-key")

        cands = [
            _make_candidate("Alignment Paper Alpha", doi="10.2/a", openalex_id="W10"),
            _make_candidate("Reward Modeling Beta", doi="10.2/b", openalex_id="W11"),
            _make_candidate("Preference Tuning Gamma", doi="10.2/c", openalex_id="W12"),
            _make_candidate("Safety Evaluation Delta", doi="10.2/d", openalex_id="W13"),
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

        class _FakeAgent:
            def __init__(self, *args, **kwargs):
                return None

            async def rerank_papers(self, papers, question):
                assert len(papers) == 2
                return [
                    {
                        "index": 1,
                        "selected": True,
                        "agent_score": 9.0,
                        "agent_rank": 1,
                        "agent_rationale": "Best fit.",
                    },
                    {
                        "index": 0,
                        "selected": True,
                        "agent_score": 8.0,
                        "agent_rank": 2,
                        "agent_rationale": "Second best.",
                    },
                ]

            async def close(self):
                return None

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        result = _run(
            run_query_pipeline(
                "explicit query theme",
                storage,
                settings=settings,
                coarse_pool_limit=2,
                agent_candidate_limit=2,
                final_limit=1,
            )
        )

        assert result.total_after_first_stage == 4
        assert result.total_agent_candidates == 2
        assert result.total_final == 1
        assert [work.title for work in result.works] == ["Reward Modeling Beta"]

    def test_run_query_pipeline_returns_only_selected_papers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-query-selected",
            document_text="selected query theme",
            parsed_queries=["selected query theme"],
        )
        settings = Settings(bigmodel_api_key="glm-key", final_limit=5, agent_candidate_limit=3)

        cands = [
            _make_candidate("Alignment Paper Alpha", doi="10.3/a", openalex_id="W20"),
            _make_candidate("Reward Modeling Beta", doi="10.3/b", openalex_id="W21"),
            _make_candidate("Preference Tuning Gamma", doi="10.3/c", openalex_id="W22"),
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

        class _FakeAgent:
            def __init__(self, *args, **kwargs):
                return None

            async def rerank_papers(self, papers, question):
                return [
                    {
                        "index": 0,
                        "selected": True,
                        "agent_score": 9.0,
                        "agent_rank": 1,
                        "agent_rationale": "Keep this paper.",
                    },
                    {
                        "index": 1,
                        "selected": False,
                        "agent_score": 8.5,
                        "agent_rank": 2,
                        "agent_rationale": "Reject this paper.",
                    },
                ]

            async def close(self):
                return None

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FakeAgent)

        result = _run(run_query_pipeline("selected query theme", storage, settings=settings))

        assert result.total_agent_candidates == 3
        assert result.total_final == 1
        assert [work.title for work in result.works] == ["Alignment Paper Alpha"]

    def test_run_query_pipeline_requires_api_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-query-no-key",
            document_text="query without key",
            parsed_queries=["query without key"],
        )
        settings = Settings(bigmodel_api_key="", final_limit=2, agent_candidate_limit=3)

        cands = [
            _make_candidate("Fallback Alpha", doi="10.4/a", openalex_id="W30"),
            _make_candidate("Fallback Beta", doi="10.4/b", openalex_id="W31"),
            _make_candidate("Fallback Gamma", doi="10.4/c", openalex_id="W32"),
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
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)

        with pytest.raises(
            QueryPipelineConfigurationError,
            match="SCHOLARTRACE_BIGMODEL_API_KEY is required",
        ):
            _run(run_query_pipeline("query without key", storage, settings=settings))

    def test_run_query_pipeline_falls_back_when_agent_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        storage = _tmp_storage(tmp_path)
        theme = Theme(
            id="theme-query-agent-fail",
            document_text="query with agent failure",
            parsed_queries=["query with agent failure"],
        )
        settings = Settings(bigmodel_api_key="glm-key", final_limit=2, agent_candidate_limit=3)

        cands = [
            _make_candidate("AgentFail Alpha", doi="10.5/a", openalex_id="W40"),
            _make_candidate("AgentFail Beta", doi="10.5/b", openalex_id="W41"),
            _make_candidate("AgentFail Gamma", doi="10.5/c", openalex_id="W42"),
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
        from scholartrace.deepxiv.agent import DeepXivAgentError

        class _FailingAgent:
            def __init__(self, *args, **kwargs):
                return None

            async def rerank_papers(self, papers, question):
                raise DeepXivAgentError("provider temporary failure")

            async def close(self):
                return None

        monkeypatch.setattr(retrieval_mod, "_build_connectors", lambda _s: mock_connectors)
        monkeypatch.setattr(retrieval_mod, "parse_theme", lambda doc: theme, raising=False)
        monkeypatch.setattr(retrieval_mod, "DeepXivAgent", _FailingAgent)

        result = _run(run_query_pipeline("query with agent failure", storage, settings=settings))

        assert result.total_agent_candidates == 3
        assert result.total_final == 2
        assert len(result.works) == 2
        assert all(
            (work.agent_rationale or "").startswith("fallback_default_ranking:agent_failure")
            for work in result.works
        )
