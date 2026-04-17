"""Tests for the public MCP product surface.

Each test calls the tool functions directly without going through MCP transport.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scholartrace.models.schemas import Section, Theme, Work
from scholartrace.services import runtime_limits
from scholartrace.services.storage import StorageService


@pytest.fixture
def test_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = StorageService(db_path=db_path)
        storage.init_db()
        yield storage


@pytest.fixture(autouse=True)
def _inject_storage(test_storage):
    import scholartrace.api.mcp_server as mod

    prev = mod._storage
    mod.set_storage(test_storage)
    yield
    mod._storage = prev


@pytest.fixture(autouse=True)
def _reset_runtime_budgets():
    asyncio.run(runtime_limits.budget_manager.reset())
    yield
    asyncio.run(runtime_limits.budget_manager.reset())


def _make_work(**overrides) -> Work:
    defaults = {
        "title": "Test Paper",
        "authors": ["Alice Smith", "Bob Jones"],
        "year": 2024,
        "venue": "NeurIPS",
        "abstract": "A test abstract about testing.",
        "composite_score": 0.85,
        "doi": None,
        "arxiv_id": None,
        "agent_score": 0.0,
        "agent_rank": None,
        "agent_rationale": None,
    }
    defaults.update(overrides)
    return Work(**defaults)


@pytest.mark.asyncio
async def test_only_query_and_read_are_public_tools():
    from scholartrace.api import mcp_server

    tools = await mcp_server.mcp.list_tools()

    assert sorted(tool.name for tool in tools) == ["query", "read"]


@pytest.mark.asyncio
async def test_query_returns_reranked_papers_with_pipeline_counts():
    from scholartrace.api.mcp_server import query

    fake_theme = Theme(
        id="theme-1",
        document_text="testing theme document",
        parsed_topics=["topic-a"],
        parsed_methods=["method-b"],
        parsed_datasets=[],
        parsed_queries=["topic-a query"],
    )
    fake_works = [
        _make_work(
            id=f"paper-{i}",
            title=f"Paper {i}",
            composite_score=0.9 - i * 0.05,
            agent_score=9.0 - i,
            agent_rank=i + 1,
            agent_rationale=f"Reason {i}",
        )
        for i in range(3)
    ]

    async def _fake_query_pipeline(
        doc_text,
        storage,
        settings=None,
        *,
        final_limit,
        agent_candidate_limit,
        coarse_pool_limit,
        include_rationale,
    ):
        assert doc_text == "Some theme document text"
        assert final_limit == 20
        assert agent_candidate_limit == 100
        assert coarse_pool_limit is None
        assert include_rationale is True
        storage.save_theme(fake_theme)
        for idx, work in enumerate(fake_works, start=1):
            storage.save_work(work)
            storage.link_theme_work(fake_theme.id, work.id, idx)
        return SimpleNamespace(
            theme=fake_theme,
            total_retrieved=140,
            total_after_dedup=60,
            total_after_first_stage=40,
            total_agent_candidates=20,
            total_final=3,
            works=fake_works,
        )

    with patch(
        "scholartrace.services.retrieval.run_query_pipeline",
        new=_fake_query_pipeline,
        create=True,
    ):
        result_str = await query("Some theme document text")

    result = json.loads(result_str)
    assert result["theme_id"] == "theme-1"
    assert result["total_retrieved"] == 140
    assert result["total_after_dedup"] == 60
    assert result["total_after_first_stage"] == 40
    assert result["total_agent_candidates"] == 20
    assert result["total_final"] == 3
    assert len(result["papers"]) == 3
    assert result["papers"][0]["paper_id"] == "theme-1:paper-0"
    assert result["papers"][0]["agent_rank"] == 1
    assert result["papers"][0]["agent_score"] == 9.0
    assert result["papers"][0]["rationale"] == "Reason 0"
    assert "fulltext_status" in result["papers"][0]


@pytest.mark.asyncio
async def test_query_passes_explicit_limits_to_pipeline():
    from scholartrace.api.mcp_server import query

    fake_theme = Theme(id="theme-explicit", document_text="explicit theme")

    async def _fake_query_pipeline(
        doc_text,
        storage,
        settings=None,
        *,
        final_limit,
        agent_candidate_limit,
        coarse_pool_limit,
        include_rationale,
    ):
        assert final_limit == 7
        assert agent_candidate_limit == 13
        assert coarse_pool_limit == 25
        assert include_rationale is False
        return SimpleNamespace(
            theme=fake_theme,
            total_retrieved=10,
            total_after_dedup=9,
            total_after_first_stage=8,
            total_agent_candidates=7,
            total_final=0,
            works=[],
        )

    with patch(
        "scholartrace.services.retrieval.run_query_pipeline",
        new=_fake_query_pipeline,
        create=True,
    ):
        result_str = await query(
            "Some theme document text",
            final_limit=7,
            agent_candidate_limit=13,
            coarse_pool_limit=25,
            include_rationale=False,
        )

    result = json.loads(result_str)
    assert result["theme_id"] == "theme-explicit"
    assert result["total_final"] == 0
    assert result["papers"] == []


@pytest.mark.asyncio
async def test_read_summary_returns_metadata_agent_state_and_fulltext_status(test_storage):
    from scholartrace.api.mcp_server import read

    work = _make_work(
        doi="10.1234/test",
        arxiv_id="2401.00001",
        agent_score=8.7,
        agent_rank=2,
        agent_rationale="Strong fit for the theme.",
    )
    test_storage.save_work(work)

    result_str = await read(work.id, depth="summary")
    data = json.loads(result_str)

    assert data["paper_id"] == work.id
    assert data["depth"] == "summary"
    assert data["title"] == "Test Paper"
    assert data["doi"] == "10.1234/test"
    assert data["agent_score"] == 8.7
    assert data["agent_rank"] == 2
    assert data["rationale"] == "Strong fit for the theme."
    assert data["fulltext_status"]["fulltext_available"] is False
    assert data["fulltext_status"]["acquisition_state"] == "missing"


@pytest.mark.asyncio
async def test_read_sections_returns_cached_sections(test_storage):
    from scholartrace.api.mcp_server import read

    work = _make_work()
    test_storage.save_work(work)
    test_storage.save_section(
        Section(
            work_id=work.id,
            section_title="Introduction",
            section_order=0,
            text_content="This is the intro.",
        )
    )
    test_storage.save_section(
        Section(
            work_id=work.id,
            section_title="Methods",
            section_order=1,
            text_content="This is the method.",
        )
    )

    result_str = await read(work.id, depth="sections")
    data = json.loads(result_str)

    assert data["paper_id"] == work.id
    assert data["depth"] == "sections"
    assert len(data["sections"]) == 2
    assert data["sections"][0]["section_title"] == "Introduction"
    assert data["sections"][1]["section_title"] == "Methods"


@pytest.mark.asyncio
async def test_read_fulltext_status_matches_cached_contract(test_storage):
    from scholartrace.api.mcp_server import read

    work = _make_work(fulltext_available=False)
    test_storage.save_work(work)

    result_str = await read(work.id, depth="fulltext_status")
    data = json.loads(result_str)

    assert data["paper_id"] == work.id
    assert data["depth"] == "fulltext_status"
    assert data["fulltext_status"]["fulltext_available"] is False
    assert data["fulltext_status"]["needs_acquisition"] is True


@pytest.mark.asyncio
async def test_read_fulltext_triggers_explicit_acquire_when_allowed(test_storage):
    from scholartrace.api.mcp_server import read

    work = _make_work(fulltext_available=False)
    test_storage.save_work(work)

    before_payload = {
        "paper_id": work.id,
        "title": work.title,
        "fulltext_available": False,
        "access_status": "unknown",
        "acquisition_state": "missing",
        "needs_acquisition": True,
        "last_attempt_at": None,
        "next_retry_at": None,
        "error_message": None,
        "artifacts": [],
        "sections": [],
        "parsed_text": None,
    }
    after_payload = dict(before_payload)
    after_payload["fulltext_available"] = True
    after_payload["acquisition_state"] = "available"
    after_payload["needs_acquisition"] = False
    after_payload["parsed_text"] = "Recovered full text."

    async def _fake_acquire(target_work, storage, settings):
        return target_work

    with patch(
        "scholartrace.services.fulltext.read_cached_fulltext",
        side_effect=[before_payload, after_payload],
    ), patch(
        "scholartrace.services.fulltext.acquire_fulltext",
        new=_fake_acquire,
    ):
        result_str = await read(work.id, depth="fulltext", allow_acquire=True)

    data = json.loads(result_str)

    assert data["paper_id"] == work.id
    assert data["depth"] == "fulltext"
    assert data["fulltext"] == "Recovered full text."
    assert data["fulltext_status"]["fulltext_available"] is True


@pytest.mark.asyncio
async def test_read_direct_evidence_uses_deepxiv_for_arxiv_backed_papers(
    test_storage,
    monkeypatch: pytest.MonkeyPatch,
):
    from scholartrace.api import mcp_server

    work = _make_work(arxiv_id="2401.00001")
    test_storage.save_work(work)

    class _FakeConnector:
        async def get_paper_metadata(self, arxiv_id):
            return {"title": "DeepXiv Head", "arxiv_id": arxiv_id}

        async def get_paper_brief(self, arxiv_id):
            return {"tldr": "Brief summary", "arxiv_id": arxiv_id}

    mcp_server._deepxiv_connector = None
    monkeypatch.setattr(
        "scholartrace.connectors.deepxiv_connector.DeepXivConnector",
        lambda settings=None: _FakeConnector(),
    )

    result_str = await mcp_server.read(work.id, depth="direct_evidence")
    data = json.loads(result_str)

    assert data["paper_id"] == work.id
    assert data["depth"] == "direct_evidence"
    assert data["available"] is True
    assert data["source"] == "deepxiv"
    assert data["arxiv_id"] == "2401.00001"
    assert data["evidence"]["metadata"]["title"] == "DeepXiv Head"
    assert data["evidence"]["brief"]["tldr"] == "Brief summary"


@pytest.mark.asyncio
async def test_read_returns_not_found_error_for_missing_paper(test_storage):
    from scholartrace.api.mcp_server import read

    result_str = await read("nonexistent-id", depth="summary")
    data = json.loads(result_str)

    assert data == {
        "error": {
            "code": "not_found",
            "message": "Paper nonexistent-id not found",
            "retryable": False,
        }
    }
