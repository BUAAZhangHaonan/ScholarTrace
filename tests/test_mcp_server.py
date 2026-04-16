"""Tests for the MCP server tools.

Each test calls the tool functions directly (as regular async functions)
without going through MCP transport.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from scholartrace.models.schemas import (
    Section,
    Theme,
    Work,
)
from scholartrace.services.storage import StorageService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def test_storage():
    """Create a temporary StorageService and initialise its schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = StorageService(db_path=db_path)
        storage.init_db()
        yield storage


@pytest.fixture(autouse=True)
def _inject_storage(test_storage):
    """Override the module-level storage in mcp_server for every test."""
    import scholartrace.api.mcp_server as mod

    prev = mod._storage
    mod.set_storage(test_storage)
    yield
    mod._storage = prev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    }
    defaults.update(overrides)
    return Work(**defaults)


# ---------------------------------------------------------------------------
# search_papers_by_theme
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_papers_by_theme(test_storage):
    from scholartrace.api.mcp_server import search_papers_by_theme

    fake_theme = Theme(
        id="theme-1",
        document_text="testing theme document",
        parsed_topics=["topic-a"],
        parsed_methods=["method-b"],
        parsed_datasets=[],
        parsed_queries=["topic-a query"],
    )
    fake_works = [
        _make_work(title=f"Paper {i}", composite_score=0.9 - i * 0.05)
        for i in range(12)
    ]

    async def _fake_retrieval(doc_text, storage, settings=None):
        # Persist theme and works so they exist in storage
        storage.save_theme(fake_theme)
        for idx, w in enumerate(fake_works):
            storage.save_work(w)
            storage.link_theme_work(fake_theme.id, w.id, idx + 1)
        return fake_theme, fake_works

    with patch(
        "scholartrace.services.retrieval.run_retrieval_for_document",
        new=_fake_retrieval,
    ):
        result_str = await search_papers_by_theme("Some theme document text")

    result = json.loads(result_str)
    assert result["theme_id"] == "theme-1"
    assert result["query_count"] == 1
    assert result["total_papers"] == 12
    assert len(result["top_10"]) == 10
    assert "title" in result["top_10"][0]
    assert "composite_score" in result["top_10"][0]


# ---------------------------------------------------------------------------
# get_ranked_papers
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_ranked_papers(test_storage):
    from scholartrace.api.mcp_server import get_ranked_papers

    theme = Theme(id="theme-rp", document_text="ranked papers theme")
    test_storage.save_theme(theme)

    works = [
        _make_work(title=f"Ranked {i}", composite_score=0.9 - i * 0.1)
        for i in range(5)
    ]
    for idx, w in enumerate(works):
        test_storage.save_work(w)
        test_storage.link_theme_work(theme.id, w.id, idx + 1)

    result_str = await get_ranked_papers("theme-rp", limit=3)
    papers = json.loads(result_str)
    assert len(papers) == 3
    assert papers[0]["title"] == "Ranked 0"
    assert papers[0]["composite_score"] == 0.9


# ---------------------------------------------------------------------------
# get_paper_metadata
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_paper_metadata(test_storage):
    from scholartrace.api.mcp_server import get_paper_metadata

    work = _make_work(doi="10.1234/test", arxiv_id="2401.00001")
    test_storage.save_work(work)

    result_str = await get_paper_metadata(work.id)
    data = json.loads(result_str)
    assert data["title"] == "Test Paper"
    assert data["doi"] == "10.1234/test"
    assert data["arxiv_id"] == "2401.00001"
    assert "composite_score" in data
    assert "citation_count" in data


@pytest.mark.asyncio
async def test_get_paper_metadata_not_found(test_storage):
    from scholartrace.api.mcp_server import get_paper_metadata

    result_str = await get_paper_metadata("nonexistent-id")
    data = json.loads(result_str)
    assert "error" in data


# ---------------------------------------------------------------------------
# get_paper_sections
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_paper_sections(test_storage):
    from scholartrace.api.mcp_server import get_paper_sections

    work = _make_work()
    test_storage.save_work(work)

    sections = [
        Section(
            work_id=work.id,
            section_title="Introduction",
            section_order=0,
            text_content="This is the intro.",
        ),
        Section(
            work_id=work.id,
            section_title="Methods",
            section_order=1,
            text_content="We used a method.",
        ),
    ]
    for s in sections:
        test_storage.save_section(s)

    result_str = await get_paper_sections(work.id)
    data = json.loads(result_str)
    assert len(data) == 2
    assert data[0]["section_title"] == "Introduction"
    assert data[1]["section_title"] == "Methods"
    assert data[0]["text_content"] == "This is the intro."


# ---------------------------------------------------------------------------
# get_paper_fulltext
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_paper_fulltext(test_storage):
    from scholartrace.api.mcp_server import get_paper_fulltext

    work = _make_work(fulltext_available=False)
    test_storage.save_work(work)

    # Mock acquire_fulltext to avoid real HTTP calls
    with patch(
        "scholartrace.services.fulltext.acquire_fulltext",
        new=AsyncMock(return_value=work),
    ):
        result_str = await get_paper_fulltext(work.id)

    data = json.loads(result_str)
    assert data["paper_id"] == work.id
    assert "fulltext_available" in data
    assert "access_status" in data


# ---------------------------------------------------------------------------
# get_related_papers
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_related_papers(test_storage):
    from scholartrace.api.mcp_server import get_related_papers

    # Seed paper
    seed = _make_work(title="Seed Paper", year=2024,
                      venue="NeurIPS", composite_score=0.9)
    test_storage.save_work(seed)

    # Related papers (same venue, close year)
    related = [
        _make_work(
            title=f"Related {i}",
            year=2023 + i,
            venue="NeurIPS",
            composite_score=0.8 - i * 0.1,
        )
        for i in range(3)
    ]
    for w in related:
        test_storage.save_work(w)

    # Unrelated (different venue)
    unrelated = _make_work(
        title="Unrelated", year=2024, venue="ICML", composite_score=0.99
    )
    test_storage.save_work(unrelated)

    result_str = await get_related_papers(seed.id, limit=5)
    data = json.loads(result_str)
    assert len(data) == 3
    titles = [p["title"] for p in data]
    assert "Unrelated" not in titles
    assert all(t.startswith("Related") for t in titles)


# ---------------------------------------------------------------------------
# export_theme_report
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_export_theme_report_json(test_storage):
    from scholartrace.api.mcp_server import export_theme_report

    theme = Theme(
        id="theme-export",
        document_text="export test",
        parsed_topics=["topic-a"],
        parsed_methods=["RL"],
        parsed_datasets=["MMLU"],
        parsed_queries=["topic-a query"],
    )
    test_storage.save_theme(theme)

    works = [
        _make_work(title=f"Export {i}", composite_score=0.7 + i * 0.05)
        for i in range(3)
    ]
    for idx, w in enumerate(works):
        test_storage.save_work(w)
        test_storage.link_theme_work(theme.id, w.id, idx + 1)

    result_str = await export_theme_report("theme-export", format="json")
    data = json.loads(result_str)
    assert data["theme_id"] == "theme-export"
    assert data["total_papers"] == 3
    assert len(data["papers"]) == 3
    assert data["parsed_topics"] == ["topic-a"]


@pytest.mark.asyncio
async def test_export_theme_report_markdown(test_storage):
    from scholartrace.api.mcp_server import export_theme_report

    theme = Theme(
        id="theme-md",
        document_text="markdown export test",
        parsed_topics=["topic-a", "topic-b"],
        parsed_methods=[],
    )
    test_storage.save_theme(theme)

    works = [_make_work(title="MD Paper 1", composite_score=0.88)]
    for idx, w in enumerate(works):
        test_storage.save_work(w)
        test_storage.link_theme_work(theme.id, w.id, idx + 1)

    result_str = await export_theme_report("theme-md", format="markdown")
    assert "# Theme Report:" in result_str
    assert "topic-a" in result_str
    assert "MD Paper 1" in result_str
    assert "0.8800" in result_str


@pytest.mark.asyncio
async def test_export_theme_report_not_found(test_storage):
    from scholartrace.api.mcp_server import export_theme_report

    result_str = await export_theme_report("nonexistent-theme")
    data = json.loads(result_str)
    assert "error" in data
