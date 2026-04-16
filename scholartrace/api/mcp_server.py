"""MCP (Model Context Protocol) server for ScholarTrace.

Exposes 7 tools that LLM agents can call to search for papers,
retrieve metadata, acquire full text, and export theme reports.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from mcp.server.fastmcp import FastMCP

from scholartrace.config import get_settings
from scholartrace.services.storage import StorageService

logger = logging.getLogger(__name__)

mcp = FastMCP("ScholarTrace")

# ---------------------------------------------------------------------------
# Lazy-initialised storage singleton (overridable in tests)
# ---------------------------------------------------------------------------
_storage: StorageService | None = None


def _get_storage() -> StorageService:
    global _storage
    if _storage is None:
        settings = get_settings()
        _storage = StorageService(db_path=settings.db_path)
        _storage.init_db()
    return _storage


def set_storage(storage: StorageService) -> None:
    """Replace the module-level storage instance (useful for testing)."""
    global _storage
    _storage = storage


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _work_to_dict(work: Any) -> dict[str, Any]:
    """Serialise a Work model to a plain dict suitable for JSON."""
    return {
        "id": work.id,
        "doi": work.doi,
        "arxiv_id": work.arxiv_id,
        "openalex_id": work.openalex_id,
        "s2_id": work.s2_id,
        "dblp_key": work.dblp_key,
        "openreview_id": work.openreview_id,
        "title": work.title,
        "authors": work.authors,
        "year": work.year,
        "venue": work.venue,
        "abstract": work.abstract,
        "relevance_score": work.relevance_score,
        "recency_score": work.recency_score,
        "influence_score": work.influence_score,
        "venue_score": work.venue_score,
        "composite_score": work.composite_score,
        "fulltext_available": work.fulltext_available,
        "access_status": work.access_status.value
        if hasattr(work.access_status, "value")
        else str(work.access_status),
        "source_provenance": work.source_provenance,
        "citation_count": work.citation_count,
        "reference_count": work.reference_count,
        "pdf_url": work.pdf_url,
        "html_url": work.html_url,
        "oa_url": work.oa_url,
        "created_at": work.created_at.isoformat() if work.created_at else None,
        "updated_at": work.updated_at.isoformat() if work.updated_at else None,
    }


def _work_summary(work: Any) -> dict[str, Any]:
    """Lightweight summary used in list-style responses."""
    return {
        "id": work.id,
        "title": work.title,
        "year": work.year,
        "venue": work.venue,
        "composite_score": work.composite_score,
        "arxiv_id": work.arxiv_id,
        "doi": work.doi,
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def search_papers_by_theme(theme_document: str) -> str:
    """Parse a theme document and run the full retrieval pipeline.

    Returns a JSON string with theme_id, query_count, total_papers, and
    the top-10 papers ranked by composite score.
    """
    storage = _get_storage()
    settings = get_settings()

    from scholartrace.services.retrieval import run_retrieval_for_document

    theme, works = await run_retrieval_for_document(
        theme_document, storage, settings
    )

    total = len(works)
    top_10 = [_work_summary(w) for w in works[:10]]

    result = {
        "theme_id": theme.id,
        "query_count": len(theme.parsed_queries),
        "total_papers": total,
        "top_10": top_10,
    }
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_ranked_papers(theme_id: str, limit: int = 50) -> str:
    """Get ranked papers for a theme from storage.

    Returns a JSON list of paper summaries ordered by rank.
    """
    storage = _get_storage()
    works = storage.list_works_by_theme(theme_id, limit=limit)
    papers = [_work_summary(w) for w in works]
    return json.dumps(papers, ensure_ascii=False)


@mcp.tool()
async def get_paper_metadata(paper_id: str) -> str:
    """Get full metadata for a single paper by its ID.

    Returns a JSON object with all Work fields.
    """
    storage = _get_storage()
    work = storage.get_work(paper_id)
    if work is None:
        return json.dumps({"error": f"Paper {paper_id} not found"})
    return json.dumps(_work_to_dict(work), ensure_ascii=False)


@mcp.tool()
async def get_paper_sections(paper_id: str) -> str:
    """Get parsed sections for a paper by its ID.

    Returns a JSON list of section objects.
    """
    storage = _get_storage()
    work = storage.get_work(paper_id)
    if work is None:
        return json.dumps({"error": f"Paper {paper_id} not found"})

    sections = storage.get_sections_by_work(paper_id)
    section_list = [
        {
            "id": s.id,
            "section_title": s.section_title,
            "section_order": s.section_order,
            "text_content": s.text_content,
        }
        for s in sections
    ]
    return json.dumps(section_list, ensure_ascii=False)


@mcp.tool()
async def get_paper_fulltext(paper_id: str) -> str:
    """Get full-text status and content for a paper.

    If the full text has not been acquired yet, attempts to acquire it.
    Returns a JSON object with acquisition status and (if available) the
    text content.
    """
    storage = _get_storage()
    settings = get_settings()

    from scholartrace.services.fulltext import acquire_fulltext

    work = storage.get_work(paper_id)
    if work is None:
        return json.dumps({"error": f"Paper {paper_id} not found"})

    # Attempt acquisition if not already done
    if not work.fulltext_available:
        work = await acquire_fulltext(work, storage, settings)

    # Gather any parsed-text artifact
    artifacts = storage.get_artifacts_by_work(paper_id)
    parsed_text = None
    for art in artifacts:
        if art.kind.value == "parsed_text" and art.local_path:
            try:
                parsed_text = open(art.local_path).read()
            except OSError:
                pass
            break

    result = {
        "paper_id": paper_id,
        "title": work.title,
        "fulltext_available": work.fulltext_available,
        "access_status": work.access_status.value
        if hasattr(work.access_status, "value")
        else str(work.access_status),
        "fulltext_content": parsed_text,
    }
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_related_papers(paper_id: str, limit: int = 10) -> str:
    """Find papers related to the given paper by shared venue and overlapping years.

    Returns a JSON list of related paper summaries.
    """
    storage = _get_storage()
    work = storage.get_work(paper_id)
    if work is None:
        return json.dumps({"error": f"Paper {paper_id} not found"})

    conn: sqlite3.Connection = storage._get_conn()

    if work.venue and work.year:
        # Same venue, +/- 2 years, excluding the paper itself
        rows = conn.execute(
            """
            SELECT * FROM works
            WHERE venue = ?
              AND year BETWEEN ? AND ?
              AND id != ?
            ORDER BY composite_score DESC
            LIMIT ?
            """,
            (work.venue, work.year - 2, work.year + 2, paper_id, limit),
        ).fetchall()
    elif work.venue:
        rows = conn.execute(
            """
            SELECT * FROM works
            WHERE venue = ?
              AND id != ?
            ORDER BY composite_score DESC
            LIMIT ?
            """,
            (work.venue, paper_id, limit),
        ).fetchall()
    else:
        rows = []

    from scholartrace.models.schemas import Work as WorkModel

    related = []
    for row in rows:
        w = storage._row_to_work(row)
        related.append(_work_summary(w))

    return json.dumps(related, ensure_ascii=False)


@mcp.tool()
async def export_theme_report(theme_id: str, format: str = "json") -> str:
    """Export all papers for a theme as JSON or Markdown.

    Parameters:
        theme_id: The theme identifier.
        format:   'json' (default) or 'markdown'.

    Returns:
        A JSON string or a Markdown-formatted string with paper details.
    """
    storage = _get_storage()

    theme = storage.get_theme(theme_id)
    if theme is None:
        return json.dumps({"error": f"Theme {theme_id} not found"})

    works = storage.list_works_by_theme(theme_id, limit=10000)

    if format == "markdown":
        lines: list[str] = [
            f"# Theme Report: {theme_id}",
            "",
            f"**Parsed topics**: {', '.join(theme.parsed_topics) if theme.parsed_topics else 'N/A'}",
            "",
            f"**Parsed methods**: {', '.join(theme.parsed_methods) if theme.parsed_methods else 'N/A'}",
            "",
            f"**Total papers**: {len(works)}",
            "",
            "---",
            "",
        ]
        for rank, w in enumerate(works, start=1):
            lines.append(f"## {rank}. {w.title}")
            lines.append("")
            if w.authors:
                lines.append(f"**Authors**: {', '.join(w.authors)}")
                lines.append("")
            lines.append(f"**Year**: {w.year or 'N/A'}")
            lines.append(f"**Venue**: {w.venue or 'N/A'}")
            lines.append(f"**Composite Score**: {w.composite_score:.4f}")
            if w.doi:
                lines.append(f"**DOI**: {w.doi}")
            if w.arxiv_id:
                lines.append(f"**arXiv**: {w.arxiv_id}")
            if w.abstract:
                lines.append("")
                lines.append(
                    f"> {w.abstract[:500]}{'...' if len(w.abstract) > 500 else ''}")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    # Default: JSON
    papers = [_work_to_dict(w) for w in works]
    report = {
        "theme_id": theme_id,
        "parsed_topics": theme.parsed_topics,
        "parsed_methods": theme.parsed_methods,
        "parsed_datasets": theme.parsed_datasets,
        "parsed_queries": theme.parsed_queries,
        "total_papers": len(papers),
        "papers": papers,
    }
    return json.dumps(report, ensure_ascii=False)
