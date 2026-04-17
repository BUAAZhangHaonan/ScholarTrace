from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _readme_text(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def _ordered_positions(text: str, terms: list[str]) -> list[int]:
    positions: list[int] = []
    start = 0
    for term in terms:
        pos = text.find(term, start)
        assert pos != -1, f"missing ordered term: {term}"
        positions.append(pos)
        start = pos + len(term)
    return positions


def test_readmes_document_explicit_acquire_and_mcp_workflow():
    rest_flow = [
        "GET /papers/{paper_id}/fulltext",
        "POST /papers/{paper_id}/fulltext/acquire",
        "GET /papers/{paper_id}/fulltext",
    ]
    mcp_flow = [
        "search_papers_by_theme",
        "get_ranked_papers",
        "get_paper_fulltext",
        "acquire_paper_fulltext",
        "get_paper_fulltext",
    ]
    required_terms = ["stdio", "SSE", "glm-5-turbo"]

    for readme_name in ("README.md", "README_CN.md"):
        text = _readme_text(readme_name)
        for term in required_terms:
            assert term in text, f"{readme_name} is missing {term}"
        _ordered_positions(text, rest_flow)
        _ordered_positions(text, mcp_flow)


def test_readmes_align_on_core_runtime_story():
    english = _readme_text("README.md")
    chinese = _readme_text("README_CN.md")

    assert "13 tools" in english
    assert "13 个工具" in chinese
    assert "182 tests" in english
    assert "182 个测试" in chinese

    aligned_terms = [
        "search_papers_by_theme",
        "get_ranked_papers",
        "get_paper_metadata",
        "get_paper_sections",
        "get_paper_fulltext",
        "acquire_paper_fulltext",
        "get_related_papers",
        "export_theme_report",
        "deepxiv_search",
        "deepxiv_paper_summary",
        "deepxiv_paper_fulltext",
        "deepxiv_paper_section",
        "deepxiv_agent_filter",
        "GET /papers/{paper_id}/fulltext",
        "POST /papers/{paper_id}/fulltext/acquire",
        "GET /deepxiv/papers/{arxiv_id}/fulltext",
    ]

    for text, name in ((english, "README.md"), (chinese, "README_CN.md")):
        for term in aligned_terms:
            assert term in text, f"{name} must include {term}"
