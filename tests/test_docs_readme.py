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
        "query",
        "read",
        "allow_acquire",
    ]
    required_terms = [
        "SSE",
        "stdio",
        "glm-5-turbo",
        "SCHOLARTRACE_ACCESS_TOKEN=g203-mcp",
        "Authorization: Bearer g203-mcp",
        "http://<server-lan-ip>:8001/sse",
        "agent_candidate_limit",
        "final_limit",
    ]

    for readme_name in ("README.md", "README_CN.md"):
        text = _readme_text(readme_name)
        for term in required_terms:
            assert term in text, f"{readme_name} is missing {term}"
        _ordered_positions(text, rest_flow)
        _ordered_positions(text, mcp_flow)


def test_readmes_align_on_core_runtime_story():
    english = _readme_text("README.md")
    chinese = _readme_text("README_CN.md")

    assert "2 public MCP tools" in english
    assert "2 个公开 MCP 工具" in chinese
    assert "182 tests" in english
    assert "182 个测试" in chinese

    aligned_terms = [
        "`query`",
        "`read`",
        "fulltext_status",
        "direct_evidence",
        "ChatBox",
        "DeepXiv Agent",
        "arXiv HTML",
        "arXiv PDF",
        "pdf_url",
        "oa_url",
        "html_url",
        "markdown fallback",
        "GET /papers/{paper_id}/fulltext",
        "POST /papers/{paper_id}/fulltext/acquire",
        "Authorization: Bearer g203-mcp",
    ]

    for text, name in ((english, "README.md"), (chinese, "README_CN.md")):
        for term in aligned_terms:
            assert term in text, f"{name} must include {term}"
        assert "13 tools" not in text
        assert "13 个工具" not in text
