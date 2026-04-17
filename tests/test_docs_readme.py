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
    startup_flow = [
        "./run_scholartrace_mcp_sse.sh",
        "./status_scholartrace_mcp_sse.sh",
        "./stop_scholartrace_mcp_sse.sh",
    ]
    exact_chatbox_json = """{
  "mcpServers": {
    "scholartrace": {
      "url": "http://172.17.194.210:8001/sse",
      "headers": {
        "Authorization": "Bearer g203-mcp"
      }
    }
  }
}"""
    shared_required_terms = [
        "SSE",
        "stdio",
        "glm-5-turbo",
        "SCHOLARTRACE_ACCESS_TOKEN=g203-mcp",
        "Authorization: Bearer g203-mcp",
        "http://172.17.194.210:8001/sse",
        "run_scholartrace_mcp_sse",
        "status_scholartrace_mcp_sse",
        "stop_scholartrace_mcp_sse",
        "agent_candidate_limit",
        "final_limit",
        "SCHOLARTRACE_BIGMODEL_API_KEY",
        ".env",
        "SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET",
        "auto-register",
    ]
    language_specific_terms = {
        "README.md": ["DeepXiv is optional"],
        "README_CN.md": ["DeepXiv 是可选的"],
    }

    for readme_name in ("README.md", "README_CN.md"):
        text = _readme_text(readme_name)
        for term in shared_required_terms:
            assert term in text, f"{readme_name} is missing {term}"
        for term in language_specific_terms[readme_name]:
            assert term in text, f"{readme_name} is missing {term}"
        _ordered_positions(text, rest_flow)
        _ordered_positions(text, mcp_flow)
        _ordered_positions(text, startup_flow)
        assert exact_chatbox_json in text, f"{readme_name} is missing the exact ChatBox JSON example"


def test_readmes_align_on_core_runtime_story():
    english = _readme_text("README.md")
    chinese = _readme_text("README_CN.md")

    assert "2 public MCP tools" in english
    assert "2 个公开 MCP 工具" in chinese

    shared_terms = [
        "`query`",
        "`read`",
        "fulltext_status",
        "direct_evidence",
        "ChatBox",
        "arXiv HTML",
        "arXiv PDF",
        "pdf_url",
        "oa_url",
        "html_url",
        "markdown fallback",
        "SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET",
        "GET /papers/{paper_id}/fulltext",
        "POST /papers/{paper_id}/fulltext/acquire",
        "Authorization: Bearer g203-mcp",
        "./run_scholartrace_mcp_sse.sh",
        "./status_scholartrace_mcp_sse.sh",
        "./stop_scholartrace_mcp_sse.sh",
        "SCHOLARTRACE_BIGMODEL_API_KEY",
        "tmux attach -t scholartrace_mcp_sse",
        "tmux capture-pane -pt scholartrace_mcp_sse",
        "ss -ltnp | grep ':8001'",
    ]

    english_terms = [
        "DeepXiv Agent",
        "DeepXiv is optional",
        "MCP clients do not pass that key in request payloads",
    ]
    chinese_terms = [
        "DeepXiv Agent",
        "DeepXiv 是可选的",
        "MCP 客户端不会在请求参数里传这个 key",
    ]

    for term in shared_terms:
        assert term in english, f"README.md must include {term}"
        assert term in chinese, f"README_CN.md must include {term}"
    for term in english_terms:
        assert term in english, f"README.md must include {term}"
    for term in chinese_terms:
        assert term in chinese, f"README_CN.md must include {term}"

    assert "13 tools" not in english
    assert "13 个工具" not in chinese
