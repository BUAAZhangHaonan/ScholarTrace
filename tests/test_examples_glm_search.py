from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from scholartrace.services.prompt_budget import DEFAULT_PROMPT_BUDGET, PromptBudget


def _load_example_module():
    path = Path(__file__).resolve().parent.parent / "examples" / "glm_scholar_search.py"
    spec = importlib.util.spec_from_file_location("glm_scholar_search", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_example_requires_explicit_bigmodel_key(monkeypatch: pytest.MonkeyPatch):
    module = _load_example_module()
    monkeypatch.delenv("SCHOLARTRACE_BIGMODEL_API_KEY", raising=False)
    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SCHOLARTRACE_BIGMODEL_API_KEY"):
        module.get_bigmodel_api_key()


def test_example_summary_prompt_stays_under_budget():
    module = _load_example_module()
    papers = [
        {
            "title": f"Paper {idx}",
            "year": 2024,
            "venue": "Venue",
            "composite_score": 0.9,
            "abstract": "A" * 12000,
        }
        for idx in range(120)
    ]

    messages = module.build_summary_messages(
        papers,
        "Theme " * 5000,
        budget=DEFAULT_PROMPT_BUDGET,
    )

    assert DEFAULT_PROMPT_BUDGET.estimate_messages(messages) <= DEFAULT_PROMPT_BUDGET.max_input_tokens


def test_example_builds_multiple_summary_batches_for_large_inputs():
    module = _load_example_module()
    small_budget = PromptBudget(
        model_context_tokens=4_000,
        response_headroom_tokens=1_000,
        tool_headroom_tokens=500,
    )
    papers = [
        {
            "title": f"Paper {idx}",
            "year": 2024,
            "venue": "Venue",
            "composite_score": 0.9,
            "abstract": "A" * 12000,
        }
        for idx in range(120)
    ]

    batches = module.build_summary_request_batches(
        papers,
        "Theme " * 200,
        budget=small_budget,
    )

    assert len(batches) > 1
    joined = "\n".join(batch[-1]["content"] for batch in batches)
    assert "Paper 0" in joined
    assert "Paper 119" in joined


def test_example_acquire_flow_uses_explicit_acquire_then_reread():
    module = _load_example_module()

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []

        def get(self, url, **kwargs):
            self.calls.append(("GET", url))
            if url.endswith("/fulltext"):
                cached_reads = sum(
                    1 for method, seen_url in self.calls if method == "GET" and seen_url.endswith("/fulltext")
                )
                if cached_reads == 1:
                    return _FakeResponse(
                        {
                            "paper_id": "paper-1",
                            "fulltext_available": False,
                            "needs_acquisition": True,
                            "access_status": "abstract_only",
                        }
                    )
                return _FakeResponse(
                    {
                        "paper_id": "paper-1",
                        "fulltext_available": True,
                        "needs_acquisition": False,
                        "access_status": "available",
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def post(self, url, **kwargs):
            self.calls.append(("POST", url))
            if url.endswith("/fulltext/acquire"):
                return _FakeResponse(
                    {
                        "paper_id": "paper-1",
                        "fulltext_available": True,
                        "needs_acquisition": False,
                        "access_status": "available",
                    }
                )
            raise AssertionError(f"unexpected POST {url}")

    client = _FakeClient()
    result = module.ensure_paper_fulltext(client, {"id": "paper-1", "title": "Paper 1"})

    assert result["fulltext_available"] is True
    assert client.calls == [
        ("GET", f"{module.SCHOLARTRACE_URL}/papers/paper-1/fulltext"),
        ("POST", f"{module.SCHOLARTRACE_URL}/papers/paper-1/fulltext/acquire"),
        ("GET", f"{module.SCHOLARTRACE_URL}/papers/paper-1/fulltext"),
    ]


def test_example_interactive_messages_are_trimmed_under_budget():
    module = _load_example_module()
    messages = [{"role": "system", "content": "system prompt"}]
    messages.extend(
        {"role": "user", "content": f"question {idx} " + ("A" * 20000)}
        for idx in range(20)
    )

    trimmed = module.prepare_chat_messages(messages, budget=DEFAULT_PROMPT_BUDGET)

    assert trimmed[0]["role"] == "system"
    assert DEFAULT_PROMPT_BUDGET.estimate_messages(trimmed) <= DEFAULT_PROMPT_BUDGET.max_input_tokens


def test_example_summary_uses_all_batches(monkeypatch: pytest.MonkeyPatch):
    module = _load_example_module()
    monkeypatch.setenv("SCHOLARTRACE_BIGMODEL_API_KEY", "test-key")
    small_budget = PromptBudget(
        model_context_tokens=4_000,
        response_headroom_tokens=1_000,
        tool_headroom_tokens=500,
    )

    papers = [
        {
            "title": f"Paper {idx}",
            "year": 2024,
            "venue": "Venue",
            "composite_score": 0.9,
            "abstract": "A" * 12000,
        }
        for idx in range(120)
    ]

    seen_prompts: list[str] = []

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_post(url, headers=None, json=None, timeout=None):
        prompt = json["messages"][-1]["content"]
        seen_prompts.append(prompt)
        if "Batch Findings:" in prompt:
            return _FakeResponse(
                {"choices": [{"message": {"content": "final synthesis"}}]}
            )
        return _FakeResponse(
            {"choices": [{"message": {"content": f"batch-{len(seen_prompts)}"}}]}
        )

    monkeypatch.setattr(module.httpx, "post", _fake_post)

    summary = module.summarize_with_glm(papers, "Theme " * 200, budget=small_budget)

    assert summary == "final synthesis"
    assert len(seen_prompts) > 2
    assert any("Paper 0" in prompt for prompt in seen_prompts)
    assert any("Paper 119" in prompt for prompt in seen_prompts)
