from __future__ import annotations

import asyncio
import json as json_lib
import os
import tempfile

import httpx
import pytest

from scholartrace.api.payloads import deepxiv_summary_payload
from scholartrace.config import Settings
from scholartrace.deepxiv.agent import DeepXivAgent, DeepXivAgentError
from scholartrace.deepxiv.token_pool import TokenPool
from scholartrace.services.prompt_budget import DEFAULT_PROMPT_BUDGET
from scholartrace.services import runtime_limits
from scholartrace.services.storage import StorageService


@pytest.fixture(autouse=True)
def _reset_runtime_budgets():
    asyncio.run(runtime_limits.budget_manager.reset())
    yield
    asyncio.run(runtime_limits.budget_manager.reset())


def test_token_pool_uses_scholartrace_settings_namespace_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPXIV_TOKEN", "legacy-token")

    pool = TokenPool.from_settings(Settings())
    assert pool.size == 0

    configured = TokenPool.from_settings(Settings(deepxiv_tokens="token-a,token-b"))
    assert configured.size == 2


@pytest.mark.asyncio
async def test_token_pool_rotates_proactively_and_tracks_cooldown_and_disable():
    pool = TokenPool.from_settings(
        Settings(
            deepxiv_tokens="token-a,token-b",
            deepxiv_auto_register=False,
            deepxiv_pool_size=2,
        )
    )

    assert await pool.get_token() == "token-a"
    assert await pool.get_token() == "token-b"

    await pool.mark_rate_limited("token-a", retry_after_seconds=60)
    assert await pool.get_token() == "token-b"

    await pool.mark_auth_failed("token-b")
    with pytest.raises(RuntimeError, match="cooldown|disabled|available"):
        await pool.get_token()


@pytest.mark.asyncio
async def test_auto_register_requires_explicit_sdk_secret():
    pool = TokenPool.from_settings(
        Settings(
            deepxiv_tokens="",
            deepxiv_auto_register=True,
            deepxiv_register_sdk_secret="",
            deepxiv_pool_size=1,
        )
    )

    with pytest.raises(RuntimeError, match="SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET"):
        await pool.get_token()


@pytest.mark.asyncio
async def test_auto_register_uses_configured_secret(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    async def _fake_register(secret: str | None = None) -> str | None:
        calls.append(secret or "")
        return "registered-token"

    monkeypatch.setattr(
        "scholartrace.deepxiv.reader.DeepXivReader.register",
        staticmethod(_fake_register),
    )

    pool = TokenPool.from_settings(
        Settings(
            deepxiv_tokens="",
            deepxiv_auto_register=True,
            deepxiv_register_sdk_secret="sdk-secret",
            deepxiv_pool_size=1,
        )
    )

    token = await pool.get_token()
    assert token == "registered-token"
    assert calls == ["sdk-secret"]


@pytest.mark.asyncio
async def test_agent_failure_does_not_select_everything(monkeypatch: pytest.MonkeyPatch):
    agent = DeepXivAgent(api_key="test-key")

    async def _failing_post(*args, **kwargs):
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(500, request=request, text="server error")
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr(agent._client, "post", _failing_post)

    papers = [
        {"title": "Paper A", "abstract": "A"},
        {"title": "Paper B", "abstract": "B"},
    ]
    filtered = await agent.filter_papers(papers, "test question")
    await agent.close()

    # When the LLM fails, the agent now falls back to deterministic ranking
    # instead of returning empty results. Verify the fallback rationale is present.
    assert len(filtered) > 0
    assert all("fallback" in (p.get("agent_reason") or "") for p in filtered)


@pytest.mark.asyncio
async def test_agent_batches_large_requests_under_prompt_budget(monkeypatch: pytest.MonkeyPatch):
    agent = DeepXivAgent(api_key="test-key")
    recorded_token_estimates: list[int] = []
    recorded_prompt_texts: list[str] = []

    async def _fake_post(url, headers=None, json=None):
        messages = json["messages"]
        recorded_token_estimates.append(
            DEFAULT_PROMPT_BUDGET.estimate_messages(messages)
        )
        user_content = messages[-1]["content"]
        recorded_prompt_texts.append(user_content)
        batch_count = user_content.count("\nAbstract:")
        response_body = {
            "choices": [
                {
                    "message": {
                        "content": str(
                            [
                                {
                                    "index": idx,
                                    "selected": False,
                                    "relevance": 0,
                                    "recency": 0,
                                    "novelty": 0,
                                    "quality": 0,
                                    "reason": "not selected",
                                }
                                for idx in range(batch_count)
                            ]
                        ).replace("'", '"')
                    }
                }
            ]
        }
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json=response_body)

    monkeypatch.setattr(agent._client, "post", _fake_post)

    papers = [
        {
            "title": f"Paper {idx}",
            "abstract": "A" * 8000,
        }
        for idx in range(80)
    ]
    filtered = await agent.filter_papers(papers, "How does this scale?")
    await agent.close()

    # LLM selected nothing, so agent falls back to deterministic ranking
    assert len(filtered) > 0
    assert all("fallback" in (p.get("agent_reason") or "") for p in filtered)
    assert len(recorded_token_estimates) > 1
    assert all(
        estimate <= DEFAULT_PROMPT_BUDGET.max_input_tokens
        for estimate in recorded_token_estimates
    )
    assert any("Paper 0" in prompt for prompt in recorded_prompt_texts)
    assert any("Paper 79" in prompt for prompt in recorded_prompt_texts)


@pytest.mark.asyncio
async def test_agent_caps_batch_size_and_sets_glm_request_options(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = DeepXivAgent(api_key="test-key")
    recorded_batch_counts: list[int] = []
    recorded_max_tokens: list[int] = []
    recorded_thinking_modes: list[str] = []

    async def _fake_post(url, headers=None, json=None):
        user_content = json["messages"][-1]["content"]
        batch_count = user_content.count("\nAbstract:")
        recorded_batch_counts.append(batch_count)
        recorded_max_tokens.append(json["max_tokens"])
        recorded_thinking_modes.append(json["thinking"]["type"])
        response_body = {
            "choices": [
                    {
                        "message": {
                            "content": json_lib.dumps(
                                [
                                    {
                                        "index": idx,
                                        "selected": idx == 0,
                                    "relevance": 8,
                                    "novelty": 7,
                                    "quality": 7,
                                    "reason": "kept",
                                }
                                for idx in range(batch_count)
                                ]
                            )
                        }
                    }
                ]
            }
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json=response_body)

    monkeypatch.setattr(agent._client, "post", _fake_post)

    papers = [
        {"title": f"Paper {idx}", "abstract": "A" * 1200}
        for idx in range(23)
    ]
    reranked = await agent.rerank_papers(papers, "How should we batch these?")
    await agent.close()

    assert len(reranked) == 23
    assert recorded_batch_counts == [10, 10, 3]
    assert recorded_max_tokens == [128_000, 128_000, 128_000]
    assert recorded_thinking_modes == ["enabled", "enabled", "enabled"]


@pytest.mark.asyncio
async def test_agent_retries_with_smaller_batches_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = DeepXivAgent(api_key="test-key")
    recorded_batch_counts: list[int] = []

    async def _fake_post(url, headers=None, json=None):
        user_content = json["messages"][-1]["content"]
        batch_count = user_content.count("\nAbstract:")
        recorded_batch_counts.append(batch_count)
        if batch_count >= 10:
            raise httpx.ReadTimeout("timed out", request=httpx.Request("POST", url))
        response_body = {
            "choices": [
                    {
                        "message": {
                            "content": json_lib.dumps(
                                [
                                    {
                                        "index": idx,
                                        "selected": True,
                                    "relevance": 8,
                                    "novelty": 7,
                                    "quality": 7,
                                    "reason": "kept",
                                }
                                for idx in range(batch_count)
                                ]
                            )
                        }
                    }
                ]
            }
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json=response_body)

    monkeypatch.setattr(agent._client, "post", _fake_post)

    papers = [
        {"title": f"Paper {idx}", "abstract": "A" * 1200}
        for idx in range(12)
    ]
    reranked = await agent.rerank_papers(papers, "How should we retry this?")
    await agent.close()

    assert len(reranked) == 12
    # Batch of 10 retries 3 times (initial + 2 retries) before shrinking to 5,
    # then two batches of 5 succeed, then the remaining 2.
    assert recorded_batch_counts == [10, 10, 10, 5, 5, 2]


@pytest.mark.asyncio
async def test_agent_includes_business_error_code_in_glm_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = DeepXivAgent(api_key="test-key")

    async def _failing_post(url, headers=None, json=None):
        request = httpx.Request("POST", url)
        response = httpx.Response(
            400,
            request=request,
            json={"error": {"code": "1261", "message": "Prompt 超长"}},
        )
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr(agent._client, "post", _failing_post)

    with pytest.raises(DeepXivAgentError, match="1261"):
        await agent.rerank_papers(
            [{"title": "Paper A", "abstract": "A"}],
            "test question",
            strict=True,
        )
    await agent.close()

def test_rest_and_mcp_wrappers_share_configured_deepxiv_settings(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = StorageService(db_path)
        storage.init_db()
        settings = Settings(
            data_dir=os.path.join(tmpdir, "data"),
            db_path=db_path,
            deepxiv_tokens="configured-token",
        )
        settings.data_dir.mkdir(parents=True, exist_ok=True)

        import scholartrace.api.rest as rest_module
        import scholartrace.api.mcp_server as mcp_module

        rest_module._storage = storage
        rest_module._settings = settings
        rest_module._deepxiv_connector_rest = None
        mcp_module._storage = storage
        mcp_module._settings = settings
        mcp_module._deepxiv_connector = None

        captured = []

        class _FakeConnector:
            def __init__(self, settings=None):
                captured.append(settings)

        monkeypatch.setattr("scholartrace.connectors.deepxiv_connector.DeepXivConnector", _FakeConnector)

        rest_module._get_deepxiv_rest()
        asyncio.run(mcp_module._get_deepxiv())

        assert captured == [settings, settings]


def test_deepxiv_summary_payload_redacts_internal_fields():
    payload = deepxiv_summary_payload(
        "2401.00001",
        {
            "title": "DeepXiv Head",
            "pdf_url": "https://example.com/paper.pdf",
            "html_url": "https://example.com/paper.html",
            "source_provenance": ["deepxiv"],
        },
        {
            "tldr": "Brief summary",
            "oa_url": "https://example.com/oa.pdf",
        },
    )

    assert payload == {
        "arxiv_id": "2401.00001",
        "metadata": {"title": "DeepXiv Head"},
        "brief": {"tldr": "Brief summary"},
    }
