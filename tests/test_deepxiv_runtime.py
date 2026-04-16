from __future__ import annotations

import asyncio
import os
import tempfile

import httpx
import pytest

from scholartrace.api.payloads import deepxiv_summary_payload
from scholartrace.config import Settings
from scholartrace.deepxiv.agent import DeepXivAgent
from scholartrace.deepxiv.token_pool import TokenPool
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

    assert filtered == []


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
