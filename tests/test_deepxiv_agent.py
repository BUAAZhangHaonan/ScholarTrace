"""Tests for DeepXivAgent timeout/retry/fallback behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from scholartrace.deepxiv.agent import DeepXivAgent


@pytest.mark.asyncio
async def test_filter_papers_fallback_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DeepXivAgent(
        api_key="test-key",
        total_timeout_seconds=1.0,
        fallback_top_k=2,
    )

    async def _slow_call(_papers, _question, *, strict: bool):
        assert strict is False
        await asyncio.sleep(2.0)
        return []

    monkeypatch.setattr(agent, "_call_llm_filter", _slow_call)

    papers = [
        {
            "title": "Transformer Language Modeling",
            "abstract": "This paper studies transformer language models.",
            "year": 2024,
            "citation_count": 12,
            "authors": ["A"],
        },
        {
            "title": "Graph Theory Basics",
            "abstract": "A graph theory introduction.",
            "year": 2024,
            "citation_count": 8,
            "authors": ["B"],
        },
        {
            "title": "Practical Prompt Engineering",
            "abstract": "Prompting methods for LLMs.",
            "year": 2023,
            "citation_count": 10,
            "authors": ["C"],
        },
    ]

    try:
        result = await agent.filter_papers(papers, "transformer language models")
    finally:
        await agent.close()

    assert len(result) == 2
    assert all("fallback_default_ranking:timeout" in paper["agent_reason"] for paper in result)


@pytest.mark.asyncio
async def test_call_llm_filter_batch_retries_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DeepXivAgent(
        api_key="test-key",
        max_retries=2,
        retry_backoff_seconds=0.01,
    )

    request = httpx.Request("POST", "https://example.com")
    success_response = httpx.Response(
        200,
        request=request,
        json={
            "choices": [
                {
                    "message": {
                        "content": '[{"index": 0, "selected": true, "relevance": 8, "novelty": 7, "quality": 9, "reason": "match"}]'
                    }
                }
            ]
        },
    )

    sleep_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("scholartrace.deepxiv.agent.asyncio.sleep", sleep_mock)

    agent._client.post = AsyncMock(
        side_effect=[httpx.ReadTimeout("timeout"), success_response]
    )

    try:
        result = await agent._call_llm_filter_batch(
            papers=[
                {
                    "title": "Paper 1",
                    "abstract": "A",
                }
            ],
            question="test",
            strict=True,
        )
    finally:
        await agent.close()

    assert len(result) == 1
    assert result[0]["index"] == 0
    assert result[0]["selected"] is True
    assert agent._client.post.await_count == 2
    sleep_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_agent_results_handles_sparse_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = DeepXivAgent(api_key="test-key")

    papers = [
        {"title": "Paper A", "abstract": "A"},
        {"title": "Paper B", "abstract": "B"},
    ]
    async def _fake_rerank(_papers, _question, *, strict: bool = True):
        assert strict is False
        return [
            {
                "index": 99,
                "selected": True,
                "agent_score": 9.0,
                "agent_rank": 1,
                "agent_rationale": "invalid",
            },
            {
                "index": 1,
                "selected": True,
                "agent_score": 8.0,
                "agent_rank": 2,
                "agent_rationale": "valid",
            },
            {
                "index": 0,
                "selected": False,
                "agent_score": 7.0,
                "agent_rank": 3,
                "agent_rationale": "rejected",
            },
        ]

    monkeypatch.setattr(agent, "rerank_papers", _fake_rerank)

    try:
        enriched = await agent.filter_papers(papers, "test")
        assert len(enriched) == 1
        assert enriched[0]["title"] == "Paper B"
        assert enriched[0]["agent_rank"] == 2
    finally:
        await agent.close()
