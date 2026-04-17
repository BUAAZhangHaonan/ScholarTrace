"""Lightweight DeepXiv agent for GLM-based reranking."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from scholartrace.services.prompt_budget import DEFAULT_PROMPT_BUDGET, PromptBudget

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a research assistant that filters academic papers for relevance.
Given a research question and a list of papers with their titles and abstracts,
you must select the most relevant papers and explain why they matter.

For each paper, assess:
1. **Relevance** (0-10): How directly does it address the research question?
2. **Novelty** (0-10): Does it introduce new methods, datasets, or insights?
3. **Quality** (0-10): Based on venue, methodology soundness, and results.

Return your analysis as a JSON array. Each element must have:
- "index": the paper index (0-based)
- "selected": true/false
- "relevance": score 0-10
- "novelty": score 0-10
- "quality": score 0-10
- "reason": one sentence explaining why selected or rejected

Only select papers with relevance >= 5. Max 20 papers can be selected.
Return ONLY the JSON array, no other text."""
_DEFAULT_BATCH_SIZE = 10
_FALLBACK_BATCH_SIZE = 5
_SINGLE_PAPER_BATCH_SIZE = 1
_GLM_MAX_TOKENS = 128_000


class DeepXivAgentError(RuntimeError):
    """Raised when GLM reranking cannot produce a trustworthy result."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        provider_code: str | None = None,
        retry_with_smaller_batch: bool = False,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.provider_code = provider_code
        self.retry_with_smaller_batch = retry_with_smaller_batch


class DeepXivAgent:
    """Agent that reranks papers using the configured BigModel GLM endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
        model: str = "glm-5-turbo",
        max_fulltext: int = 20,
        prompt_budget: PromptBudget = DEFAULT_PROMPT_BUDGET,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._max_fulltext = max_fulltext
        self._prompt_budget = prompt_budget
        self._client = httpx.AsyncClient(timeout=120.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def rerank_papers(
        self,
        papers: list[dict[str, Any]],
        question: str,
        *,
        strict: bool = True,
    ) -> list[dict[str, Any]]:
        """Return all candidate papers in reranked order with agent scores."""
        if not papers:
            return []
        if not self._api_key:
            raise DeepXivAgentError("BigModel API key is not configured")

        filter_results = await self._call_llm_filter(papers, question, strict=strict)
        reranked: list[dict[str, Any]] = []
        for result in filter_results:
            score = (
                float(result.get("relevance", 0) or 0)
                + float(result.get("novelty", 0) or 0) * 0.5
                + float(result.get("quality", 0) or 0) * 0.3
            )
            reranked.append(
                {
                    "index": result.get("index"),
                    "selected": bool(result.get("selected")),
                    "agent_score": score,
                    "agent_rationale": result.get("reason", ""),
                }
            )

        reranked.sort(
            key=lambda item: (
                not item.get("selected", False),
                -float(item.get("agent_score", 0.0) or 0.0),
                item.get("index", 0),
            )
        )
        for rank, item in enumerate(reranked, start=1):
            item["agent_rank"] = rank
        return reranked

    async def filter_papers(
        self,
        papers: list[dict[str, Any]],
        question: str,
    ) -> list[dict[str, Any]]:
        """Preserve the legacy direct-agent behavior for REST-only flows."""
        if not papers:
            return []

        try:
            reranked = await self.rerank_papers(papers, question, strict=False)
        except DeepXivAgentError as exc:
            logger.error("DeepXiv agent filtering failed: %s", exc)
            return []

        enriched: list[dict[str, Any]] = []
        for item in reranked:
            if not item.get("selected"):
                continue
            index = item.get("index")
            if not isinstance(index, int) or not (0 <= index < len(papers)):
                continue
            enriched.append(
                {
                    **papers[index],
                    "agent_score": item["agent_score"],
                    "agent_rank": item["agent_rank"],
                    "agent_reason": item["agent_rationale"],
                }
            )
        return enriched[: self._max_fulltext]

    async def _call_llm_filter(
        self,
        papers: list[dict[str, Any]],
        question: str,
        *,
        strict: bool,
    ) -> list[dict[str, Any]]:
        fixed_messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        question_text = self._prompt_budget.truncate_text(question, 2_000)

        merged_results = self._default_filter(len(papers))
        start = 0
        active_batch_limit = _DEFAULT_BATCH_SIZE
        while start < len(papers):
            batch_results, consumed, active_batch_limit = await self._run_adaptive_batch(
                papers,
                question_text,
                start=start,
                active_batch_limit=active_batch_limit,
                strict=strict,
                fixed_messages=fixed_messages,
            )
            for result in batch_results:
                local_index = result.get("index")
                if not isinstance(local_index, int):
                    continue
                global_index = start + local_index
                if 0 <= global_index < len(merged_results):
                    merged_results[global_index] = {
                        **result,
                        "index": global_index,
                    }
            start += consumed

        return merged_results

    async def _run_adaptive_batch(
        self,
        papers: list[dict[str, Any]],
        question: str,
        *,
        start: int,
        active_batch_limit: int,
        strict: bool,
        fixed_messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, int]:
        remaining = len(papers) - start
        initial_size = min(remaining, max(_SINGLE_PAPER_BATCH_SIZE, active_batch_limit))
        attempted_sizes: list[int] = []
        batch_sizes = self._batch_size_attempts(initial_size)
        last_error: DeepXivAgentError | None = None

        for batch_size in batch_sizes:
            batch_papers = papers[start:start + batch_size]
            if self._estimate_batch_messages(batch_papers, question, fixed_messages) > self._prompt_budget.max_input_tokens:
                logger.warning(
                    "GLM batch size %d exceeds prompt budget for reranking; retrying with a smaller batch",
                    batch_size,
                )
                attempted_sizes.append(batch_size)
                continue

            try:
                results = await self._call_llm_filter_batch(
                    batch_papers,
                    question,
                    strict=strict,
                )
                next_active_limit = min(active_batch_limit, batch_size)
                return results, batch_size, next_active_limit
            except DeepXivAgentError as exc:
                last_error = exc
                attempted_sizes.append(batch_size)
                has_smaller_batch = any(size < batch_size for size in batch_sizes)
                if exc.retry_with_smaller_batch and has_smaller_batch:
                    logger.warning(
                        "GLM batch size %d failed (%s); retrying with a smaller batch",
                        batch_size,
                        exc,
                    )
                    continue
                if strict:
                    raise
                return self._default_filter(batch_size), batch_size, min(active_batch_limit, batch_size)

        fallback_size = attempted_sizes[-1] if attempted_sizes else initial_size
        if last_error is not None:
            if strict:
                raise last_error
            return self._default_filter(fallback_size), fallback_size, min(active_batch_limit, fallback_size)

        message = (
            f"GLM rerank request exceeded the prompt budget even after shrinking to batch size {fallback_size}"
        )
        if strict:
            raise DeepXivAgentError(message, retry_with_smaller_batch=False)
        return self._default_filter(fallback_size), fallback_size, min(active_batch_limit, fallback_size)

    def _batch_size_attempts(self, initial_size: int) -> list[int]:
        attempts = [initial_size]
        if initial_size > _FALLBACK_BATCH_SIZE:
            attempts.append(_FALLBACK_BATCH_SIZE)
        if initial_size > _SINGLE_PAPER_BATCH_SIZE:
            attempts.append(_SINGLE_PAPER_BATCH_SIZE)
        return attempts

    def _estimate_batch_messages(
        self,
        papers: list[dict[str, Any]],
        question: str,
        fixed_messages: list[dict[str, Any]],
    ) -> int:
        paper_lines = [f"[{i}] {self._paper_snippet(paper)}" for i, paper in enumerate(papers)]
        user_msg = f"Research Question: {question}\n\nPapers:\n{chr(10).join(paper_lines)}"
        messages = fixed_messages + [{"role": "user", "content": user_msg}]
        return self._prompt_budget.estimate_messages(messages)

    def _request_payload(self, user_msg: str) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "thinking": {"type": "enabled"},
            "max_tokens": _GLM_MAX_TOKENS,
            "temperature": 0.3,
        }

    def _format_http_error(self, response: httpx.Response) -> tuple[str, str | None]:
        provider_code: str | None = None
        provider_message: str | None = None

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                raw_code = error_payload.get("code")
                if raw_code is not None:
                    provider_code = str(raw_code)
                raw_message = error_payload.get("message")
                if raw_message:
                    provider_message = str(raw_message)

        if provider_code and provider_message:
            return (
                f"GLM request failed with status {response.status_code} "
                f"(business code {provider_code}: {provider_message})",
                provider_code,
            )
        if provider_code:
            return (
                f"GLM request failed with status {response.status_code} "
                f"(business code {provider_code})",
                provider_code,
            )
        if provider_message:
            return (
                f"GLM request failed with status {response.status_code}: {provider_message}",
                None,
            )

        raw_text = response.text.strip()
        if raw_text:
            return (
                f"GLM request failed with status {response.status_code}: {raw_text[:500]}",
                None,
            )
        return f"GLM request failed with status {response.status_code}", None

    def _paper_snippet(self, paper: dict[str, Any]) -> str:
        title = paper.get("title", "Unknown")
        abstract = self._prompt_budget.truncate_text(paper.get("abstract") or "", 1_500)
        return f"{title}\nAbstract: {abstract}"

    async def _call_llm_filter_batch(
        self,
        papers: list[dict[str, Any]],
        question: str,
        *,
        strict: bool,
    ) -> list[dict[str, Any]]:
        paper_lines = [f"[{i}] {self._paper_snippet(paper)}" for i, paper in enumerate(papers)]
        user_msg = f"Research Question: {question}\n\nPapers:\n{chr(10).join(paper_lines)}"
        payload = self._request_payload(user_msg)

        try:
            resp = await self._client.post(
                self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except httpx.TimeoutException as exc:
            logger.error("GLM API timeout while reranking batch of %d papers", len(papers))
            if not strict:
                raise DeepXivAgentError(
                    "GLM request timed out while reranking papers",
                    retry_with_smaller_batch=len(papers) > 1,
                ) from exc
            raise DeepXivAgentError(
                "GLM request timed out while reranking papers",
                retry_with_smaller_batch=len(papers) > 1,
            ) from exc
        except httpx.HTTPStatusError as exc:
            error_message, provider_code = self._format_http_error(exc.response)
            logger.error("GLM API error: %s", error_message)
            retry_with_smaller_batch = (
                len(papers) > 1
                and (
                    exc.response.status_code == 400
                    or provider_code == "1261"
                )
            )
            raise DeepXivAgentError(
                error_message,
                http_status=exc.response.status_code,
                provider_code=provider_code,
                retry_with_smaller_batch=retry_with_smaller_batch,
            ) from exc
        except (KeyError, IndexError) as exc:
            logger.error("Invalid GLM response structure: %s", exc)
            if not strict:
                return self._default_filter(len(papers))
            raise DeepXivAgentError("GLM response was missing the expected content") from exc

        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        try:
            results = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM filter response: %s", exc)
            if not strict:
                return self._default_filter(len(papers))
            raise DeepXivAgentError("GLM reranking returned invalid JSON") from exc

        if isinstance(results, list):
            return results

        logger.warning("Unexpected LLM response format: %s", type(results))
        if not strict:
            return self._default_filter(len(papers))
        raise DeepXivAgentError("GLM reranking returned an unexpected payload format")

    @staticmethod
    def _default_filter(count: int) -> list[dict[str, Any]]:
        return [
            {
                "index": i,
                "selected": False,
                "relevance": 0,
                "novelty": 0,
                "quality": 0,
                "reason": "Filtering unavailable",
            }
            for i in range(count)
        ]
