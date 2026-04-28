"""Lightweight DeepXiv agent for GLM-based reranking."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

import httpx

from scholartrace.models.schemas import Theme, Work
from scholartrace.services.prompt_budget import DEFAULT_PROMPT_BUDGET, PromptBudget
from scholartrace.services.ranking import rank_papers

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """You are a research assistant that filters academic papers for relevance.
Given a research question and a list of papers with their titles, abstracts, and publication years,
you must select the most relevant papers and explain why they matter.

Current date: {current_date}
Current year: {current_year}

For each paper, assess:
1. **Relevance** (0-10): How directly does it address the research question? Be strict — a paper must
   substantively engage with the core topics of the research question, not merely share a broad field.
   Papers that only tangentially mention keywords should score <= 3.
2. **Recency** (0-10): How recent is this paper? Strongly prefer recent work:
   - Published in {current_year} or same month as current date: 9-10
   - Published in {current_year_minus_1}: 7-8
   - Published in {current_year_minus_2}: 5-6
   - Published in {current_year_minus_3}: 3-4
   - Older than {current_year_minus_3}: 0-2, unless it is a seminal/highly-cited foundational work
3. **Novelty** (0-10): Does it introduce new methods, datasets, or insights?
4. **Quality** (0-10): Based on venue, methodology soundness, and results.

Return your analysis as a JSON array. Each element must have:
- "index": the paper index (0-based)
- "selected": true/false
- "relevance": score 0-10
- "recency": score 0-10
- "novelty": score 0-10
- "quality": score 0-10
- "reason": one sentence explaining why selected or rejected

SELECTION RULES:
- Only select papers with relevance >= 5 AND recency >= 3.
- EXCEPTION: Older papers (recency < 3) may be selected ONLY if relevance >= 8 AND quality >= 8
  (i.e., they are foundational/seminal works that are essential to the topic).
- Max 20 papers can be selected.
- When deciding between two papers of similar relevance, always prefer the more recent one.
Return ONLY the JSON array, no other text."""
_STAGE1_SCORING_PROMPT_TEMPLATE = """You are a research assistant scoring academic papers for relevance to a research question.
Your task is ONLY to score each paper — do NOT decide whether to select or reject.

Current date: {current_date}
Current year: {current_year}

IMPORTANT: You must use a CONSISTENT scoring standard across all batches. Apply these calibration anchors:
- A paper that is the exact topic of the research question should score relevance 9-10.
- A paper in the same broad field but not addressing the core question should score relevance 4-6.
- A paper only tangentially related should score relevance 1-3.

For each paper, assess:
1. **Relevance** (0-10): How directly does it address the research question? Be strict — a paper must
   substantively engage with the core topics, not merely share a broad field.
2. **Recency** (0-10): How recent is this paper?
   - Published in {current_year} or same month as current date: 9-10
   - Published in {current_year_minus_1}: 7-8
   - Published in {current_year_minus_2}: 5-6
   - Published in {current_year_minus_3}: 3-4
   - Older than {current_year_minus_3}: 0-2, unless seminal/highly-cited foundational work
3. **Novelty** (0-10): Does it introduce new methods, datasets, or insights?
4. **Quality** (0-10): Based on venue, methodology soundness, and results.

Return your analysis as a JSON array. Each element must have:
- "index": the paper index (0-based)
- "relevance": score 0-10
- "recency": score 0-10
- "novelty": score 0-10
- "quality": score 0-10
- "reason": one sentence explaining the score

Return ONLY the JSON array, no other text."""

_STAGE2_SELECTION_PROMPT_TEMPLATE = """You are a research assistant performing FINAL selection of academic papers for a research question.
You will receive papers that have already been pre-screened and scored. Your job is to:

1. Compare papers AGAINST EACH OTHER (not in isolation).
2. Select the top {max_select} most valuable papers.
3. Rank them from most to least valuable.

Current date: {current_date}
Current year: {current_year}

For each paper you will see pre-computed scores. Use them as a guide but apply your own judgment:
- Prefer papers that form a coherent set covering different aspects of the research question.
- Avoid selecting papers with nearly identical content — prefer diversity.
- A highly relevant older paper beats a marginally relevant newer paper.
- Among equally relevant papers, prefer higher quality venues and more citations.

Return your analysis as a JSON array. Each element must have:
- "index": the paper index (0-based in the provided list)
- "selected": true if this paper should be in the final set, false otherwise
- "final_rank": rank among selected papers (1 = best), null if not selected
- "relevance": your reassessed relevance score 0-10
- "reason": one sentence explaining why selected or rejected

SELECTION RULES:
- Select exactly {max_select} papers (or fewer if not enough qualify).
- Only select papers with relevance >= 5.
- Max selection: {max_select} papers.
Return ONLY the JSON array, no other text."""

_DEFAULT_BATCH_SIZE = 10
_FALLBACK_BATCH_SIZE = 5
_SINGLE_PAPER_BATCH_SIZE = 1
_GLM_MAX_TOKENS = 128_000
_QWEN_MAX_TOKENS = 32_768
_DEEPSEEK_MAX_TOKENS = 128_000


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
    """Agent that reranks papers using the configured LLM endpoint.

    Supports two backends:
    - "glm": Zhipu BigModel GLM API (sends thinking: enabled)
    - "qwen": Local Qwen via vLLM (sends chat_template_kwargs to disable thinking)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
        model: str = "glm-5-turbo",
        backend: str = "glm",
        max_fulltext: int = 20,
        prompt_budget: PromptBudget = DEFAULT_PROMPT_BUDGET,
        request_timeout_seconds: float = 45.0,
        total_timeout_seconds: float = 120.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 2.0,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        fallback_top_k: int = 20,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._backend = backend
        self._max_fulltext = max_fulltext
        self._prompt_budget = prompt_budget
        self._request_timeout_seconds = max(1.0, request_timeout_seconds)
        self._total_timeout_seconds = (
            max(1.0, total_timeout_seconds) if total_timeout_seconds is not None else None
        )
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.1, retry_backoff_seconds)
        self._batch_size = max(_SINGLE_PAPER_BATCH_SIZE, batch_size)
        self._fallback_top_k = max(1, fallback_top_k)
        # Local backends (Qwen/vLLM) bypass proxy to avoid SOCKS issues
        # Timeout strategy: 5s connect, unlimited read (let model process)
        client_kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(
                connect=self._request_timeout_seconds,  # 5s: fail fast if API unreachable
                read=120.0,   # 2min: balance between model processing and responsiveness
                write=5.0,
                pool=5.0,
            ),
        }
        if backend == "qwen":
            client_kwargs["trust_env"] = False
        self._client = httpx.AsyncClient(**client_kwargs)

    async def close(self) -> None:
        await self._client.aclose()

    async def score_papers_batch(
        self,
        papers: list[dict[str, Any]],
        question: str,
        *,
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """Score a single batch of papers using a custom system prompt (Stage 1).

        Returns a list of scored result dicts with index, relevance, recency,
        novelty, quality, and reason.  Raises DeepXivAgentError on failure.
        """
        if not papers:
            return []
        if not self._api_key:
            raise DeepXivAgentError("API key not configured for scoring")

        paper_lines = [
            f"[{i}] {self.paper_snippet(paper, self._prompt_budget)}"
            for i, paper in enumerate(papers)
        ]
        user_msg = f"Research Question: {question}\n\nPapers:\n{chr(10).join(paper_lines)}"
        payload = self._request_payload_with_system_prompt(user_msg, system_prompt)

        content = await self._raw_llm_call(payload, len(papers))
        results = self._parse_json_response(content, len(papers), strict=True)
        return results

    async def select_papers(
        self,
        papers: list[dict[str, Any]],
        question: str,
        *,
        max_select: int,
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """Select and rank papers from a pre-screened candidate set (Stage 2).

        Returns a list of result dicts with index, selected, final_rank,
        relevance, and reason.  Raises DeepXivAgentError on failure.
        """
        if not papers:
            return []
        if not self._api_key:
            raise DeepXivAgentError("API key not configured for selection")

        paper_lines = [
            f"[{i}] {self.paper_snippet(paper, self._prompt_budget)}"
            for i, paper in enumerate(papers)
        ]
        user_msg = f"Research Question: {question}\n\nPapers:\n{chr(10).join(paper_lines)}"
        payload = self._request_payload_with_system_prompt(user_msg, system_prompt)

        content = await self._raw_llm_call(payload, len(papers))
        results = self._parse_json_response(content, len(papers), strict=True)
        return results

    def _request_payload_with_system_prompt(
        self, user_msg: str, system_prompt: str,
    ) -> dict[str, Any]:
        """Build a request payload with an explicit system prompt override."""
        if self._backend == "qwen":
            return {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": _QWEN_MAX_TOKENS,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        if self._backend == "deepseek":
            return {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": _DEEPSEEK_MAX_TOKENS,
                "temperature": 0.3,
            }
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "thinking": {"type": "enabled"},
            "max_tokens": _GLM_MAX_TOKENS,
            "temperature": 0.3,
        }

    async def _raw_llm_call(
        self, payload: dict[str, Any], paper_count: int,
    ) -> str:
        """Execute a single LLM call and return the content string.

        Retries up to max_retries times on transient errors.
        Raises DeepXivAgentError if all attempts fail.
        """
        content: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post(
                    self._base_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

                # Handle GLM business errors returned with HTTP 200
                if "error" in data and "choices" not in data:
                    error_obj = data["error"]
                    biz_code = error_obj.get("code", "unknown") if isinstance(error_obj, dict) else "unknown"
                    biz_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
                    error_message = f"GLM business error (code {biz_code}): {biz_msg}"
                    if attempt < self._max_retries:
                        await self._sleep_before_retry(attempt, f"business error {biz_code}")
                        continue
                    raise DeepXivAgentError(error_message, provider_code=str(biz_code))

                content = data["choices"][0]["message"]["content"]
                break
            except httpx.TimeoutException as exc:
                if attempt < self._max_retries:
                    await self._sleep_before_retry(attempt, "timed out")
                    continue
                raise DeepXivAgentError(
                    "LLM request timed out",
                    retry_with_smaller_batch=paper_count > 1,
                ) from exc
            except httpx.RequestError as exc:
                if attempt < self._max_retries:
                    await self._sleep_before_retry(attempt, "failed")
                    continue
                raise DeepXivAgentError(
                    "LLM request failed",
                    retry_with_smaller_batch=paper_count > 1,
                ) from exc
            except httpx.HTTPStatusError as exc:
                error_message, provider_code = self._format_http_error(exc.response)
                retryable = exc.response.status_code in {429, 500, 502, 503, 504}
                if retryable and attempt < self._max_retries:
                    await self._sleep_before_retry(attempt, f"HTTP {exc.response.status_code}")
                    continue
                raise DeepXivAgentError(
                    error_message,
                    http_status=exc.response.status_code,
                    provider_code=provider_code,
                    retry_with_smaller_batch=paper_count > 1,
                ) from exc
            except (KeyError, IndexError) as exc:
                if attempt < self._max_retries:
                    await self._sleep_before_retry(attempt, "malformed response")
                    continue
                raise DeepXivAgentError("LLM response was missing expected content") from exc

        if content is None:
            raise DeepXivAgentError("LLM response content was empty")

        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        return content.strip()

    def _parse_json_response(
        self, content: str, paper_count: int, *, strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Parse JSON from LLM response content."""
        try:
            results = json.loads(content)
        except json.JSONDecodeError as exc:
            if not strict:
                return self._default_filter(paper_count)
            raise DeepXivAgentError("LLM returned invalid JSON") from exc

        if isinstance(results, list):
            return results
        if not strict:
            return self._default_filter(paper_count)
        raise DeepXivAgentError("LLM returned an unexpected payload format")

    async def rerank_papers(
        self,
        papers: list[dict[str, Any]],
        question: str,
        *,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Return all candidate papers in reranked order with agent scores."""
        if not papers:
            return []
        if not self._api_key:
            if strict:
                raise DeepXivAgentError("BigModel API key is not configured")
            logger.warning("BigModel API key is not configured, using fallback ranking")
            return self._fallback_rerank(papers, question, reason_tag="missing_api_key")

        try:
            if self._total_timeout_seconds is not None:
                async with asyncio.timeout(self._total_timeout_seconds):
                    filter_results = await self._call_llm_filter(papers, question, strict=strict)
            else:
                # No total timeout — let the model process without time limit
                filter_results = await self._call_llm_filter(papers, question, strict=strict)
        except TimeoutError as exc:
            message = (
                f"GLM reranking timed out after {self._total_timeout_seconds:.1f}s"
            )
            logger.warning(message)
            if strict:
                raise DeepXivAgentError(
                    message,
                    retry_with_smaller_batch=True,
                ) from exc
            return self._fallback_rerank(papers, question, reason_tag="timeout")
        except DeepXivAgentError as exc:
            if strict:
                raise
            logger.warning("DeepXiv agent reranking failed: %s", exc)
            return self._fallback_rerank(papers, question, reason_tag="agent_error")
        except Exception as exc:
            if strict:
                raise DeepXivAgentError(
                    f"GLM reranking failed unexpectedly: {exc}",
                    retry_with_smaller_batch=False,
                ) from exc
            logger.exception("Unexpected DeepXiv reranking failure")
            return self._fallback_rerank(papers, question, reason_tag="unexpected_error")

        reranked: list[dict[str, Any]] = []
        for result in filter_results:
            recency = float(result.get("recency", 0) or 0)
            score = (
                float(result.get("relevance", 0) or 0)
                + recency * 0.6
                + float(result.get("novelty", 0) or 0) * 0.3
                + float(result.get("quality", 0) or 0) * 0.2
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

        if not strict and not any(item.get("selected", False) for item in reranked):
            logger.warning("Agent selected no papers, using fallback ranking")
            return self._fallback_rerank(papers, question, reason_tag="no_selected")

        return reranked

    async def filter_papers(
        self,
        papers: list[dict[str, Any]],
        question: str,
    ) -> list[dict[str, Any]]:
        """Preserve the legacy direct-agent behavior for REST-only flows."""
        if not papers:
            return []

        reranked = await self.rerank_papers(papers, question, strict=False)

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

        if not enriched:
            return []

        return enriched[: self._max_fulltext]

    def _fallback_rerank(
        self,
        papers: list[dict[str, Any]],
        question: str,
        *,
        reason_tag: str,
    ) -> list[dict[str, Any]]:
        """Fallback to deterministic local ranking when GLM is unavailable."""
        theme = Theme(parsed_queries=[question] if question else [])

        works: list[Work] = []
        index_by_work_id: dict[str, int] = {}
        for idx, paper in enumerate(papers):
            citation_count = paper.get("citation_count", 0)
            if not isinstance(citation_count, int):
                citation_count = 0
            citation_count = max(citation_count, 0)

            year = paper.get("year")
            if not isinstance(year, int):
                year = None

            work = Work(
                title=str(paper.get("title") or ""),
                abstract=paper.get("abstract") or None,
                authors=list(paper.get("authors") or []),
                year=year,
                venue=paper.get("venue") or None,
                citation_count=citation_count,
                source_provenance=["deepxiv"],
            )
            works.append(work)
            index_by_work_id[work.id] = idx

        ranked = rank_papers(works, theme)
        selected_cap = min(self._fallback_top_k, len(ranked))

        reranked: list[dict[str, Any]] = []
        for rank, work in enumerate(ranked, start=1):
            original_index = index_by_work_id[work.id]
            reranked.append(
                {
                    "index": original_index,
                    "selected": rank <= selected_cap,
                    "agent_score": round(max(work.composite_score, 0.0) * 10, 4),
                    "agent_rationale": f"fallback_default_ranking:{reason_tag}",
                    "agent_rank": rank,
                }
            )
        return reranked

    @staticmethod
    def build_stage1_prompt() -> str:
        """Build the Stage 1 scoring system prompt with date injected."""
        now = datetime.now()
        current_year = now.year
        return _STAGE1_SCORING_PROMPT_TEMPLATE.format(
            current_date=now.strftime("%Y-%m-%d"),
            current_year=current_year,
            current_year_minus_1=current_year - 1,
            current_year_minus_2=current_year - 2,
            current_year_minus_3=current_year - 3,
        )

    @staticmethod
    def build_stage2_prompt(max_select: int) -> str:
        """Build the Stage 2 selection system prompt with date injected."""
        now = datetime.now()
        current_year = now.year
        return _STAGE2_SELECTION_PROMPT_TEMPLATE.format(
            current_date=now.strftime("%Y-%m-%d"),
            current_year=current_year,
            current_year_minus_1=current_year - 1,
            current_year_minus_2=current_year - 2,
            current_year_minus_3=current_year - 3,
            max_select=max_select,
        )

    @staticmethod
    def paper_snippet(paper: dict[str, Any], budget: PromptBudget | None = None) -> str:
        """Format a paper dict into a concise text snippet for LLM prompts."""
        budget = budget or DEFAULT_PROMPT_BUDGET
        title = paper.get("title", "Unknown")
        year = paper.get("year")
        year_str = f" (Year: {year})" if isinstance(year, int) else " (Year: unknown)"
        venue = paper.get("venue")
        venue_str = f" [Venue: {venue}]" if venue else ""
        citations = paper.get("citation_count")
        cite_str = f" [Citations: {citations}]" if isinstance(citations, int) and citations > 0 else ""
        abstract = budget.truncate_text(paper.get("abstract") or "", 1_500)
        return f"{title}{year_str}{venue_str}{cite_str}\nAbstract: {abstract}"

    @staticmethod
    def _build_system_prompt() -> str:
        """Build the system prompt with the current date injected."""
        now = datetime.now()
        current_year = now.year
        return _SYSTEM_PROMPT_TEMPLATE.format(
            current_date=now.strftime("%Y-%m-%d"),
            current_year=current_year,
            current_year_minus_1=current_year - 1,
            current_year_minus_2=current_year - 2,
            current_year_minus_3=current_year - 3,
        )

    async def _call_llm_filter(
        self,
        papers: list[dict[str, Any]],
        question: str,
        *,
        strict: bool,
    ) -> list[dict[str, Any]]:
        system_prompt = self._build_system_prompt()
        fixed_messages = [{"role": "system", "content": system_prompt}]
        question_text = self._prompt_budget.truncate_text(question, 2_000)

        merged_results = self._default_filter(len(papers))
        start = 0
        active_batch_limit = self._batch_size
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
        system_prompt = self._build_system_prompt()
        if self._backend == "qwen":
            return {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": _QWEN_MAX_TOKENS,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        if self._backend == "deepseek":
            return {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": _DEEPSEEK_MAX_TOKENS,
                "temperature": 0.3,
            }
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
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
        return self.paper_snippet(paper, self._prompt_budget)

    async def _sleep_before_retry(self, attempt: int, reason: str) -> None:
        backoff = self._retry_backoff_seconds * (2 ** attempt)
        logger.warning(
            "GLM request %s, retrying in %.2fs (attempt %d/%d)",
            reason,
            backoff,
            attempt + 1,
            self._max_retries,
        )
        await asyncio.sleep(backoff)

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

        content: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post(
                    self._base_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

                # Handle GLM business errors returned with HTTP 200
                # (e.g. rate limit 1302, quota errors, etc.)
                if "error" in data and "choices" not in data:
                    error_obj = data["error"]
                    biz_code = error_obj.get("code", "unknown") if isinstance(error_obj, dict) else "unknown"
                    biz_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
                    error_message = f"GLM business error (code {biz_code}): {biz_msg}"
                    logger.error("GLM API error: %s", error_message)
                    if attempt < self._max_retries:
                        await self._sleep_before_retry(attempt, f"business error {biz_code}")
                        continue
                    raise DeepXivAgentError(
                        error_message,
                        provider_code=str(biz_code),
                    )

                content = data["choices"][0]["message"]["content"]
                break
            except httpx.TimeoutException as exc:
                logger.error("GLM API timeout while reranking batch of %d papers", len(papers))
                if attempt < self._max_retries:
                    await self._sleep_before_retry(attempt, "timed out")
                    continue
                raise DeepXivAgentError(
                    "GLM request timed out while reranking papers",
                    retry_with_smaller_batch=len(papers) > 1,
                ) from exc
            except httpx.RequestError as exc:
                logger.error("GLM API request error while reranking batch: %s", exc)
                if attempt < self._max_retries:
                    await self._sleep_before_retry(attempt, "failed")
                    continue
                raise DeepXivAgentError(
                    "GLM request failed while reranking papers",
                    retry_with_smaller_batch=len(papers) > 1,
                ) from exc
            except httpx.HTTPStatusError as exc:
                error_message, provider_code = self._format_http_error(exc.response)
                logger.error("GLM API error: %s", error_message)
                retryable_status = exc.response.status_code in {429, 500, 502, 503, 504}
                if retryable_status and attempt < self._max_retries:
                    await self._sleep_before_retry(attempt, f"returned HTTP {exc.response.status_code}")
                    continue
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
                if attempt < self._max_retries:
                    await self._sleep_before_retry(attempt, "returned malformed payload")
                    continue
                if not strict:
                    return self._default_filter(len(papers))
                raise DeepXivAgentError("GLM response was missing the expected content") from exc

        if content is None:
            if not strict:
                return self._default_filter(len(papers))
            raise DeepXivAgentError("GLM response content was empty")

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
                "recency": 0,
                "novelty": 0,
                "quality": 0,
                "reason": "Filtering unavailable",
            }
            for i in range(count)
        ]
