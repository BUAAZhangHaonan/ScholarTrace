"""Core retrieval orchestration for ScholarTrace."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from scholartrace.config import Settings, get_settings
from scholartrace.connectors.arxiv import ArxivConnector
from scholartrace.connectors.base import BaseConnector
from scholartrace.connectors.crossref import CrossrefConnector
from scholartrace.connectors.deepxiv_connector import DeepXivConnector
from scholartrace.connectors.dblp import DblpConnector
from scholartrace.connectors.openalex import OpenAlexConnector
from scholartrace.connectors.openreview import OpenReviewConnector
from scholartrace.connectors.semantic_scholar import SemanticScholarConnector
from scholartrace.deepxiv.agent import DeepXivAgent, DeepXivAgentError
from scholartrace.models.schemas import RawCandidate, Theme, Work
from scholartrace.services.dedup import deduplicate_candidates_async
from scholartrace.services.prompt_budget import DEFAULT_PROMPT_BUDGET, PromptBudget
from scholartrace.services.ranking import rank_papers_async
from scholartrace.services.storage import StorageService
from scholartrace.services.theme_parser import parse_theme
from scholartrace.services.timing import PipelineTimer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi-model concurrent pool
# ---------------------------------------------------------------------------
# Each model gets its own semaphore with a configured concurrency limit.
# When a request arrives, we try models in priority order; if the top model's
# semaphore is exhausted (or the model is in cooldown after a recent error),
# we move on to the next one.  This lets different models handle different
# requests *simultaneously*, maximising throughput under load.


@dataclass
class ModelPoolEntry:
    """One model slot in the concurrent pool."""

    backend: str
    model: str
    api_key: str
    base_url: str
    max_concurrent: int
    semaphore: asyncio.Semaphore
    last_error_time: float = 0.0
    error_count: int = 0

    @property
    def in_cooldown(self) -> bool:
        settings = get_settings()
        return (time.monotonic() - self.last_error_time) < settings.model_pool_cooldown_seconds

    def record_error(self) -> None:
        self.error_count += 1
        self.last_error_time = time.monotonic()

    def record_success(self) -> None:
        self.error_count = 0


class ModelPool:
    """Priority-ordered pool of LLM models with per-model concurrency control."""

    _instance: ModelPool | None = None

    def __init__(self, entries: list[ModelPoolEntry]) -> None:
        self._entries = entries

    @classmethod
    def get(cls, settings: Settings) -> ModelPool:
        """Return (and lazily create) the singleton ModelPool."""
        if cls._instance is None:
            cls._instance = cls._build(settings)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton (useful in tests)."""
        cls._instance = None

    # ---- acquisition ---------------------------------------------------

    async def acquire(self, timeout: float = 5.0) -> ModelPoolEntry:
        """Acquire the first available (not in cooldown, semaphore free) model.

        Waits up to *timeout* seconds.  Raises RuntimeError if nothing frees
        up in time.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for entry in self._entries:
                if entry.in_cooldown:
                    continue
                # Non-blocking try-acquire
                try:
                    await asyncio.wait_for(entry.semaphore.acquire(), timeout=0.05)
                    return entry
                except asyncio.TimeoutError:
                    continue
            # Nothing free right now — brief pause before retrying
            await asyncio.sleep(0.05)
        raise RuntimeError("ModelPool: no model available within timeout")

    def release(self, entry: ModelPoolEntry, *, success: bool) -> None:
        """Release a model slot back to the pool."""
        if success:
            entry.record_success()
        else:
            entry.record_error()
        entry.semaphore.release()

    @property
    def entries(self) -> list[ModelPoolEntry]:
        return list(self._entries)

    # ---- build ---------------------------------------------------------

    @staticmethod
    def _build(settings: Settings) -> ModelPool:
        entries: list[ModelPoolEntry] = []
        model_path = settings.model_path

        if model_path == "deepseek_flash":
            # DeepSeek Flash path: single model with 1M context
            entries.append(ModelPoolEntry(
                backend="deepseek",
                model=settings.deepseek_flash_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                max_concurrent=settings.deepseek_flash_max_concurrent,
                semaphore=asyncio.Semaphore(settings.deepseek_flash_max_concurrent),
            ))
        elif model_path == "glm_extended":
            # GLM Extended path: glm-4.6 + glm-4.5 pool
            for model_name in settings.glm_extended_models.split(","):
                model_name = model_name.strip()
                if not model_name:
                    continue
                entries.append(ModelPoolEntry(
                    backend="glm",
                    model=model_name,
                    api_key=settings.bigmodel_api_key,
                    base_url=settings.bigmodel_base_url,
                    max_concurrent=settings.glm_extended_max_concurrent,
                    semaphore=asyncio.Semaphore(settings.glm_extended_max_concurrent),
                ))
        else:
            # Default path: glm-5-turbo primary + fallbacks + deepseek + qwen
            entries.append(ModelPoolEntry(
                backend="glm",
                model=settings.bigmodel_model,
                api_key=settings.bigmodel_api_key,
                base_url=settings.bigmodel_base_url,
                max_concurrent=settings.glm_primary_max_concurrent,
                semaphore=asyncio.Semaphore(settings.glm_primary_max_concurrent),
            ))

            for model_name in settings.bigmodel_fallback_models.split(","):
                model_name = model_name.strip()
                if not model_name:
                    continue
                entries.append(ModelPoolEntry(
                    backend="glm",
                    model=model_name,
                    api_key=settings.bigmodel_api_key,
                    base_url=settings.bigmodel_base_url,
                    max_concurrent=settings.glm_fallback_max_concurrent,
                    semaphore=asyncio.Semaphore(settings.glm_fallback_max_concurrent),
                ))

            if settings.deepseek_api_key.strip():
                entries.append(ModelPoolEntry(
                    backend="deepseek",
                    model=settings.deepseek_model,
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                    max_concurrent=settings.deepseek_max_concurrent,
                    semaphore=asyncio.Semaphore(settings.deepseek_max_concurrent),
                ))

            if settings.qwen_base_url.strip():
                entries.append(ModelPoolEntry(
                    backend="qwen",
                    model=settings.qwen_model,
                    api_key=settings.qwen_api_key,
                    base_url=settings.qwen_base_url,
                    max_concurrent=settings.qwen_max_concurrent,
                    semaphore=asyncio.Semaphore(settings.qwen_max_concurrent),
                ))

        logger.info(
            "[PIPELINE] ModelPool initialised (path=%s) with %d models: %s",
            model_path,
            len(entries),
            ", ".join(f"{e.model}({e.max_concurrent})" for e in entries),
        )
        return ModelPool(entries)


@dataclass(frozen=True)
class QueryPipelineResult:
    theme: Theme
    total_retrieved: int
    total_after_dedup: int
    total_after_first_stage: int
    total_agent_candidates: int
    total_final: int
    works: list[Work]


class QueryPipelineConfigurationError(ValueError):
    """Raised when the MCP query pipeline lacks required configuration."""


class QueryPipelineRuntimeError(RuntimeError):
    """Raised when the MCP query pipeline cannot finish honestly."""


def _has_usable_deepxiv_tokens(settings: Settings) -> bool:
    return any(token.strip() for token in settings.deepxiv_tokens.split(","))


def _candidate_to_work(candidate: RawCandidate) -> Work:
    """Convert a deduplicated RawCandidate into a Work object."""
    return Work(
        doi=candidate.doi,
        arxiv_id=candidate.arxiv_id,
        openalex_id=candidate.openalex_id,
        s2_id=candidate.s2_id,
        dblp_key=candidate.dblp_key,
        openreview_id=candidate.openreview_id,
        title=candidate.title,
        authors=candidate.authors,
        year=candidate.year,
        venue=candidate.venue,
        abstract=candidate.abstract,
        citation_count=candidate.citation_count,
        reference_count=candidate.reference_count,
        source_provenance=candidate.source_provenance,
        pdf_url=candidate.pdf_url,
        html_url=candidate.html_url,
        oa_url=candidate.oa_url,
    )


def _build_connectors(settings: Settings) -> list[BaseConnector]:
    """Instantiate the unified source connectors for retrieval."""
    connectors: list[BaseConnector] = [
        OpenAlexConnector(settings=settings),
        ArxivConnector(settings=settings),
        SemanticScholarConnector(settings=settings),
        DblpConnector(settings=settings),
        OpenReviewConnector(settings=settings),
        CrossrefConnector(settings=settings),
    ]

    deepxiv_configured = _has_usable_deepxiv_tokens(settings) or (
        settings.deepxiv_auto_register
        and bool(settings.deepxiv_register_sdk_secret.strip())
    )
    if deepxiv_configured:
        connectors.append(DeepXivConnector(settings=settings))
    else:
        logger.info(
            "DeepXiv is not configured for unified retrieval; skipping DeepXiv connector"
        )

    return connectors


async def _fan_out_query(
    connectors: list[BaseConnector],
    query: str,
    max_results: int,
    *,
    connector_timeout: float = 45.0,
) -> list[RawCandidate]:
    """Run one query against all connectors concurrently, tolerating individual failures.

    Each connector is wrapped with *connector_timeout* so a slow source cannot
    block the entire retrieval.
    """
    timer = PipelineTimer(f"fan_out_query({query[:50]})")

    async def _safe_search(connector: BaseConnector) -> list[RawCandidate]:
        with timer.stage(f"connector: {connector.source_name}"):
            try:
                return await asyncio.wait_for(
                    connector.search(query, max_results=max_results),
                    timeout=connector_timeout,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"{connector.source_name} timed out after {connector_timeout:.0f}s"
                )

    tasks = [_safe_search(c) for c in connectors]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: list[RawCandidate] = []
    for connector, result in zip(connectors, results):
        if isinstance(result, Exception):
            logger.warning(
                "Connector %s failed for query %r: %s",
                connector.source_name,
                query,
                result,
            )
            continue
        candidates.extend(result)
    timer.log_summary()
    return candidates


async def _collect_ranked_works(
    theme: Theme,
    storage: StorageService,
    settings: Settings,
) -> tuple[list[Work], int, int]:
    max_attempts = 1 + max(0, settings.retrieval_query_max_retries)
    timer = PipelineTimer("collect_ranked_works")

    for attempt in range(max_attempts):
        connectors = _build_connectors(settings)
        try:
            if attempt == 0:
                storage.save_theme(theme)

            all_candidates: list[RawCandidate] = []
            with timer.stage("all_queries_fan_out"):
                query_results = await asyncio.gather(
                    *[
                        _fan_out_query(
                            connectors,
                            query,
                            settings.max_results_per_source_per_query,
                            connector_timeout=settings.retrieval_connector_timeout_seconds,
                        )
                        for query in theme.parsed_queries
                    ],
                    return_exceptions=True,
                )
                for result in query_results:
                    if isinstance(result, Exception):
                        logger.warning("Query fan-out failed: %s", result)
                        continue
                    all_candidates.extend(result)

            if not all_candidates and attempt < max_attempts - 1:
                logger.warning(
                    "[PIPELINE] All connectors returned 0 candidates (attempt %d/%d), retrying",
                    attempt + 1, max_attempts,
                )
                continue

            logger.info(
                "[PIPELINE] Collected %d raw candidates across %d queries",
                len(all_candidates),
                len(theme.parsed_queries),
            )

            deduped = await deduplicate_candidates_async(all_candidates)
            logger.info("[PIPELINE] After dedup: %d candidates", len(deduped))

            with timer.stage("ranking"):
                works = [_candidate_to_work(candidate) for candidate in deduped]
                ranked = await rank_papers_async(works, theme)
            timer.log_summary()
            return ranked, len(all_candidates), len(deduped)
        finally:
            await asyncio.gather(
                *[connector.close() for connector in connectors],
                return_exceptions=True,
            )

    # All retries exhausted — return empty
    logger.warning("[PIPELINE] All retrieval attempts exhausted, returning empty")
    return [], 0, 0


def _estimate_papers_tokens(
    papers: list[dict[str, Any]],
    question: str,
    system_prompt: str,
) -> int:
    """Roughly estimate token cost for a set of papers + system/user messages."""
    budget = DEFAULT_PROMPT_BUDGET
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    fixed_cost = budget.estimate_messages(messages)
    paper_cost = sum(
        budget.estimate_text(DeepXivAgent.paper_snippet(p))
        for p in papers
    )
    return fixed_cost + paper_cost


def _get_model_context_tokens(entry: ModelPoolEntry, settings: Settings) -> int:
    """Return the context window size for a given model pool entry.

    Used for PromptBudget batch sizing — this is the total input+output budget.
    """
    if entry.backend == "deepseek" and settings.model_path == "deepseek_flash":
        return 1_000_000
    if entry.backend == "glm":
        if "4.6" in entry.model:
            return 200_000
        if "4.5" in entry.model:
            return 128_000
        return 128_000  # glm-5-turbo and other GLM models
    if entry.backend == "deepseek":
        return 128_000
    if entry.backend == "qwen":
        return 32_768
    return 128_000


def _get_model_max_output_tokens(entry: ModelPoolEntry, settings: Settings) -> int:
    """Return the maximum output tokens (max_tokens) for a given model pool entry.

    This is NOT the same as context window — it's the API's max_tokens limit.
    For GLM models with thinking enabled, max_tokens is capped at 128K even when
    the context window is 200K.
    """
    if entry.backend == "deepseek":
        # deepseek-v4-flash/v4-pro: max output = 384K (393216)
        # deepseek-chat (legacy): max output = 128K
        if "v4" in entry.model:
            return 393_216
        return 128_000
    if entry.backend == "glm":
        # glm-4.5 max output is 96K (98304); glm-4.6 max output is 128K (131072)
        if "4.5" in entry.model:
            return 98_304
        return 131_072
    if entry.backend == "qwen":
        return 32_768
    return 128_000


async def _batched_select_papers(
    agent: DeepXivAgent,
    paper_dicts: list[dict[str, Any]],
    theme_description: str,
    system_prompt: str,
    context_tokens: int,
    requested_final: int,
    timer: PipelineTimer,
) -> list[dict[str, Any]]:
    """Select papers using batched LLM calls if papers exceed context window.

    For models with large context (e.g. deepseek_flash 1M), fits everything
    in a single call.  For smaller context windows, splits papers into
    batches, calls select_papers() per batch, then merges results.
    """
    budget = PromptBudget(model_context_tokens=context_tokens)

    # Estimate each paper's snippet cost
    snippets = [DeepXivAgent.paper_snippet(p, budget) for p in paper_dicts]
    fixed_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Research Question: {theme_description}\n\nPapers:\n"},
    ]
    batches = budget.pack_items(
        snippets,
        fixed_messages=fixed_messages,
        prefix=f"Research Question: {theme_description}\n\nPapers:\n",
        separator="\n",
    )

    if len(batches) <= 1:
        # Single batch — direct call, no index adjustment needed
        with timer.stage("select_papers(single_batch)"):
            return await agent.select_papers(
                paper_dicts, theme_description,
                max_select=requested_final,
                system_prompt=system_prompt,
            )

    # Multiple batches — call per batch, adjust indices, merge
    logger.info(
        "[PIPELINE] Batching %d papers into %d calls (context=%dK)",
        len(paper_dicts), len(batches), context_tokens // 1000,
    )
    merged: dict[int, dict[str, Any]] = {}  # original_index -> result
    offset = 0  # running index into original paper_dicts

    for batch_idx, batch in enumerate(batches):
        batch_size = len(batch)
        batch_papers = paper_dicts[offset : offset + batch_size]
        offset += batch_size

        if not batch_papers:
            continue

        per_batch_select = max(1, requested_final // len(batches) + 2)
        with timer.stage(f"select_papers(batch_{batch_idx+1}/{len(batches)})"):
            try:
                batch_results = await agent.select_papers(
                    batch_papers, theme_description,
                    max_select=per_batch_select,
                    system_prompt=system_prompt,
                )
            except (DeepXivAgentError, Exception) as exc:
                logger.warning(
                    "[PIPELINE] Batch %d/%d failed: %s",
                    batch_idx + 1, len(batches), exc,
                )
                continue

        # Adjust indices back to original paper_dicts
        batch_offset = offset - batch_size
        for item in batch_results:
            batch_local_idx = item.get("index")
            if not isinstance(batch_local_idx, int) or batch_local_idx >= batch_size:
                continue
            orig_idx = batch_offset + batch_local_idx
            item["index"] = orig_idx
            # Keep highest relevance if same paper appears in multiple batches
            if orig_idx in merged:
                old_rel = float(merged[orig_idx].get("relevance", 0) or 0)
                new_rel = float(item.get("relevance", 0) or 0)
                if new_rel > old_rel:
                    merged[orig_idx] = item
            else:
                merged[orig_idx] = item

    return list(merged.values())


def _collect_selection_works(
    results: list[dict[str, Any]],
    agent_candidates: list[Work],
) -> tuple[list[Work], list[Work]]:
    """Convert select_papers() results into (all_scored, selected) works."""
    scored: list[Work] = []
    selected: list[Work] = []
    for item in results:
        idx = item.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(agent_candidates)):
            continue
        # Map select_papers() output fields to the annotation format
        relevance = float(item.get("relevance", 0) or 0)
        final_rank = item.get("final_rank")
        annotated_item = {
            "agent_score": relevance + (10 - (final_rank or 20)) * 0.5,
            "agent_rank": final_rank,
            "agent_rationale": item.get("reason", ""),
        }
        work = _annotated_work(agent_candidates[idx], annotated_item)
        scored.append(work)
        if item.get("selected"):
            selected.append(work)
    return scored, selected


def _annotated_work(work: Work, rerank: dict[str, Any]) -> Work:
    annotated = work.model_copy(deep=True)
    annotated.agent_score = float(rerank.get("agent_score", 0.0) or 0.0)
    annotated.agent_rank = rerank.get("agent_rank")
    annotated.agent_rationale = rerank.get("agent_rationale")
    return annotated


async def run_retrieval(
    theme: Theme,
    storage: StorageService,
    settings: Settings | None = None,
) -> list[Work]:
    """Execute the broad retrieval pipeline used by the REST layer."""
    resolved = settings or get_settings()

    try:
        ranked, _, _ = await _collect_ranked_works(theme, storage, resolved)
        works = storage.replace_theme_results(theme.id, ranked)
        logger.info("Retrieval complete: %d works saved for theme %s", len(works), theme.id)
        return works
    except Exception:
        logger.exception("Retrieval failed for theme %s", theme.id)
        raise


async def run_query_pipeline(
    document_text: str,
    storage: StorageService,
    settings: Settings | None = None,
    *,
    final_limit: int | None = None,
    agent_candidate_limit: int | None = None,
    coarse_pool_limit: int | None = None,
    include_rationale: bool = True,
) -> QueryPipelineResult:
    """Execute the MCP query pipeline with built-in DeepXiv agent reranking.

    This function NEVER raises exceptions. All errors are caught and converted
    to fallback results so the MCP client always receives papers.
    """
    del include_rationale  # payload shaping decides whether rationale is exposed.

    resolved = settings or get_settings()

    pipeline_start = time.monotonic()
    pipeline_timer = PipelineTimer("query_pipeline")
    logger.info(
        "[PIPELINE] === Query pipeline started === "
        "final_limit=%s agent_candidate_limit=%s coarse_pool_limit=%s",
        final_limit, agent_candidate_limit, coarse_pool_limit,
    )

    with pipeline_timer.stage("parse_theme"):
        theme = parse_theme(document_text)
    logger.info(
        "[PIPELINE] Theme parsed in %.2fs: id=%s queries=%d",
        time.monotonic() - pipeline_start,
        theme.id,
        len(theme.parsed_queries),
    )

    # --- Stage 1: Retrieval ---
    with pipeline_timer.stage("retrieval"):
        try:
            async with asyncio.timeout(resolved.retrieval_total_timeout_seconds):
                ranked, total_retrieved, total_after_dedup = await _collect_ranked_works(
                    theme,
                    storage,
                    resolved,
                )
            logger.info(
                "[PIPELINE] Retrieval done: raw=%d dedup=%d ranked=%d",
                total_retrieved, total_after_dedup, len(ranked),
            )
        except Exception:
            logger.exception(
                "[PIPELINE] Retrieval FAILED for theme %s",
                theme.id,
            )
            saved = storage.replace_theme_results(theme.id, [])
            return QueryPipelineResult(
                theme=theme,
                total_retrieved=0,
                total_after_dedup=0,
                total_after_first_stage=0,
                total_agent_candidates=0,
                total_final=0,
                works=saved,
            )

    total_after_first_stage = len(ranked)
    coarse_limit = coarse_pool_limit or resolved.target_candidate_pool
    coarse_pool = ranked[: max(1, coarse_limit)] if ranked else []
    candidate_limit = agent_candidate_limit or resolved.agent_candidate_limit
    agent_candidates = coarse_pool[: max(1, candidate_limit)] if coarse_pool else []
    requested_final = final_limit or resolved.final_limit

    if not agent_candidates:
        saved = storage.replace_theme_results(theme.id, [])
        return QueryPipelineResult(
            theme=theme,
            total_retrieved=total_retrieved,
            total_after_dedup=total_after_dedup,
            total_after_first_stage=total_after_first_stage,
            total_agent_candidates=0,
            total_final=0,
            works=saved,
        )

    # Use compressed summary for Agent if available, otherwise full document text
    theme_description = (
        theme.compressed_summary
        if theme.compressed_summary
        else theme.document_text
    )

    # Build paper dicts for the agent
    paper_dicts = [
        {
            "title": work.title,
            "abstract": work.abstract or "",
            "authors": work.authors,
            "year": work.year,
            "venue": work.venue,
            "citation_count": work.citation_count,
        }
        for work in agent_candidates
    ]

    # --- Agent refinement: batched select_papers() via ModelPool ---
    pool = ModelPool.get(resolved)
    system_prompt = DeepXivAgent.build_stage2_prompt(max_select=requested_final)

    logger.info(
        "[PIPELINE] Agent refinement: %d papers, pool=%s",
        len(paper_dicts),
        ", ".join(f"{e.model}({e.max_concurrent})" for e in pool.entries),
    )

    selection_results: list[dict[str, Any]] | None = None
    max_attempts = len(pool.entries) * 2  # try each model up to twice
    last_error = ""

    try:
        with pipeline_timer.stage("agent_refinement"):
            async with asyncio.timeout(resolved.agent_total_timeout_seconds):
                for attempt in range(max_attempts):
                    try:
                        entry = await pool.acquire(timeout=5.0)
                    except RuntimeError:
                        logger.warning("[PIPELINE] ModelPool exhausted, using deterministic fallback")
                        break

                    logger.info(
                        "[PIPELINE] Attempt %d/%d: acquired %s/%s (cooldown=%s)",
                        attempt + 1, max_attempts,
                        entry.backend, entry.model, entry.in_cooldown,
                    )

                    context_tokens = _get_model_context_tokens(entry, resolved)
                    max_output_tokens = _get_model_max_output_tokens(entry, resolved)
                    agent = DeepXivAgent(
                        api_key=entry.api_key,
                        base_url=entry.base_url,
                        model=entry.model,
                        backend=entry.backend,
                        request_timeout_seconds=resolved.deepxiv_agent_http_timeout_seconds,
                        total_timeout_seconds=resolved.agent_total_timeout_seconds,
                        max_retries=0,
                        retry_backoff_seconds=resolved.deepxiv_agent_retry_backoff_seconds,
                        context_tokens=context_tokens,
                        max_output_tokens=max_output_tokens,
                    )
                    try:
                        selection_results = await _batched_select_papers(
                            agent, paper_dicts, theme_description,
                            system_prompt=system_prompt,
                            context_tokens=context_tokens,
                            requested_final=requested_final,
                            timer=pipeline_timer,
                        )
                        pool.release(entry, success=True)
                        logger.info(
                            "[PIPELINE] %s/%s SUCCEEDED, selected=%d",
                            entry.backend, entry.model,
                            len(selection_results) if selection_results else 0,
                        )
                        break
                    except (DeepXivAgentError, Exception) as exc:
                        last_error = str(exc)
                        pool.release(entry, success=False)
                        logger.warning(
                            "[PIPELINE] %s/%s FAILED: %s",
                            entry.backend, entry.model,
                            exc,
                        )
                    finally:
                        await agent.close()
    except TimeoutError:
        logger.warning(
            "[PIPELINE] Agent refinement timed out after %ds",
            resolved.agent_total_timeout_seconds,
        )

    # Deterministic fallback if all model attempts failed
    if selection_results is None:
        logger.warning(
            "[PIPELINE] ALL models FAILED; deterministic fallback. Last error: %s",
            last_error,
        )
        fallback_agent = DeepXivAgent(
            api_key="",
            base_url=resolved.bigmodel_base_url,
            model="fallback",
            fallback_top_k=resolved.deepxiv_agent_fallback_top_k,
        )
        try:
            selection_results = await fallback_agent.rerank_papers(
                paper_dicts,
                theme_description,
            )
        finally:
            await fallback_agent.close()

    if not selection_results:
        saved = storage.replace_theme_results(theme.id, [])
        return QueryPipelineResult(
            theme=theme,
            total_retrieved=total_retrieved,
            total_after_dedup=total_after_dedup,
            total_after_first_stage=total_after_first_stage,
            total_agent_candidates=len(agent_candidates),
            total_final=0,
            works=saved,
        )

    scored, selected = _collect_selection_works(selection_results, agent_candidates)

    if not selected:
        logger.warning("[PIPELINE] Agent selected nothing; using top-scored fallback")
        selected = scored[: max(1, resolved.deepxiv_agent_fallback_top_k)]
    elif len(selected) < requested_final:
        # Supplement with top-scored papers the agent didn't pick
        selected_ids = {id(w) for w in selected}
        fill = [w for w in scored if id(w) not in selected_ids]
        fill_count = min(len(fill), requested_final - len(selected))
        if fill_count > 0:
            logger.info(
                "[PIPELINE] Agent selected %d < final_limit=%d; supplementing %d from scored",
                len(selected), requested_final, fill_count,
            )
            selected = selected + fill[:fill_count]

    final_works = selected[: max(1, requested_final)]
    with pipeline_timer.stage("storage_save"):
        saved = storage.replace_theme_results(theme.id, final_works)
    pipeline_timer.log_summary()
    logger.info(
        "[PIPELINE] === DONE in %.2fs total === theme=%s final=%d papers",
        time.monotonic() - pipeline_start,
        theme.id,
        len(saved),
    )
    return QueryPipelineResult(
        theme=theme,
        total_retrieved=total_retrieved,
        total_after_dedup=total_after_dedup,
        total_after_first_stage=total_after_first_stage,
        total_agent_candidates=len(agent_candidates),
        total_final=len(saved),
        works=saved,
    )


async def run_retrieval_for_document(
    document_text: str,
    storage: StorageService,
    settings: Settings | None = None,
) -> tuple[Theme, list[Work]]:
    """Convenience: parse a document into a Theme, then run the broad retrieval pipeline."""
    theme = parse_theme(document_text)
    storage.save_theme(theme)

    works = await run_retrieval(theme, storage, settings=settings)
    return theme, works
