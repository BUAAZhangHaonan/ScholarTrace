"""Core retrieval orchestration for ScholarTrace."""

from __future__ import annotations

import asyncio
import logging
import math
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
from scholartrace.services.dedup import deduplicate_candidates
from scholartrace.services.prompt_budget import DEFAULT_PROMPT_BUDGET
from scholartrace.services.ranking import rank_papers
from scholartrace.services.storage import StorageService
from scholartrace.services.theme_parser import parse_theme

logger = logging.getLogger(__name__)


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
) -> list[RawCandidate]:
    """Run one query against all connectors concurrently, tolerating individual failures."""
    tasks = [c.search(query, max_results=max_results) for c in connectors]
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
    return candidates


async def _collect_ranked_works(
    theme: Theme,
    storage: StorageService,
    settings: Settings,
) -> tuple[list[Work], int, int]:
    connectors = _build_connectors(settings)
    try:
        storage.save_theme(theme)

        all_candidates: list[RawCandidate] = []
        for query in theme.parsed_queries:
            query_candidates = await _fan_out_query(
                connectors,
                query,
                settings.max_results_per_source_per_query,
            )
            all_candidates.extend(query_candidates)

        logger.info(
            "Collected %d raw candidates across %d queries",
            len(all_candidates),
            len(theme.parsed_queries),
        )

        deduped = deduplicate_candidates(all_candidates)
        logger.info("After dedup: %d candidates", len(deduped))

        works = [_candidate_to_work(candidate) for candidate in deduped]
        ranked = rank_papers(works, theme)
        return ranked, len(all_candidates), len(deduped)
    finally:
        await asyncio.gather(
            *[connector.close() for connector in connectors],
            return_exceptions=True,
        )


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


def _compute_stage1_scores(
    batch_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Stage 1 raw LLM output into scored items with agent_score."""
    scored: list[dict[str, Any]] = []
    for result in batch_results:
        idx = result.get("index")
        if not isinstance(idx, int):
            continue
        relevance = float(result.get("relevance", 0) or 0)
        recency = float(result.get("recency", 0) or 0)
        novelty = float(result.get("novelty", 0) or 0)
        quality = float(result.get("quality", 0) or 0)
        score = relevance + recency * 0.6 + novelty * 0.3 + quality * 0.2
        scored.append({
            "index": idx,
            "agent_score": score,
            "agent_rationale": result.get("reason", ""),
            "relevance": relevance,
            "recency": recency,
            "novelty": novelty,
            "quality": quality,
        })
    return scored


async def _run_stage1_scoring(
    paper_dicts: list[dict[str, Any]],
    question: str,
    settings: Settings,
    final_limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Stage 1: distribute papers across small batches, score concurrently with glm-4.6.

    Returns (top_candidates_with_scores, stage1_success).
    Each item in top_candidates has 'original_index', 'agent_score', 'agent_rationale'.
    """
    batch_size = settings.stage1_batch_size
    num_batches = math.ceil(len(paper_dicts) / batch_size) if paper_dicts else 1
    K = math.ceil(final_limit * 2 / num_batches) if num_batches > 0 else final_limit * 2

    logger.info(
        "[PIPELINE] Stage 1: %d papers, batch_size=%d, %d batches, K=%d per batch",
        len(paper_dicts), batch_size, num_batches, K,
    )

    # Split papers into batches
    batches: list[tuple[int, list[dict[str, Any]]]] = []
    for i in range(0, len(paper_dicts), batch_size):
        batches.append((i, paper_dicts[i:i + batch_size]))

    system_prompt = DeepXivAgent.build_stage1_prompt()
    semaphore = asyncio.Semaphore(settings.stage1_concurrency)

    async def _score_batch(
        batch_offset: int,
        batch_papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score one batch, return scored items with global indices."""
        async with semaphore:
            agent = DeepXivAgent(
                api_key=settings.bigmodel_api_key,
                base_url=settings.bigmodel_base_url,
                model=settings.stage1_model,
                backend="glm",
                request_timeout_seconds=settings.deepxiv_agent_http_timeout_seconds,
                total_timeout_seconds=None,
                max_retries=1,
                retry_backoff_seconds=settings.deepxiv_agent_retry_backoff_seconds,
                batch_size=len(batch_papers),
            )
            try:
                results = await agent.score_papers_batch(
                    batch_papers,
                    question,
                    system_prompt=system_prompt,
                )
            except DeepXivAgentError:
                logger.warning(
                    "[PIPELINE] Stage 1 batch at offset %d FAILED, using deterministic fallback",
                    batch_offset,
                )
                # Deterministic fallback for this batch
                theme = Theme(parsed_queries=[question])
                works = []
                for j, p in enumerate(batch_papers):
                    works.append(Work(
                        title=str(p.get("title") or ""),
                        abstract=p.get("abstract"),
                        authors=list(p.get("authors") or []),
                        year=p.get("year") if isinstance(p.get("year"), int) else None,
                        venue=p.get("venue"),
                        citation_count=max(0, p.get("citation_count") or 0) if isinstance(p.get("citation_count"), int) else 0,
                        source_provenance=["stage1_fallback"],
                    ))
                ranked = rank_papers(works, theme)
                results = []
                for rank, w in enumerate(ranked):
                    results.append({
                        "index": j,
                        "relevance": 5.0,
                        "recency": 5.0,
                        "novelty": 5.0,
                        "quality": 5.0,
                        "reason": "stage1_deterministic_fallback",
                    })
                    # Find original j for this work
                    for jj, p in enumerate(batch_papers):
                        if p.get("title") == w.title:
                            results[-1]["index"] = jj
                            break
            finally:
                await agent.close()

        # Map local batch indices to global indices
        scored = _compute_stage1_scores(results)
        for item in scored:
            item["original_index"] = batch_offset + item["index"]
        return scored

    # Run all batches with concurrency control
    retry_queue: list[tuple[int, list[dict[str, Any]]]] = list(batches)
    all_scored: list[dict[str, Any]] = []
    stage1_success = True

    for retry_round in range(settings.stage1_max_retries + 1):
        if not retry_queue:
            break
        tasks = [_score_batch(offset, papers) for offset, papers in retry_queue]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        next_retry: list[tuple[int, list[dict[str, Any]]]] = []
        for (offset, papers), result in zip(retry_queue, batch_results):
            if isinstance(result, Exception):
                logger.warning(
                    "[PIPELINE] Stage 1 batch at offset %d failed (round %d): %s",
                    offset, retry_round, result,
                )
                if retry_round < settings.stage1_max_retries:
                    next_retry.append((offset, papers))
                else:
                    stage1_success = False
                    # Final fallback: assign low scores
                    for j in range(len(papers)):
                        all_scored.append({
                            "original_index": offset + j,
                            "agent_score": 1.0,
                            "agent_rationale": "stage1_batch_failed",
                        })
            else:
                all_scored.extend(result)

        retry_queue = next_retry

    # Per-batch top-K selection, then global sort
    batch_top_k: list[dict[str, Any]] = []
    for i in range(0, len(paper_dicts), batch_size):
        batch_items = [
            item for item in all_scored
            if i <= item["original_index"] < i + batch_size
        ]
        batch_items.sort(key=lambda x: -x["agent_score"])
        batch_top_k.extend(batch_items[:K])

    batch_top_k.sort(key=lambda x: -x["agent_score"])
    logger.info(
        "[PIPELINE] Stage 1 done: %d scored, %d top-K candidates, success=%s",
        len(all_scored), len(batch_top_k), stage1_success,
    )
    return batch_top_k, stage1_success


async def _run_stage2_selection(
    candidates: list[dict[str, Any]],
    paper_dicts: list[dict[str, Any]],
    question: str,
    settings: Settings,
    final_limit: int,
) -> list[dict[str, Any]] | None:
    """Stage 2: use glm-5-turbo with 128K context for final comparative selection.

    Returns reranked list of dicts with index (global), selected, agent_score, etc.
    Returns None if Stage 2 fails entirely.
    """
    # Collect the actual paper dicts for Stage 2 candidates
    stage2_papers: list[dict[str, Any]] = []
    index_map: list[int] = []  # maps stage2 local index → original paper_dicts index
    for candidate in candidates:
        orig_idx = candidate["original_index"]
        if 0 <= orig_idx < len(paper_dicts):
            stage2_papers.append(paper_dicts[orig_idx])
            index_map.append(orig_idx)

    if not stage2_papers:
        return None

    system_prompt = DeepXivAgent.build_stage2_prompt(max_select=final_limit)
    max_ctx = settings.stage2_max_context_tokens

    # Check if all candidates fit in one call
    estimated_tokens = _estimate_papers_tokens(stage2_papers, question, system_prompt)

    if estimated_tokens <= max_ctx:
        # Single call
        return await _stage2_single_call(
            stage2_papers, index_map, question, system_prompt, settings, final_limit,
        )

    # Split into multiple calls, then merge
    logger.info(
        "[PIPELINE] Stage 2: %d papers (%d est tokens) > %d limit, splitting",
        len(stage2_papers), estimated_tokens, max_ctx,
    )
    # Binary search for split point
    batches = _split_papers_for_stage2(stage2_papers, question, system_prompt, max_ctx)
    all_selections: list[dict[str, Any]] = []

    for batch_offset, batch_papers in batches:
        batch_index_map = index_map[batch_offset:batch_offset + len(batch_papers)]
        result = await _stage2_single_call(
            batch_papers, batch_index_map, question, system_prompt, settings, final_limit,
        )
        if result is not None:
            all_selections.extend(result)

    if not all_selections:
        return None

    # Merge: deduplicate by original_index, sort by final_rank
    seen: set[int] = set()
    merged: list[dict[str, Any]] = []
    for item in all_selections:
        if item.get("original_index") not in seen:
            seen.add(item["original_index"])
            merged.append(item)

    merged.sort(key=lambda x: (x.get("final_rank") or 999))
    return merged[:final_limit]


def _split_papers_for_stage2(
    papers: list[dict[str, Any]],
    question: str,
    system_prompt: str,
    max_tokens: int,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Split papers into batches that fit within token limits."""
    budget = DEFAULT_PROMPT_BUDGET
    fixed_cost = budget.estimate_messages([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ])
    available = max_tokens - fixed_cost
    if available <= 0:
        # Can't even fit the prompt, use single paper batches
        return [(i, [p]) for i, p in enumerate(papers)]

    batches: list[tuple[int, list[dict[str, Any]]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    offset = 0

    for i, paper in enumerate(papers):
        paper_tokens = budget.estimate_text(DeepXivAgent.paper_snippet(paper))
        if current and current_tokens + paper_tokens > available:
            batches.append((offset, current))
            offset = i
            current = [paper]
            current_tokens = paper_tokens
        else:
            current.append(paper)
            current_tokens += paper_tokens

    if current:
        batches.append((offset, current))
    return batches


async def _stage2_single_call(
    papers: list[dict[str, Any]],
    index_map: list[int],
    question: str,
    system_prompt: str,
    settings: Settings,
    final_limit: int,
) -> list[dict[str, Any]] | None:
    """Execute a single Stage 2 LLM call."""
    agent = DeepXivAgent(
        api_key=settings.bigmodel_api_key,
        base_url=settings.bigmodel_base_url,
        model=settings.stage2_model,
        backend="glm",
        request_timeout_seconds=settings.deepxiv_agent_http_timeout_seconds,
        total_timeout_seconds=None,
        max_retries=1,
        retry_backoff_seconds=settings.deepxiv_agent_retry_backoff_seconds,
    )
    try:
        results = await agent.select_papers(
            papers,
            question,
            max_select=final_limit,
            system_prompt=system_prompt,
        )
    except DeepXivAgentError as exc:
        logger.warning("[PIPELINE] Stage 2 call FAILED: %s", exc)
        return None
    finally:
        await agent.close()

    # Convert Stage 2 results to reranked format with global indices
    reranked: list[dict[str, Any]] = []
    for result in results:
        local_idx = result.get("index")
        if not isinstance(local_idx, int) or not (0 <= local_idx < len(index_map)):
            continue
        selected = bool(result.get("selected"))
        relevance = float(result.get("relevance", 0) or 0)
        final_rank = result.get("final_rank")
        reranked.append({
            "original_index": index_map[local_idx],
            "selected": selected,
            "agent_score": relevance + (10 - (final_rank or 20)) * 0.5,
            "agent_rationale": result.get("reason", ""),
            "agent_rank": final_rank,
            "final_rank": final_rank,
        })
    return reranked


async def _run_two_stage_pipeline(
    *,
    theme: Theme,
    theme_description: str,
    agent_candidates: list[Work],
    paper_dicts: list[dict[str, Any]],
    total_retrieved: int,
    total_after_dedup: int,
    total_after_first_stage: int,
    requested_final: int,
    storage: StorageService,
    settings: Settings,
    pipeline_start: float,
) -> QueryPipelineResult:
    """Execute the two-stage pipeline: distributed scoring + centralized selection."""
    logger.info(
        "[PIPELINE] === TWO-STAGE pipeline: %d candidates, final=%d ===",
        len(paper_dicts), requested_final,
    )

    # --- Stage 1: Distributed scoring with glm-4.6 ---
    stage1_start = time.monotonic()
    top_candidates, stage1_ok = await _run_stage1_scoring(
        paper_dicts, theme_description, settings, requested_final,
    )
    logger.info(
        "[PIPELINE] Stage 1 done in %.2fs: %d candidates, ok=%s",
        time.monotonic() - stage1_start, len(top_candidates), stage1_ok,
    )

    if not top_candidates:
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

    # --- Stage 2: Centralized selection with glm-5-turbo ---
    stage2_start = time.monotonic()
    reranked = await _run_stage2_selection(
        top_candidates, paper_dicts, theme_description, settings, requested_final,
    )
    logger.info(
        "[PIPELINE] Stage 2 done in %.2fs: reranked=%s",
        time.monotonic() - stage2_start,
        len(reranked) if reranked else "None",
    )

    # If Stage 2 failed, use Stage 1 scores directly
    if reranked is None:
        logger.warning("[PIPELINE] Stage 2 FAILED; using Stage 1 scores directly")
        reranked = [
            {
                "original_index": c["original_index"],
                "selected": True,
                "agent_score": c["agent_score"],
                "agent_rationale": c.get("agent_rationale", ""),
            }
            for c in top_candidates[:requested_final]
        ]

    # Convert reranked results to final works
    scored_works: list[Work] = []
    selected_works: list[Work] = []

    for item in reranked:
        orig_idx = item.get("original_index")
        if not isinstance(orig_idx, int) or not (0 <= orig_idx < len(agent_candidates)):
            continue
        work = _annotated_work(agent_candidates[orig_idx], item)
        scored_works.append(work)
        if item.get("selected"):
            selected_works.append(work)

    # If nothing selected, use top-scored
    if not selected_works:
        logger.warning("[PIPELINE] Two-stage: nothing selected, using top-scored")
        selected_works = scored_works[:max(1, settings.deepxiv_agent_fallback_top_k)]
    elif len(selected_works) < requested_final:
        # Supplement with scored works
        selected_ids = {id(w) for w in selected_works}
        fill = [w for w in scored_works if id(w) not in selected_ids]
        fill_count = min(len(fill), requested_final - len(selected_works))
        if fill_count > 0:
            selected_works = selected_works + fill[:fill_count]

    final_works = selected_works[:max(1, requested_final)]
    saved = storage.replace_theme_results(theme.id, final_works)
    logger.info(
        "[PIPELINE] === TWO-STAGE DONE in %.2fs === theme=%s final=%d papers",
        time.monotonic() - pipeline_start, theme.id, len(saved),
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
    logger.info(
        "[PIPELINE] === Query pipeline started === "
        "final_limit=%s agent_candidate_limit=%s coarse_pool_limit=%s",
        final_limit, agent_candidate_limit, coarse_pool_limit,
    )

    theme = parse_theme(document_text)
    logger.info(
        "[PIPELINE] Theme parsed in %.2fs: id=%s queries=%d",
        time.monotonic() - pipeline_start,
        theme.id,
        len(theme.parsed_queries),
    )

    # --- Stage 1: Retrieval ---
    retrieval_start = time.monotonic()
    try:
        ranked, total_retrieved, total_after_dedup = await _collect_ranked_works(
            theme,
            storage,
            resolved,
        )
        logger.info(
            "[PIPELINE] Retrieval done in %.2fs: raw=%d dedup=%d ranked=%d",
            time.monotonic() - retrieval_start,
            total_retrieved, total_after_dedup, len(ranked),
        )
    except Exception:
        logger.exception(
            "[PIPELINE] Retrieval FAILED in %.2fs for theme %s",
            time.monotonic() - retrieval_start, theme.id,
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

    # Build paper dicts for agent (used by both single-stage and two-stage)
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

    # --- Two-stage pipeline branch ---
    if resolved.two_stage_enabled:
        return await _run_two_stage_pipeline(
            theme=theme,
            theme_description=theme_description,
            agent_candidates=agent_candidates,
            paper_dicts=paper_dicts,
            total_retrieved=total_retrieved,
            total_after_dedup=total_after_dedup,
            total_after_first_stage=total_after_first_stage,
            requested_final=requested_final,
            storage=storage,
            settings=resolved,
            pipeline_start=pipeline_start,
        )

    # --- Original single-stage pipeline (below) ---
    # Build model chain: GLM primary → GLM fallbacks → DeepSeek → Qwen
    model_chain: list[dict[str, Any]] = []

    # GLM models as primary chain
    glm_primary = resolved.bigmodel_model
    glm_fallbacks = [
        m.strip()
        for m in resolved.bigmodel_fallback_models.split(",")
        if m.strip()
    ]
    for model_name in [glm_primary] + glm_fallbacks:
        model_chain.append({
            "backend": "glm",
            "model": model_name,
            "api_key": resolved.bigmodel_api_key,
            "base_url": resolved.bigmodel_base_url,
        })

    # DeepSeek as fallback
    if resolved.deepseek_api_key.strip():
        model_chain.append({
            "backend": "deepseek",
            "model": resolved.deepseek_model,
            "api_key": resolved.deepseek_api_key,
            "base_url": resolved.deepseek_base_url,
        })

    # Local Qwen as last resort
    if resolved.qwen_base_url.strip():
        model_chain.append({
            "backend": "qwen",
            "model": resolved.qwen_model,
            "api_key": resolved.qwen_api_key,
            "base_url": resolved.qwen_base_url,
        })

    def _collect_reranked_works(
        reranked_items: list[dict[str, Any]],
    ) -> tuple[list[Work], list[Work]]:
        scored_works: list[Work] = []
        selected_works: list[Work] = []
        for rerank in reranked_items:
            index = rerank.get("index")
            if not isinstance(index, int) or not (0 <= index < len(agent_candidates)):
                continue
            annotated = _annotated_work(agent_candidates[index], rerank)
            scored_works.append(annotated)
            if rerank.get("selected"):
                selected_works.append(annotated)
        return scored_works, selected_works

    # Try each model in the chain until one succeeds
    reranked: list[dict[str, Any]] | None = None
    last_model_error: str = ""
    agent_start = time.monotonic()
    logger.info(
        "[PIPELINE] Starting agent reranking: %d papers, %d models in chain",
        len(paper_dicts), len(model_chain),
    )

    for i, model_cfg in enumerate(model_chain):
        model_attempt_start = time.monotonic()
        # Timeout strategy: 5s connect timeout, unlimited processing time
        # If the model API is unreachable in 5s, try next model.
        # Once connected, let the model process papers without time limit.
        http_timeout = resolved.deepxiv_agent_http_timeout_seconds
        # No total timeout limit — model processing time is not restricted
        model_total_timeout = None
        agent = DeepXivAgent(
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            model=model_cfg["model"],
            backend=model_cfg["backend"],
            request_timeout_seconds=http_timeout,
            total_timeout_seconds=model_total_timeout,
            max_retries=0,
            retry_backoff_seconds=resolved.deepxiv_agent_retry_backoff_seconds,
            batch_size=resolved.deepxiv_agent_batch_size,
            fallback_top_k=resolved.deepxiv_agent_fallback_top_k,
        )
        try:
            logger.info(
                "[PIPELINE] Model %d/%d: attempting %s/%s (connect_timeout=%.1fs total_timeout=%s)",
                i + 1, len(model_chain),
                model_cfg["backend"], model_cfg["model"], http_timeout,
                "none" if model_total_timeout is None else f"{model_total_timeout:.1f}s",
            )
            reranked = await agent.rerank_papers(
                paper_dicts,
                theme_description,
                strict=True,
            )
            if reranked is not None:
                logger.info(
                    "[PIPELINE] Model %d/%d SUCCEEDED: %s/%s in %.2fs, reranked=%d papers",
                    i + 1, len(model_chain),
                    model_cfg["backend"], model_cfg["model"],
                    time.monotonic() - model_attempt_start,
                    len(reranked),
                )
                break
        except DeepXivAgentError as exc:
            last_model_error = str(exc)
            logger.warning(
                "[PIPELINE] Model %d/%d FAILED: %s/%s in %.2fs: %s",
                i + 1, len(model_chain),
                model_cfg["backend"], model_cfg["model"],
                time.monotonic() - model_attempt_start,
                exc,
            )
        except Exception as exc:
            last_model_error = str(exc)
            logger.warning(
                "[PIPELINE] Model %d/%d UNEXPECTED FAILURE: %s/%s in %.2fs: %s",
                i + 1, len(model_chain),
                model_cfg["backend"], model_cfg["model"],
                time.monotonic() - model_attempt_start,
                exc,
            )
        finally:
            await agent.close()

    # If all models failed, use deterministic fallback
    if reranked is None:
        logger.warning(
            "[PIPELINE] ALL %d MODELS FAILED in %.2fs; falling back to deterministic. Last error: %s",
            len(model_chain),
            time.monotonic() - agent_start,
            last_model_error,
        )
        fallback_agent = DeepXivAgent(
            api_key="",
            base_url=resolved.bigmodel_base_url,
            model="fallback",
            fallback_top_k=resolved.deepxiv_agent_fallback_top_k,
        )
        try:
            reranked = await fallback_agent.rerank_papers(
                paper_dicts,
                theme_description,
            )
        finally:
            await fallback_agent.close()

    if not reranked:
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

    scored, selected = _collect_reranked_works(reranked)

    if not selected:
        # Agent selected nothing — use top scored works as fallback
        logger.warning("Agent selected no papers; using top-scored works as fallback")
        selected = scored[: max(1, resolved.deepxiv_agent_fallback_top_k)]
    elif len(selected) < requested_final:
        # Agent selected fewer than requested — supplement with top-scored papers
        selected_ids = {id(w) for w in selected}
        fill = [w for w in scored if id(w) not in selected_ids]
        filled_count = min(len(fill), requested_final - len(selected))
        if filled_count > 0:
            logger.info(
                "[PIPELINE] Agent selected %d papers < final_limit=%d; "
                "supplementing with %d top-scored papers",
                len(selected), requested_final, filled_count,
            )
            selected = selected + fill[:filled_count]

    final_works = selected[: max(1, requested_final)]
    saved = storage.replace_theme_results(theme.id, final_works)
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
