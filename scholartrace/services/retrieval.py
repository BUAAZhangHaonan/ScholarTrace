"""Core retrieval orchestration for ScholarTrace."""

from __future__ import annotations

import asyncio
import logging
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
from scholartrace.deepxiv.agent import DeepXivAgent
from scholartrace.models.schemas import RawCandidate, Theme, Work
from scholartrace.services.dedup import deduplicate_candidates
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
    """Execute the MCP query pipeline with built-in DeepXiv agent reranking."""
    del include_rationale  # payload shaping decides whether rationale is exposed.

    resolved = settings or get_settings()
    if not resolved.bigmodel_api_key.strip():
        raise ValueError("SCHOLARTRACE_BIGMODEL_API_KEY is required for MCP query reranking")

    theme = parse_theme(document_text)
    ranked, total_retrieved, total_after_dedup = await _collect_ranked_works(
        theme,
        storage,
        resolved,
    )

    total_after_first_stage = len(ranked)
    coarse_limit = coarse_pool_limit or resolved.target_candidate_pool
    coarse_pool = ranked[: max(1, coarse_limit)] if ranked else []
    candidate_limit = agent_candidate_limit or resolved.agent_candidate_limit
    agent_candidates = coarse_pool[: max(1, candidate_limit)] if coarse_pool else []

    agent = DeepXivAgent(
        api_key=resolved.bigmodel_api_key,
        base_url=resolved.bigmodel_base_url,
        model=resolved.bigmodel_model,
    )
    try:
        reranked = await agent.rerank_papers(
            [
                {
                    "title": work.title,
                    "abstract": work.abstract or "",
                    "authors": work.authors,
                    "year": work.year,
                    "venue": work.venue,
                }
                for work in agent_candidates
            ],
            theme.document_text,
        )
    finally:
        await agent.close()

    annotated: list[Work] = []
    for rerank in reranked:
        index = rerank.get("index")
        if not isinstance(index, int) or not (0 <= index < len(agent_candidates)):
            continue
        annotated.append(_annotated_work(agent_candidates[index], rerank))

    if not annotated:
        raise ValueError("Agent reranking returned no scored papers")

    requested_final = final_limit or resolved.final_limit
    final_works = annotated[: max(1, requested_final)]
    saved = storage.replace_theme_results(theme.id, final_works)
    logger.info(
        "MCP query pipeline complete: %d final works saved for theme %s",
        len(saved),
        theme.id,
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
