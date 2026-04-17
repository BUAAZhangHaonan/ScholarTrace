"""Core retrieval orchestration: fan-out queries across all sources, dedup, rank, and store."""

from __future__ import annotations

import asyncio
import logging

from scholartrace.config import Settings, get_settings
from scholartrace.connectors.arxiv import ArxivConnector
from scholartrace.connectors.base import BaseConnector
from scholartrace.connectors.crossref import CrossrefConnector
from scholartrace.connectors.deepxiv_connector import DeepXivConnector
from scholartrace.connectors.dblp import DblpConnector
from scholartrace.connectors.openalex import OpenAlexConnector
from scholartrace.connectors.openreview import OpenReviewConnector
from scholartrace.connectors.semantic_scholar import SemanticScholarConnector
from scholartrace.models.schemas import RawCandidate, Theme, Work
from scholartrace.services.dedup import deduplicate_candidates
from scholartrace.services.ranking import rank_papers
from scholartrace.services.storage import StorageService

logger = logging.getLogger(__name__)


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


async def run_retrieval(
    theme: Theme,
    storage: StorageService,
    settings: Settings | None = None,
) -> list[Work]:
    """Execute the full retrieval pipeline for a Theme.

    Steps:
    1. Fan out each parsed query across the configured unified connectors concurrently.
       DeepXiv joins this fan-out when its runtime is configured.
    2. Aggregate, deduplicate, convert to Work objects.
    3. Rank by composite score, persist to storage atomically, link to theme.
    4. Return the ranked list.
    """
    if settings is None:
        settings = get_settings()

    connectors = _build_connectors(settings)

    try:
        storage.save_theme(theme)

        # --- Fan-out across queries and sources ---
        all_candidates: list[RawCandidate] = []
        for query in theme.parsed_queries:
            query_candidates = await _fan_out_query(
                connectors, query, settings.max_results_per_source_per_query
            )
            all_candidates.extend(query_candidates)

        logger.info(
            "Collected %d raw candidates across %d queries",
            len(all_candidates),
            len(theme.parsed_queries),
        )

        # --- Deduplicate ---
        deduped = deduplicate_candidates(all_candidates)
        logger.info("After dedup: %d candidates", len(deduped))

        # --- Convert to Work objects ---
        works = [_candidate_to_work(c) for c in deduped]

        # --- Rank ---
        works = rank_papers(works, theme)

        works = storage.replace_theme_results(theme.id, works)

        logger.info("Retrieval complete: %d works saved for theme %s", len(works), theme.id)
        return works

    except Exception:
        logger.exception("Retrieval failed for theme %s", theme.id)
        raise

    finally:
        # Always close connectors to release httpx clients
        await asyncio.gather(
            *[c.close() for c in connectors], return_exceptions=True
        )


async def run_retrieval_for_document(
    document_text: str,
    storage: StorageService,
    settings: Settings | None = None,
) -> tuple[Theme, list[Work]]:
    """Convenience: parse a document into a Theme, then run the full retrieval pipeline."""
    from scholartrace.services.theme_parser import parse_theme

    theme = parse_theme(document_text)
    storage.save_theme(theme)

    works = await run_retrieval(theme, storage, settings=settings)
    return theme, works
