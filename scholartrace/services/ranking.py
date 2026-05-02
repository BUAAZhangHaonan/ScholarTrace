"""Multi-objective ranking service for ScholarTrace.

Computes individual scores (relevance, recency, influence, venue)
and a weighted composite score for each work, then returns works
sorted by composite_score descending.
"""

from __future__ import annotations

import asyncio
import math
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scholartrace.config import get_settings
from scholartrace.models.schemas import Theme, Work

_RANK_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ranking")

# ── Venue tiers ──────────────────────────────────────────────────────
TOP_TIER_VENUES: set[str] = {
    v.lower()
    for v in (
        "NeurIPS",
        "ICML",
        "ICLR",
        "ACL",
        "EMNLP",
        "AAAI",
        "CVPR",
        "IJCAI",
        "NAACL",
        "COLING",
        "Interspeech",
        "SIGCHI",
        "CHI",
        "TACL",
        "CL",
        "JMLR",
        "Nature",
        "Science",
        "PNAS",
    )
}

GOOD_VENUE_KEYWORDS: tuple[str, ...] = (
    "workshop",
    "symposium",
    "conference",
    "journal",
    "transactions",
    "letters",
    "proceedings",
)

ARXIV_NAMES: set[str] = {"arxiv", "corr", "coRR"}


def _venue_score(venue: str | None) -> float:
    """Tiered venue scoring."""
    if venue is None:
        return 0.3
    v_lower = venue.lower().strip()
    if not v_lower:
        return 0.3
    # Top-tier: exact match (ignoring case)
    if v_lower in TOP_TIER_VENUES:
        return 1.0
    # Good venues: keyword match or arXiv
    if any(kw in v_lower for kw in GOOD_VENUE_KEYWORDS):
        return 0.7
    if any(arxiv in v_lower for arxiv in ARXIV_NAMES):
        return 0.7
    # Everything else
    return 0.5


def _relevance_scores(works: Sequence[Work], theme: Theme) -> list[float]:
    """TF-IDF cosine similarity between theme queries and each paper."""
    # If there are no works, return empty list
    if not works:
        return []

    # Theme document: prefer structured anchors (topics/methods/datasets),
    # then fall back to generated queries.
    anchor_fragments: list[str] = []
    anchor_fragments.extend(theme.parsed_topics[:12])
    anchor_fragments.extend(theme.parsed_methods[:8])
    anchor_fragments.extend(theme.parsed_datasets[:8])
    if not anchor_fragments:
        anchor_fragments.extend(theme.parsed_queries[:8])

    anchor_tokens: list[str] = []
    for fragment in anchor_fragments:
        anchor_tokens.extend(re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", fragment.lower()))
    theme_doc = " ".join(anchor_tokens)

    # Paper documents: title repeated 2x + abstract for stronger title signal
    paper_docs: list[str] = []
    for w in works:
        parts: list[str] = []
        if w.title:
            parts.append(w.title)
            parts.append(w.title)  # title 2x for emphasis
        if w.abstract:
            parts.append(w.abstract)
        paper_docs.append(" ".join(parts))

    # Build corpus: theme first, then papers
    corpus = [theme_doc] + paper_docs

    # If theme doc is empty and all paper docs are empty, all zeros
    if not any(corpus):
        return [0.0] * len(works)

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Happens when all documents are empty after tokenization
        return [0.0] * len(works)

    theme_vec = tfidf_matrix[0:1]
    scores: list[float] = []
    for i in range(len(works)):
        paper_vec = tfidf_matrix[i + 1: i + 2]
        sim = cosine_similarity(theme_vec, paper_vec)[0][0]
        # Clip to [0, 1]
        scores.append(float(np.clip(sim, 0.0, 1.0)))

    return scores


def _recency_score(year: int | None, current_year: int = 2026) -> float:
    """Exponential decay with ~1.5 year half-life, stronger recency preference.

    Scoring guide (current_year=2026):
      2026: 1.00   2025: 0.65   2024: 0.42   2023: 0.27
      2022: 0.18   2021: 0.12   2020: 0.08   older: <0.05
    """
    if year is None:
        return 0.0
    if year > current_year:
        return 1.0
    return math.exp(-0.43 * (current_year - year))


def _influence_score(
    citation_count: int, max_citation_count: int
) -> float:
    """Log-normalized citations."""
    if citation_count == 0:
        return 0.0
    return math.log(1 + citation_count) / math.log(1 + max_citation_count)


def _fulltext_score(fulltext_available: bool) -> float:
    """Bonus for having full text."""
    return 1.0 if fulltext_available else 0.0


def _source_agreement_score(source_provenance: list[str]) -> float:
    """More sources = more confidence, capped at 1.0."""
    return min(len(source_provenance) / 3, 1.0)


def _staleness_penalty(
    year: int | None,
    influence_score: float,
    *,
    current_year: int = 2026,
) -> float:
    """Softly penalize older low-impact papers without hard date cutoffs.

    No penalty is applied for papers published within the most recent 2 years.
    For older papers, stronger influence mitigates the penalty.
    """
    if year is None:
        return 0.0

    age = max(0, current_year - year)
    if age <= 2:
        return 0.0

    age_factor = min((age - 2) / 6.0, 1.0)
    influence_shield = float(np.clip(influence_score, 0.0, 1.0))
    return float(np.clip(age_factor * (1.0 - influence_shield), 0.0, 1.0))


# ── Main entry point ─────────────────────────────────────────────────
def rank_papers(
    works: list[Work],
    theme: Theme,
    weights: dict[str, float] | None = None,
) -> list[Work]:
    """Score and rank papers by composite score.

    Updates each Work's score fields in-place and returns the list
    sorted by composite_score descending.
    """
    if not works:
        return []

    # Resolve weights: caller override > settings defaults
    settings = get_settings()
    w: dict[str, float] = {
        "weight_relevance": settings.weight_relevance,
        "weight_recency": settings.weight_recency,
        "weight_influence": settings.weight_influence,
        "weight_venue": settings.weight_venue,
        "weight_fulltext": settings.weight_fulltext,
        "weight_source_agreement": settings.weight_source_agreement,
    }
    if weights:
        w.update(weights)

    # Pre-compute max citation count for influence normalization
    max_citations = max((wk.citation_count for wk in works), default=0)
    if max_citations == 0:
        max_citations = 1

    # Compute relevance scores in batch (TF-IDF needs the full corpus)
    relevance_scores = _relevance_scores(works, theme)

    for i, work in enumerate(works):
        rel = relevance_scores[i]
        rec = _recency_score(work.year)
        inf = _influence_score(work.citation_count, max_citations)
        ven = _venue_score(work.venue)
        ft = _fulltext_score(work.fulltext_available)
        sa = _source_agreement_score(work.source_provenance)

        composite = (
            w["weight_relevance"] * rel
            + w["weight_recency"] * rec
            + w["weight_influence"] * inf
            + w["weight_venue"] * ven
            + w["weight_fulltext"] * ft
            + w["weight_source_agreement"] * sa
        )

        stale_penalty = _staleness_penalty(work.year, inf)
        composite *= 1.0 - 0.30 * stale_penalty
        composite = float(np.clip(composite, 0.0, 1.0))

        work.relevance_score = rel
        work.recency_score = rec
        work.influence_score = inf
        work.venue_score = ven
        work.composite_score = composite

    works.sort(key=lambda wk: wk.composite_score, reverse=True)
    return works


async def rank_papers_async(
    works: list[Work],
    theme: Theme,
    weights: dict[str, float] | None = None,
) -> list[Work]:
    """Async wrapper that runs ranking in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _RANK_EXECUTOR,
        partial(rank_papers, works, theme, weights),
    )
