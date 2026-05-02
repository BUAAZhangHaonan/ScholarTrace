"""Multi-key deduplication service for RawCandidate objects.

Uses exact identifier matching (DOI, arXiv, S2, OpenAlex, DBLP, OpenReview)
via Union-Find, followed by fuzzy title matching for remaining ungrouped
candidates. Merges each group into the richest single candidate.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from rapidfuzz import fuzz

from scholartrace.models.schemas import RawCandidate, SourceName

_DEDUP_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dedup")


class UnionFind:
    """Disjoint Set Union with path halving and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def _merge_group(group: list[RawCandidate]) -> RawCandidate:
    """Merge a list of candidates that represent the same paper.

    Keeps the richest metadata across all candidates in the group.
    """
    title = max((c.title for c in group), key=len, default="")
    authors = max((c.authors for c in group), key=len, default=[])
    abstract = max((c.abstract for c in group if c.abstract), key=len, default=None)
    year = next((c.year for c in group if c.year is not None), None)
    venue = next((c.venue for c in group if c.venue is not None), None)
    doi = next((c.doi for c in group if c.doi), None)
    arxiv_id = next((c.arxiv_id for c in group if c.arxiv_id), None)
    openalex_id = next((c.openalex_id for c in group if c.openalex_id), None)
    s2_id = next((c.s2_id for c in group if c.s2_id), None)
    dblp_key = next((c.dblp_key for c in group if c.dblp_key), None)
    openreview_id = next((c.openreview_id for c in group if c.openreview_id), None)
    citation_count = max((c.citation_count for c in group), default=0)
    reference_count = max((c.reference_count for c in group), default=0)
    fulltext_url = next((c.fulltext_url for c in group if c.fulltext_url), None)
    pdf_url = next((c.pdf_url for c in group if c.pdf_url), None)
    html_url = next((c.html_url for c in group if c.html_url), None)
    oa_url = next((c.oa_url for c in group if c.oa_url), None)
    license_val = next((c.license for c in group if c.license), None)

    seen: set[str] = set()
    source_provenance: list[str] = []
    for c in group:
        name = c.source.value if isinstance(c.source, SourceName) else str(c.source)
        if name not in seen:
            seen.add(name)
            source_provenance.append(name)

    return RawCandidate(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        abstract=abstract,
        doi=doi,
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        s2_id=s2_id,
        dblp_key=dblp_key,
        openreview_id=openreview_id,
        source=group[0].source,
        citation_count=citation_count,
        reference_count=reference_count,
        fulltext_url=fulltext_url,
        pdf_url=pdf_url,
        html_url=html_url,
        oa_url=oa_url,
        license=license_val,
        source_provenance=source_provenance,
    )


def deduplicate_candidates(candidates: list[RawCandidate]) -> list[RawCandidate]:
    """Deduplicate a list of RawCandidate objects.

    1. Exact-match on shared identifiers via Union-Find.
    2. Fuzzy title match (token_sort_ratio >= 0.85) + same year for
       remaining ungrouped candidates.
    3. Merge each group into one candidate with the richest metadata.
    """
    if not candidates:
        return []

    n = len(candidates)
    uf = UnionFind(n)

    # --- Step 1: exact identifier matching ---
    id_fields = [
        "doi",
        "arxiv_id",
        "s2_id",
        "openalex_id",
        "dblp_key",
        "openreview_id",
    ]
    for field in id_fields:
        index: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(candidates):
            val = getattr(c, field, None)
            if val:
                index[val].append(i)
        for indices in index.values():
            for j in range(1, len(indices)):
                uf.union(indices[0], indices[j])

    # --- Step 2: fuzzy title matching for ungrouped candidates ---
    # Build groups from current UF state
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    # Only consider groups that are singletons (not yet merged by IDs)
    singletons = [root for root, members in groups.items() if len(members) == 1]
    singleton_titles = {
        root: candidates[root].title.lower() for root in singletons
    }

    for ai_pos in range(len(singletons)):
        a_root = singletons[ai_pos]
        a_title = singleton_titles[a_root]
        if not a_title:
            continue
        a_year = candidates[a_root].year
        for bi_pos in range(ai_pos + 1, len(singletons)):
            b_root = singletons[bi_pos]
            if uf.find(a_root) == uf.find(b_root):
                continue
            b_title = singleton_titles[b_root]
            if not b_title:
                continue
            b_year = candidates[b_root].year
            # Year must match (or both None)
            if a_year is not None and b_year is not None and a_year != b_year:
                continue
            if a_year is None and b_year is not None:
                continue
            if a_year is not None and b_year is None:
                continue
            score = fuzz.token_sort_ratio(a_title, b_title) / 100.0
            if score >= 0.85:
                uf.union(a_root, b_root)

    # --- Step 3: build final groups and merge ---
    final_groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        final_groups[uf.find(i)].append(i)

    results: list[RawCandidate] = []
    for members in final_groups.values():
        group_candidates = [candidates[i] for i in members]
        results.append(_merge_group(group_candidates))

    return results


async def deduplicate_candidates_async(
    candidates: list[RawCandidate],
) -> list[RawCandidate]:
    """Async wrapper that runs dedup in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _DEDUP_EXECUTOR,
        partial(deduplicate_candidates, candidates),
    )
