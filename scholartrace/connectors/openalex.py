from __future__ import annotations

import logging
from typing import Any

import httpx

from scholartrace.config import Settings
from scholartrace.connectors.base import BaseConnector
from scholartrace.models.schemas import RawCandidate, SourceName

logger = logging.getLogger(__name__)

_PER_PAGE = 200
_FIELDS = (
    "id,doi,title,publication_year,cited_by_count,"
    "authorships,abstract_inverted_index,locations,"
    "open_access,type"
)


class OpenAlexConnector(BaseConnector):
    """Connector for the OpenAlex scholarly API."""

    source_name = "openalex"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client = httpx.AsyncClient(
            base_url="https://api.openalex.org",
            params={"mailto": self._settings.openalex_mailto},
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, query: str, max_results: int = 200) -> list[RawCandidate]:
        results: list[RawCandidate] = []
        cursor = "*"

        while len(results) < max_results and cursor:
            remaining = max_results - len(results)
            per_page = min(_PER_PAGE, remaining)

            params = {
                "search": query,
                "per_page": per_page,
                "cursor": cursor,
                "select": _FIELDS,
            }
            resp = await self._client.get("/works", params=params)
            resp.raise_for_status()
            body = resp.json()

            for work in body.get("results", []):
                candidate = self._parse_work(work)
                if candidate:
                    results.append(candidate)

            cursor = body.get("meta", {}).get("next_cursor")
            if not body.get("results"):
                break

        return results[:max_results]

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_work(work: dict[str, Any]) -> RawCandidate | None:
        title = work.get("title")
        if not title:
            return None

        # Authors
        authors: list[str] = []
        for auth in work.get("authorships", []):
            name = auth.get("author", {}).get("display_name")
            if name:
                authors.append(name)

        # DOI
        doi: str | None = work.get("doi")
        if doi and not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"

        # OpenAlex ID — strip URL prefix
        openalex_id: str | None = None
        raw_id = work.get("id", "")
        if raw_id:
            openalex_id = raw_id.rsplit("/", 1)[-1] if "/" in raw_id else raw_id

        # Abstract from inverted index
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

        # Open access URL
        oa_url: str | None = None
        oa_info = work.get("open_access")
        if oa_info:
            oa_url = oa_info.get("oa_url")

        return RawCandidate(
            title=title,
            authors=authors,
            year=work.get("publication_year"),
            venue=_extract_venue(work),
            abstract=abstract,
            doi=doi,
            openalex_id=openalex_id,
            source=SourceName.OPENALEX,
            citation_count=work.get("cited_by_count", 0),
            oa_url=oa_url,
        )


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """Convert an OpenAlex inverted abstract index back into plain text."""
    if not inverted:
        return None
    length = max(pos for positions in inverted.values() for pos in positions) + 1
    words: list[str] = [""] * length
    for word, positions in inverted.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


def _extract_venue(work: dict[str, Any]) -> str | None:
    for loc in work.get("locations", []):
        source = loc.get("source") or {}
        name = source.get("display_name")
        if name:
            return name
    return None
