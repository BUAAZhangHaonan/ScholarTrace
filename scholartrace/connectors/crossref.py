from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from scholartrace.config import Settings
from scholartrace.connectors.base import BaseConnector
from scholartrace.models.schemas import RawCandidate, SourceName

logger = logging.getLogger(__name__)

_PAGE_SIZE = 1000
_JATS_RE = re.compile(r"<\/?jats:[^>]*>")


class CrossrefConnector(BaseConnector):
    """Connector for the Crossref Works API."""

    source_name = "crossref"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client = httpx.AsyncClient(
            base_url="https://api.crossref.org",
            params={"mailto": self._settings.crossref_mailto},
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
            rows = min(_PAGE_SIZE, remaining)

            params = {
                "query": query,
                "rows": rows,
                "cursor": cursor,
            }
            resp = await self._http_get_with_retry(self._client, "/works", params=params)
            body = resp.json()

            message = body.get("message", {})
            items = message.get("items", [])
            if not items:
                break

            for item in items:
                candidate = self._parse_item(item)
                if candidate:
                    results.append(candidate)

            cursor = message.get("next-cursor")

        return results[:max_results]

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_item(item: dict[str, Any]) -> RawCandidate | None:
        # Title is an array — take the first element
        raw_titles = item.get("title", [])
        title = raw_titles[0].strip() if raw_titles else ""
        if not title:
            return None

        # Authors
        authors: list[str] = []
        for a in item.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            full = f"{given} {family}".strip()
            if full:
                authors.append(full)

        # Year from published-print or published-online
        year: int | None = None
        for key in ("published-print", "published-online"):
            parts = item.get(key, {}).get("date-parts", [[]])
            if parts and parts[0]:
                try:
                    year = int(parts[0][0])
                except (ValueError, IndexError, TypeError):
                    pass
                else:
                    break

        # Venue
        containers = item.get("container-title", [])
        venue = containers[0] if containers else None

        # DOI
        doi: str | None = item.get("DOI")
        if doi and not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"

        # Abstract — strip JATS XML tags
        abstract: str | None = None
        raw_abstract = item.get("abstract")
        if raw_abstract:
            abstract = _JATS_RE.sub("", raw_abstract).strip() or None

        return RawCandidate(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            abstract=abstract,
            doi=doi,
            source=SourceName.CROSSREF,
            citation_count=item.get("is-referenced-by-count", 0),
        )
