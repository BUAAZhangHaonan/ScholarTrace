from __future__ import annotations

import logging
from typing import Any

import httpx

from scholartrace.config import Settings
from scholartrace.connectors.base import BaseConnector
from scholartrace.models.schemas import RawCandidate, SourceName

logger = logging.getLogger(__name__)

_PAGE_SIZE = 1000


class DblpConnector(BaseConnector):
    """Connector for the DBLP search API."""

    source_name = "dblp"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client = httpx.AsyncClient(
            base_url="https://dblp.org/search/publ/api",
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, query: str, max_results: int = 200) -> list[RawCandidate]:
        results: list[RawCandidate] = []
        first = 0

        while len(results) < max_results:
            remaining = max_results - len(results)
            h = min(_PAGE_SIZE, remaining)

            params = {
                "q": query,
                "h": h,
                "f": first,
                "format": "json",
            }
            resp = await self._client.get("", params=params)
            resp.raise_for_status()
            body = resp.json()

            result = body.get("result", {})
            hits_info = result.get("hits", {})

            total = int(hits_info.get("@total", 0))
            if total == 0:
                break

            hits = hits_info.get("hit", [])
            if not hits:
                break

            for hit in hits:
                info = hit.get("info", {})
                candidate = self._parse_info(info)
                if candidate:
                    results.append(candidate)

            sent = int(hits_info.get("@sent", 0))
            first += sent
            if first >= total:
                break

        return results[:max_results]

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_info(info: dict[str, Any]) -> RawCandidate | None:
        title = info.get("title", "").strip()
        if not title:
            return None

        # Authors — can be a single dict or a list
        authors = _extract_authors(info.get("authors", {}).get("author"))

        # Year
        year: int | None = None
        raw_year = info.get("year")
        if raw_year:
            try:
                year = int(raw_year)
            except (ValueError, TypeError):
                pass

        # DOI
        doi: str | None = info.get("doi") or None
        if doi and not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"

        # DBLP key and URL
        dblp_key = info.get("key") or None
        url = info.get("url") or None

        return RawCandidate(
            title=title,
            authors=authors,
            year=year,
            venue=info.get("venue") or None,
            doi=doi,
            dblp_key=dblp_key,
            source=SourceName.DBLP,
            html_url=url,
        )


def _extract_authors(
    author: dict[str, str] | list[dict[str, str]] | None,
) -> list[str]:
    """Normalise DBLP author field to a list of names.

    DBLP returns a single author as ``{"text": "Name"}`` or a list of such dicts.
    """
    if author is None:
        return []
    if isinstance(author, list):
        return [a.get("text", "") for a in author if a.get("text")]
    # Single dict
    name = author.get("text", "")
    return [name] if name else []
