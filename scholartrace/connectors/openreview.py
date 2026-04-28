from __future__ import annotations

import logging
from typing import Any

import httpx

from scholartrace.config import Settings
from scholartrace.connectors.base import BaseConnector
from scholartrace.models.schemas import RawCandidate, SourceName

logger = logging.getLogger(__name__)

_PAGE_SIZE = 50


class OpenReviewConnector(BaseConnector):
    """Connector for the OpenReview API v2."""

    source_name = "openreview"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client = httpx.AsyncClient(
            base_url="https://api2.openreview.net",
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def search(self, query: str, max_results: int = 200) -> list[RawCandidate]:
        results: list[RawCandidate] = []
        offset = 0

        while len(results) < max_results:
            remaining = max_results - len(results)
            limit = min(_PAGE_SIZE, remaining)

            params = {
                "query": query,
                "source": "forum",
                "limit": limit,
                "offset": offset,
            }
            resp = await self._http_get_with_retry(self._client, "/notes/search", params=params)
            body = resp.json()

            notes = body.get("notes", [])
            if not notes:
                break

            for note in notes:
                candidate = self._parse_note(note)
                if candidate:
                    results.append(candidate)

            offset += len(notes)
            if len(notes) < limit:
                break

        return results[:max_results]

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_note(note: dict[str, Any]) -> RawCandidate | None:
        content = note.get("content", {})

        title = _get_value(content, "title")
        if not title:
            return None

        abstract = _get_value(content, "abstract") or None

        # Authors can be a list or a string
        raw_authors = _get_value(content, "authors")
        if isinstance(raw_authors, list):
            authors = [str(a) for a in raw_authors]
        elif isinstance(raw_authors, str) and raw_authors:
            authors = [raw_authors]
        else:
            authors = []

        venue = _get_value(content, "venue") or _get_value(content, "venueid") or None
        forum_id = note.get("forum") or None

        return RawCandidate(
            title=title,
            authors=authors,
            abstract=abstract,
            venue=venue,
            openreview_id=forum_id,
            source=SourceName.OPENREVIEW,
        )


def _get_value(content: dict[str, Any], field: str) -> str | list | None:
    """Extract a value from OpenReview content dict.

    Content fields are dicts like ``{"value": "some text"}``.
    """
    entry = content.get(field)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("value")
    return entry
