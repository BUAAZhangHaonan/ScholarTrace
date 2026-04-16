from __future__ import annotations

import asyncio
import itertools
import logging
import time
from typing import Any

import httpx

from scholartrace.config import Settings
from scholartrace.connectors.base import BaseConnector
from scholartrace.models.schemas import RawCandidate, SourceName

logger = logging.getLogger(__name__)

_FIELDS = (
    "title,year,abstract,authors,citationCount,venue,"
    "externalIds,url,openAccessPdf,fieldsOfStudy"
)
_PAGE_SIZE = 100
_MAX_RETRIES = 3
_BASE_BACKOFF = 2.0


class SemanticScholarConnector(BaseConnector):
    """Connector for the Semantic Scholar Graph API with multi-key rotation."""

    source_name = "semantic_scholar"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._keys = self._resolve_keys()
        self._key_cycle = itertools.cycle(self._keys) if self._keys else None
        self._current_key: str | None = None
        self._client = httpx.AsyncClient(
            base_url="https://api.semanticscholar.org/graph/v1",
            timeout=30.0,
        )

    def _resolve_keys(self) -> list[str]:
        """Collect all API keys from single and multi-key config."""
        keys: list[str] = []
        if self._settings.semantic_scholar_api_key:
            keys.append(self._settings.semantic_scholar_api_key)
        if self._settings.semantic_scholar_api_keys:
            for k in self._settings.semantic_scholar_api_keys.split(","):
                k = k.strip()
                if k and k not in keys:
                    keys.append(k)
        return keys

    def _next_headers(self) -> dict[str, str]:
        """Get headers with the next API key in rotation."""
        if self._key_cycle:
            self._current_key = next(self._key_cycle)
            return {"x-api-key": self._current_key}
        return {}

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
                "offset": offset,
                "limit": limit,
                "fields": _FIELDS,
            }

            body = await self._request_with_retry("/paper/search", params)
            if body is None:
                break

            papers = body.get("data", [])
            if not papers:
                break

            for paper in papers:
                candidate = self._parse_paper(paper)
                if candidate:
                    results.append(candidate)

            next_token = body.get("next")
            if not next_token:
                break
            offset = next_token if isinstance(next_token, int) else offset + limit

        return results[:max_results]

    async def _request_with_retry(
        self, path: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Make a request with multi-key rotation and exponential backoff on 429."""
        n_keys = len(self._keys) if self._keys else 1
        max_attempts = _MAX_RETRIES * n_keys

        for attempt in range(max_attempts):
            headers = self._next_headers()
            try:
                resp = await self._client.get(path, params=params, headers=headers)
                if resp.status_code == 429:
                    backoff = _BASE_BACKOFF * (2 ** (attempt // n_keys))
                    logger.warning(
                        "S2 rate limited (key=%s...), retrying in %.1fs (attempt %d/%d)",
                        (self._current_key or "")[:8],
                        backoff,
                        attempt + 1,
                        max_attempts,
                    )
                    await asyncio.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    backoff = _BASE_BACKOFF * (2 ** (attempt // n_keys))
                    logger.warning(
                        "S2 429, retrying in %.1fs (attempt %d/%d)",
                        backoff,
                        attempt + 1,
                        max_attempts,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise
            except httpx.TimeoutException:
                backoff = _BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "S2 timeout, retrying in %.1fs (attempt %d/%d)",
                    backoff,
                    attempt + 1,
                    max_attempts,
                )
                await asyncio.sleep(backoff)
                continue

        logger.error("S2 all retries exhausted for %s", path)
        return None

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_paper(paper: dict[str, Any]) -> RawCandidate | None:
        title = paper.get("title")
        if not title:
            return None

        authors: list[str] = []
        for a in paper.get("authors", []):
            name = a.get("name")
            if name:
                authors.append(name)

        ext = paper.get("externalIds") or {}

        doi: str | None = None
        raw_doi = ext.get("DOI")
        if raw_doi:
            doi = raw_doi if raw_doi.startswith("http") else f"https://doi.org/{raw_doi}"

        pdf_url: str | None = None
        oa_pdf = paper.get("openAccessPdf")
        if oa_pdf:
            pdf_url = oa_pdf.get("url")

        return RawCandidate(
            title=title,
            authors=authors,
            year=paper.get("year"),
            venue=paper.get("venue") or None,
            abstract=paper.get("abstract") or None,
            doi=doi,
            arxiv_id=ext.get("ArXiv"),
            s2_id=str(paper.get("paperId", "")) or None,
            dblp_key=ext.get("DBLP"),
            source=SourceName.SEMANTIC_SCHOLAR,
            citation_count=paper.get("citationCount", 0),
            pdf_url=pdf_url,
        )
