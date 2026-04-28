from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import httpx

from scholartrace.models.schemas import RawCandidate

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 2
_DEFAULT_BASE_BACKOFF = 1.0


class BaseConnector(ABC):
    """Abstract base class for all source connectors."""

    source_name: str = ""

    @abstractmethod
    async def search(self, query: str, max_results: int = 200) -> list[RawCandidate]:
        """Search for papers matching *query*, returning up to *max_results* candidates."""
        ...

    async def get_fulltext_url(self, paper_id: str) -> str | None:
        """Return a direct URL to the full text of *paper_id*, if available."""
        return None

    async def close(self) -> None:
        """Clean up resources (HTTP clients, etc.)."""
        pass

    async def _http_get_with_retry(
        self,
        client: httpx.AsyncClient,
        path: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_backoff: float = _DEFAULT_BASE_BACKOFF,
    ) -> httpx.Response:
        """Make an HTTP GET with automatic retry on transient errors.

        Retries on: 429 (rate limit), 5xx (server error), httpx.TimeoutException.
        Uses exponential backoff between retries.
        """
        for attempt in range(max_retries + 1):
            try:
                resp = await client.get(path, params=params, headers=headers)
                if resp.status_code == 429 and attempt < max_retries:
                    backoff = base_backoff * (2 ** attempt)
                    logger.warning(
                        "%s rate limited (429), retrying in %.1fs (attempt %d/%d)",
                        self.source_name, backoff, attempt + 1, max_retries + 1,
                    )
                    await asyncio.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {500, 502, 503, 504} and attempt < max_retries:
                    backoff = base_backoff * (2 ** attempt)
                    logger.warning(
                        "%s server error (%d), retrying in %.1fs (attempt %d/%d)",
                        self.source_name, exc.response.status_code,
                        backoff, attempt + 1, max_retries + 1,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise
            except httpx.TimeoutException:
                if attempt < max_retries:
                    backoff = base_backoff * (2 ** attempt)
                    logger.warning(
                        "%s timeout, retrying in %.1fs (attempt %d/%d)",
                        self.source_name, backoff, attempt + 1, max_retries + 1,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise
        # Should not reach here, but just in case
        raise RuntimeError(f"{self.source_name}: all retries exhausted")
