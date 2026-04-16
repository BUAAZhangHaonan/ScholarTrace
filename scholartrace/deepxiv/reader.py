"""Async DeepXiv Reader for ScholarTrace.

Adapted from deepxiv_sdk.reader but uses httpx.AsyncClient for
non-blocking I/O and integrates with ScholarTrace's token pool.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://data.rag.ac.cn"
_REGISTER_URL = f"{_BASE_URL}/api/register/sdk"
_SDK_SECRET = "UuZp0i83svQU7_naUEexczc-X3NWv7lvNkD8e3sPyng"

_DEFAULT_TIMEOUT = 30.0
_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0


class DeepXivAPIError(Exception):
    """Base exception for DeepXiv API errors."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class DeepXivRateLimitError(DeepXivAPIError):
    """Rate limit exceeded (HTTP 429)."""


class DeepXivAuthError(DeepXivAPIError):
    """Authentication failed (HTTP 401)."""


class DeepXivNotFoundError(DeepXivAPIError):
    """Resource not found (HTTP 404)."""


def _raise_for_status(resp: httpx.Response) -> None:
    """Raise appropriate exception based on HTTP status."""
    code = resp.status_code
    if 200 <= code < 300:
        return
    body = resp.text[:500]
    if code == 429:
        raise DeepXivRateLimitError(f"Rate limit exceeded: {body}", code)
    if code == 401:
        raise DeepXivAuthError(f"Authentication failed: {body}", code)
    if code == 404:
        raise DeepXivNotFoundError(f"Not found: {body}", code)
    raise DeepXivAPIError(f"HTTP {code}: {body}", code)


class DeepXivReader:
    """Async reader for the DeepXiv API (data.rag.ac.cn).

    Uses a token pool for automatic rotation on rate limits.
    All methods are async and use httpx.AsyncClient.
    """

    def __init__(
        self,
        token_or_pool: str | Any | None = None,
    ):
        """Initialize the reader.

        Args:
            token_or_pool: Either a single token string, a TokenPool instance,
                           or None (will try env var DEEPXIV_TOKEN).
        """
        # Import here to avoid circular dependency
        from .token_pool import TokenPool

        if isinstance(token_or_pool, TokenPool):
            self._pool = token_or_pool
        elif isinstance(token_or_pool, str) and token_or_pool:
            self._pool = TokenPool(initial_tokens=[token_or_pool])
        else:
            self._pool = TokenPool.from_env()

        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=_DEFAULT_TIMEOUT,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------
    async def _headers(self) -> dict[str, str]:
        """Get authorization headers with the current best token."""
        token = await self._pool.get_token()
        return {"Authorization": f"Bearer {token}"}

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        params: Any | None = None,
    ) -> httpx.Response:
        """Make a request with automatic retry and token rotation on 429."""
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            headers = await self._headers()
            try:
                resp = await self._client.request(
                    method, url, json=json, params=params, headers=headers,
                )
                if resp.status_code == 429:
                    # Rotate token and retry
                    await self._pool.rotate()
                    backoff = _BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        "DeepXiv rate limit, rotating token, retry %.1fs", backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                _raise_for_status(resp)
                return resp
            except DeepXivAuthError:
                # Bad token, rotate immediately
                await self._pool.rotate()
                logger.warning("DeepXiv auth error, rotating token")
                if attempt == _MAX_RETRIES - 1:
                    raise
                continue
            except DeepXivAPIError as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    backoff = _BASE_BACKOFF * (2 ** attempt)
                    await asyncio.sleep(backoff)
                    continue
                raise

        raise last_error or DeepXivAPIError("Max retries exceeded")

    # ------------------------------------------------------------------
    # arXiv search
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        *,
        size: int = 10,
        offset: int = 0,
        search_mode: str = "hybrid",
        categories: list[str] | None = None,
        authors: list[str] | None = None,
        min_citation: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """Search arXiv papers via DeepXiv (BM25 / Vector / Hybrid).

        Returns dict with 'total' and 'results' list.
        """
        body: dict[str, Any] = {
            "query": query,
            "size": size,
            "offset": offset,
            "search_mode": search_mode,
        }
        if categories:
            body["categories"] = categories
        if authors:
            body["authors"] = authors
        if min_citation is not None:
            body["min_citation"] = min_citation
        if date_from:
            body["date_from"] = date_from
        if date_to:
            body["date_to"] = date_to

        resp = await self._request_with_retry("POST", "/api/arxiv/search", json=body)
        return resp.json()

    # ------------------------------------------------------------------
    # Paper metadata
    # ------------------------------------------------------------------
    async def head(self, arxiv_id: str) -> dict[str, Any] | None:
        """Get paper metadata: title, abstract, authors, sections with TLDRs."""
        try:
            resp = await self._request_with_retry(
                "GET", f"/api/arxiv/{arxiv_id}/head",
            )
            return resp.json()
        except DeepXivNotFoundError:
            return None

    async def brief(self, arxiv_id: str) -> dict[str, Any] | None:
        """Get brief paper info: title, TLDR, keywords, citations."""
        try:
            resp = await self._request_with_retry(
                "GET", f"/api/arxiv/{arxiv_id}/brief",
            )
            return resp.json()
        except DeepXivNotFoundError:
            return None

    # ------------------------------------------------------------------
    # Full text
    # ------------------------------------------------------------------
    async def raw(self, arxiv_id: str) -> str | None:
        """Get full paper text in markdown format."""
        try:
            resp = await self._request_with_retry(
                "GET", f"/api/arxiv/{arxiv_id}/raw",
            )
            return resp.text
        except DeepXivNotFoundError:
            return None

    async def preview(self, arxiv_id: str) -> dict[str, Any] | None:
        """Get first ~10k characters of a paper."""
        try:
            resp = await self._request_with_retry(
                "GET", f"/api/arxiv/{arxiv_id}/preview",
            )
            return resp.json()
        except DeepXivNotFoundError:
            return None

    async def section(self, arxiv_id: str, section_name: str) -> str | None:
        """Get a specific section's content."""
        try:
            resp = await self._request_with_retry(
                "GET", f"/api/arxiv/{arxiv_id}/section/{section_name}",
            )
            return resp.text
        except DeepXivNotFoundError:
            return None

    # ------------------------------------------------------------------
    # Semantic Scholar via DeepXiv
    # ------------------------------------------------------------------
    async def semantic_scholar_search(
        self,
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        fields: str = "title,abstract,year,citationCount,externalIds",
    ) -> dict[str, Any]:
        """Search Semantic Scholar via DeepXiv (free, no personal API key needed)."""
        body = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": fields,
        }
        resp = await self._request_with_retry(
            "POST", "/api/semantic_scholar/search", json=body,
        )
        return resp.json()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    @staticmethod
    async def register() -> str | None:
        """Auto-register a new DeepXiv account and return the token.

        Uses the SDK secret to create a random account.
        """
        import string
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        username = f"deepxiv_{suffix}"
        email = f"{suffix}@example.com"

        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                _REGISTER_URL,
                json={
                    "username": username,
                    "email": email,
                    "sdk_secret": _SDK_SECRET,
                },
            )
            if resp.status_code != 200:
                logger.error("DeepXiv registration failed: %s", resp.text[:200])
                return None
            data = resp.json()
            token = data.get("token")
            if token:
                logger.info("DeepXiv registered: %s", username)
            return token
