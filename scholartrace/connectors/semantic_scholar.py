from __future__ import annotations

import logging
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


class SemanticScholarConnector(BaseConnector):
    """Connector for the Semantic Scholar Graph API."""

    source_name = "semantic_scholar"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        headers: dict[str, str] = {}
        if self._settings.semantic_scholar_api_key:
            headers["x-api-key"] = self._settings.semantic_scholar_api_key
        self._client = httpx.AsyncClient(
            base_url="https://api.semanticscholar.org/graph/v1",
            headers=headers,
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
                "offset": offset,
                "limit": limit,
                "fields": _FIELDS,
            }
            resp = await self._client.get("/paper/search", params=params)
            resp.raise_for_status()
            body = resp.json()

            papers = body.get("data", [])
            if not papers:
                break

            for paper in papers:
                candidate = self._parse_paper(paper)
                if candidate:
                    results.append(candidate)

            # Check for continuation token
            next_token = body.get("next")
            if not next_token:
                break
            offset = next_token if isinstance(next_token, int) else offset + limit

        return results[:max_results]

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

        # DOI
        doi: str | None = None
        raw_doi = ext.get("DOI")
        if raw_doi:
            doi = raw_doi if raw_doi.startswith("http") else f"https://doi.org/{raw_doi}"

        # Open access PDF
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
