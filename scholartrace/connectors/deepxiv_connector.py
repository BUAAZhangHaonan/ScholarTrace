"""DeepXiv connector for ScholarTrace.

Wraps the async DeepXivReader to implement the BaseConnector interface,
providing arXiv search via DeepXiv (hybrid BM25 + vector) and Semantic Scholar
search through the DeepXiv proxy.
"""

from __future__ import annotations

import logging
from typing import Any

from scholartrace.config import Settings
from scholartrace.connectors.base import BaseConnector
from scholartrace.deepxiv.reader import DeepXivReader
from scholartrace.models.schemas import RawCandidate, SourceName

logger = logging.getLogger(__name__)


class DeepXivConnector(BaseConnector):
    """Connector that searches arXiv via DeepXiv (data.rag.ac.cn).

    Uses DeepXiv's hybrid search (BM25 + vector) and can also query
    Semantic Scholar through the DeepXiv proxy.

    Falls back to DeepXiv token auto-registration if no token is configured.
    """

    source_name = "deepxiv"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._reader = DeepXivReader(settings=self._settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        max_results: int = 200,
        *,
        search_mode: str = "hybrid",
        categories: list[str] | None = None,
        authors: list[str] | None = None,
    ) -> list[RawCandidate]:
        """Search arXiv papers via DeepXiv.

        Args:
            query: Search query string.
            max_results: Maximum number of results.
            search_mode: One of 'bm25', 'vector', 'hybrid'.
            categories: Optional arXiv category filter (e.g. ['cs.AI']).
            authors: Optional author name filter.

        Returns:
            List of RawCandidate objects.
        """
        results: list[RawCandidate] = []
        offset = 0
        page_size = min(50, max_results)

        while len(results) < max_results:
            remaining = max_results - len(results)
            fetch_size = min(page_size, remaining)

            try:
                data = await self._reader.search(
                    query,
                    size=fetch_size,
                    offset=offset,
                    search_mode=search_mode,
                    categories=categories,
                    authors=authors,
                )
            except RuntimeError:
                raise
            except Exception:
                logger.exception("DeepXiv search failed at offset %d", offset)
                break

            hits = data.get("results") or data.get("hits") or []
            if not hits:
                break

            for hit in hits:
                candidate = self._hit_to_candidate(hit)
                if candidate:
                    results.append(candidate)

            offset += fetch_size
            # DeepXiv returns fewer results than requested when exhausted
            if len(hits) < fetch_size:
                break

        return results[:max_results]

    async def get_paper_metadata(self, arxiv_id: str) -> dict[str, Any] | None:
        """Get paper metadata (head) from DeepXiv."""
        try:
            return await self._reader.head(arxiv_id)
        except RuntimeError:
            raise
        except Exception:
            logger.exception("DeepXiv head failed for %s", arxiv_id)
            return None

    async def get_paper_brief(self, arxiv_id: str) -> dict[str, Any] | None:
        """Get brief paper info from DeepXiv."""
        try:
            return await self._reader.brief(arxiv_id)
        except RuntimeError:
            raise
        except Exception:
            logger.exception("DeepXiv brief failed for %s", arxiv_id)
            return None

    async def get_fulltext(self, arxiv_id: str) -> str | None:
        """Get full paper text in markdown from DeepXiv."""
        try:
            return await self._reader.raw(arxiv_id)
        except RuntimeError:
            raise
        except Exception:
            logger.exception("DeepXiv raw failed for %s", arxiv_id)
            return None

    async def get_section(self, arxiv_id: str, section_name: str) -> str | None:
        """Get a specific section's content from DeepXiv."""
        try:
            return await self._reader.section(arxiv_id, section_name)
        except RuntimeError:
            raise
        except Exception:
            logger.exception("DeepXiv section failed for %s/%s", arxiv_id, section_name)
            return None

    async def get_preview(self, arxiv_id: str) -> dict[str, Any] | None:
        """Get paper preview from DeepXiv."""
        try:
            return await self._reader.preview(arxiv_id)
        except RuntimeError:
            raise
        except Exception:
            logger.exception("DeepXiv preview failed for %s", arxiv_id)
            return None

    async def semantic_scholar_search(
        self,
        query: str,
        limit: int = 10,
        fields: str = "title,abstract,year,citationCount,externalIds",
    ) -> dict[str, Any]:
        """Search Semantic Scholar via DeepXiv proxy."""
        try:
            return await self._reader.semantic_scholar_search(
                query, limit=limit, fields=fields,
            )
        except RuntimeError:
            raise

    async def get_fulltext_url(self, paper_id: str) -> str | None:
        """DeepXiv doesn't provide direct fulltext URLs; use get_fulltext instead."""
        return None

    async def close(self) -> None:
        await self._reader.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _hit_to_candidate(hit: dict[str, Any]) -> RawCandidate | None:
        """Convert a DeepXiv search hit to a RawCandidate."""
        title = (hit.get("title") or "").strip()
        if not title:
            return None

        arxiv_id = hit.get("arxiv_id") or hit.get("id") or ""
        # If arxiv_id is a full URL, extract just the ID
        if arxiv_id.startswith("http"):
            arxiv_id = arxiv_id.rsplit("/", 1)[-1]

        year: int | None = None
        pub_date = hit.get("published") or hit.get("date") or ""
        if isinstance(pub_date, str) and len(pub_date) >= 4:
            try:
                year = int(pub_date[:4])
            except ValueError:
                pass
        elif isinstance(pub_date, int):
            year = pub_date

        authors = hit.get("authors") or []
        if isinstance(authors, list):
            author_names = []
            for a in authors:
                if isinstance(a, str):
                    author_names.append(a)
                elif isinstance(a, dict):
                    author_names.append(a.get("name", ""))
            authors = author_names

        return RawCandidate(
            title=title,
            authors=authors,
            year=year,
            abstract=hit.get("abstract") or hit.get("summary"),
            doi=hit.get("doi"),
            arxiv_id=arxiv_id or None,
            source=SourceName.DEEPXIV,
            citation_count=hit.get("citation_count", 0),
            pdf_url=hit.get("pdf_url"),
        )
