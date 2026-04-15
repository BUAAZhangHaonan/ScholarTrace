from __future__ import annotations

from abc import ABC, abstractmethod

from scholartrace.models.schemas import RawCandidate


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
