from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from scholartrace.config import Settings
from scholartrace.connectors.base import BaseConnector
from scholartrace.models.schemas import RawCandidate, SourceName

logger = logging.getLogger(__name__)

_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_PAGE_SIZE = 200
_DELAY_SECONDS = 3


class ArxivConnector(BaseConnector):
    """Connector for the arXiv export API."""

    source_name = "arxiv"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client = httpx.AsyncClient(
            base_url="https://export.arxiv.org/api",
            timeout=60.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def search(self, query: str, max_results: int = 200) -> list[RawCandidate]:
        results: list[RawCandidate] = []
        start = 0

        while len(results) < max_results:
            remaining = max_results - len(results)
            page_size = min(_PAGE_SIZE, remaining)

            params = {
                "search_query": f"all:{query}",
                "start": start,
                "max_results": page_size,
            }
            resp = await self._http_get_with_retry(self._client, "/query", params=params)

            entries = _parse_atom_entries(resp.text)
            if not entries:
                break

            for entry in entries:
                candidate = _entry_to_candidate(entry)
                if candidate:
                    results.append(candidate)

            start += page_size
            if len(entries) < page_size:
                break

            # arXiv asks for a polite delay between paginated requests
            await asyncio.sleep(_DELAY_SECONDS)

        return results[:max_results]

    async def close(self) -> None:
        await self._client.aclose()


# ------------------------------------------------------------------
# XML parsing helpers
# ------------------------------------------------------------------
def _parse_atom_entries(xml_text: str) -> list[dict[str, Any]]:
    """Parse the Atom feed and return a list of entry element trees."""
    root = ET.fromstring(xml_text)
    entries: list[dict[str, Any]] = []
    for entry_el in root.findall("atom:entry", _NS):
        entries.append(_element_to_dict(entry_el))
    return entries


def _element_to_dict(entry: ET.Element) -> dict[str, Any]:
    d: dict[str, Any] = {}
    d["title"] = _text(entry, "atom:title")
    d["summary"] = _text(entry, "atom:summary")
    d["id"] = _text(entry, "atom:id")
    d["published"] = _text(entry, "atom:published")

    # Authors
    authors: list[str] = []
    for author_el in entry.findall("atom:author", _NS):
        name = _text(author_el, "atom:name")
        if name:
            authors.append(name)
    d["authors"] = authors

    # Links
    links: list[dict[str, str]] = []
    for link_el in entry.findall("atom:link", _NS):
        links.append(dict(link_el.attrib))
    d["links"] = links

    # DOI from arxiv:doi
    doi_el = entry.find(f"{{{_ARXIV_NS}}}doi")
    d["doi"] = doi_el.text if doi_el is not None and doi_el.text else None

    return d


def _text(parent: ET.Element, tag: str) -> str:
    el = parent.find(tag, _NS)
    return (el.text or "").strip() if el is not None else ""


def _entry_to_candidate(entry: dict[str, Any]) -> RawCandidate | None:
    title = entry.get("title", "").strip()
    if not title:
        return None

    # Extract arXiv ID from the id URL
    raw_id = entry.get("id", "")
    arxiv_id = _extract_arxiv_id(raw_id)

    # Year from published date
    year: int | None = None
    pub = entry.get("published", "")
    if len(pub) >= 4:
        try:
            year = int(pub[:4])
        except ValueError:
            pass

    # PDF URL from links
    pdf_url: str | None = None
    for link in entry.get("links", []):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
            break

    return RawCandidate(
        title=title,
        authors=entry.get("authors", []),
        year=year,
        abstract=entry.get("summary") or None,
        doi=entry.get("doi"),
        arxiv_id=arxiv_id,
        source=SourceName.ARXIV,
        pdf_url=pdf_url,
    )


def _extract_arxiv_id(url: str) -> str:
    """Strip version and base URL to get the bare arXiv ID.

    ``http://arxiv.org/abs/2301.12345v1`` -> ``2301.12345``
    """
    segment = url.rsplit("/", 1)[-1]
    # Strip version suffix like v1, v2
    if "v" in segment:
        parts = segment.split("v")
        # Only strip if the part after 'v' is numeric
        if parts[-1].isdigit():
            segment = "v".join(parts[:-1])
    return segment
