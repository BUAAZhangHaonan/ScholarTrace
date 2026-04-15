"""Full-text acquisition cascade service.

Tries multiple sources in order to obtain full text for a Work:
1. arXiv HTML
2. arXiv PDF
3. OA / pdf_url from metadata
4. Fall back to abstract-only
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

import fitz

from scholartrace.config import Settings
from scholartrace.models.schemas import (
    AccessStatus,
    Artifact,
    ArtifactKind,
    Section,
    Work,
)
from scholartrace.services.storage import StorageService

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30.0


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


async def _fetch_html(url: str, client: httpx.AsyncClient) -> str | None:
    """Fetch HTML content from *url*. Returns ``None`` on any error."""
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception:
        logger.debug("HTML fetch failed for %s", url, exc_info=True)
        return None


async def _fetch_pdf(url: str, client: httpx.AsyncClient) -> bytes | None:
    """Fetch PDF bytes from *url*. Returns ``None`` on any error."""
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception:
        logger.debug("PDF fetch failed for %s", url, exc_info=True)
        return None


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    text_parts: list[str] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
    except Exception:
        logger.debug("PDF text extraction failed", exc_info=True)
    return "\n".join(text_parts)


def _parse_html_sections(html: str) -> list[tuple[str, str]]:
    """Parse HTML into ``(title, content)`` pairs by scanning heading tags.

    Any content before the first heading is grouped under a synthetic
    section titled ``"Preamble"``.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Collect all heading and content elements in document order.
    headings = soup.find_all(["h1", "h2", "h3"])

    sections: list[tuple[str, str]] = []

    if not headings:
        # No headings at all -- return the full text as a single section.
        full_text = soup.get_text(separator="\n", strip=True)
        if full_text.strip():
            sections.append(("Full Text", full_text))
        return sections

    # Capture any content before the first heading.
    first_heading = headings[0]
    preamble_parts: list[str] = []
    for sibling in first_heading.previous_siblings:
        if hasattr(sibling, "get_text"):
            preamble_parts.append(sibling.get_text(separator="\n", strip=True))
        elif isinstance(sibling, str) and sibling.strip():
            preamble_parts.append(sibling.strip())
    preamble = "\n".join(reversed(preamble_parts)).strip()
    if preamble:
        sections.append(("Preamble", preamble))

    # For each heading, gather following siblings until the next heading.
    for idx, heading in enumerate(headings):
        title = heading.get_text(strip=True) or f"Section {idx}"
        content_parts: list[str] = []
        for sibling in heading.next_siblings:
            if sibling in headings:
                break
            if hasattr(sibling, "get_text"):
                content_parts.append(sibling.get_text(separator="\n", strip=True))
            elif isinstance(sibling, str) and sibling.strip():
                content_parts.append(sibling.strip())
        content = "\n".join(content_parts).strip()
        if title or content:
            sections.append((title, content))

    return sections


def _resolve_settings(settings: Settings | None) -> Settings:
    return settings if settings is not None else Settings()


def _save_artifact(
    work_id: str,
    kind: ArtifactKind,
    content: bytes | str,
    source_url: str,
    storage: StorageService,
    settings: Settings,
) -> Artifact:
    """Persist an artifact to disk and to the database."""
    ext = "html" if kind == ArtifactKind.HTML else "txt" if kind == ArtifactKind.PARSED_TEXT else "pdf"
    raw_dir = settings.data_dir / "artifacts" / "raw"
    parsed_dir = settings.data_dir / "artifacts" / "parsed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # Save raw artifact.
    raw_path = raw_dir / f"{work_id}.{ext}"
    if isinstance(content, bytes):
        raw_path.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
    else:
        raw_bytes = content.encode("utf-8")
        raw_path.write_bytes(raw_bytes)
        sha = hashlib.sha256(raw_bytes).hexdigest()

    artifact = Artifact(
        id=str(uuid.uuid4()),
        work_id=work_id,
        kind=kind,
        source_url=source_url,
        local_path=str(raw_path),
        sha256=sha,
        access_status=AccessStatus.AVAILABLE,
        created_at=datetime.utcnow(),
    )
    storage.save_artifact(artifact)
    return artifact


def _save_sections(
    work_id: str,
    artifact_id: str,
    sections: list[tuple[str, str]],
    storage: StorageService,
    settings: Settings,
) -> list[Section]:
    """Persist extracted sections to disk and database."""
    sections_dir = settings.data_dir / "artifacts" / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Section] = []
    for i, (title, text) in enumerate(sections):
        section_path = sections_dir / f"{work_id}_section_{i}.json"
        section_path.write_text(
            json.dumps({"title": title, "text": text}, ensure_ascii=False, indent=2)
        )

        sec = Section(
            id=str(uuid.uuid4()),
            work_id=work_id,
            artifact_id=artifact_id,
            section_title=title,
            section_order=i,
            text_content=text,
            created_at=datetime.utcnow(),
        )
        storage.save_section(sec)
        saved.append(sec)
    return saved


def _save_parsed_text(
    work_id: str,
    text: str,
    storage: StorageService,
    settings: Settings,
) -> Artifact:
    """Save the concatenated parsed text as a PARSED_TEXT artifact."""
    parsed_dir = settings.data_dir / "artifacts" / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    parsed_path = parsed_dir / f"{work_id}.txt"
    parsed_bytes = text.encode("utf-8")
    parsed_path.write_bytes(parsed_bytes)

    artifact = Artifact(
        id=str(uuid.uuid4()),
        work_id=work_id,
        kind=ArtifactKind.PARSED_TEXT,
        local_path=str(parsed_path),
        sha256=hashlib.sha256(parsed_bytes).hexdigest(),
        access_status=AccessStatus.AVAILABLE,
        created_at=datetime.utcnow(),
    )
    storage.save_artifact(artifact)
    return artifact


# ------------------------------------------------------------------
# Main cascade
# ------------------------------------------------------------------


async def acquire_fulltext(
    work: Work,
    storage: StorageService,
    settings: Settings | None = None,
) -> Work:
    """Attempt to acquire full text for *work* via a cascade of sources.

    Returns the updated ``Work`` object (also persisted via *storage*).
    """
    settings = _resolve_settings(settings)
    updated = work.updated_at = datetime.utcnow()

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        # ----------------------------------------------------------
        # 1. arXiv HTML
        # ----------------------------------------------------------
        if work.arxiv_id:
            html_url = f"https://arxiv.org/html/{work.arxiv_id}"
            html = await _fetch_html(html_url, client)
            if html:
                sections = _parse_html_sections(html)
                if sections:
                    # Save raw HTML artifact.
                    html_artifact = _save_artifact(
                        work.id, ArtifactKind.HTML, html, html_url, storage, settings
                    )
                    # Save parsed text.
                    full_text = "\n\n".join(content for _, content in sections)
                    _save_parsed_text(work.id, full_text, storage, settings)
                    # Save individual sections.
                    _save_sections(work.id, html_artifact.id, sections, storage, settings)

                    work.fulltext_available = True
                    work.access_status = AccessStatus.AVAILABLE
                    work.updated_at = datetime.utcnow()
                    storage.save_work(work)
                    return work

        # ----------------------------------------------------------
        # 2. arXiv PDF
        # ----------------------------------------------------------
        if work.arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{work.arxiv_id}"
            pdf_bytes = await _fetch_pdf(pdf_url, client)
            if pdf_bytes:
                text = _extract_pdf_text(pdf_bytes)
                if text.strip():
                    _save_artifact(
                        work.id, ArtifactKind.PDF, pdf_bytes, pdf_url, storage, settings
                    )
                    _save_parsed_text(work.id, text, storage, settings)

                    work.fulltext_available = True
                    work.access_status = AccessStatus.AVAILABLE
                    work.updated_at = datetime.utcnow()
                    storage.save_work(work)
                    return work

        # ----------------------------------------------------------
        # 3. OA / pdf_url from metadata
        # ----------------------------------------------------------
        if work.pdf_url:
            pdf_bytes = await _fetch_pdf(work.pdf_url, client)
            if pdf_bytes:
                text = _extract_pdf_text(pdf_bytes)
                if text.strip():
                    _save_artifact(
                        work.id, ArtifactKind.PDF, pdf_bytes, work.pdf_url, storage, settings
                    )
                    _save_parsed_text(work.id, text, storage, settings)

                    work.fulltext_available = True
                    work.access_status = AccessStatus.AVAILABLE
                    work.updated_at = datetime.utcnow()
                    storage.save_work(work)
                    return work

    # ----------------------------------------------------------
    # 4. Abstract-only fallback
    # ----------------------------------------------------------
    work.fulltext_available = False
    work.access_status = AccessStatus.ABSTRACT_ONLY
    work.updated_at = datetime.utcnow()
    storage.save_work(work)
    return work
