"""Full-text acquisition service with explicit cache and acquire semantics."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import socket
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import fitz
import httpx
from bs4 import BeautifulSoup

from scholartrace.config import Settings
from scholartrace.connectors.deepxiv_connector import DeepXivConnector
from scholartrace.models.schemas import (
    AccessStatus,
    AcquisitionState,
    Artifact,
    ArtifactKind,
    FullTextState,
    Section,
    Work,
)
from scholartrace.services.storage import StorageService

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30.0
_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
_MAX_REDIRECTS = 3
_NEGATIVE_CACHE_TTL = timedelta(minutes=15)
_ACQUIRE_SEMAPHORE = asyncio.Semaphore(4)
_INFLIGHT_ACQUIRES: dict[str, asyncio.Task[Work]] = {}
_INFLIGHT_LOCK = asyncio.Lock()


class FullTextFetchRejected(Exception):
    """Raised when a target URL or response is unsafe for acquisition."""


def _utcnow() -> datetime:
    return datetime.utcnow()


def _resolve_settings(settings: Settings | None) -> Settings:
    return settings if settings is not None else Settings()


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _is_public_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return ip.is_global


def _resolve_host_addresses(host: str) -> set[str]:
    try:
        return {
            result[4][0]
            for result in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as exc:
        raise FullTextFetchRejected("unresolvable host") from exc


def _validate_outbound_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise FullTextFetchRejected("unsupported scheme")
    if not parts.hostname:
        raise FullTextFetchRejected("missing host")
    host = parts.hostname
    if host.lower() == "localhost":
        raise FullTextFetchRejected("loopback host")
    try:
        ip = ipaddress.ip_address(host)
        if not ip.is_global:
            raise FullTextFetchRejected("non-public address")
    except ValueError:
        addresses = _resolve_host_addresses(host)
        if not addresses or any(not _is_public_ip(address) for address in addresses):
            raise FullTextFetchRejected("non-public address")
    return url


async def _download_response(
    url: str,
    client: httpx.AsyncClient,
) -> tuple[object, str]:
    current_url = url
    for _ in range(_MAX_REDIRECTS + 1):
        safe_url = _validate_outbound_url(current_url)
        response = await client.get(safe_url, follow_redirects=False)
        if 300 <= response.status_code < 400:
            location = (getattr(response, "headers", {}) or {}).get("location")
            if not location:
                raise FullTextFetchRejected("redirect missing location")
            current_url = urljoin(safe_url, location)
            continue
        response.raise_for_status()
        content = getattr(response, "content", b"")
        if len(content) > _MAX_DOWNLOAD_BYTES:
            raise FullTextFetchRejected("response too large")
        return response, safe_url
    raise FullTextFetchRejected("too many redirects")


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
    """Parse HTML into ``(title, content)`` pairs by scanning heading tags."""
    soup = BeautifulSoup(html, "html.parser")
    headings = soup.find_all(["h1", "h2", "h3"])
    sections: list[tuple[str, str]] = []

    if not headings:
        full_text = soup.get_text(separator="\n", strip=True)
        if full_text.strip():
            sections.append(("Full Text", full_text))
        return sections

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


def _parse_markdown_sections(markdown: str) -> list[tuple[str, str]]:
    """Parse markdown headings into ``(title, content)`` pairs."""
    sections: list[tuple[str, str]] = []
    preamble: list[str] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def _flush_current() -> None:
        nonlocal current_title, current_lines
        if current_title is None:
            return
        sections.append((current_title, "\n".join(current_lines).strip()))
        current_title = None
        current_lines = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("#"):
            _flush_current()
            current_title = line.lstrip("#").strip() or "Section"
            current_lines = []
            continue
        if current_title is None:
            if line.strip():
                preamble.append(line)
            continue
        current_lines.append(line)

    if preamble:
        sections.append(("Preamble", "\n".join(preamble).strip()))
    _flush_current()

    if sections:
        return sections

    full_text = markdown.strip()
    if full_text:
        return [("Full Text", full_text)]
    return []


def _save_artifact(
    work_id: str,
    kind: ArtifactKind,
    content: bytes | str,
    source_url: str,
    storage: StorageService,
    settings: Settings,
    *,
    conn,
) -> Artifact:
    if kind == ArtifactKind.HTML:
        ext = "html"
    elif kind in {ArtifactKind.PARSED_TEXT, ArtifactKind.MARKDOWN}:
        ext = "txt"
    else:
        ext = "pdf"
    raw_dir = settings.data_dir / "artifacts" / "raw"
    parsed_dir = settings.data_dir / "artifacts" / "parsed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{work_id}_{kind.value}.{ext}"
    raw_bytes = content.encode("utf-8") if isinstance(content, str) else content
    raw_path.write_bytes(raw_bytes)
    artifact = Artifact(
        id=str(uuid.uuid4()),
        work_id=work_id,
        kind=kind,
        source_url=source_url,
        local_path=str(raw_path),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        access_status=AccessStatus.AVAILABLE,
        created_at=_utcnow(),
    )
    storage.save_artifact(artifact, conn=conn)
    return artifact


def _save_sections(
    work_id: str,
    artifact_id: str,
    sections: list[tuple[str, str]],
    storage: StorageService,
    settings: Settings,
    *,
    conn,
) -> list[Section]:
    sections_dir = settings.data_dir / "artifacts" / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Section] = []
    for index, (title, text) in enumerate(sections):
        section_path = sections_dir / f"{work_id}_section_{index}.json"
        section_path.write_text(
            json.dumps({"title": title, "text": text}, ensure_ascii=False, indent=2)
        )
        section = Section(
            id=str(uuid.uuid4()),
            work_id=work_id,
            artifact_id=artifact_id,
            section_title=title,
            section_order=index,
            text_content=text,
            created_at=_utcnow(),
        )
        storage.save_section(section, conn=conn)
        saved.append(section)
    return saved


def _save_parsed_text(
    work_id: str,
    text: str,
    storage: StorageService,
    settings: Settings,
    *,
    conn,
) -> Artifact:
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
        created_at=_utcnow(),
    )
    storage.save_artifact(artifact, conn=conn)
    return artifact


def _make_fulltext_state(
    work_id: str,
    acquisition_state: AcquisitionState,
    *,
    last_attempt_at: datetime | None = None,
    next_retry_at: datetime | None = None,
    error_message: str | None = None,
) -> FullTextState:
    return FullTextState(
        work_id=work_id,
        acquisition_state=acquisition_state,
        last_attempt_at=last_attempt_at,
        next_retry_at=next_retry_at,
        error_message=error_message,
        updated_at=_utcnow(),
    )


def _persist_negative_cache(
    work: Work,
    storage: StorageService,
    error_message: str,
) -> Work:
    now = _utcnow()
    work.fulltext_available = False
    work.access_status = AccessStatus.ABSTRACT_ONLY
    work.updated_at = now
    with storage.transaction(immediate=True) as conn:
        saved_work = storage.save_work(work, conn=conn)
        storage.save_fulltext_state(
            _make_fulltext_state(
                work.id,
                AcquisitionState.NEGATIVE_CACHED,
                last_attempt_at=now,
                next_retry_at=now + _NEGATIVE_CACHE_TTL,
                error_message=error_message,
            ),
            conn=conn,
        )
        return saved_work


def _persist_available_text(
    work: Work,
    storage: StorageService,
    settings: Settings,
    *,
    kind: ArtifactKind,
    content: bytes | str,
    source_url: str,
    sections: list[tuple[str, str]] | None = None,
    parsed_text: str,
) -> Work:
    now = _utcnow()
    work.fulltext_available = True
    work.access_status = AccessStatus.AVAILABLE
    work.updated_at = now
    with storage.transaction(immediate=True) as conn:
        artifact = _save_artifact(
            work.id,
            kind,
            content,
            source_url,
            storage,
            settings,
            conn=conn,
        )
        _save_parsed_text(work.id, parsed_text, storage, settings, conn=conn)
        if sections:
            _save_sections(work.id, artifact.id, sections, storage, settings, conn=conn)
        saved_work = storage.save_work(work, conn=conn)
        storage.save_fulltext_state(
            _make_fulltext_state(
                work.id,
                AcquisitionState.AVAILABLE,
                last_attempt_at=now,
                next_retry_at=None,
                error_message=None,
            ),
            conn=conn,
        )
        return saved_work


def _has_usable_deepxiv_tokens(settings: Settings) -> bool:
    return any(token.strip() for token in settings.deepxiv_tokens.split(","))


def _deepxiv_acquisition_enabled(settings: Settings) -> bool:
    return _has_usable_deepxiv_tokens(settings) or (
        settings.deepxiv_auto_register
        and bool(settings.deepxiv_register_sdk_secret.strip())
    )


def _load_parsed_text(work_id: str, storage: StorageService) -> str | None:
    artifacts = storage.get_artifacts_by_work(work_id)
    for artifact in artifacts:
        if artifact.kind != ArtifactKind.PARSED_TEXT or not artifact.local_path:
            continue
        path = Path(artifact.local_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


def _artifact_payload(artifact: Artifact) -> dict:
    return {
        "id": artifact.id,
        "kind": artifact.kind.value,
        "source_url": artifact.source_url,
        "access_status": artifact.access_status.value,
    }


def _section_payload(section: Section) -> dict:
    return {
        "id": section.id,
        "section_title": section.section_title,
        "section_order": section.section_order,
        "text_content": section.text_content,
    }


def read_cached_fulltext(
    work: Work,
    storage: StorageService,
    settings: Settings | None = None,
) -> dict:
    settings = _resolve_settings(settings)
    current = storage.get_work(work.id) or work
    state = storage.get_fulltext_state(current.id)
    if state is None:
        acquisition_state = (
            AcquisitionState.AVAILABLE if current.fulltext_available else AcquisitionState.MISSING
        )
        state = _make_fulltext_state(current.id, acquisition_state)
    artifacts = storage.get_artifacts_by_work(current.id)
    sections = storage.get_sections_by_work(current.id)
    parsed_text = _load_parsed_text(current.id, storage)
    now = _utcnow()
    needs_acquisition = (
        not current.fulltext_available
        and state.acquisition_state != AcquisitionState.ACQUIRING
        and (state.next_retry_at is None or state.next_retry_at <= now)
    )
    return {
        "paper_id": current.id,
        "title": current.title,
        "fulltext_available": current.fulltext_available,
        "access_status": current.access_status.value,
        "acquisition_state": state.acquisition_state.value,
        "needs_acquisition": needs_acquisition,
        "last_attempt_at": state.last_attempt_at.isoformat() if state.last_attempt_at else None,
        "next_retry_at": state.next_retry_at.isoformat() if state.next_retry_at else None,
        "error_message": state.error_message,
        "artifacts": [_artifact_payload(artifact) for artifact in artifacts],
        "sections": [_section_payload(section) for section in sections],
        "parsed_text": parsed_text,
    }


async def _attempt_html_acquisition(
    work: Work,
    storage: StorageService,
    settings: Settings,
    client: httpx.AsyncClient,
) -> Work | None:
    if not work.arxiv_id:
        return None
    html_url = f"https://arxiv.org/html/{work.arxiv_id}"
    response, safe_url = await _download_response(html_url, client)
    html = response.text
    sections = _parse_html_sections(html)
    if not sections:
        return None
    full_text = "\n\n".join(content for _, content in sections)
    return _persist_available_text(
        work,
        storage,
        settings,
        kind=ArtifactKind.HTML,
        content=html,
        source_url=safe_url,
        sections=sections,
        parsed_text=full_text,
    )


async def _attempt_pdf_acquisition(
    work: Work,
    storage: StorageService,
    settings: Settings,
    client: httpx.AsyncClient,
    url: str,
) -> Work | None:
    response, safe_url = await _download_response(url, client)
    pdf_bytes = response.content
    text = _extract_pdf_text(pdf_bytes)
    if not text.strip():
        return None
    return _persist_available_text(
        work,
        storage,
        settings,
        kind=ArtifactKind.PDF,
        content=pdf_bytes,
        source_url=safe_url,
        parsed_text=text,
    )


async def _attempt_metadata_url_acquisition(
    work: Work,
    storage: StorageService,
    settings: Settings,
    client: httpx.AsyncClient,
    url: str,
) -> Work | None:
    response, safe_url = await _download_response(url, client)
    content = response.content
    content_type = ((getattr(response, "headers", {}) or {}).get("content-type") or "").lower()

    if (
        "pdf" in content_type
        or safe_url.lower().endswith(".pdf")
        or content.startswith(b"%PDF")
    ):
        text = _extract_pdf_text(content)
        if not text.strip():
            return None
        return _persist_available_text(
            work,
            storage,
            settings,
            kind=ArtifactKind.PDF,
            content=content,
            source_url=safe_url,
            parsed_text=text,
        )

    html = response.text
    sections = _parse_html_sections(html)
    if not sections:
        return None
    full_text = "\n\n".join(content for _, content in sections)
    return _persist_available_text(
        work,
        storage,
        settings,
        kind=ArtifactKind.HTML,
        content=html,
        source_url=safe_url,
        sections=sections,
        parsed_text=full_text,
    )


async def _attempt_deepxiv_markdown_acquisition(
    work: Work,
    storage: StorageService,
    settings: Settings,
) -> Work | None:
    if not work.arxiv_id or not _deepxiv_acquisition_enabled(settings):
        return None

    connector = DeepXivConnector(settings=settings)
    try:
        markdown = await connector.get_fulltext(work.arxiv_id)
    finally:
        await connector.close()

    if not markdown or not markdown.strip():
        return None

    sections = _parse_markdown_sections(markdown)
    parsed_text = "\n\n".join(
        content for _, content in sections if content.strip()
    ).strip() or markdown.strip()
    return _persist_available_text(
        work,
        storage,
        settings,
        kind=ArtifactKind.MARKDOWN,
        content=markdown,
        source_url=f"https://data.rag.ac.cn/api/arxiv/{work.arxiv_id}/raw",
        sections=sections,
        parsed_text=parsed_text,
    )


async def _acquire_fulltext_inner(
    work: Work,
    storage: StorageService,
    settings: Settings,
) -> Work:
    current = storage.get_work(work.id) or work
    state = storage.get_fulltext_state(current.id)
    now = _utcnow()

    if current.fulltext_available:
        if state is None or state.acquisition_state != AcquisitionState.AVAILABLE:
            with storage.transaction(immediate=True) as conn:
                storage.save_fulltext_state(
                    _make_fulltext_state(
                        current.id,
                        AcquisitionState.AVAILABLE,
                        last_attempt_at=now,
                    ),
                    conn=conn,
                )
        return current

    if (
        state is not None
        and state.acquisition_state == AcquisitionState.NEGATIVE_CACHED
        and state.next_retry_at is not None
        and state.next_retry_at > now
    ):
        logger.info(
            "Negative-cache hit for full-text acquisition on %s until %s",
            current.id,
            state.next_retry_at.isoformat(),
        )
        return current

    with storage.transaction(immediate=True) as conn:
        storage.save_fulltext_state(
            _make_fulltext_state(
                current.id,
                AcquisitionState.ACQUIRING,
                last_attempt_at=now,
            ),
            conn=conn,
        )

    async with _ACQUIRE_SEMAPHORE:
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                if current.arxiv_id:
                    try:
                        result = await _attempt_html_acquisition(current, storage, settings, client)
                        if result is not None:
                            return result
                    except FullTextFetchRejected as exc:
                        logger.warning(
                            "Blocked unsafe full-text fetch for %s via %s: %s",
                            current.id,
                            _redact_url(f"https://arxiv.org/html/{current.arxiv_id}"),
                            exc,
                        )
                    except Exception:
                        logger.debug("HTML acquisition failed for %s", current.id, exc_info=True)

                    try:
                        result = await _attempt_pdf_acquisition(
                            current,
                            storage,
                            settings,
                            client,
                            f"https://arxiv.org/pdf/{current.arxiv_id}",
                        )
                        if result is not None:
                            return result
                    except FullTextFetchRejected as exc:
                        logger.warning(
                            "Blocked unsafe full-text fetch for %s via %s: %s",
                            current.id,
                            _redact_url(f"https://arxiv.org/pdf/{current.arxiv_id}"),
                            exc,
                        )
                    except Exception:
                        logger.debug("arXiv PDF acquisition failed for %s", current.id, exc_info=True)

                if current.pdf_url:
                    try:
                        result = await _attempt_metadata_url_acquisition(
                            current,
                            storage,
                            settings,
                            client,
                            current.pdf_url,
                        )
                        if result is not None:
                            return result
                    except FullTextFetchRejected as exc:
                        logger.warning(
                            "Blocked unsafe full-text fetch for %s via %s: %s",
                            current.id,
                            _redact_url(current.pdf_url),
                            exc,
                        )
                    except Exception:
                        logger.debug("Metadata PDF acquisition failed for %s", current.id, exc_info=True)

                for metadata_url in (current.oa_url, current.html_url):
                    if not metadata_url or metadata_url == current.pdf_url:
                        continue
                    try:
                        result = await _attempt_metadata_url_acquisition(
                            current,
                            storage,
                            settings,
                            client,
                            metadata_url,
                        )
                        if result is not None:
                            return result
                    except FullTextFetchRejected as exc:
                        logger.warning(
                            "Blocked unsafe full-text fetch for %s via %s: %s",
                            current.id,
                            _redact_url(metadata_url),
                            exc,
                        )
                    except Exception:
                        logger.debug(
                            "Metadata URL acquisition failed for %s",
                            current.id,
                            exc_info=True,
                        )

                if current.arxiv_id:
                    try:
                        result = await _attempt_deepxiv_markdown_acquisition(
                            current,
                            storage,
                            settings,
                        )
                        if result is not None:
                            return result
                    except Exception:
                        logger.debug(
                            "DeepXiv markdown acquisition failed for %s",
                            current.id,
                            exc_info=True,
                        )
        except Exception:
            logger.warning("Full-text acquisition failed for %s", current.id, exc_info=True)

    logger.warning("Negative-caching full-text miss for %s", current.id)
    return _persist_negative_cache(current, storage, "Full-text source unavailable")


async def acquire_fulltext(
    work: Work,
    storage: StorageService,
    settings: Settings | None = None,
) -> Work:
    """Explicitly acquire full text for *work* when cache state allows it."""
    resolved_settings = _resolve_settings(settings)
    async with _INFLIGHT_LOCK:
        task = _INFLIGHT_ACQUIRES.get(work.id)
        if task is None:
            task = asyncio.create_task(_acquire_fulltext_inner(work, storage, resolved_settings))
            _INFLIGHT_ACQUIRES[work.id] = task
        else:
            logger.info("Collapsing duplicate full-text acquisition for %s", work.id)
    try:
        return await task
    finally:
        async with _INFLIGHT_LOCK:
            if _INFLIGHT_ACQUIRES.get(work.id) is task and task.done():
                _INFLIGHT_ACQUIRES.pop(work.id, None)
