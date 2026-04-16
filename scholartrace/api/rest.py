"""FastAPI REST API for ScholarTrace."""

from __future__ import annotations

import asyncio
import logging

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Query
from fastapi.responses import PlainTextResponse

from scholartrace.config import Settings, get_settings
from scholartrace.jobs.manager import JobManager
from scholartrace.models.schemas import RetrievalJob, Theme, Work
from scholartrace.services.storage import StorageService
from scholartrace.services.theme_parser import parse_theme

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------
app = FastAPI(title="ScholarTrace", version="0.1.0")

# Module-level singletons – initialised lazily on first request.
_storage: StorageService | None = None
_settings: Settings | None = None


def _get_storage() -> StorageService:
    global _storage, _settings
    if _storage is None:
        _settings = get_settings()
        _settings.data_dir.mkdir(parents=True, exist_ok=True)
        _storage = StorageService(_settings.db_path)
        _storage.init_db()
    return _storage


def _get_settings() -> Settings:
    _get_storage()  # ensures _settings is populated
    return _settings  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
@app.post("/themes", response_model=Theme)
def create_theme(text: str = Form(...)) -> Theme:
    storage = _get_storage()
    theme = parse_theme(text)
    storage.save_theme(theme)
    return theme


# ---------------------------------------------------------------------------
# Retrieval jobs
# ---------------------------------------------------------------------------
@app.post("/retrieval/jobs", response_model=RetrievalJob)
def create_retrieval_job(
    background_tasks: BackgroundTasks,
    theme_id: str = Form(...),
) -> RetrievalJob:
    storage = _get_storage()
    settings = _get_settings()

    theme = storage.get_theme(theme_id)
    if theme is None:
        raise HTTPException(status_code=404, detail="Theme not found")

    job_manager = JobManager(storage)
    job = job_manager.create_job(theme_id)

    # Import here to avoid heavy module load at import time.
    from scholartrace.services.retrieval import run_retrieval

    async def _run_retrieval_background(theme_id: str, job_id: str) -> None:
        storage = _get_storage()
        job_manager = JobManager(storage)
        theme = storage.get_theme(theme_id)
        if theme is None:
            job_manager.fail_job(job_id, "Theme disappeared")
            return
        try:
            job_manager.start_job(job_id)
            works = await run_retrieval(theme, storage, settings)
            job_manager.complete_job(job_id, len(works))
        except Exception as exc:
            logger.exception("Background retrieval failed for job %s", job_id)
            job_manager.fail_job(job_id, str(exc))

    # FastAPI BackgroundTasks only supports sync callables, so we
    # wrap the async function with asyncio.run via a small helper.
    def _sync_wrapper() -> None:
        asyncio.run(_run_retrieval_background(theme_id, job.id))

    background_tasks.add_task(_sync_wrapper)

    return job


@app.get("/retrieval/jobs/{job_id}", response_model=RetrievalJob)
def get_job_status(job_id: str) -> RetrievalJob:
    storage = _get_storage()
    job = storage.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------
@app.get("/themes/{theme_id}/papers", response_model=list[Work])
def list_papers(
    theme_id: str,
    limit: int = Query(default=50, ge=1),
    offset: int = Query(default=0, ge=0),
) -> list[Work]:
    storage = _get_storage()
    theme = storage.get_theme(theme_id)
    if theme is None:
        raise HTTPException(status_code=404, detail="Theme not found")
    return storage.list_works_by_theme(theme_id, limit, offset)


@app.get("/papers/{paper_id}", response_model=Work)
def get_paper(paper_id: str) -> Work:
    storage = _get_storage()
    work = storage.get_work(paper_id)
    if work is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return work


@app.get("/papers/{paper_id}/sections")
def get_sections(paper_id: str) -> list[dict]:
    storage = _get_storage()
    work = storage.get_work(paper_id)
    if work is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    sections = storage.get_sections_by_work(paper_id)
    return [s.model_dump() for s in sections]


@app.get("/papers/{paper_id}/fulltext")
def get_fulltext(paper_id: str) -> dict:
    storage = _get_storage()
    work = storage.get_work(paper_id)
    if work is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    artifacts = storage.get_artifacts_by_work(paper_id)
    sections = storage.get_sections_by_work(paper_id)

    artifact_list = []
    for a in artifacts:
        artifact_list.append(
            {
                "id": a.id,
                "kind": a.kind.value,
                "source_url": a.source_url,
                "local_path": a.local_path,
                "access_status": a.access_status.value,
            }
        )

    section_list = []
    for s in sections:
        section_list.append(
            {
                "id": s.id,
                "section_title": s.section_title,
                "section_order": s.section_order,
                "text_content": s.text_content,
            }
        )

    return {
        "work_id": work.id,
        "fulltext_available": work.fulltext_available,
        "access_status": work.access_status.value,
        "sections": section_list,
        "artifacts": artifact_list,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
@app.get("/themes/{theme_id}/export", response_model=None)
def export_theme(
    theme_id: str,
    format: str = Query(default="json"),
) -> dict | PlainTextResponse:
    storage = _get_storage()
    theme = storage.get_theme(theme_id)
    if theme is None:
        raise HTTPException(status_code=404, detail="Theme not found")

    works = storage.list_works_by_theme(theme_id, limit=10000, offset=0)

    if format == "json":
        return {
            "theme": theme.model_dump(),
            "papers": [w.model_dump() for w in works],
        }

    if format == "markdown":
        theme_excerpt = theme.document_text[:
                                            80] if theme.document_text else theme.id
        lines: list[str] = []
        lines.append(f"# ScholarTrace Report: {theme_excerpt}")
        lines.append("")
        lines.append(f"## Papers ({len(works)} total)")
        lines.append("")
        for idx, w in enumerate(works, start=1):
            lines.append(f"### {idx}. {w.title or 'Untitled'}")
            lines.append(
                f"- Authors: {', '.join(w.authors) if w.authors else 'N/A'}")
            lines.append(f"- Year: {w.year or 'N/A'}")
            lines.append(f"- Venue: {w.venue or 'N/A'}")
            lines.append(f"- Score: {w.composite_score:.4f}")
            if w.abstract:
                lines.append(f"- Abstract: {w.abstract}")
            lines.append("---")
            lines.append("")
        return PlainTextResponse("\n".join(lines), media_type="text/markdown")

    raise HTTPException(
        status_code=400,
        detail="Unsupported format. Use 'json' or 'markdown'.",
    )
