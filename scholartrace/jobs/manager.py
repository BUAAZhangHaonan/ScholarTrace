"""Simple job manager wrapping StorageService for RetrievalJob lifecycle."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

from scholartrace.models.schemas import JobStatus, RetrievalJob
from scholartrace.services.storage import StorageService

logger = logging.getLogger(__name__)


class JobManager:
    """Manages RetrievalJob lifecycle: create, start, complete, fail."""

    def __init__(self, storage: StorageService) -> None:
        self.storage = storage

    def create_job(self, theme_id: str) -> RetrievalJob:
        job = RetrievalJob(theme_id=theme_id, status=JobStatus.PENDING)
        return self.storage.save_job(job)

    def create_or_get_active_job(
        self,
        theme_id: str,
        *,
        return_created: bool = False,
    ):
        active = self.storage.get_active_job_by_theme(theme_id)
        if active is not None:
            logger.info("Reusing active retrieval job %s for theme %s", active.id, theme_id)
            return (active, False) if return_created else active

        job = RetrievalJob(theme_id=theme_id, status=JobStatus.PENDING)
        try:
            saved = self.storage.save_job(job)
            return (saved, True) if return_created else saved
        except sqlite3.IntegrityError:
            active = self.storage.get_active_job_by_theme(theme_id)
            if active is None:
                raise
            logger.info("Collapsed duplicate retrieval job creation onto %s", active.id)
            return (active, False) if return_created else active

    def start_job(self, job_id: str) -> None:
        self.storage.update_job_status(job_id, JobStatus.RUNNING)

    def complete_job(self, job_id: str, result_count: int) -> None:
        self.storage.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            result_count=result_count,
            completed_at=datetime.utcnow(),
        )

    def fail_job(self, job_id: str, error_message: str) -> None:
        self.storage.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=error_message,
        )

    def get_job(self, job_id: str) -> RetrievalJob | None:
        return self.storage.get_job(job_id)

    def get_active_job(self, theme_id: str) -> RetrievalJob | None:
        return self.storage.get_active_job_by_theme(theme_id)
