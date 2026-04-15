"""Simple job manager wrapping StorageService for RetrievalJob lifecycle."""

from __future__ import annotations

from datetime import datetime

from scholartrace.models.schemas import JobStatus, RetrievalJob
from scholartrace.services.storage import StorageService


class JobManager:
    """Manages RetrievalJob lifecycle: create, start, complete, fail."""

    def __init__(self, storage: StorageService) -> None:
        self.storage = storage

    def create_job(self, theme_id: str) -> RetrievalJob:
        job = RetrievalJob(theme_id=theme_id, status=JobStatus.PENDING)
        return self.storage.save_job(job)

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
