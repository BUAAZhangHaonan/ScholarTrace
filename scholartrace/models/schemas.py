from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceName(str, Enum):
    OPENALEX = "openalex"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    DBLP = "dblp"
    OPENREVIEW = "openreview"
    CROSSREF = "crossref"
    DEEPXIV = "deepxiv"


class ArtifactKind(str, Enum):
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    SOURCE_TAR = "source_tar"
    PARSED_TEXT = "parsed_text"


class AccessStatus(str, Enum):
    AVAILABLE = "available"
    ABSTRACT_ONLY = "abstract_only"
    PAYWALL = "paywall"
    UNKNOWN = "unknown"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AcquisitionState(str, Enum):
    MISSING = "missing"
    ACQUIRING = "acquiring"
    AVAILABLE = "available"
    NEGATIVE_CACHED = "negative_cached"


class Work(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    s2_id: str | None = None
    dblp_key: str | None = None
    openreview_id: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    relevance_score: float = 0.0
    recency_score: float = 0.0
    influence_score: float = 0.0
    venue_score: float = 0.0
    composite_score: float = 0.0
    agent_score: float = 0.0
    agent_rank: int | None = None
    agent_rationale: str | None = None
    fulltext_available: bool = False
    access_status: AccessStatus = AccessStatus.UNKNOWN
    source_provenance: list[str] = Field(default_factory=list)
    citation_count: int = 0
    reference_count: int = 0
    pdf_url: str | None = None
    html_url: str | None = None
    oa_url: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Artifact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    work_id: str = ""
    kind: ArtifactKind = ArtifactKind.PDF
    source_url: str | None = None
    local_path: str | None = None
    sha256: str | None = None
    license: str | None = None
    access_status: AccessStatus = AccessStatus.UNKNOWN
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Section(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    work_id: str = ""
    artifact_id: str = ""
    section_title: str = ""
    section_order: int = 0
    text_content: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Theme(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_text: str = ""
    parsed_topics: list[str] = Field(default_factory=list)
    parsed_methods: list[str] = Field(default_factory=list)
    parsed_datasets: list[str] = Field(default_factory=list)
    parsed_queries: list[str] = Field(default_factory=list)
    compressed_summary: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievalJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    theme_id: str = ""
    status: JobStatus = JobStatus.PENDING
    query_count: int = 0
    candidate_count: int = 0
    result_count: int = 0
    error_message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class FullTextState(BaseModel):
    work_id: str = ""
    acquisition_state: AcquisitionState = AcquisitionState.MISSING
    last_attempt_at: datetime | None = None
    next_retry_at: datetime | None = None
    error_message: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RawCandidate(BaseModel):
    """Intermediate model from source connectors, before dedup."""

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    s2_id: str | None = None
    dblp_key: str | None = None
    openreview_id: str | None = None
    source: SourceName = SourceName.OPENALEX
    citation_count: int = 0
    reference_count: int = 0
    fulltext_url: str | None = None
    pdf_url: str | None = None
    html_url: str | None = None
    oa_url: str | None = None
    license: str | None = None
    source_provenance: list[str] = Field(default_factory=list)
