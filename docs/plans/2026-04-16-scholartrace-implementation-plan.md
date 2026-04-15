# ScholarTrace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a theme-guided multi-source scholarly retrieval system that retrieves 100+ papers, ranks them, acquires full text when available, and exposes REST API + MCP server interfaces.

**Architecture:** Python monolith with FastAPI REST + MCP server sharing a core service layer. SQLite for metadata, filesystem for artifacts. Async HTTP for all source connectors. Multi-source fan-out retrieval with multi-key dedup and weighted multi-objective ranking.

**Tech Stack:** Python 3.13, FastAPI, uvicorn, httpx, Pydantic, mcp SDK, SQLite, PyMuPDF, beautifulsoup4, scikit-learn, rapidfuzz

---

## Task 1: Project Scaffolding and Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `scholartrace/__init__.py`
- Create: `scholartrace/config.py`
- Create: `scholartrace/main.py`
- Create: `data/.gitkeep`
- Create: `data/artifacts/raw/.gitkeep`
- Create: `data/artifacts/parsed/.gitkeep`
- Create: `data/artifacts/sections/.gitkeep`

**Step 1: Create project structure**

```bash
cd /home/g203/zhanghaonan/ScholarTrace
mkdir -p scholartrace/{models,services,connectors,api,jobs} tests data/artifacts/{raw,parsed,sections}
touch scholartrace/__init__.py scholartrace/models/__init__.py scholartrace/services/__init__.py scholartrace/connectors/__init__.py scholartrace/api/__init__.py scholartrace/jobs/__init__.py tests/__init__.py
touch data/.gitkeep data/artifacts/raw/.gitkeep data/artifacts/parsed/.gitkeep data/artifacts/sections/.gitkeep
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "scholartrace"
version = "0.1.0"
description = "Theme-guided multi-source paper discovery and full-text evidence access"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "httpx>=0.28.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
    "mcp>=1.6.0",
    "aiofiles>=24.1.0",
    "PyMuPDF>=1.25.0",
    "beautifulsoup4>=4.12.0",
    "scikit-learn>=1.6.0",
    "rapidfuzz>=3.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.25.0",
    "pytest-httpx>=0.35.0",
    "httpx",
]

[project.scripts]
scholartrace-api = "scholartrace.main:run_api"
scholartrace-mcp = "scholartrace.main:run_mcp"
```

**Step 3: Create config.py**

```python
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Storage
    data_dir: Path = Path("data")
    db_path: Path = Path("data/scholartrace.db")

    # Source API keys (optional)
    semantic_scholar_api_key: str = ""
    openalex_mailto: str = ""
    crossref_mailto: str = ""
    openreview_username: str = ""
    openreview_password: str = ""

    # Retrieval
    max_results_per_source_per_query: int = 200
    target_candidate_pool: int = 500
    max_fulltext_downloads: int = 50

    # Ranking weights
    weight_relevance: float = 0.35
    weight_recency: float = 0.20
    weight_influence: float = 0.20
    weight_venue: float = 0.10
    weight_fulltext: float = 0.10
    weight_source_agreement: float = 0.05

    # Server
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8001

    model_config = {"env_prefix": "SCHOLARTRACE_", "env_file": ".env"}

settings = Settings()
```

**Step 4: Create minimal main.py**

```python
"""ScholarTrace: Theme-guided multi-source paper discovery."""

def run_api():
    import uvicorn
    uvicorn.run("scholartrace.api.rest:app", host="127.0.0.1", port=8000, reload=False)

def run_mcp():
    from scholartrace.api.mcp_server import mcp
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_api()
```

**Step 5: Install dependencies**

```bash
cd /home/g203/zhanghaonan/ScholarTrace
pip install -e ".[dev]"
```

**Step 6: Verify installation**

```bash
python -c "import fastapi; import httpx; import mcp; print('All deps OK')"
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with dependencies"
```

---

## Task 2: Canonical Schema (Pydantic Models + SQLite)

**Files:**
- Create: `scholartrace/models/schemas.py`
- Create: `scholartrace/services/storage.py`

**Step 1: Create schemas.py with all Pydantic models**

```python
"""Canonical data models for ScholarTrace."""
from __future__ import annotations
import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class SourceName(str, Enum):
    OPENALEX = "openalex"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    DBLP = "dblp"
    OPENREVIEW = "openreview"
    CROSSREF = "crossref"

class ArtifactKind(str, Enum):
    PDF = "pdf"
    HTML = "html"
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
    fulltext_available: bool = False
    access_status: AccessStatus = AccessStatus.UNKNOWN
    source_provenance: list[str] = Field(default_factory=list)
    citation_count: int = 0
    reference_count: int = 0
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
```

**Step 2: Create storage.py with SQLite schema init and CRUD**

Implement `StorageService` class with:
- `init_db()` — creates tables if not exist using raw SQL
- `save_work(work: Work) -> Work`
- `get_work(work_id: str) -> Work | None`
- `get_work_by_doi(doi: str) -> Work | None` (and similar for arxiv_id, etc.)
- `list_works_by_theme(theme_id: str, limit, offset) -> list[Work]`
- `count_works_by_theme(theme_id: str) -> int`
- `save_artifact(artifact: Artifact) -> Artifact`
- `get_artifacts_by_work(work_id: str) -> list[Artifact]`
- `save_section(section: Section) -> Section`
- `get_sections_by_work(work_id: str) -> list[Section]`
- `save_theme(theme: Theme) -> Theme`
- `get_theme(theme_id: str) -> Theme | None`
- `save_job(job: RetrievalJob) -> RetrievalJob`
- `get_job(job_id: str) -> RetrievalJob | None`
- `update_job_status(job_id: str, status, **kwargs)`

Also create a `theme_works` junction table linking themes to works with ranking info.

SQL schema:
```sql
CREATE TABLE IF NOT EXISTS works (
    id TEXT PRIMARY KEY,
    doi TEXT, arxiv_id TEXT, openalex_id TEXT, s2_id TEXT, dblp_key TEXT, openreview_id TEXT,
    title TEXT NOT NULL, authors TEXT, year INTEGER, venue TEXT, abstract TEXT,
    relevance_score REAL DEFAULT 0, recency_score REAL DEFAULT 0, influence_score REAL DEFAULT 0,
    venue_score REAL DEFAULT 0, composite_score REAL DEFAULT 0,
    fulltext_available INTEGER DEFAULT 0, access_status TEXT DEFAULT 'unknown',
    source_provenance TEXT, citation_count INTEGER DEFAULT 0, reference_count INTEGER DEFAULT 0,
    created_at TEXT, updated_at TEXT,
    UNIQUE(doi), UNIQUE(arxiv_id), UNIQUE(openalex_id), UNIQUE(s2_id), UNIQUE(dblp_key)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY, work_id TEXT NOT NULL,
    kind TEXT, source_url TEXT, local_path TEXT, sha256 TEXT,
    license TEXT, access_status TEXT DEFAULT 'unknown', created_at TEXT,
    FOREIGN KEY (work_id) REFERENCES works(id)
);

CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY, work_id TEXT NOT NULL, artifact_id TEXT,
    section_title TEXT, section_order INTEGER, text_content TEXT, created_at TEXT,
    FOREIGN KEY (work_id) REFERENCES works(id)
);

CREATE TABLE IF NOT EXISTS themes (
    id TEXT PRIMARY KEY, document_text TEXT,
    parsed_topics TEXT, parsed_methods TEXT, parsed_datasets TEXT, parsed_queries TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS theme_works (
    theme_id TEXT NOT NULL, work_id TEXT NOT NULL, rank_order INTEGER,
    PRIMARY KEY (theme_id, work_id),
    FOREIGN KEY (theme_id) REFERENCES themes(id),
    FOREIGN KEY (work_id) REFERENCES works(id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, theme_id TEXT NOT NULL, status TEXT DEFAULT 'pending',
    query_count INTEGER DEFAULT 0, candidate_count INTEGER DEFAULT 0, result_count INTEGER DEFAULT 0,
    error_message TEXT, created_at TEXT, updated_at TEXT, completed_at TEXT,
    FOREIGN KEY (theme_id) REFERENCES themes(id)
);
```

**Step 3: Test storage service**

```python
# tests/test_storage.py
import tempfile, os
from scholartrace.services.storage import StorageService
from scholartrace.models.schemas import Work, Theme

def test_init_db_creates_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = StorageService(db_path=db_path)
        store.init_db()
        # Insert and retrieve a work
        work = Work(title="Test Paper", doi="10.1234/test")
        saved = store.save_work(work)
        retrieved = store.get_work(saved.id)
        assert retrieved is not None
        assert retrieved.title == "Test Paper"
        assert retrieved.doi == "10.1234/test"

def test_get_work_by_doi():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = StorageService(db_path=db_path)
        store.init_db()
        work = Work(title="Test", doi="10.1234/test2")
        store.save_work(work)
        found = store.get_work_by_doi("10.1234/test2")
        assert found is not None
        assert found.title == "Test"
```

**Step 4: Run tests**

```bash
pytest tests/test_storage.py -v
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: canonical schema models and SQLite storage service"
```

---

## Task 3: Theme Parser

**Files:**
- Create: `scholartrace/services/theme_parser.py`
- Create: `tests/test_theme_parser.py`

**Step 1: Implement theme_parser.py**

The theme parser takes a document string and extracts structured query formulations. It uses keyword extraction (TF-IDF or simple NLP) to identify core topics, methods, datasets, then generates 6-8 different query strings optimized for different retrieval strategies.

Key functions:
- `parse_theme(document_text: str) -> Theme` — extracts topics/methods/datasets, generates queries
- `_extract_key_phrases(text: str, n: int) -> list[str]` — extracts top N key phrases using word frequency after stopword removal
- `_generate_queries(topics, methods, datasets) -> list[str]` — generates query formulations

Query generation strategies:
1. Core topic query: top 2-3 topics combined with AND
2. Broad recall query: all topics OR'd
3. Recent query: core topic + year filter hint
4. Method query: method names combined
5. Domain + topic query: domain keyword + core topic
6. Citation seed queries: if specific paper titles are mentioned

**Step 2: Test theme parser**

```python
# tests/test_theme_parser.py
from scholartrace.services.theme_parser import parse_theme

def test_parse_theme_extracts_topics():
    text = "RLHF sycophancy in language models: when reward models amplify agreeable behavior"
    theme = parse_theme(text)
    assert len(theme.parsed_topics) > 0
    assert len(theme.parsed_queries) >= 5

def test_parse_theme_from_research_brief():
    # Use the actual research brief
    with open("docs/examples/sycophancy_affective_hallucination_research_brief.md") as f:
        text = f.read()
    theme = parse_theme(text)
    assert len(theme.parsed_queries) >= 5
    assert any("sycophancy" in q.lower() for q in theme.parsed_queries)
```

**Step 3: Run tests**

```bash
pytest tests/test_theme_parser.py -v
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: theme parser with keyword extraction and query generation"
```

---

## Task 4: Source Connectors

**Files:**
- Create: `scholartrace/connectors/base.py`
- Create: `scholartrace/connectors/openalex.py`
- Create: `scholartrace/connectors/arxiv.py`
- Create: `scholartrace/connectors/semantic_scholar.py`
- Create: `scholartrace/connectors/dblp.py`
- Create: `scholartrace/connectors/openreview.py`
- Create: `scholartrace/connectors/crossref.py`
- Create: `tests/test_connectors.py`

**Step 1: Create base connector interface**

```python
# scholartrace/connectors/base.py
from abc import ABC, abstractmethod
from scholartrace.models.schemas import RawCandidate

class BaseConnector(ABC):
    source_name: str = ""

    @abstractmethod
    async def search(self, query: str, max_results: int = 200) -> list[RawCandidate]:
        """Search for papers matching the query. Return up to max_results candidates."""
        ...

    async def get_fulltext_url(self, paper_id: str) -> str | None:
        """Get a URL for full-text access if available."""
        return None

    async def close(self):
        pass
```

**Step 2: Implement OpenAlex connector**

Key details from API research:
- Base URL: `https://api.openalex.org/works`
- Cursor pagination: `cursor=*` for first page, then use `meta.next_cursor`
- Abstract comes as inverted index, need to reconstruct
- Add `mailto` param for polite pool
- Per page max: 200
- Fields: id, doi, title, publication_year, cited_by_count, authorships, abstract_inverted_index, locations, open_access, type

Implement `_reconstruct_abstract(inverted_index)` helper to convert OpenAlex's inverted index format back to plain text.

Implement pagination loop that pages through cursor until reaching max_results or exhausting results.

**Step 3: Implement arXiv connector**

Key details:
- Base URL: `http://export.arxiv.org/api/query`
- Returns Atom XML, parse with `xml.etree.ElementTree`
- Pagination: offset-based (`start` + `max_results`)
- Max 2000 per page, max 30000 total
- Add 3-second delay between requests
- Extract: title, authors, abstract, arxiv_id (from id URL), pdf_url, published date, doi, categories

**Step 4: Implement Semantic Scholar connector**

Key details:
- Base URL: `https://api.semanticscholar.org/graph/v1/paper/search`
- Pagination: offset + limit, also `next` token in response
- Fields: `title,year,abstract,authors,citationCount,venue,externalIds,url,openAccessPdf,fieldsOfStudy`
- API key via `x-api-key` header (optional but recommended)
- Rate limit: 1 RPS without key

**Step 5: Implement DBLP connector**

Key details:
- Base URL: `https://dblp.org/search/publ/api`
- Pagination: `h` (hits per page, max 1000) + `f` (first offset)
- Handle single-author edge case (author can be dict instead of list)
- Extract: title, authors, year, venue, doi, key, url

**Step 6: Implement Crossref connector**

Key details:
- Base URL: `https://api.crossref.org/works`
- Cursor pagination: `cursor=*`, then `next-cursor`
- Add `mailto` for polite pool
- Abstract wrapped in JATS XML tags, strip with regex or BS4
- Title is array, take first element
- Extract: DOI, title, author, year, venue, citation count, abstract

**Step 7: Implement OpenReview connector**

Key details:
- Base URL: `https://api2.openreview.net`
- Search: `GET /notes/search?query=...&source=forum&limit=50&offset=0`
- Content fields are dicts with `value` key
- Auth optional for public search
- Extract: title, abstract, authors, venue, keywords, forum id

**Step 8: Write connector tests**

Test each connector with mocked HTTP responses using `pytest-httpx` or manual mocks. Verify:
- Correct API URL construction
- Pagination handling
- Response parsing into RawCandidate models
- Edge cases (empty results, single author in DBLP, missing fields)

**Step 9: Run tests**

```bash
pytest tests/test_connectors.py -v
```

**Step 10: Commit**

```bash
git add -A
git commit -m "feat: all 6 source connectors with pagination and error handling"
```

---

## Task 5: Deduplication Service

**Files:**
- Create: `scholartrace/services/dedup.py`
- Create: `tests/test_dedup.py`

**Step 1: Implement dedup.py**

Key function: `deduplicate_candidates(candidates: list[RawCandidate]) -> list[RawCandidate]`

Algorithm:
1. Build index maps: doi -> candidate, arxiv_id -> candidate, s2_id -> candidate, openalex_id -> candidate, dblp_key -> candidate
2. Group candidates that share any exact ID
3. For remaining ungrouped candidates, do fuzzy title matching (threshold 0.85) with same-year check
4. For each group, merge into one candidate keeping richest metadata
5. Track provenance: merged candidate lists all sources

Use `rapidfuzz.fuzz.token_sort_ratio` for title similarity.

**Step 2: Test dedup**

Test cases:
- Same DOI different sources → merged
- Same arXiv ID → merged
- Fuzzy title match same year → merged
- Completely different papers → separate
- Merged candidate keeps best metadata

**Step 3: Run tests and commit**

```bash
pytest tests/test_dedup.py -v
git add -A && git commit -m "feat: multi-key deduplication with fuzzy title matching"
```

---

## Task 6: Ranking Service

**Files:**
- Create: `scholartrace/services/ranking.py`
- Create: `tests/test_ranking.py`

**Step 1: Implement ranking.py**

Key function: `rank_papers(works: list[Work], theme: Theme) -> list[Work]`

Score components:
1. **relevance_score**: TF-IDF cosine similarity between theme queries and paper (title + abstract)
2. **recency_score**: `exp(-0.23 * (current_year - year))` — half-life of 3 years
3. **influence_score**: `log(1 + citation_count) / log(1 + max_citations)` normalized to [0,1]
4. **venue_score**: Check venue name against a list of top-tier venues (NeurIPS, ICML, ICLR, ACL, EMNLP, AAAI, CVPR, etc.) — 1.0 for top, 0.7 for good, 0.5 for unknown
5. **fulltext_score**: 1.0 if fulltext_available else 0.0
6. **source_agreement_score**: `min(len(provenance) / 3, 1.0)` — more sources = more confidence

Composite: `sum(weight_i * score_i)` using configurable weights from settings.

**Step 2: Test ranking**

Test cases:
- All scores are in [0, 1]
- Composite score is weighted sum
- Higher relevance ranks higher
- Recent papers get recency boost
- Highly cited papers get influence boost
- Papers from multiple sources get agreement boost

**Step 3: Run tests and commit**

```bash
pytest tests/test_ranking.py -v
git add -A && git commit -m "feat: multi-objective ranking with TF-IDF relevance"
```

---

## Task 7: Full-Text Acquisition Cascade

**Files:**
- Create: `scholartrace/services/fulltext.py`
- Create: `tests/test_fulltext.py`

**Step 1: Implement fulltext.py**

Key function: `async acquire_fulltext(work: Work, storage: StorageService) -> Work`

Cascade:
1. If arXiv ID exists:
   a. Try `https://arxiv.org/html/{arxiv_id}` — fetch and parse HTML with BS4
   b. If HTML fails, try `https://arxiv.org/pdf/{arxiv_id}` — download PDF, extract text with PyMuPDF
2. If Semantic Scholar OA PDF URL exists: download and extract
3. If oa_url exists from any source: try downloading
4. Mark access_status accordingly

For PDF text extraction:
```python
import fitz  # PyMuPDF
def extract_pdf_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
```

For HTML section extraction: parse with BS4, extract sections by heading tags, build Section objects.

Store raw artifact (PDF bytes / HTML string) to `data/artifacts/raw/`, parsed text to `data/artifacts/parsed/`, sections as JSON to `data/artifacts/sections/`.

**Step 2: Test fulltext**

Test with mock HTTP:
- arXiv HTML path works
- arXiv PDF path works (with small test PDF)
- Cascade falls through correctly
- Abstract-only papers marked correctly

**Step 3: Run tests and commit**

```bash
pytest tests/test_fulltext.py -v
git add -A && git commit -m "feat: full-text acquisition cascade with PDF/HTML extraction"
```

---

## Task 8: Retrieval Orchestration Service

**Files:**
- Create: `scholartrace/services/retrieval.py`
- Create: `scholartrace/jobs/manager.py`
- Create: `tests/test_retrieval.py`

**Step 1: Implement retrieval.py**

This is the core orchestration service. Key function:

```python
async def run_retrieval(theme: Theme, storage: StorageService, settings: Settings) -> list[Work]
```

Flow:
1. Create all connectors
2. For each query in `theme.parsed_queries`:
   a. Run search on all connectors concurrently (asyncio.gather)
   b. Collect all RawCandidate results
   c. Respect per-source max_results limit
3. Aggregate all candidates into one list
4. Run deduplication
5. Convert deduplicated candidates to Work objects
6. Run ranking
7. Save all works to storage
8. Link works to theme in theme_works table
9. Return ranked works

Handle errors per-source: if one source fails, log and continue with others.

**Step 2: Implement jobs/manager.py**

Simple in-memory job tracker. `JobManager` class:
- `create_job(theme_id) -> RetrievalJob`
- `start_job(job_id)`
- `complete_job(job_id, result_count)`
- `fail_job(job_id, error_message)`
- `get_job(job_id) -> RetrievalJob | None`

**Step 3: Test retrieval orchestration**

Test with mocked connectors returning canned data. Verify:
- Multiple queries fan out correctly
- Dedup runs
- Ranking produces ordered results
- Works saved to storage
- Errors in one source don't kill the whole job

**Step 4: Run tests and commit**

```bash
pytest tests/test_retrieval.py -v
git add -A && git commit -m "feat: retrieval orchestration with async multi-source fan-out"
```

---

## Task 9: REST API (FastAPI)

**Files:**
- Create: `scholartrace/api/rest.py`
- Create: `tests/test_rest_api.py`

**Step 1: Implement rest.py**

Endpoints:

```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query, HTTPException
from scholartrace.models.schemas import Work, Theme, RetrievalJob, Section, Artifact

app = FastAPI(title="ScholarTrace", version="0.1.0")

# GET /health
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}

# POST /themes — accept theme document text, parse it, return theme
@app.post("/themes", response_model=Theme)
async def create_theme(text: str = "", file: UploadFile | None = None):
    ...

# POST /retrieval/jobs — launch retrieval for a theme
@app.post("/retrieval/jobs", response_model=RetrievalJob)
async def create_retrieval_job(theme_id: str, background_tasks: BackgroundTasks):
    ...

# GET /retrieval/jobs/{job_id}
@app.get("/retrieval/jobs/{job_id}", response_model=RetrievalJob)
async def get_job_status(job_id: str):
    ...

# GET /themes/{theme_id}/papers
@app.get("/themes/{theme_id}/papers", response_model=list[Work])
async def list_papers(theme_id: str, limit: int = 50, offset: int = 0):
    ...

# GET /papers/{paper_id}
@app.get("/papers/{paper_id}", response_model=Work)
async def get_paper(paper_id: str):
    ...

# GET /papers/{paper_id}/sections
@app.get("/papers/{paper_id}/sections", response_model=list[Section])
async def get_sections(paper_id: str):
    ...

# GET /papers/{paper_id}/fulltext
@app.get("/papers/{paper_id}/fulltext")
async def get_fulltext(paper_id: str):
    ...

# GET /themes/{theme_id}/export
@app.get("/themes/{theme_id}/export")
async def export_theme(theme_id: str, format: str = "json"):
    ...
```

Use FastAPI's `BackgroundTasks` to run retrieval jobs asynchronously. The `create_retrieval_job` endpoint returns immediately with a job_id, and the actual retrieval runs in the background.

**Step 2: Test REST API**

Use FastAPI's `TestClient` (backed by httpx) to test all endpoints:
- Health check returns 200
- Create theme returns theme with parsed queries
- Launch retrieval job returns job_id
- List papers returns paginated results
- Get paper detail returns full metadata
- Export works in JSON and markdown

**Step 3: Run tests and commit**

```bash
pytest tests/test_rest_api.py -v
git add -A && git commit -m "feat: FastAPI REST API with all required endpoints"
```

---

## Task 10: MCP Server

**Files:**
- Create: `scholartrace/api/mcp_server.py`
- Create: `tests/test_mcp_server.py`

**Step 1: Implement mcp_server.py**

```python
from mcp.server.fastmcp import FastMCP
from scholartrace.services.storage import StorageService
from scholartrace.services.theme_parser import parse_theme
from scholartrace.services.retrieval import run_retrieval
from scholartrace.config import settings

mcp = FastMCP("ScholarTrace", json_response=True)
storage = StorageService(db_path=settings.db_path)

@mcp.tool()
async def search_papers_by_theme(theme_document: str) -> dict:
    """Parse a theme document and search for relevant papers across multiple sources.
    Returns a theme_id and summary of found papers."""
    ...

@mcp.tool()
async def get_ranked_papers(theme_id: str, limit: int = 50) -> list[dict]:
    """Get ranked papers for a previously processed theme."""
    ...

@mcp.tool()
async def get_paper_metadata(paper_id: str) -> dict:
    """Get full metadata for a specific paper."""
    ...

@mcp.tool()
async def get_paper_sections(paper_id: str) -> list[dict]:
    """Get section-level content for a paper if full text is available."""
    ...

@mcp.tool()
async def get_paper_fulltext(paper_id: str) -> dict:
    """Get full text content for a paper if available."""
    ...

@mcp.tool()
async def get_related_papers(paper_id: str, limit: int = 10) -> list[dict]:
    """Get papers related to a given paper via citation graph."""
    ...

@mcp.tool()
async def export_theme_report(theme_id: str, format: str = "json") -> str:
    """Export a complete report for a theme as JSON or Markdown."""
    ...
```

Each tool should call the corresponding service method and return structured data. The `search_papers_by_theme` tool is the main entry point — it runs the full pipeline (parse theme → retrieve → dedup → rank → store) and returns a summary.

**Step 2: Test MCP server**

Use the mcp client to test tool calls:
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_search():
    server_params = StdioServerParameters(command="python", args=["-m", "scholartrace.api.mcp_server"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_papers_by_theme", {"theme_document": "RLHF sycophancy"})
            assert result is not None
```

Alternative: test the tool functions directly by importing them.

**Step 3: Run tests and commit**

```bash
pytest tests/test_mcp_server.py -v
git add -A && git commit -m "feat: MCP server with 7 tools for paper discovery"
```

---

## Task 11: End-to-End Test and Verification Script

**Files:**
- Create: `tests/e2e_test.py`
- Create: `verify.py`

**Step 1: Create e2e_test.py**

Full end-to-end test using the sycophancy research brief:
1. Read `docs/examples/sycophancy_affective_hallucination_research_brief.md`
2. Parse theme document
3. Run retrieval (with real API calls)
4. Verify >= 100 unique papers found
5. Verify dedup works (no duplicate DOIs)
6. Verify ranking produces ordered results
7. Verify some papers have fulltext_available=True
8. Test REST API endpoints with TestClient
9. Test MCP tool functions directly
10. Verify both interfaces return consistent data

**Step 2: Create verify.py**

Standalone verification script that:
1. Starts the REST API server
2. Runs health check
3. Submits the theme document
4. Waits for retrieval to complete
5. Checks paper count >= 100
6. Exports results to JSON and markdown
7. Prints summary statistics

**Step 3: Run e2e test**

```bash
pytest tests/e2e_test.py -v --timeout=300
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: end-to-end tests and verification script"
```

---

## Task 12: Documentation and Final Polish

**Files:**
- Create: `README.md` (overwrite placeholder)
- Create: `API.md`
- Create: `MCP.md`

**Step 1: Write README.md**

Sections:
- What is ScholarTrace
- Quick start (install, configure, run)
- REST API usage examples
- MCP server usage examples
- Architecture overview
- Configuration options
- Limitations and legal notes

**Step 2: Write API.md**

Full REST API documentation with request/response examples for each endpoint.

**Step 3: Write MCP.md**

MCP server setup and tool documentation with usage examples.

**Step 4: Final commit**

```bash
git add -A
git commit -m "docs: README, API docs, MCP usage guide"
```

---

## Summary

| Task | Description | Est. Steps |
|------|-------------|------------|
| 1 | Project scaffolding + deps | 7 |
| 2 | Canonical schema + SQLite | 5 |
| 3 | Theme parser | 4 |
| 4 | 6 source connectors | 10 |
| 5 | Deduplication | 3 |
| 6 | Ranking | 3 |
| 7 | Full-text cascade | 3 |
| 8 | Retrieval orchestration | 4 |
| 9 | REST API (FastAPI) | 3 |
| 10 | MCP server | 3 |
| 11 | E2E test + verification | 4 |
| 12 | Documentation | 4 |
| **Total** | | **53 steps** |
