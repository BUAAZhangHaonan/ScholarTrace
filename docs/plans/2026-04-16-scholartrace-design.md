# ScholarTrace System Design

## 1. Problem Statement

Build ScholarTrace: a theme-guided, multi-source, full-text-aware scholarly retrieval service. Given a user-provided theme document, the system retrieves 100+ relevant papers, ranks them by relevance/recency/influence, acquires full text when available, and exposes both a REST API and an MCP server interface.

## 2. Architecture Overview

```
Theme Document
    |
    v
Theme Parser (extract topics, methods, datasets, generate queries)
    |
    v
Multi-Source Fan-Out (async parallel queries to all sources)
    |   OpenAlex / arXiv / Semantic Scholar / DBLP / OpenReview / Crossref
    v
Candidate Aggregation + Multi-Key Dedup
    |
    v
Multi-Objective Ranking (relevance + recency + influence + venue + OA + source agreement)
    |
    v
Full-Text Acquisition Cascade (arXiv HTML -> PDF text -> OA -> abstract only)
    |
    v
Storage (SQLite metadata + filesystem artifacts)
    |
    v
Exposure Layer: REST API (FastAPI) + MCP Server (mcp SDK)
```

## 3. Storage

- **SQLite** for canonical metadata: works, identifiers, artifacts, sections, jobs
- **Filesystem** for raw artifacts: PDFs, HTML, parsed text, sections
- Directory layout:
  ```
  data/
    scholartrace.db
    artifacts/
      raw/       # downloaded PDFs, HTML
      parsed/    # extracted full text
      sections/  # section-level JSON
  ```

## 4. Canonical Schema

### works table
- id (internal UUID)
- doi, arxiv_id, openalex_id, s2_id, dblp_key, openreview_id
- title, authors (JSON), year, venue, abstract
- relevance_score, recency_score, influence_score, composite_score
- fulltext_available (bool)
- source_provenance (JSON list of sources)
- created_at, updated_at

### artifacts table
- id, work_id
- kind (pdf, html, source_tar, parsed_text)
- source_url, local_path, sha256
- license, access_status
- created_at

### sections table
- id, work_id, artifact_id
- section_title, section_order
- text_content
- created_at

### jobs table
- id, theme_id, status (pending/running/completed/failed)
- query_count, candidate_count, result_count
- created_at, updated_at, completed_at

### themes table
- id, document_path, document_text
- parsed_topics, parsed_methods, parsed_queries (JSON)
- created_at

## 5. Source Connectors

Each connector implements:
```python
class BaseConnector(ABC):
    async def search(self, query: str, max_results: int) -> list[RawCandidate]
    async def get_paper(self, paper_id: str) -> RawCandidate | None
    async def get_fulltext_url(self, paper_id: str) -> str | None
```

### Connectors
1. **OpenAlex** — primary recall source. Uses REST API with cursor-based pagination. Rate limit: 10 req/s (polite pool with mailto).
2. **arXiv** — open full-text source. Uses arXiv API + HTML access at `arxiv.org/html/{id}`. Bulk via OAI-PMH if needed.
3. **Semantic Scholar** — enrichment + secondary recall. API with key for higher rate limits. Used for citation counts, related papers, and extra recall.
4. **DBLP** — CS venue normalization. Public search API, max 1000 results per query.
5. **OpenReview** — ML conference submissions/reviews. Python client or REST API.
6. **Crossref** — DOI normalization and metadata. Public REST API.

## 6. Theme Parser

Takes a theme document (markdown or plain text). Extracts:
- Core topic phrases
- Methods mentioned
- Datasets/benchmarks
- Domain keywords
- Exclusion terms

Generates 5-8 query formulations:
1. High-precision topic query (2-3 key terms ANDed)
2. Broader recall query (more terms, ORed)
3. Recent-trend query (filtered to last 3 years)
4. Method-centric query
5. Citation-chaining seeds (from key papers mentioned in theme)
6. Domain-scoped query
7. Dataset/benchmark query

## 7. Retrieval Orchestration

```
for each query formulation:
    for each enabled source:
        page through results (up to 200 per source per query)
        collect raw candidates
        merge into global candidate pool with dedup

dedup candidates (DOI, arXiv ID, S2 ID, OpenAlex ID, DBLP key, title fuzzy match)

rank remaining candidates

attempt full-text acquisition for top candidates
```

## 8. Deduplication

Multi-key canonical dedup:
1. Exact DOI match
2. Exact arXiv ID match
3. Exact Semantic Scholar ID match
4. Exact OpenAlex ID match
5. Exact DBLP key match
6. Fuzzy title + first-author + year match (threshold 0.85)

When merging, keep the richest metadata record and track all source provenance.

## 9. Ranking

Composite score = weighted sum:
- **Relevance** (0.35): TF-IDF cosine similarity between theme keywords and paper title+abstract
- **Recency** (0.20): Exponential decay from current year. Half-life ~3 years
- **Influence** (0.20): Log-normalized citation count
- **Venue quality** (0.10): Tiered venue scoring (top venues get bonus)
- **Full-text availability** (0.10): Binary boost for papers with accessible full text
- **Source agreement** (0.05): Papers found by multiple sources get a boost

Weights are configurable via environment variables.

## 10. Full-Text Acquisition Cascade

For each ranked paper:
1. Check arXiv HTML access (`arxiv.org/html/{id}`)
2. Download arXiv PDF, extract text via PyMuPDF
3. Check Semantic Scholar open-access PDF link
4. Check Unpaywall OA URL
5. Mark as abstract-only if no full text available

Store raw artifacts and parsed text separately.

## 11. REST API (FastAPI)

Endpoints:
- `GET /health` — health check
- `POST /themes` — upload theme document, return theme_id
- `POST /retrieval/jobs` — launch retrieval for a theme, return job_id
- `GET /retrieval/jobs/{job_id}` — get job status
- `GET /themes/{theme_id}/papers` — list ranked papers (paginated)
- `GET /papers/{paper_id}` — get paper detail
- `GET /papers/{paper_id}/sections` — get section-level content
- `GET /papers/{paper_id}/fulltext` — get full-text availability and content
- `GET /themes/{theme_id}/export?format=json|markdown` — export results

## 12. MCP Server

Tools:
- `search_papers_by_theme(theme_document: str)` — parse theme, run retrieval, return ranked papers
- `get_ranked_papers(theme_id: str, limit: int)` — get ranked papers for a theme
- `get_paper_metadata(paper_id: str)` — get full metadata for one paper
- `get_paper_sections(paper_id: str)` — get section-level content
- `get_paper_fulltext(paper_id: str)` — get full text if available
- `get_related_papers(paper_id: str, limit: int)` — get related papers via citation graph
- `export_theme_report(theme_id: str, format: str)` — export theme results

## 13. Testing Strategy

- Unit tests for each connector (mocked HTTP)
- Integration tests for retrieval pipeline (with caching)
- API tests for REST endpoints
- MCP tests using mcp client
- End-to-end test with sycophancy research brief as theme document
- Verification script that confirms 100+ papers retrieved

## 14. Dependencies

Core:
- fastapi, uvicorn, httpx, pydantic
- mcp (Python MCP SDK)
- sqlite3 (stdlib)
- aiofiles (async file I/O)
- aiohttp or httpx for async HTTP

Parsing:
- pymupdf (PDF text extraction)
- beautifulsoup4 (HTML parsing)

Ranking:
- scikit-learn (TF-IDF, cosine similarity)
- rapidfuzz (fuzzy string matching)

## 15. Out of Scope for V1

- Vector search / embedding index (Qdrant, OpenSearch)
- Grobid / Docling service dependencies
- Zotero integration
- Neo4j graph database
- Docker deployment (provide instructions but no compose file)
- Web UI
