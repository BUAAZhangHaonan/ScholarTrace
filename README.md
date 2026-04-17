# ScholarTrace

**English** | [中文](README_CN.md)

> Theme-guided scholarly retrieval, cached evidence access, and explicit full-text acquisition

## Overview

ScholarTrace turns a research brief into ranked papers, cached evidence, and exportable reports.

- **Unified retrieval**: 6 core scholarly sources by default: OpenAlex, arXiv, Semantic Scholar, DBLP, OpenReview, and Crossref.
- **DeepXiv joins unified retrieval when configured**: if `SCHOLARTRACE_DEEPXIV_TOKENS` is set, or explicit auto-register is enabled with `SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET`, DeepXiv becomes a normal retrieval source in the same fan-out, dedup, ranking, and storage path.
- **One ranking path**: all candidates go through the same deduplication, provenance merge, and composite ranking logic. DeepXiv is not a side ranking system.
- **Cache-first full-text model**: `GET /papers/{paper_id}/fulltext` and `get_paper_fulltext` only read cached state. Missing full text is fetched only through the explicit acquire path.
- **Direct DeepXiv evidence access**: dedicated DeepXiv REST endpoints and MCP tools still exist, but they are direct DeepXiv reads, not ScholarTrace cache reads.
- **LLM-facing interfaces**: REST API plus a 13-tool MCP server. MCP defaults to local `stdio`. SSE is optional and token-gated.
- **BigModel GLM example**: `examples/glm_scholar_search.py` uses `glm-5-turbo` by default and keeps every single request under the model context window with bounded prompt packing.

## Quick Start

```bash
conda create -n ScholarTrace python=3.13 -y
conda activate ScholarTrace

cd ScholarTrace
python -m pip install -r requirements-dev.txt

scholartrace-check-env --include-dev --pytest-collect
pytest tests/ -q

cp .env.example .env
# Edit .env for your local machine

scholartrace-api
# -> http://127.0.0.1:9000

scholartrace-mcp
# -> local stdio MCP server
```

Current local validation collects **182 tests**.

## Configuration

All runtime settings use the `SCHOLARTRACE_` prefix. Use `.env` only for local development. For a deployed service, keep secrets outside the repo tree.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `SCHOLARTRACE_API_HOST` | No | `127.0.0.1` | REST bind host |
| `SCHOLARTRACE_API_PORT` | No | `9000` | REST bind port |
| `SCHOLARTRACE_MCP_HOST` | No | `127.0.0.1` | MCP SSE bind host |
| `SCHOLARTRACE_MCP_PORT` | No | `8001` | MCP SSE bind port |
| `SCHOLARTRACE_MCP_TRANSPORT` | No | `stdio` | MCP transport: `stdio` or `sse` |
| `SCHOLARTRACE_REMOTE_ACCESS_ENABLED` | No | `false` | Required before non-loopback REST or MCP SSE |
| `SCHOLARTRACE_ACCESS_TOKEN` | Remote only | | Shared bearer token for REST and MCP SSE |
| `SCHOLARTRACE_SEMANTIC_SCHOLAR_API_KEY` | No | | Optional higher Semantic Scholar rate limits |
| `SCHOLARTRACE_OPENALEX_MAILTO` | No | | OpenAlex polite-pool email |
| `SCHOLARTRACE_CROSSREF_MAILTO` | No | | Crossref polite-pool email |
| `SCHOLARTRACE_MAX_RESULTS_PER_SOURCE_PER_QUERY` | No | `200` | Per-source retrieval cap |
| `SCHOLARTRACE_TARGET_CANDIDATE_POOL` | No | `500` | Target merged candidate pool |
| `SCHOLARTRACE_MAX_FULLTEXT_DOWNLOADS` | No | `50` | Retrieval-time full-text cap |
| `SCHOLARTRACE_BIGMODEL_API_KEY` | Example only | | BigModel key for the GLM example and DeepXiv agent filtering |
| `SCHOLARTRACE_BIGMODEL_BASE_URL` | No | `https://open.bigmodel.cn/api/coding/paas/v4/chat/completions` | BigModel endpoint |
| `SCHOLARTRACE_BIGMODEL_MODEL` | No | `glm-5-turbo` | Default GLM model |
| `SCHOLARTRACE_DEEPXIV_TOKENS` | DeepXiv only | | Comma-separated DeepXiv tokens |
| `SCHOLARTRACE_DEEPXIV_AUTO_REGISTER` | No | `false` | Explicit opt-in auto-register |
| `SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET` | Auto-register only | | SDK secret used only when auto-register is enabled |

## Runtime Model

### Unified retrieval and DeepXiv

Theme retrieval has one main path:

1. parse the theme document into queries
2. fan out each query across the configured connectors
3. merge duplicate papers by stable identifiers and fuzzy title match
4. rank the merged papers
5. persist canonical works, links, artifacts, and sections

DeepXiv is part of that path when configured. If it is not configured, ScholarTrace keeps the 6-source flow and skips DeepXiv cleanly.

### Cached reads vs explicit acquire

ScholarTrace now uses one clear full-text model:

1. **Read cached state** with `GET /papers/{paper_id}/fulltext` or `get_paper_fulltext`
2. **Acquire missing full text explicitly** with `POST /papers/{paper_id}/fulltext/acquire` or `acquire_paper_fulltext`
3. **Read cached state again** with `GET /papers/{paper_id}/fulltext` or `get_paper_fulltext`

Important details:

- cache-only reads do not perform network fetches
- explicit acquire is the only path that can trigger network work
- public-source acquisition is tried first
- for arXiv papers, DeepXiv markdown can be used as a later fallback during explicit acquire when DeepXiv is configured
- expensive operations are budgeted and rate-limited; cache reads are cheap

### Direct DeepXiv reads

The dedicated DeepXiv REST endpoints and MCP tools are still useful, but they are different from ScholarTrace cache reads:

- `GET /deepxiv/papers/{arxiv_id}/fulltext` and `deepxiv_paper_fulltext` return direct DeepXiv markdown
- they do not replace `GET /papers/{paper_id}/fulltext`
- they are best used for direct arXiv evidence access, summaries, sections, and agent-assisted filtering

## REST API

### Core REST endpoints

```text
GET  /health
POST /themes
POST /retrieval/jobs
GET  /retrieval/jobs/{job_id}
GET  /themes/{theme_id}/papers
GET  /papers/{paper_id}
GET  /papers/{paper_id}/sections
GET  /papers/{paper_id}/fulltext
POST /papers/{paper_id}/fulltext/acquire
GET  /themes/{theme_id}/export
```

### Direct DeepXiv REST endpoints

```text
POST /deepxiv/search
GET  /deepxiv/papers/{arxiv_id}/summary
GET  /deepxiv/papers/{arxiv_id}/fulltext
GET  /deepxiv/papers/{arxiv_id}/sections/{section_name}
POST /deepxiv/agent/filter
```

### End-to-end REST workflow

This is the normal REST flow for a client or script:

```bash
# 1. Create a theme
curl -s -X POST http://127.0.0.1:9000/themes \
  -F 'text=RLHF sycophancy and affective hallucination in language models'

# 2. Launch retrieval
curl -s -X POST http://127.0.0.1:9000/retrieval/jobs \
  -F 'theme_id=<theme-id>'

# 3. Poll the job
curl -s http://127.0.0.1:9000/retrieval/jobs/<job-id>

# 4. Get ranked papers
curl -s 'http://127.0.0.1:9000/themes/<theme-id>/papers?limit=20'

# 5. Read cached full-text state
curl -s http://127.0.0.1:9000/papers/<paper-id>/fulltext

# 6. Explicitly acquire missing full text when needed
curl -s -X POST http://127.0.0.1:9000/papers/<paper-id>/fulltext/acquire

# 7. Re-read the cached state
curl -s http://127.0.0.1:9000/papers/<paper-id>/fulltext
```

The full-text payload tells you whether the paper is already cached, whether it still needs acquisition, and whether the last attempt ended in a negative cache window.

## MCP Server

ScholarTrace exposes **13 tools** over MCP. Local `stdio` is the default. SSE is available only when you opt in and provide an access token.

| # | Tool | Purpose |
|---|---|---|
| 1 | `search_papers_by_theme` | Parse a theme document, run unified retrieval, rank results, and return top papers |
| 2 | `get_ranked_papers` | Read ranked papers for a stored theme |
| 3 | `get_paper_metadata` | Read redacted public metadata for one paper |
| 4 | `get_paper_sections` | Read cached section content |
| 5 | `get_paper_fulltext` | Read cached full-text state only |
| 6 | `acquire_paper_fulltext` | Explicitly acquire full text, then return the refreshed cached state |
| 7 | `get_related_papers` | Read nearby papers by venue and year |
| 8 | `export_theme_report` | Export a JSON or Markdown report |
| 9 | `deepxiv_search` | Search arXiv through DeepXiv |
| 10 | `deepxiv_paper_summary` | Read direct DeepXiv metadata and TLDR |
| 11 | `deepxiv_paper_fulltext` | Read direct DeepXiv markdown full text |
| 12 | `deepxiv_paper_section` | Read one direct DeepXiv section |
| 13 | `deepxiv_agent_filter` | Search with DeepXiv, then filter with the GLM agent |

### Local `stdio` and optional SSE

Recommended local mode:

```bash
scholartrace-mcp
```

Optional network mode:

```bash
SCHOLARTRACE_MCP_TRANSPORT=sse \
SCHOLARTRACE_MCP_HOST=0.0.0.0 \
SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true \
SCHOLARTRACE_ACCESS_TOKEN=change-me \
scholartrace-mcp
```

Use SSE only when you really need a network endpoint. Remote startup is rejected unless remote access is explicitly enabled and a token is present.

### ChatBox-style MCP workflow

For ChatBox or any other MCP client, the normal flow is:

1. call `search_papers_by_theme` with the full research brief
2. call `get_ranked_papers` for the returned `theme_id`
3. call `get_paper_fulltext` for a chosen paper to inspect cached state
4. if `needs_acquisition` is `true`, call `acquire_paper_fulltext`
5. call `get_paper_fulltext` again to read the updated cache

That is the main MCP story. Use the dedicated DeepXiv tools only when you want direct DeepXiv search, summary, markdown full text, or section access.

## Example Script

`examples/glm_scholar_search.py` follows the same model as the API:

1. create a theme
2. launch unified retrieval
3. read ranked papers
4. read cached full-text state
5. explicitly acquire missing full text
6. re-read cached state
7. summarize papers with BigModel GLM

Notes:

- the default model stays `glm-5-turbo`
- `SCHOLARTRACE_BIGMODEL_API_KEY` is required
- no repository fallback key is used
- prompt construction is bounded per request, not globally capped
- the script trims message history and packs paper batches so one call stays inside the model context window

Interactive commands in the example:

- `papers` shows the current ranked list
- `fulltext N` reads cached state for paper `N`
- `acquire N` explicitly acquires full text for paper `N`, then re-reads the cache
- `chat` opens the bounded interactive GLM loop

## Architecture Snapshot

```text
theme document
    -> parsed queries
    -> unified fan-out across configured sources
    -> dedup + provenance merge
    -> composite ranking
    -> canonical storage
    -> cached sections / cached full-text state

default sources:
    OpenAlex, arXiv, Semantic Scholar, DBLP, OpenReview, Crossref

optional configured source:
    DeepXiv

explicit evidence path:
    cache read
    -> explicit acquire
    -> cache re-read
```

## Validation

Useful local checks:

```bash
scholartrace-check-env --include-dev --pytest-collect
pytest tests/ -q
python -m compileall scholartrace examples/glm_scholar_search.py
```

## License

MIT
