# ScholarTrace

**English** | [中文](README_CN.md)

> Clean LAN-served scholarly MCP for ChatBox, with honest full-text status and only two public tools.

## Overview

ScholarTrace takes a theme document, retrieves papers from multiple scholarly sources, reranks them, and lets the MCP client read deeper only when needed.

- **2 public MCP tools**: `query` and `read`
- **Main deployment mode**: SSE over LAN for shared team use in ChatBox
- **Default rerank model**: `glm-5-turbo`
- **Default second-stage pool**: `agent_candidate_limit=100`
- **Default final output**: `final_limit=20`
- **Current local validation still collects 182 tests**

REST stays broad for now. This pass simplifies the MCP product surface only.

## LAN SSE Quick Start

This is the main deployment story.

```bash
conda create -n ScholarTrace python=3.13 -y
conda activate ScholarTrace

cd ScholarTrace
python -m pip install -r requirements-dev.txt

scholartrace-check-env --include-dev --pytest-collect
pytest tests/ -q

export SCHOLARTRACE_MCP_TRANSPORT=sse
export SCHOLARTRACE_MCP_HOST=0.0.0.0
export SCHOLARTRACE_MCP_PORT=8001
export SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true
export SCHOLARTRACE_ACCESS_TOKEN=g203-mcp
export SCHOLARTRACE_BIGMODEL_API_KEY=<your-bigmodel-key>

scholartrace-mcp
```

Use this URL from a client on the same LAN:

- `http://<server-lan-ip>:8001/sse`

The token is user-defined. It is not auto-generated.

The MCP client must send:

- `Authorization: Bearer g203-mcp`

## ChatBox JSON

Paste or import this in ChatBox:

```json
{
  "name": "ScholarTrace LAN",
  "type": "sse",
  "url": "http://<server-lan-ip>:8001/sse",
  "headers": {
    "Authorization": "Bearer g203-mcp"
  }
}
```

## Public MCP Surface

ScholarTrace now exposes only two public MCP tools.

### `query`

Suggested call:

```json
{
  "theme_document": "your theme document text",
  "final_limit": 20,
  "agent_candidate_limit": 100,
  "coarse_pool_limit": 500,
  "include_rationale": true
}
```

What `query` does by default:

1. Parse the theme document
2. Run unified retrieval across the configured scholarly sources
3. Deduplicate the raw candidates
4. Run first-stage composite ranking
5. Keep a coarse candidate pool
6. Run the built-in DeepXiv Agent second-stage rerank with `glm-5-turbo`
7. Return the final selected papers

Important notes:

- DeepXiv still joins unified retrieval as a source when it is configured
- The DeepXiv Agent is no longer a separate MCP step in the normal user flow
- If `final_limit` is not given, `query` returns 20 papers by default
- If the client asks for more, ScholarTrace returns that many when possible
- `agent_candidate_limit` defaults to 100
- `coarse_pool_limit` is optional

The `query` response includes:

- `theme_id`
- `total_retrieved`
- `total_after_dedup`
- `total_after_first_stage`
- `total_agent_candidates`
- `total_final`
- `papers`

Each paper summary includes:

- `paper_id`
- `title`
- `authors`
- `year`
- `venue`
- `abstract`
- `composite_score`
- `agent_score`
- `agent_rank`
- `rationale`
- `fulltext_status`

### `read`

`read` is the single layered access tool.

Supported depths:

- `summary`
- `sections`
- `fulltext_status`
- `fulltext`
- `direct_evidence`

Normal MCP flow:

1. Call `query`
2. Pick a paper from the returned list
3. Call `read`
4. If `fulltext_status` shows the paper is not cached, call `read` again with `allow_acquire=true`

Example:

```json
{
  "paper_id": "paper-id-from-query",
  "depth": "fulltext",
  "allow_acquire": true
}
```

What each depth means:

- `summary`: metadata, abstract, ranking state, agent state, and compact full-text status
- `sections`: cached sections only
- `fulltext_status`: the honest cache and acquisition state
- `fulltext`: cached parsed text when present; if missing and `allow_acquire=true`, ScholarTrace attempts explicit acquisition and returns the resulting state
- `direct_evidence`: direct DeepXiv metadata and brief for arXiv-backed papers, still under the same `read` tool

## Full-Text Honesty

ScholarTrace does have a real explicit acquire path. It is not an empty shell.

Today the explicit acquire path really tries, in this order:

1. arXiv HTML
2. arXiv PDF
3. metadata `pdf_url`
4. metadata `oa_url` or `html_url`
5. DeepXiv markdown fallback for arXiv-backed papers when DeepXiv is configured

What is confirmed working today:

- arXiv HTML fetch with heading-based section parsing
- arXiv PDF fetch with plain-text extraction
- metadata PDF and HTML fetch
- DeepXiv markdown fallback with heading-based section parsing
- explicit negative-cache states when acquisition fails

What is still limited:

- New-paper full-text retrieval is not guaranteed
- PDF parsing is mostly plain-text extraction, not strong section recovery
- No OCR for scanned or image-only PDFs
- HTML parsing is a simple heading-based split
- markdown fallback parsing is a simple heading-based split
- retrieval does not auto-download full text during search

This is why `read` is honest:

- if full text is cached, it returns it
- if only sections are cached, it returns sections
- if only abstract and metadata are available, it says so
- if explicit acquisition fails, it says so clearly

## REST API

REST stays broad in this round.

Core REST endpoints:

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

The explicit full-text REST flow is still:

1. `GET /papers/{paper_id}/fulltext`
2. `POST /papers/{paper_id}/fulltext/acquire`
3. `GET /papers/{paper_id}/fulltext`

## Config Notes

Key runtime settings:

| Variable | Default | Purpose |
|---|---|---|
| `SCHOLARTRACE_MCP_TRANSPORT` | `stdio` | Use `sse` for LAN serving |
| `SCHOLARTRACE_MCP_HOST` | `127.0.0.1` | Set `0.0.0.0` for LAN serving |
| `SCHOLARTRACE_MCP_PORT` | `8001` | MCP SSE port |
| `SCHOLARTRACE_REMOTE_ACCESS_ENABLED` | `false` | Must be `true` for non-loopback SSE |
| `SCHOLARTRACE_ACCESS_TOKEN` | | Shared bearer token for network MCP |
| `SCHOLARTRACE_BIGMODEL_API_KEY` | | Required for the built-in query rerank |
| `SCHOLARTRACE_BIGMODEL_MODEL` | `glm-5-turbo` | Default rerank model |
| `SCHOLARTRACE_AGENT_CANDIDATE_LIMIT` | `100` | Default second-stage candidate count |
| `SCHOLARTRACE_FINAL_LIMIT` | `20` | Default final result count |
| `SCHOLARTRACE_TARGET_CANDIDATE_POOL` | `500` | Default coarse pool before rerank |

## `stdio` Debug Mode

`stdio` is still available, but only as a local debugging or development mode.

```bash
SCHOLARTRACE_MCP_TRANSPORT=stdio scholartrace-mcp
```

Do not treat `stdio` as the main team deployment story. The main story is LAN SSE.

## Example systemd Service

The repository includes `scripts/scholartrace-mcp.service` as an SSE-first example.

The key values are:

- `SCHOLARTRACE_MCP_TRANSPORT=sse`
- `SCHOLARTRACE_MCP_HOST=0.0.0.0`
- `SCHOLARTRACE_MCP_PORT=8001`
- `SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true`
- `SCHOLARTRACE_ACCESS_TOKEN=g203-mcp`

## Validation

Useful local checks:

```bash
scholartrace-check-env --include-dev --pytest-collect
pytest tests/ -q
python -m compileall scholartrace examples/glm_scholar_search.py
```

## License

MIT
