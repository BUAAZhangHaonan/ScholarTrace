# ScholarTrace

**English** | [中文](README_CN.md)

> Multi-source scholarly paper retrieval and LLM-based reranking, served via MCP for ChatBox and other AI clients.

## What It Does

ScholarTrace takes a theme document, retrieves papers from 6+ scholarly sources, ranks them with a multi-model LLM pool, and exposes two MCP tools for downstream clients.

**Two MCP tools**: `query` (search & rank) and `read` (layered paper access).

**Pipeline**: theme parsing → multi-source retrieval → dedup → composite scoring → ModelPool LLM rerank → final papers.

**Model pool**: glm-5-turbo(5) → glm-4.7(20) → glm-4.6(20) → deepseek(10) → qwen(10), with automatic failover and cooldown.

**Sources**: OpenAlex, Semantic Scholar, arXiv, DBLP, CrossRef, OpenReview, DeepXiv (optional).

## Quick Start

### 1. Configure `.env`

```bash
SCHOLARTRACE_BIGMODEL_API_KEY=<your-bigmodel-key>
SCHOLARTRACE_ACCESS_TOKEN=g203-mcp

# LAN SSE defaults (scripts set these automatically)
SCHOLARTRACE_MCP_TRANSPORT=sse
SCHOLARTRACE_MCP_HOST=0.0.0.0
SCHOLARTRACE_MCP_PORT=8001
SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true

# Optional: DeepXiv
# SCHOLARTRACE_DEEPXIV_TOKENS=token-a,token-b
```

### 2. Start the server

```bash
./run_scholartrace_mcp_sse.sh        # start
./status_scholartrace_mcp_sse.sh     # check status
./stop_scholartrace_mcp_sse.sh       # stop
```

### 3. Connect from ChatBox

Replace `<server-ip>` with your machine's LAN IP:

```json
{
  "mcpServers": {
    "scholartrace": {
      "url": "http://<server-ip>:8001/sse",
      "headers": {
        "Authorization": "Bearer g203-mcp"
      }
    }
  }
}
```

## MCP Tools

### `query` — Search and Rank

```json
{
  "theme_document": "your research topic description",
  "final_limit": 25,
  "agent_candidate_limit": 200,
  "include_rationale": true
}
```

Returns ranked papers with scores, rationales, and full-text status.

### `read` — Layered Paper Access

```json
{
  "paper_id": "theme-id:paper-id",
  "depth": "fulltext",
  "allow_acquire": true
}
```

Depths: `summary` → `sections` → `fulltext` → `direct_evidence`. Each depth returns honest status about what's available.

## Architecture

```
theme document
    │
    ▼
parse_theme (extract queries + compress summary)
    │
    ▼
fan-out retrieval (6+ sources, per-connector timeout 45s, retry on 429/5xx)
    │
    ▼
dedup → composite scoring (relevance + recency + influence + venue + ...)
    │
    ▼
ModelPool agent rerank (multi-model failover, total timeout 180s)
    │
    ▼
final papers (top 25)
```

Flow diagrams: [`docs/architecture/pipeline_flow.md`](docs/architecture/pipeline_flow.md)

Design docs: [`docs/plans/`](docs/plans/)

## Reliability Features

- **Connector retry**: exponential backoff on 429/5xx/timeout for all 6 sources
- **Per-connector timeout**: each source capped at 45s, won't block others
- **Query retry**: auto-retry when all connectors return empty
- **Overall retrieval timeout**: 300s cap on the entire retrieval stage
- **Model pool failover**: automatic cooldown + next-model fallback on LLM errors
- **Deterministic fallback**: when all LLM models fail, composite scores still return papers

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SCHOLARTRACE_MCP_TRANSPORT` | `stdio` | `sse` for LAN serving |
| `SCHOLARTRACE_MCP_HOST` | `127.0.0.1` | `0.0.0.0` for LAN |
| `SCHOLARTRACE_MCP_PORT` | `8001` | SSE port |
| `SCHOLARTRACE_REMOTE_ACCESS_ENABLED` | `false` | Required for LAN SSE |
| `SCHOLARTRACE_ACCESS_TOKEN` | | Bearer token |
| `SCHOLARTRACE_BIGMODEL_API_KEY` | | Primary LLM API key |
| `SCHOLARTRACE_AGENT_CANDIDATE_LIMIT` | `200` | Papers sent to LLM rerank |
| `SCHOLARTRACE_FINAL_LIMIT` | `25` | Final paper count |
| `SCHOLARTRACE_RETRIEVAL_CONNECTOR_TIMEOUT_SECONDS` | `45` | Per-connector timeout |
| `SCHOLARTRACE_RETRIEVAL_TOTAL_TIMEOUT_SECONDS` | `300` | Overall retrieval timeout |
| `SCHOLARTRACE_AGENT_TOTAL_TIMEOUT_SECONDS` | `180` | LLM rerank timeout |
| `SCHOLARTRACE_DEEPXIV_TOKENS` | | Optional DeepXiv tokens |

## Deployment

### tmux (recommended)

The root-level shell scripts handle `.env` loading, transport defaults, and error checking.

### systemd

See `scripts/scholartrace-mcp.service`. Place secrets in `/etc/scholartrace/scholartrace.env`.

### stdio (debug only)

```bash
SCHOLARTRACE_MCP_TRANSPORT=stdio scholartrace-mcp
```

## License

MIT
