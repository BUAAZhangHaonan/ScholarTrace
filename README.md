# ScholarTrace

**English** | [中文](README_CN.md)

> Theme-guided multi-source scholarly paper discovery and full-text evidence access

## Features

- **Multi-source retrieval** from 7 scholarly databases: OpenAlex, arXiv, Semantic Scholar, DBLP, OpenReview, Crossref, DeepXiv
- **Theme document parsing** — understands full research briefs, not just keywords
- **Multi-key deduplication** — exact ID matching (DOI, arXiv ID, S2 ID, etc.) + fuzzy title matching (threshold 0.85)
- **Multi-objective ranking** — relevance (TF-IDF), recency (exponential decay), influence (log-normalized citations), venue quality, open-access bonus, source agreement
- **Full-text acquisition cascade** — arXiv HTML → arXiv PDF (PyMuPDF) → OA URL → abstract-only fallback
- **Dual interface** — REST API (FastAPI, port 9000) and MCP server (stdio transport, port 8001)
- **BigModel GLM integration** — example script for intelligent literature analysis with `glm-5-turbo`

## Quick Start

```bash
# Create conda environment
conda create -n ScholarTrace python=3.13 -y
conda activate ScholarTrace

# Install with pinned dev constraints
cd ScholarTrace
python -m pip install -r requirements-dev.txt

# Verify imports and pytest collection before running the app
scholartrace-check-env --include-dev --pytest-collect

# Configure API keys for local development
cp .env.example .env
# Edit .env with your API keys

# Run tests (117 tests)
pytest tests/ -v

# Start REST API
scholartrace-api
# -> http://localhost:9000

# Start MCP server over stdio (default)
scholartrace-mcp

# Start MCP over SSE only when you also set an access token
SCHOLARTRACE_MCP_TRANSPORT=sse \
SCHOLARTRACE_ACCESS_TOKEN=change-me \
scholartrace-mcp
```

## Reproducible Environment

Use the pinned dev install path when you need a clean local setup:

```bash
python -m pip install -r requirements-dev.txt
scholartrace-check-env --include-dev --pytest-collect
```

- `constraints-dev.txt` pins the local-friendly package set used for development and test runs.
- `requirements-dev.txt` installs the project in editable mode with dev extras under those constraints.
- `scripts/check_environment.py` and `scholartrace-check-env` validate declared imports from `pyproject.toml` and can run `pytest --collect-only`.

## Configuration (.env)

All settings use the `SCHOLARTRACE_` prefix. Copy `.env.example` to `.env` for local development only. For a deployed service, keep secrets in `/etc/scholartrace/scholartrace.env` or another external env file with restrictive permissions.

| Variable | Required | Default | Description |
|---|---|---|---|
| `SCHOLARTRACE_SEMANTIC_SCHOLAR_API_KEY` | No | | Semantic Scholar API key (higher rate limits) |
| `SCHOLARTRACE_OPENALEX_MAILTO` | No | | Email for OpenAlex polite pool |
| `SCHOLARTRACE_CROSSREF_MAILTO` | No | | Email for Crossref polite pool |
| `SCHOLARTRACE_API_HOST` | No | `127.0.0.1` | REST API bind host |
| `SCHOLARTRACE_API_PORT` | No | `9000` | REST API bind port |
| `SCHOLARTRACE_MCP_HOST` | No | `127.0.0.1` | MCP SSE bind host when `SCHOLARTRACE_MCP_TRANSPORT=sse` |
| `SCHOLARTRACE_MCP_PORT` | No | `8001` | MCP SSE bind port |
| `SCHOLARTRACE_MCP_TRANSPORT` | No | `stdio` | MCP transport (`stdio` or `sse`) |
| `SCHOLARTRACE_REMOTE_ACCESS_ENABLED` | No | `false` | Must be `true` before binding REST or MCP SSE beyond loopback |
| `SCHOLARTRACE_ACCESS_TOKEN` | Remote only | | Shared bearer token for REST and MCP SSE |
| `SCHOLARTRACE_MAX_RESULTS_PER_SOURCE_PER_QUERY` | No | `200` | Results per source per query |
| `SCHOLARTRACE_TARGET_CANDIDATE_POOL` | No | `500` | Target total candidate papers |
| `SCHOLARTRACE_MAX_FULLTEXT_DOWNLOADS` | No | `50` | Maximum full-text downloads per retrieval |
| `SCHOLARTRACE_BIGMODEL_API_KEY` | Example only | | BigModel GLM API key used by `examples/glm_scholar_search.py` |
| `SCHOLARTRACE_BIGMODEL_BASE_URL` | No | `https://open.bigmodel.cn/api/coding/paas/v4/chat/completions` | BigModel GLM API endpoint |
| `SCHOLARTRACE_BIGMODEL_MODEL` | No | `glm-5-turbo` | BigModel GLM model name |
| `SCHOLARTRACE_DEEPXIV_TOKENS` | DeepXiv only | | Comma-separated DeepXiv tokens |
| `SCHOLARTRACE_DEEPXIV_AUTO_REGISTER` | No | `false` | Opt-in auto-registration for DeepXiv |
| `SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET` | Auto-register only | | DeepXiv SDK secret used only when auto-register is enabled |

### Network Exposure Defaults

- REST binds to `127.0.0.1` by default.
- MCP defaults to `stdio`; SSE is opt-in.
- Remote REST or MCP SSE startup is rejected unless `SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true` and `SCHOLARTRACE_ACCESS_TOKEN` is set.
- Client-facing REST and MCP errors use a stable safe shape like `{"error":{"code":"not_found","message":"...","retryable":false}}`.

### Ranking Weights (configurable)

| Component | Default Weight | Method |
|---|---|---|
| Relevance | 0.35 | TF-IDF cosine similarity |
| Recency | 0.20 | Exponential decay (half-life 2 years) |
| Influence | 0.20 | Log-normalized citation count |
| Venue | 0.10 | Tiered venue scoring |
| Fulltext | 0.10 | Open-access availability bonus |
| Source Agreement | 0.05 | Papers found by multiple sources |

## REST API Endpoints

```
GET  /health                          — Health check
POST /themes                          — Create theme from text
POST /retrieval/jobs                  — Launch retrieval (background)
GET  /retrieval/jobs/{job_id}         — Job status
GET  /themes/{theme_id}/papers        — Ranked papers (paginated, ?limit=N)
GET  /papers/{paper_id}              — Paper metadata
GET  /papers/{paper_id}/sections     — Section-level content
GET  /papers/{paper_id}/fulltext     — Full-text status and cached content
GET  /themes/{theme_id}/export       — Export (JSON/Markdown)

# DeepXiv endpoints
POST /deepxiv/search                  — Search arXiv via DeepXiv (BM25+vector)
GET  /deepxiv/papers/{arxiv_id}/summary     — Paper summary & TLDR
GET  /deepxiv/papers/{arxiv_id}/fulltext    — Full text from DeepXiv
GET  /deepxiv/papers/{arxiv_id}/sections/{name} — Specific section
POST /deepxiv/agent/filter            — Agent-filtered search (GLM scoring)
```

## MCP Server

The MCP server provides 12 tools for LLM agent integration. The default transport is `stdio`. SSE is available only when you opt in with `SCHOLARTRACE_MCP_TRANSPORT=sse` and set `SCHOLARTRACE_ACCESS_TOKEN`.

| # | Tool | Description |
|---|---|---|
| 1 | `search_papers_by_theme` | Full pipeline: parse theme → retrieve → rank → return top 10 |
| 2 | `get_ranked_papers` | Get ranked papers for a stored theme |
| 3 | `get_paper_metadata` | Full paper metadata by ID |
| 4 | `get_paper_sections` | Section-level content extraction |
| 5 | `get_paper_fulltext` | Full text (triggers download cascade) |
| 6 | `get_related_papers` | Related papers by shared venue and year |
| 7 | `export_theme_report` | Export full report as JSON or Markdown |
| 8 | `deepxiv_search` | Search arXiv via DeepXiv (hybrid BM25 + vector) |
| 9 | `deepxiv_paper_summary` | Paper metadata & TLDR from DeepXiv |
| 10 | `deepxiv_paper_fulltext` | Full paper text (markdown) from DeepXiv |
| 11 | `deepxiv_paper_section` | Specific section content from DeepXiv |
| 12 | `deepxiv_agent_filter` | Agent-filtered search: GLM scores & filters papers |

### MCP Client Configuration

`stdio` is the default and recommended local setup. Use SSE only when you need a long-running network endpoint and have an access token in place.

**Claude Desktop** — add to `claude_desktop_config.json` (local machine):

```json
{
  "mcpServers": {
    "scholartrace": {
      "command": "conda",
      "args": ["run", "-n", "ScholarTrace", "scholartrace-mcp"]
    }
  }
}
```

**LAN / Remote access** — enable SSE explicitly and require a bearer token:

```bash
SCHOLARTRACE_MCP_TRANSPORT=sse \
SCHOLARTRACE_MCP_HOST=0.0.0.0 \
SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true \
SCHOLARTRACE_ACCESS_TOKEN=change-me \
scholartrace-mcp
```

Then connect via SSE URL:

```json
{
  "mcpServers": {
    "scholartrace": {
      "url": "http://127.0.0.1:8001/sse",
      "headers": {
        "Authorization": "Bearer change-me"
      }
    }
  }
}
```

## Python API Usage

```python
import httpx

client = httpx.Client(timeout=30)

# 1. Create theme from a research brief
resp = client.post("http://localhost:9000/themes",
                   data={"text": "RLHF sycophancy in language models..."})
theme = resp.json()

# 2. Launch retrieval job
job = client.post("http://localhost:9000/retrieval/jobs",
                  data={"theme_id": theme["id"]}).json()

# 3. Poll for completion
import time
while True:
    status = client.get(f"http://localhost:9000/retrieval/jobs/{job['id']}").json()
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(2)

# 4. Get ranked papers (default limit 50)
papers = client.get(f"http://localhost:9000/themes/{theme['id']}/papers",
                    params={"limit": 50}).json()

for p in papers[:10]:
    print(f"[{p['composite_score']:.3f}] {p['title']} ({p['year']})")
```

## MCP Call Example

```python
import asyncio, json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def search_papers():
    server_params = StdioServerParameters(
        command="conda",
        args=["run", "-n", "ScholarTrace", "scholartrace-mcp"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()

            # Search papers
            result = await session.call_tool("search_papers_by_theme", {
                "theme_document": "RLHF sycophancy and affective hallucination..."
            })
            data = json.loads(result.content[0].text)
            print(f"Found {data['total_papers']} papers")

            # Get ranked papers
            result = await session.call_tool("get_ranked_papers", {
                "theme_id": data["theme_id"],
                "limit": 50
            })
            papers = json.loads(result.content[0].text)

            # Get full text for a specific paper
            result = await session.call_tool("get_paper_fulltext", {
                "paper_id": papers[0]["id"]
            })

asyncio.run(search_papers())
```

## BigModel GLM Integration

`examples/glm_scholar_search.py` combines ScholarTrace retrieval with BigModel GLM (`glm-5-turbo`) for intelligent literature analysis:

```bash
# Default: search with the bundled sycophancy research brief
export SCHOLARTRACE_BIGMODEL_API_KEY=your-key
python examples/glm_scholar_search.py

# Custom query
python examples/glm_scholar_search.py --query "your research topic"

# Adjust paper count
python examples/glm_scholar_search.py --limit 100

# Interactive chat mode (ask follow-up questions about papers)
python examples/glm_scholar_search.py --interactive
```

The script workflow:
1. Creates a theme and launches retrieval via the REST API
2. Fetches top N ranked papers
3. Sends paper metadata to BigModel GLM for landscape analysis
4. (Optional) Enters interactive mode for follow-up questions
5. Exports results to `scholartrace_results.json`

The example now fails closed if `SCHOLARTRACE_BIGMODEL_API_KEY` is missing instead of sending a baked-in fallback credential.

## Architecture

```
Theme Document → Theme Parser → Multi-Source Fan-Out → Dedup → Rank → Fulltext → Storage
         │                                       │          │        │          │
    parsed_queries     ┌─────────────────┐   Union-Find  Multi-obj  Cascade   SQLite
    parsed_topics      │ OpenAlex        │   + rapidfuzz  scoring   BS4/PyMuPDF
    parsed_methods     │ arXiv           │
    parsed_datasets    │ Semantic Scholar │
                       │ DBLP            │
                       │ OpenReview      │
                       │ Crossref        │
                       │ DeepXiv (BM25+V)│
                       └─────────────────┘
                                                    REST API (FastAPI) + MCP Server
```

**Data flow:**

1. A theme document (research brief) is parsed into structured queries (7-8 per theme).
2. Queries fan out to all 6 source connectors in parallel (async httpx).
3. Raw candidates are deduplicated: exact ID matching (DOI, arXiv ID, S2 ID, etc.) first, then fuzzy title matching (rapidfuzz token_sort_ratio ≥ 0.85).
4. Papers are ranked by composite score across 6 weighted dimensions.
5. Full-text content is acquired through a cascade: arXiv HTML (BeautifulSoup) → arXiv PDF (PyMuPDF) → OA URL → abstract-only fallback.
6. Everything is stored in SQLite (WAL mode) with filesystem artifacts.

### About Paper Counts

The number of retrieved papers depends on the **input theme document**. A longer, more detailed research brief produces more parsed queries, which fetch more candidates across sources. After deduplication, this typically yields 1500-3000 unique papers. A short query string produces fewer queries and proportionally fewer results.

## Auto-Start on Boot (systemd)

A systemd service file is provided at `scripts/scholartrace-mcp.service`:

```bash
# Install
sudo cp scripts/scholartrace-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable scholartrace-mcp
sudo systemctl start scholartrace-mcp

# Check status
sudo systemctl status scholartrace-mcp
```

The service runs the MCP server persistently and restarts automatically on failure.

## Examples & Scripts

| Path | Description |
|---|---|
| `docs/examples/sycophancy_affective_hallucination_research_brief.md` | Example theme document |
| `examples/glm_scholar_search.py` | GLM-powered intelligent literature search |
| `scripts/scholartrace-mcp.service` | systemd unit file for MCP auto-start |
| `scripts/verify_scholartrace.py` | End-to-end verification script |

## Tested With

- Python 3.13, conda environment `ScholarTrace`
- 117 unit + integration tests (`pytest tests/ -v`)
- E2E verification: 100+ papers from multi-source retrieval

## License

MIT
