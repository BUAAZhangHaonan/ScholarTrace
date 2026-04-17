# MCP Product Reshape Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify ScholarTrace MCP to two public tools, `query` and `read`, while making DeepXiv Agent reranking part of the default query flow and rewriting the docs around LAN SSE for ChatBox.

**Architecture:** Keep REST and the internal service layer broad. Reshape only the MCP product surface. Reuse the current retrieval, full-text, prompt-budget, and runtime-hardening layers, then add agent reranking and layered read payloads on top.

**Tech Stack:** Python 3.13, FastMCP, FastAPI, httpx, Pydantic, SQLite, PyMuPDF, BeautifulSoup, pytest, pytest-asyncio

---

### Task 1: Add the product-shape docs and lock the planned nodes

**Files:**
- Create: `docs/plans/2026-04-17-mcp-product-reshape-design.md`
- Create: `docs/plans/2026-04-17-mcp-product-reshape.md`

**Step 1: Write the design doc**

Write the approved MCP-only product design into `docs/plans/2026-04-17-mcp-product-reshape-design.md`.

**Step 2: Write the implementation plan**

Write this plan into `docs/plans/2026-04-17-mcp-product-reshape.md`.

**Step 3: Verify the files exist**

Run: `ls docs/plans`
Expected: both new `2026-04-17` plan files are listed

**Step 4: Commit**

Run:

```bash
git add docs/plans/2026-04-17-mcp-product-reshape-design.md docs/plans/2026-04-17-mcp-product-reshape.md
git commit -m "docs: plan mcp product reshape"
git push origin master
```

### Task 2: Write failing tests for the new two-tool MCP contract

**Files:**
- Modify: `tests/test_mcp_server.py`
- Modify: `tests/test_docs_readme.py`
- Modify: `tests/test_fulltext_contract.py`

**Step 1: Add failing MCP tests for `query`**

Cover:

- default `final_limit=20`
- configurable `agent_candidate_limit`
- configurable `final_limit`
- response counts and final paper summaries

**Step 2: Add failing MCP tests for `read`**

Cover:

- `summary`
- `sections`
- `fulltext_status`
- `fulltext`
- `allow_acquire=true`
- `direct_evidence`
- honest missing/fulltext-unavailable payloads

**Step 3: Update failing README contract tests**

Change the expectations from 13 tools to 2 tools and from `stdio`-first to SSE-over-LAN-first.

**Step 4: Run the focused tests and verify they fail for the right reasons**

Run:

```bash
pytest tests/test_mcp_server.py tests/test_docs_readme.py tests/test_fulltext_contract.py -q
```

Expected: failures mention missing `query` / `read` behavior or stale README contract text

### Task 3: Persist agent rerank state on `Work`

**Files:**
- Modify: `scholartrace/models/schemas.py`
- Modify: `scholartrace/services/storage.py`

**Step 1: Write the failing storage test**

Add a test that saves and reloads a `Work` with:

- `agent_score`
- `agent_rank`
- `agent_rationale`

**Step 2: Run the focused test to verify failure**

Run: `pytest tests/test_storage.py -q`
Expected: fail because the fields are not stored yet

**Step 3: Add the minimal schema and storage changes**

- extend `Work`
- add DB columns
- map row read/write logic
- keep existing merge behavior safe

**Step 4: Run the focused test to verify pass**

Run: `pytest tests/test_storage.py -q`
Expected: pass

### Task 4: Add second-stage DeepXiv Agent reranking to the default retrieval path

**Files:**
- Modify: `scholartrace/config.py`
- Modify: `scholartrace/services/retrieval.py`
- Modify: `scholartrace/deepxiv/agent.py`
- Modify: `scholartrace/api/payloads.py`
- Modify: `tests/test_retrieval.py`

**Step 1: Write the failing retrieval tests**

Cover:

- default rerank path uses `agent_candidate_limit=100`
- default final output uses `final_limit=20`
- configurable `coarse_pool_limit`
- agent rerank preserves the runtime budget behavior and batches safely

**Step 2: Run the focused test to verify failure**

Run: `pytest tests/test_retrieval.py -q`
Expected: fail because the default retrieval path has no agent rerank

**Step 3: Implement the minimal rerank changes**

- add retrieval options/config knobs
- rank first-stage results
- slice the coarse pool and agent candidate pool
- call DeepXiv Agent on title + abstract
- persist `agent_score`, `agent_rank`, and `agent_rationale`
- return final results capped by `final_limit`

**Step 4: Run the focused test to verify pass**

Run: `pytest tests/test_retrieval.py -q`
Expected: pass

### Task 5: Collapse MCP to `query` and `read`

**Files:**
- Modify: `scholartrace/api/mcp_server.py`
- Modify: `scholartrace/api/payloads.py`
- Modify: `tests/test_mcp_server.py`

**Step 1: Write the failing MCP behavior tests if anything is still missing**

Cover:

- only `query` and `read` are public tools
- `query` returns the final reranked paper summaries
- `read` returns the right depth payloads

**Step 2: Run the focused tests to verify failure**

Run: `pytest tests/test_mcp_server.py -q`
Expected: fail because the old tool surface is still public

**Step 3: Implement the minimal MCP surface change**

- remove public registration from old MCP tool entry points
- add public `query`
- add public `read`
- keep runtime-limit enforcement
- keep REST unchanged

**Step 4: Run the focused tests to verify pass**

Run: `pytest tests/test_mcp_server.py -q`
Expected: pass

**Step 5: Commit**

Run:

```bash
git add scholartrace/config.py scholartrace/models/schemas.py scholartrace/services/storage.py scholartrace/deepxiv/agent.py scholartrace/services/retrieval.py scholartrace/api/payloads.py scholartrace/api/mcp_server.py tests/test_mcp_server.py tests/test_retrieval.py tests/test_storage.py tests/test_fulltext_contract.py
git commit -m "feat: simplify mcp query and read flow"
git push origin master
```

### Task 6: Rewrite the READMEs around LAN SSE and ChatBox

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `scripts/scholartrace-mcp.service`
- Modify: `tests/test_docs_readme.py`

**Step 1: Write the failing README tests if anything is still missing**

Cover:

- LAN SSE first
- token example `g203-mcp`
- only `query` and `read`
- built-in DeepXiv Agent rerank
- layered `read`
- honest full-text story
- ChatBox JSON example
- `stdio` only as debug mode

**Step 2: Run the focused docs tests to verify failure**

Run: `pytest tests/test_docs_readme.py -q`
Expected: fail because the old README story still leads with `stdio` and 13 tools

**Step 3: Rewrite the docs**

- move LAN SSE to the main quickstart
- show `Authorization: Bearer g203-mcp`
- include paste-ready ChatBox JSON
- keep `stdio` as a short local-debug note
- state exactly what full-text acquisition works today and what may fail

**Step 4: Run the focused docs tests to verify pass**

Run: `pytest tests/test_docs_readme.py -q`
Expected: pass

**Step 5: Commit**

Run:

```bash
git add README.md README_CN.md scripts/scholartrace-mcp.service tests/test_docs_readme.py
git commit -m "docs: reshape scholartrace mcp lan sse story"
git push origin master
```

### Task 7: Final verification

**Files:**
- Verify only

**Step 1: Run the focused suite**

Run:

```bash
pytest tests/test_mcp_server.py tests/test_retrieval.py tests/test_storage.py tests/test_fulltext_contract.py tests/test_docs_readme.py -q
```

Expected: pass

**Step 2: Run the full suite**

Run:

```bash
pytest -q
```

Expected: pass

**Step 3: Verify the public MCP surface**

Run:

```bash
python - <<'PY'
import asyncio
from scholartrace.api.mcp_server import mcp

async def main():
    tools = await mcp.list_tools()
    print(sorted(tool.name for tool in tools))

asyncio.run(main())
PY
```

Expected: only `['query', 'read']`

**Step 4: Verify git state**

Run:

```bash
git status --short
git log --oneline -3
```

Expected: only the unrelated pre-existing local change remains, and the new commits are present on `master`
