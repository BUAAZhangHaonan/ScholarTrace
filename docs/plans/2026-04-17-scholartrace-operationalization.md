# ScholarTrace Operationalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep ScholarTrace on the current two-tool MCP shape and make the LAN SSE deployment operational, honest, and easy to run with tmux-backed startup scripts, explicit environment handling, and docs that match the real DeepXiv and BigModel behavior.

**Architecture:** Leave the REST API broad and leave the MCP product surface at `query` and `read`. Build this pass around operational wrappers and documentation. Reuse the current hardened runtime checks, the built-in `glm-5-turbo` rerank path, and the existing optional DeepXiv retrieval and explicit full-text acquisition flow.

**Tech Stack:** Python 3.13, FastMCP, FastAPI, uvicorn, tmux, bash, Pydantic Settings, SQLite, pytest

---

### Task 1: Lock the operationalization plan in the repo

**Files:**
- Create: `docs/plans/2026-04-17-scholartrace-operationalization.md`

**Step 1: Write the implementation plan**

Write this plan into `docs/plans/2026-04-17-scholartrace-operationalization.md`.

**Step 2: Verify the file exists**

Run:

```bash
ls docs/plans
```

Expected: `2026-04-17-scholartrace-operationalization.md` is listed.

**Step 3: Commit**

Run:

```bash
git add docs/plans/2026-04-17-scholartrace-operationalization.md
git commit -m "docs: plan scholartrace operationalization"
git push origin master
```

### Task 2: Add tmux-backed LAN SSE operational scripts

**Files:**
- Create: `run_scholartrace_mcp_sse.sh`
- Create: `stop_scholartrace_mcp_sse.sh`
- Create: `status_scholartrace_mcp_sse.sh`

**Step 1: Write failing operational tests**

Add focused tests for the script contract:

- repo-root `.env` is loaded automatically when present
- default SSE variables are set when missing
- missing `SCHOLARTRACE_BIGMODEL_API_KEY` fails clearly
- missing DeepXiv config does not block startup
- `SCHOLARTRACE_DEEPXIV_AUTO_REGISTER=true` without `SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET` fails clearly
- status output shows tmux session name, LAN URL, and bearer token header

**Step 2: Run the focused tests and verify failure**

Run:

```bash
pytest tests/test_ops_scripts.py -q
```

Expected: failures point to missing scripts or missing script behavior.

**Step 3: Implement the scripts**

Build the scripts with these rules:

- use `tmux` so the SSE server survives terminal disconnect
- load repo-root `.env` automatically when it exists
- default to:
  - `SCHOLARTRACE_MCP_TRANSPORT=sse`
  - `SCHOLARTRACE_MCP_HOST=0.0.0.0`
  - `SCHOLARTRACE_MCP_PORT=8001`
  - `SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true`
  - `SCHOLARTRACE_ACCESS_TOKEN=g203-mcp`
- print the real LAN URL `http://10.134.132.166:8001/sse`
- print `Authorization: Bearer g203-mcp`
- print basic `tmux` and socket verification commands
- fail fast if `SCHOLARTRACE_BIGMODEL_API_KEY` is still missing after `.env` loading
- fail fast if auto-register is enabled without the SDK secret
- do not fail if DeepXiv is otherwise unconfigured; print that DeepXiv retrieval and evidence will be skipped

**Step 4: Run the focused tests and verify pass**

Run:

```bash
pytest tests/test_ops_scripts.py -q
```

Expected: pass.

**Step 5: Run live operational verification**

Run:

```bash
./run_scholartrace_mcp_sse.sh
./status_scholartrace_mcp_sse.sh
./stop_scholartrace_mcp_sse.sh
```

Expected:

- start launches a tmux session
- status reports the running session and SSE URL
- stop terminates the tmux session cleanly

**Step 6: Commit**

Run:

```bash
git add run_scholartrace_mcp_sse.sh stop_scholartrace_mcp_sse.sh status_scholartrace_mcp_sse.sh tests/test_ops_scripts.py
git commit -m "ops: add background sse startup scripts and env loading"
git push origin master
```

### Task 3: Document optional DeepXiv and environment-based key handling honestly

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `tests/test_docs_readme.py`

**Step 1: Write failing docs tests**

Add or update docs assertions for:

- DeepXiv is optional
- ScholarTrace still runs without DeepXiv
- query still works and built-in `glm-5-turbo` rerank still works without DeepXiv
- direct DeepXiv evidence and DeepXiv markdown fallback may be unavailable without DeepXiv
- auto-register requires `SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET`
- the SDK secret cannot be self-discovered by code
- MCP clients do not pass `SCHOLARTRACE_BIGMODEL_API_KEY` in request parameters
- runtime loads `SCHOLARTRACE_BIGMODEL_API_KEY` from environment

**Step 2: Run the focused docs tests and verify failure**

Run:

```bash
pytest tests/test_docs_readme.py -q
```

Expected: failures point to missing honesty and key-handling statements.

**Step 3: Rewrite the docs**

Update both READMEs so they say clearly:

- public MCP surface remains `query` and `read`
- `query` still performs built-in `glm-5-turbo` second-stage reranking
- `read` stays layered and honest
- DeepXiv is optional for retrieval and evidence
- auto-register can generate usernames and emails, but still requires a real SDK secret from prior DeepXiv setup
- BigModel API key is loaded from environment or `.env`, not from MCP request parameters

**Step 4: Run the focused docs tests and verify pass**

Run:

```bash
pytest tests/test_docs_readme.py -q
```

Expected: pass.

### Task 4: Finalize tmux-first LAN SSE docs and keep systemd secondary

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `scripts/scholartrace-mcp.service`
- Modify: `tests/test_docs_readme.py`

**Step 1: Add the tmux workflow to the docs**

Document:

- `./run_scholartrace_mcp_sse.sh`
- `./status_scholartrace_mcp_sse.sh`
- `./stop_scholartrace_mcp_sse.sh`
- how to inspect tmux
- how to check the SSE port is listening

**Step 2: Add the exact LAN example**

Use:

- server LAN IP `10.134.132.166`
- token `g203-mcp`
- ChatBox JSON:

```json
{
  "name": "ScholarTrace LAN",
  "type": "sse",
  "url": "http://10.134.132.166:8001/sse",
  "headers": {
    "Authorization": "Bearer g203-mcp"
  }
}
```

**Step 3: Keep systemd as a secondary managed option**

Document `scripts/scholartrace-mcp.service` as the managed deployment example, but keep the quick tmux workflow first.

**Step 4: Run the focused docs tests and verify pass**

Run:

```bash
pytest tests/test_docs_readme.py -q
```

Expected: pass.

**Step 5: Commit**

Run:

```bash
git add README.md README_CN.md scripts/scholartrace-mcp.service tests/test_docs_readme.py
git commit -m "docs: finalize lan sse and chatbox setup guidance"
git push origin master
```

### Task 5: Run full verification and confirm the two-tool MCP contract still holds

**Files:**
- Modify if needed: `tests/test_mcp_server.py`
- Modify if needed: `tests/test_retrieval.py`
- Modify if needed: `tests/test_deepxiv_runtime.py`
- Modify if needed: `tests/test_fulltext_contract.py`

**Step 1: Run the focused operational checks**

Run:

```bash
pytest tests/test_ops_scripts.py tests/test_docs_readme.py tests/test_deepxiv_runtime.py -q
```

Expected: pass.

**Step 2: Run the MCP contract checks**

Run:

```bash
pytest tests/test_mcp_server.py tests/test_retrieval.py tests/test_fulltext_contract.py -q
```

Expected: pass.

**Step 3: Run the full suite**

Run:

```bash
pytest -q
```

Expected: pass with the current suite count, aside from any existing intentional skips.

**Step 4: Manual spot checks**

Verify:

- `query` and `read` are still the only public MCP tools
- `query` still uses built-in `glm-5-turbo` reranking
- `read` still reports full-text availability honestly
- LAN SSE remains the main documented deployment mode
- `stdio` remains a local debug note only

