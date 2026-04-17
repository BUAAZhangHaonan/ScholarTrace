# ScholarTrace MCP Product Reshape Design

## Goal

Turn ScholarTrace into a clean LAN-served MCP product for team use in ChatBox.
The public MCP surface becomes:

- `query`
- `read`

REST stays broad for now. Internal services stay broad too. This pass changes the MCP product shape, not the whole backend shape.

## Product Decision

### Keep

- Unified multi-source retrieval across the current scholarly connectors
- DeepXiv as an optional source in first-stage retrieval when configured
- Existing runtime hardening, request budgets, rate limits, SSE auth, and full-text negative cache
- Explicit full-text acquisition with honest state reporting
- `stdio` transport as a local debugging mode

### Change

- Remove the 13-tool MCP product story
- Expose only `query` and `read` as public MCP tools
- Move DeepXiv Agent reranking into the default `query` path
- Make LAN SSE the main documented deployment mode
- Rewrite both READMEs around ChatBox-over-LAN usage

## Query Design

### Public interface

`query(theme_document, final_limit=20, agent_candidate_limit=100, coarse_pool_limit=None, include_rationale=True)`

### Default flow

1. Parse the theme document into a `Theme`
2. Run unified retrieval across configured sources
3. Deduplicate raw candidates
4. Convert to `Work`
5. Apply first-stage composite ranking
6. Keep a coarse pool for agent reranking
7. Run DeepXiv Agent second-stage reranking on title + abstract only
8. Return the final selected papers

### Behavioral rules

- DeepXiv Agent is part of the normal `query` workflow, not a separate manual MCP step
- `agent_candidate_limit` defaults to `100`
- `final_limit` defaults to `20`
- If the client asks for more results, return that many when possible
- The final response must include enough summary detail for the MCP client AI to reason over the results without a separate theme-summary tool

### Response shape

Return a single payload that includes pipeline counts and final paper summaries:

- `theme_id`
- `total_retrieved`
- `total_after_dedup`
- `total_after_first_stage`
- `total_agent_candidates`
- `total_final`
- `papers`

Each paper summary should include:

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

## Read Design

### Public interface

`read(paper_id, depth="summary", allow_acquire=False)`

### Supported depths

- `summary`
- `sections`
- `fulltext_status`
- `fulltext`
- `direct_evidence`

### Behavioral rules

- `summary` returns metadata, abstract, ranking state, agent state, and honest full-text status
- `sections` returns cached sections when present
- `fulltext_status` returns the explicit cache and acquisition state
- `fulltext` returns cached full text when present; if missing and `allow_acquire=true`, it triggers the current explicit acquire path and then returns the resulting state
- `direct_evidence` exposes DeepXiv-backed direct evidence when applicable, but only through `read`

## Honest Full-Text Story

The current explicit acquisition path is real and should be preserved.
It currently attempts, in order:

1. arXiv HTML
2. arXiv PDF
3. metadata `pdf_url`
4. metadata `oa_url` or `html_url`
5. DeepXiv markdown fallback for arXiv-backed papers when DeepXiv is configured

### Confirmed working paths

- arXiv HTML fetch plus heading-based section parsing
- arXiv PDF fetch plus plain-text extraction
- metadata PDF or HTML fetch
- DeepXiv markdown fetch plus heading-based section parsing
- explicit negative-cache state when acquisition fails

### Known limits that docs and API must state clearly

- New-paper full-text retrieval is not guaranteed
- PDF parsing is mostly plain-text extraction, not strong structure recovery
- No OCR for scanned or image-only PDFs
- HTML and markdown section parsing are simple heading-based splits
- Retrieval does not auto-acquire full text during the search pipeline

## Internal Changes

### MCP layer

- Replace the 13 public `@mcp.tool()` functions with only `query` and `read`
- Keep helper functions private inside the module as needed
- Keep runtime-limit enforcement on the public tools

### Retrieval layer

- Add second-stage agent reranking after first-stage ranking
- Reuse the existing prompt budgeting so every GLM request stays under the 128K context window
- Batch safely; do not impose a new global total-token cap

### Persistence

- Extend stored `Work` state so later `read` calls can return agent score, agent rank, and rationale
- Keep the current full-text state model and negative-cache behavior

## Deployment Story

### Primary mode

LAN SSE is the primary deployment mode for shared team use in ChatBox.

Required example:

- `SCHOLARTRACE_MCP_TRANSPORT=sse`
- `SCHOLARTRACE_MCP_HOST=0.0.0.0`
- `SCHOLARTRACE_MCP_PORT=8001`
- `SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true`
- `SCHOLARTRACE_ACCESS_TOKEN=g203-mcp`

### Client expectation

The token is user-defined, not auto-generated.
The client must send:

- `Authorization: Bearer g203-mcp`

### Secondary mode

`stdio` remains available only as a local debugging/development mode.

## README Rewrite Targets

Both `README.md` and `README_CN.md` must:

- Document only two public MCP tools
- Show the built-in DeepXiv Agent second-stage rerank in the default `query` flow
- Explain the layered `read` model
- Explain the honest full-text capability story
- Lead with LAN SSE, not `stdio`
- Use `SCHOLARTRACE_ACCESS_TOKEN=g203-mcp` in examples
- Include a paste-ready ChatBox JSON example
- Keep `stdio` only as a short debug note

## Planned Commit Nodes

1. Add the design doc and implementation plan doc
2. Implement MCP `query` and `read`, plus default agent reranking and layered read behavior
3. Rewrite both READMEs and update contract tests to match the final product surface
