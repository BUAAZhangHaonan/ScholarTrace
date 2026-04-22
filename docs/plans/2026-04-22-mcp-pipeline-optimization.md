# MCP Pipeline Optimization Design

**Date**: 2026-04-22
**Status**: Approved
**Scope**: retrieval.py, ranking.py, theme_parser.py, agent.py, config.py, schemas.py

## Problem Statement

The MCP query pipeline has three critical issues:

1. **Agent failure crashes the pipeline**: `strict=True` in MCP path means any Agent failure (timeout, API error, zero selections) propagates as `QueryPipelineRuntimeError` instead of gracefully degrading.

2. **First-stage ranking returns irrelevant papers**: TF-IDF cosine similarity cannot distinguish between semantically different papers that share vocabulary (e.g., "RLHF sycophancy" vs "LLM base model training" both contain "model", "training", "reinforcement").

3. **No topic compression**: Theme parsing relies entirely on frequency-based phrase extraction, missing semantic nuance. Agent receives full research brief text (thousands of tokens) instead of a focused summary.

## Design Decisions

### Approach: A+B Combination

Combines layered degradation (A) with agent-first relaxation (B):

- Layered fallback chain: glm-5-turbo → glm-4-plus → deterministic fallback
- LLM-based topic compression at both query generation and Agent call stages
- Relevance weight boost + title amplification in TF-IDF
- Agent candidate pool relaxed from 100 to 150, batch size from 30 to 40

---

## 1. Layered Degradation Chain

### Flow

```
1. glm-5-turbo (strict=False, 45s timeout)
   ├─ success → return results
   └─ failure →
2. glm-4-plus (strict=False, 45s timeout)
   ├─ success → return results
   └─ failure →
3. _fallback_rerank() deterministic fallback
   └─ return top-k by composite_score
```

### Changes

**`config.py`**:
- Add `bigmodel_fallback_models: str = "glm-4-plus,glm-4-flash"`
- Add `llm_compression_model: str = "glm-4-flash"` (cheaper model for topic compression)

**`agent.py`**:
- `rerank_papers()` default `strict=False` (was `True`)
- New `model_chain` parameter: list of model names to try in order
- Each model in chain attempted once (retries only within `_call_llm_filter_batch` for same model)

**`retrieval.py`**:
- Remove all `QueryPipelineRuntimeError` raises on Agent failure
- Agent failure → try next model in chain → final fallback to `_fallback_rerank()`
- Agent selects zero papers → fallback to `_fallback_rerank()` instead of raising error
- `_fallback_rerank()` uses `deepxiv_agent_fallback_top_k` (20)

---

## 2. LLM Topic Compression

### Stage 1: Query Generation (`theme_parser.py`)

New function `_compress_with_llm(document_text: str) -> str | None`:
- Calls GLM (glm-4-flash, cheaper) with prompt: "Compress this research brief into 1-2 concise search sentences focusing on the core research topic, specific methods, and unique aspects. Output only the sentences, nothing else."
- Result added as first query in `parsed_queries` (highest priority)
- Result stored in new Theme field `compressed_summary: str = ""`
- If LLM call fails, skip silently — existing deterministic queries remain

**`schemas.py`**:
- Add `compressed_summary: str = ""` to Theme model

### Stage 2: Agent Call (`retrieval.py`)

- Agent's `theme_description` changed from `theme.document_text` (full original) to `theme.compressed_summary` (if non-empty)
- Reduces token consumption and improves Agent focus on core topic

---

## 3. Agent Candidate Pool Relaxation

**`config.py`**:
- `agent_candidate_limit: int = 150` (was 100)
- `deepxiv_agent_batch_size: int = 40` (was 30)

**`retrieval.py`**: No code changes needed — already uses config values.

---

## 4. Relevance Weight Restructuring

### Weight Changes

```
Current:  relevance=0.35  recency=0.30  influence=0.10  venue=0.10  fulltext=0.10  source=0.05
New:      relevance=0.45  recency=0.20  influence=0.15  venue=0.10  fulltext=0.05  source=0.05
```

Rationale: Relevance must dominate. Recency at 0.30 was over-promoting new but irrelevant papers.

**`config.py`**: Update default weight values.

### Title Amplification (`ranking.py`)

In `_relevance_scores()`, change paper doc construction:
```python
# Before: title + " " + abstract
# After:  title + " " + title + " " + abstract  (title repeated for 2x weight)
```

### Staleness Penalty Reduction (`ranking.py`)

```python
# Before: composite *= 1.0 - 0.50 * stale_penalty
# After:  composite *= 1.0 - 0.30 * stale_penalty
```

Rationale: High-influence older papers (>1000 citations) should not be excessively penalized.

---

## Files to Modify

| File | Changes |
|------|---------|
| `scholartrace/config.py` | Add `bigmodel_fallback_models`, `llm_compression_model`; update weights and limits |
| `scholartrace/models/schemas.py` | Add `compressed_summary` field to Theme |
| `scholartrace/services/theme_parser.py` | Add `_compress_with_llm()`; integrate into `parse_theme()` |
| `scholartrace/services/ranking.py` | Update weight defaults; title amplification; staleness reduction |
| `scholartrace/services/retrieval.py` | Layered degradation; use compressed summary; remove error raises |
| `scholartrace/deepxiv/agent.py` | `strict=False` default; model chain support |
| `tests/test_ranking.py` | Update weight expectations |
| `tests/test_retrieval.py` | Update mock expectations for degradation chain |
| `tests/test_mcp_server.py` | Update if needed |

## Testing Plan

1. Unit tests for `_compress_with_llm()` with mock LLM
2. Unit tests for layered degradation chain (mock each model failing)
3. Integration test with sycophancy research brief via `examples/mcp_query_simulation.py`
4. Compare results: before vs after optimization quality
