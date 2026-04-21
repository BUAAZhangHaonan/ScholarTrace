# ScholarTrace MCP Auth Simulation Summary

## Scope

This document summarizes a real end-to-end MCP simulation run against the `query` and `read` tools with bearer authentication.

- Theme input: `docs/examples/sycophancy_affective_hallucination_research_brief.md`
- Simulation script: `examples/mcp_query_simulation.py`
- Output artifact: `docs/mcp_query_simulation_result.json`

## Run Configuration

- MCP transport: SSE
- Endpoint: `http://127.0.0.1:8001/sse`
- Authorization header: `Authorization: Bearer g203-mcp`
- Query arguments:
  - `final_limit = 10`
  - `agent_candidate_limit = 40`
  - `coarse_pool_limit = 120`
  - `include_rationale = true`
- Query retries in final successful run: `0`

Run metadata captured in JSON:

- `timestamp`: `2026-04-21T14:59:41.491415+00:00`
- `tools`: `['query', 'read']`
- `query_attempts`: `[{"attempt": 1, "agent_candidate_limit": 40, "coarse_pool_limit": 120}]`

## Verification Outcome

- `query` call succeeded.
- Returned final papers: `10`.
- `read` call succeeded for the first paper ID.
- `query_result.error`: `None`.

## Top 10 Papers Returned

| Rank | Year | Venue | Agent Score | Title |
|---|---:|---|---:|---|
| 1 | 2024 | arXiv.org | 0.0000 | Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution |
| 2 | 2026 | Science | 0.0000 | Sycophantic AI decreases prosocial intentions and promotes dependence |
| 3 | 2023 | arXiv.org | 0.4144 | Qwen Technical Report |
| 4 | 2025 | arXiv.org | 0.4058 | Qwen-Image Technical Report |
| 5 | 2023 | arXiv.org | 0.3848 | Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities |
| 6 | 2023 | None | 0.3761 | Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond |
| 7 | 2026 | AAAI | 0.0000 | Detecting Emotional Dynamic Trajectories: An Evaluation Framework for Emotional Support in Language Models. |
| 8 | 2025 | arXiv.org | 0.3667 | Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model |
| 9 | 2026 | arXiv.org | 0.3651 | LTX-2: Efficient Joint Audio-Visual Foundation Model |
| 10 | 2023 | arXiv.org | 0.0000 | Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models |

## Notes

- The run demonstrates that authenticated MCP calls can return 10 final papers and execute a follow-up `read` call successfully.
- Some returned papers are highly recent and high-visibility (for example Science/AAAI/arXiv technical reports), but thematic alignment varies because the brief also contains model-family and multimodal context.
- If stricter topical purity is needed, further narrowing the query text or adding explicit exclusion constraints in the theme document is recommended.
