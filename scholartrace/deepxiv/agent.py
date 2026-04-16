"""Lightweight DeepXiv agent for filtering papers via BigModel GLM.

Unlike the full DeepXiv SDK agent (which depends on langgraph + langchain),
this uses direct OpenAI-compatible API calls to the BigModel GLM endpoint
that ScholarTrace already has configured.

The agent takes a list of paper summaries and a research question,
then returns which papers are relevant and why.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a research assistant that filters academic papers for relevance.
Given a research question and a list of papers with their titles and abstracts,
you must select the most relevant papers and explain why they matter.

For each paper, assess:
1. **Relevance** (0-10): How directly does it address the research question?
2. **Novelty** (0-10): Does it introduce new methods, datasets, or insights?
3. **Quality** (0-10): Based on venue, methodology soundness, and results.

Return your analysis as a JSON array. Each element must have:
- "index": the paper index (0-based)
- "selected": true/false
- "relevance": score 0-10
- "novelty": score 0-10
- "quality": score 0-10
- "reason": one sentence explaining why selected or rejected

Only select papers with relevance >= 5. Max 20 papers can be selected.
Return ONLY the JSON array, no other text."""


class DeepXivAgent:
    """Agent that filters papers using GLM LLM.

    Uses the BigModel GLM API (OpenAI-compatible) that ScholarTrace
    already has configured in .env.

    Args:
        api_key: BigModel API key
        base_url: BigModel API base URL
        model: Model name (default: glm-5-turbo)
        max_fulltext: Max number of full texts to fetch for detailed analysis
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
        model: str = "glm-5-turbo",
        max_fulltext: int = 20,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._max_fulltext = max_fulltext
        self._client = httpx.AsyncClient(timeout=120.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def filter_papers(
        self,
        papers: list[dict[str, Any]],
        question: str,
    ) -> list[dict[str, Any]]:
        """Filter papers by relevance to a research question.

        Args:
            papers: List of dicts with at least 'title' and 'abstract'.
            question: The research question or topic.

        Returns:
            Filtered list with added 'agent_score' and 'agent_reason' fields.
        """
        if not papers:
            return []
        if not self._api_key:
            logger.error("DeepXiv agent filtering refused: BigModel API key is not configured")
            return []

        # Step 1: Filter based on title + abstract
        filter_results = await self._call_llm_filter(papers, question)

        # Step 2: For selected papers, optionally get full text for deeper analysis
        selected_indices = [r["index"] for r in filter_results if r.get("selected")]

        # Enrich selected papers with DeepXiv content if they have arxiv_id
        enriched = []
        for idx in selected_indices[:self._max_fulltext]:
            paper = papers[idx]
            enriched.append({
                **paper,
                "agent_score": {
                    "relevance": filter_results[idx].get("relevance", 0),
                    "novelty": filter_results[idx].get("novelty", 0),
                    "quality": filter_results[idx].get("quality", 0),
                },
                "agent_reason": filter_results[idx].get("reason", ""),
            })

        # Sort by combined score
        enriched.sort(
            key=lambda p: (
                p.get("agent_score", {}).get("relevance", 0)
                + p.get("agent_score", {}).get("novelty", 0) * 0.5
                + p.get("agent_score", {}).get("quality", 0) * 0.3
            ),
            reverse=True,
        )

        return enriched

    async def _call_llm_filter(
        self,
        papers: list[dict[str, Any]],
        question: str,
    ) -> list[dict[str, Any]]:
        """Call GLM to filter papers. Returns list of filter result dicts."""
        # Build paper list for the prompt
        paper_lines = []
        for i, p in enumerate(papers):
            title = p.get("title", "Unknown")
            abstract = (p.get("abstract") or "")[:500]
            paper_lines.append(f"[{i}] {title}\nAbstract: {abstract}")

        user_msg = (
            f"Research Question: {question}\n\n"
            f"Papers:\n{chr(10).join(paper_lines)}"
        )

        try:
            resp = await self._client.post(
                self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            # Handle cases where LLM wraps in markdown code blocks
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

            results = json.loads(content)
            if isinstance(results, list):
                return results

            logger.warning("Unexpected LLM response format: %s", type(results))
            return self._default_filter(len(papers))

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error("Failed to parse LLM filter response: %s", e)
            return self._default_filter(len(papers))
        except httpx.HTTPStatusError as e:
            logger.error("GLM API error: %s", e.response.status_code)
            return self._default_filter(len(papers))

    @staticmethod
    def _default_filter(count: int) -> list[dict[str, Any]]:
        """Safe degraded fallback: select nothing when filtering fails."""
        return [
            {
                "index": i,
                "selected": False,
                "relevance": 0,
                "novelty": 0,
                "quality": 0,
                "reason": "Filtering unavailable",
            }
            for i in range(count)
        ]
