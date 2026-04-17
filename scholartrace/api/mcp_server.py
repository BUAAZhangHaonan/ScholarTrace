"""MCP server for the public ScholarTrace product surface."""

from __future__ import annotations

import json
import logging
from functools import wraps
from typing import Any

from mcp.server.fastmcp import FastMCP

from scholartrace.api.contracts import tool_error_json
from scholartrace.api.payloads import deepxiv_summary_payload, public_work_payload
from scholartrace.api.security import AccessTokenMiddleware
from scholartrace.config import Settings, get_settings
from scholartrace.services import runtime_limits
from scholartrace.services.storage import StorageService

logger = logging.getLogger(__name__)


def create_mcp(settings: Settings | None = None) -> FastMCP:
    runtime_settings = settings or get_settings()
    return FastMCP(
        "ScholarTrace",
        host=runtime_settings.mcp_host,
        port=runtime_settings.mcp_port,
    )


def create_mcp_sse_app(settings: Settings | None = None):
    runtime_settings = settings or get_settings()
    app = mcp.sse_app()
    if runtime_settings.access_token:
        return AccessTokenMiddleware(app, runtime_settings.access_token)
    return app


_settings = get_settings()
mcp = create_mcp(_settings)
_storage: StorageService | None = None
_deepxiv_connector: Any | None = None


def _get_settings() -> Settings:
    global _settings
    return _settings


def _get_storage() -> StorageService:
    global _storage
    if _storage is None:
        settings = get_settings()
        _storage = StorageService(db_path=settings.db_path)
        _storage.init_db()
    return _storage


def set_storage(storage: StorageService) -> None:
    global _storage
    _storage = storage


async def _get_deepxiv() -> Any:
    global _deepxiv_connector
    if _deepxiv_connector is None:
        from scholartrace.connectors.deepxiv_connector import DeepXivConnector

        _deepxiv_connector = DeepXivConnector(settings=_get_settings())
    return _deepxiv_connector


def safe_tool(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except runtime_limits.RateLimitExceeded as exc:
            return tool_error_json("rate_limited", str(exc), retryable=True)
        except Exception:
            logger.exception("MCP tool %s failed", func.__name__)
            return tool_error_json(
                "internal_error",
                "Internal server error",
                retryable=False,
            )

    return wrapper


def _fulltext_status_summary(fulltext_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "fulltext_available": fulltext_payload["fulltext_available"],
        "access_status": fulltext_payload["access_status"],
        "acquisition_state": fulltext_payload["acquisition_state"],
        "needs_acquisition": fulltext_payload["needs_acquisition"],
        "error_message": fulltext_payload["error_message"],
    }


def _theme_scoped_paper_id(theme_id: str, work_id: str) -> str:
    return f"{theme_id}:{work_id}"


def _resolve_public_paper_id(storage: StorageService, paper_id: str) -> tuple[str | None, Any | None]:
    if ":" in paper_id:
        theme_id, work_id = paper_id.split(":", 1)
        work = storage.get_work(work_id)
        return theme_id, work
    return None, storage.get_work(paper_id)


def _apply_theme_context(
    storage: StorageService,
    theme_id: str | None,
    work: Any,
) -> Any:
    if theme_id is None:
        return work
    context = storage.get_theme_work_context(theme_id, work.id)
    if context is None:
        return work
    themed = work.model_copy(deep=True)
    if context.get("composite_score") is not None:
        themed.composite_score = float(context["composite_score"])
    if context.get("agent_score") is not None:
        themed.agent_score = float(context["agent_score"])
    if context.get("agent_rank") is not None:
        themed.agent_rank = int(context["agent_rank"])
    if context.get("agent_rationale"):
        themed.agent_rationale = str(context["agent_rationale"])
    return themed


def _query_paper_payload(
    theme_id: str,
    work: Any,
    fulltext_payload: dict[str, Any],
    *,
    include_rationale: bool,
) -> dict[str, Any]:
    return {
        "paper_id": _theme_scoped_paper_id(theme_id, work.id),
        "title": work.title,
        "authors": work.authors,
        "year": work.year,
        "venue": work.venue,
        "abstract": work.abstract,
        "composite_score": work.composite_score,
        "agent_score": work.agent_score,
        "agent_rank": work.agent_rank,
        "rationale": work.agent_rationale if include_rationale else None,
        "fulltext_status": _fulltext_status_summary(fulltext_payload),
    }


def _summary_payload(paper_id: str, work: Any, fulltext_payload: dict[str, Any]) -> dict[str, Any]:
    payload = public_work_payload(work)
    payload["paper_id"] = paper_id
    payload.pop("id")
    payload["depth"] = "summary"
    payload["rationale"] = payload.pop("agent_rationale")
    payload["fulltext_status"] = _fulltext_status_summary(fulltext_payload)
    return payload


def _fulltext_message(fulltext_payload: dict[str, Any], *, attempted_acquire: bool) -> str | None:
    if fulltext_payload.get("parsed_text"):
        return None
    if attempted_acquire:
        return "Full text is unavailable after the explicit acquisition attempt."
    if fulltext_payload.get("needs_acquisition"):
        return "Full text is not cached. Set allow_acquire=true to attempt explicit acquisition."
    if fulltext_payload.get("error_message"):
        return str(fulltext_payload["error_message"])
    return "Full text is unavailable."


@mcp.tool()
@safe_tool
async def query(
    theme_document: str,
    final_limit: int = 20,
    agent_candidate_limit: int = 100,
    coarse_pool_limit: int | None = None,
    include_rationale: bool = True,
) -> str:
    """Run the full MCP query pipeline and return the final reranked papers."""
    async with runtime_limits.budget_manager.enforce(
        runtime_limits.RETRIEVAL_JOB_POLICY,
        "mcp",
    ):
        storage = _get_storage()
        settings = get_settings()
        from scholartrace.services.fulltext import read_cached_fulltext
        from scholartrace.services.retrieval import (
            QueryPipelineConfigurationError,
            QueryPipelineRuntimeError,
            run_query_pipeline,
        )

        try:
            result = await run_query_pipeline(
                theme_document,
                storage,
                settings=settings,
                final_limit=final_limit,
                agent_candidate_limit=agent_candidate_limit,
                coarse_pool_limit=coarse_pool_limit,
                include_rationale=include_rationale,
            )
        except QueryPipelineConfigurationError as exc:
            return tool_error_json("configuration_error", str(exc), retryable=False)
        except QueryPipelineRuntimeError as exc:
            return tool_error_json("query_failed", str(exc), retryable=True)

        papers = [
            _query_paper_payload(
                result.theme.id,
                work,
                read_cached_fulltext(work, storage, settings),
                include_rationale=include_rationale,
            )
            for work in result.works
        ]

    payload = {
        "theme_id": result.theme.id,
        "total_retrieved": result.total_retrieved,
        "total_after_dedup": result.total_after_dedup,
        "total_after_first_stage": result.total_after_first_stage,
        "total_agent_candidates": result.total_agent_candidates,
        "total_final": result.total_final,
        "papers": papers,
    }
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
@safe_tool
async def read(
    paper_id: str,
    depth: str = "summary",
    allow_acquire: bool = False,
) -> str:
    """Read a paper through a unified layered interface."""
    storage = _get_storage()
    settings = get_settings()
    theme_id, work = _resolve_public_paper_id(storage, paper_id)
    if work is None:
        return tool_error_json("not_found", f"Paper {paper_id} not found")
    work = _apply_theme_context(storage, theme_id, work)

    from scholartrace.services.fulltext import acquire_fulltext, read_cached_fulltext

    allowed_depths = {
        "summary",
        "sections",
        "fulltext_status",
        "fulltext",
        "direct_evidence",
    }
    if depth not in allowed_depths:
        return tool_error_json(
            "invalid_request",
            f"Unsupported read depth '{depth}'",
            retryable=False,
        )

    if depth == "direct_evidence":
        if not work.arxiv_id:
            return json.dumps(
                {
                    "paper_id": paper_id,
                    "depth": depth,
                    "available": False,
                    "source": "deepxiv",
                    "reason": "Direct DeepXiv evidence is only available for arXiv-backed papers.",
                },
                ensure_ascii=False,
            )

        connector = await _get_deepxiv()
        metadata = await connector.get_paper_metadata(work.arxiv_id)
        brief = await connector.get_paper_brief(work.arxiv_id)
        available = metadata is not None or brief is not None
        payload = {
            "paper_id": paper_id,
            "depth": depth,
            "available": available,
            "source": "deepxiv",
            "arxiv_id": work.arxiv_id,
        }
        if available:
            payload["evidence"] = deepxiv_summary_payload(work.arxiv_id, metadata, brief)
        else:
            payload["reason"] = "No DeepXiv evidence is available for this paper."
        return json.dumps(payload, ensure_ascii=False)

    fulltext_payload = read_cached_fulltext(work, storage, settings)

    if depth == "summary":
        return json.dumps(
            _summary_payload(paper_id, work, fulltext_payload),
            ensure_ascii=False,
        )

    if depth == "sections":
        return json.dumps(
            {
                "paper_id": paper_id,
                "depth": depth,
                "sections": fulltext_payload["sections"],
                "fulltext_status": _fulltext_status_summary(fulltext_payload),
            },
            ensure_ascii=False,
        )

    if depth == "fulltext_status":
        return json.dumps(
            {
                "paper_id": paper_id,
                "depth": depth,
                "fulltext_status": fulltext_payload,
            },
            ensure_ascii=False,
        )

    attempted_acquire = False
    if not fulltext_payload["fulltext_available"] and allow_acquire:
        async with runtime_limits.budget_manager.enforce(
            runtime_limits.FULLTEXT_ACQUIRE_POLICY,
            "mcp",
        ):
            attempted_acquire = True
            updated = await acquire_fulltext(work, storage, settings)
            fulltext_payload = read_cached_fulltext(updated, storage, settings)

    return json.dumps(
        {
            "paper_id": paper_id,
            "depth": depth,
            "fulltext": fulltext_payload.get("parsed_text"),
            "sections": fulltext_payload.get("sections", []),
            "fulltext_status": fulltext_payload,
            "message": _fulltext_message(
                fulltext_payload,
                attempted_acquire=attempted_acquire,
            ),
        },
        ensure_ascii=False,
    )
