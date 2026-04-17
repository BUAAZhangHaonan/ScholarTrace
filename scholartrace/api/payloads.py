from __future__ import annotations

from typing import Any, Iterable

from scholartrace.models.schemas import Work


_SENSITIVE_KEYS = {"source_provenance", "pdf_url", "html_url", "oa_url", "local_path"}


def _access_status_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _redact_sensitive_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _redact_sensitive_fields(item)
            for key, item in value.items()
            if key not in _SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [_redact_sensitive_fields(item) for item in value]
    return value


def public_work_payload(work: Work | Any) -> dict[str, Any]:
    payload = {
        "id": work.id,
        "doi": work.doi,
        "arxiv_id": work.arxiv_id,
        "openalex_id": work.openalex_id,
        "s2_id": work.s2_id,
        "dblp_key": work.dblp_key,
        "openreview_id": work.openreview_id,
        "title": work.title,
        "authors": work.authors,
        "year": work.year,
        "venue": work.venue,
        "abstract": work.abstract,
        "relevance_score": work.relevance_score,
        "recency_score": work.recency_score,
        "influence_score": work.influence_score,
        "venue_score": work.venue_score,
        "composite_score": work.composite_score,
        "agent_score": work.agent_score,
        "agent_rank": work.agent_rank,
        "agent_rationale": work.agent_rationale,
        "fulltext_available": work.fulltext_available,
        "access_status": _access_status_value(work.access_status),
        "citation_count": work.citation_count,
        "reference_count": work.reference_count,
        "created_at": work.created_at.isoformat() if work.created_at else None,
        "updated_at": work.updated_at.isoformat() if work.updated_at else None,
    }
    return payload


def public_work_list_payload(works: Iterable[Work | Any]) -> list[dict[str, Any]]:
    return [public_work_payload(work) for work in works]


def theme_export_json_payload(theme: Any, works: Iterable[Work | Any]) -> dict[str, Any]:
    paper_list = public_work_list_payload(works)
    return {
        "theme": theme.model_dump(),
        "papers": paper_list,
    }


def theme_report_json_payload(theme_id: str, theme: Any, works: Iterable[Work | Any]) -> dict[str, Any]:
    paper_list = public_work_list_payload(works)
    return {
        "theme_id": theme_id,
        "parsed_topics": theme.parsed_topics,
        "parsed_methods": theme.parsed_methods,
        "parsed_datasets": theme.parsed_datasets,
        "parsed_queries": theme.parsed_queries,
        "total_papers": len(paper_list),
        "papers": paper_list,
    }


def deepxiv_search_payload(candidates: Iterable[Any]) -> dict[str, Any]:
    papers: list[dict[str, Any]] = []
    for candidate in candidates:
        papers.append(
            {
                "title": candidate.title,
                "authors": candidate.authors,
                "year": candidate.year,
                "abstract": candidate.abstract,
                "arxiv_id": candidate.arxiv_id,
                "doi": candidate.doi,
                "citation_count": candidate.citation_count,
            }
        )
    return {"total": len(papers), "papers": papers}


def deepxiv_summary_payload(
    arxiv_id: str,
    metadata: dict[str, Any] | None,
    brief: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"arxiv_id": arxiv_id}
    if metadata is not None:
        payload["metadata"] = _redact_sensitive_fields(metadata)
    if brief is not None:
        payload["brief"] = _redact_sensitive_fields(brief)
    return payload
