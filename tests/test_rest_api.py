"""Tests for the FastAPI REST API."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from scholartrace.api.rest import app
from scholartrace.models.schemas import AccessStatus, Artifact, ArtifactKind, Work, Theme
from scholartrace.services.storage import StorageService


@pytest.fixture(autouse=True)
def _reset_module_singletons():
    """Reset module-level singletons so each test gets a fresh storage."""
    import scholartrace.api.rest as rest_module
    rest_module._storage = None
    rest_module._settings = None
    yield
    rest_module._storage = None
    rest_module._settings = None


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        # Monkey-patch the settings to point at our temp database.
        import scholartrace.api.rest as rest_module
        from scholartrace.config import Settings

        test_settings = Settings(
            data_dir=os.path.join(tmpdir, "data"),
            db_path=db_path,
        )
        test_settings.data_dir.mkdir(parents=True, exist_ok=True)

        storage = StorageService(db_path)
        storage.init_db()

        rest_module._storage = storage
        rest_module._settings = test_settings

        yield TestClient(app)

        storage.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_theme(storage: StorageService, text: str = "RLHF and reward hacking") -> Theme:
    from scholartrace.services.theme_parser import parse_theme
    theme = parse_theme(text)
    storage.save_theme(theme)
    return theme


def _make_work(storage: StorageService, title: str = "Test Paper", theme_id: str | None = None) -> Work:
    work = Work(title=title, authors=["Alice", "Bob"], year=2024, venue="ICML",
                abstract="A test abstract.", composite_score=0.85)
    storage.save_work(work)
    if theme_id:
        storage.link_theme_work(theme_id, work.id, 1)
    return work


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_health_check(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_create_theme(client: TestClient):
    resp = client.post(
        "/themes", data={"text": "Reinforcement learning from human feedback and reward hacking in large language models"})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert isinstance(data["parsed_queries"], list)
    assert len(data["parsed_queries"]) > 0


def test_list_papers_empty(client: TestClient):
    import scholartrace.api.rest as rest_module
    theme = _make_theme(rest_module._storage)
    resp = client.get(f"/themes/{theme.id}/papers")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_paper_not_found(client: TestClient):
    resp = client.get("/papers/nonexistent-id")
    assert resp.status_code == 404


def test_create_retrieval_job(client: TestClient):
    import scholartrace.api.rest as rest_module
    theme = _make_theme(rest_module._storage)
    with patch(
        "scholartrace.services.retrieval.run_retrieval",
        new=AsyncMock(return_value=[]),
    ):
        resp = client.post("/retrieval/jobs", data={"theme_id": theme.id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["theme_id"] == theme.id
    assert data["status"] == "pending"


def test_get_job_status(client: TestClient):
    import scholartrace.api.rest as rest_module
    theme = _make_theme(rest_module._storage)
    # Create a job first
    with patch(
        "scholartrace.services.retrieval.run_retrieval",
        new=AsyncMock(return_value=[]),
    ):
        create_resp = client.post("/retrieval/jobs", data={"theme_id": theme.id})
    job_id = create_resp.json()["id"]

    resp = client.get(f"/retrieval/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == job_id
    assert data["theme_id"] == theme.id


def test_export_json(client: TestClient):
    import scholartrace.api.rest as rest_module
    theme = _make_theme(rest_module._storage)
    _make_work(rest_module._storage, title="Export Paper 1", theme_id=theme.id)

    resp = client.get(f"/themes/{theme.id}/export?format=json")
    assert resp.status_code == 200
    data = resp.json()
    assert "theme" in data
    assert "papers" in data
    assert len(data["papers"]) == 1
    assert data["papers"][0]["title"] == "Export Paper 1"


def test_export_markdown(client: TestClient):
    import scholartrace.api.rest as rest_module
    theme = _make_theme(rest_module._storage)
    _make_work(rest_module._storage, title="Markdown Paper", theme_id=theme.id)

    resp = client.get(f"/themes/{theme.id}/export?format=markdown")
    assert resp.status_code == 200
    assert "text/" in resp.headers["content-type"]
    text = resp.text
    assert "ScholarTrace Report" in text
    assert "Markdown Paper" in text
    assert "Alice" in text


def test_theme_not_found(client: TestClient):
    resp = client.get("/themes/nonexistent-id/papers")
    assert resp.status_code == 404


def test_protected_route_requires_bearer_token_when_configured(client: TestClient):
    import scholartrace.api.rest as rest_module

    rest_module._settings.access_token = "phase1-secret"

    unauth = client.get("/papers/nonexistent-id")
    assert unauth.status_code == 401
    assert unauth.json() == {
        "error": {
            "code": "unauthorized",
            "message": "Authentication required",
            "retryable": False,
        }
    }

    health = client.get("/health")
    assert health.status_code == 200

    authed = client.get(
        "/papers/nonexistent-id",
        headers={"Authorization": "Bearer phase1-secret"},
    )
    assert authed.status_code == 404
    assert authed.json() == {
        "error": {
            "code": "not_found",
            "message": "Paper not found",
            "retryable": False,
        }
    }


def test_fulltext_response_redacts_local_path(client: TestClient):
    import scholartrace.api.rest as rest_module

    theme = _make_theme(rest_module._storage)
    work = _make_work(rest_module._storage, title="Redacted Paper", theme_id=theme.id)
    artifact = Artifact(
        work_id=work.id,
        kind=ArtifactKind.PDF,
        source_url="https://example.com/paper.pdf",
        local_path="/tmp/should-not-leak.pdf",
        access_status=AccessStatus.AVAILABLE,
    )
    rest_module._storage.save_artifact(artifact)

    resp = client.get(f"/papers/{work.id}/fulltext")
    assert resp.status_code == 200
    data = resp.json()
    assert data["artifacts"][0]["source_url"] == "https://example.com/paper.pdf"
    assert "local_path" not in data["artifacts"][0]
