from __future__ import annotations

from starlette.testclient import TestClient
import pytest

from scholartrace.config import Settings


def test_validate_runtime_settings_rejects_remote_api_without_token(caplog: pytest.LogCaptureFixture):
    from scholartrace.main import validate_runtime_settings

    settings = Settings(
        api_host="0.0.0.0",
        remote_access_enabled=False,
        access_token="",
    )

    with pytest.raises(ValueError, match="remote API access is disabled"):
        validate_runtime_settings(settings, service="api")

    assert "Denied API startup" in caplog.text


def test_validate_runtime_settings_rejects_remote_mcp_sse_without_token():
    from scholartrace.main import validate_runtime_settings

    settings = Settings(
        mcp_host="0.0.0.0",
        mcp_transport="sse",
        remote_access_enabled=True,
        access_token="",
    )

    with pytest.raises(ValueError, match="access token is required"):
        validate_runtime_settings(settings, service="mcp")


def test_create_mcp_sse_app_requires_auth_header():
    from scholartrace.api.mcp_server import create_mcp_sse_app

    settings = Settings(
        mcp_transport="sse",
        access_token="phase1-secret",
    )
    client = TestClient(create_mcp_sse_app(settings))

    unauth = client.get("/sse")
    assert unauth.status_code == 401

    authed = client.get(
        "/missing",
        headers={"Authorization": "Bearer phase1-secret"},
    )
    assert authed.status_code == 404
