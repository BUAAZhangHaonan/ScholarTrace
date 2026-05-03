"""CLI entry points for ScholarTrace."""

import logging

import uvicorn

from scholartrace.api.security import is_loopback_host
from scholartrace.config import get_settings

logger = logging.getLogger(__name__)


def validate_runtime_settings(settings, *, service: str) -> None:
    """Reject unsafe runtime configurations before the server starts."""
    if service == "api":
        host = settings.api_host
        if is_loopback_host(host):
            return
        if not settings.remote_access_enabled:
            logger.error("Denied API startup for remote host %s: remote access disabled", host)
            raise ValueError("remote API access is disabled")
        if not settings.access_token:
            logger.error("Denied API startup for remote host %s: missing access token", host)
            raise ValueError("access token is required for remote API access")
        return

    if service == "mcp":
        if settings.mcp_transport == "stdio":
            return
        host = settings.mcp_host
        if not is_loopback_host(host) and not settings.remote_access_enabled:
            logger.error("Denied MCP startup for remote host %s: remote access disabled", host)
            raise ValueError("remote MCP access is disabled")
        if not settings.access_token:
            logger.error(
                "Denied MCP startup for transport %s on host %s: missing access token",
                settings.mcp_transport,
                host,
            )
            raise ValueError("access token is required for network MCP access")
        return

    raise ValueError(f"Unknown service '{service}'")


def run_api():
    """Run the FastAPI API server."""
    settings = get_settings()
    validate_runtime_settings(settings, service="api")
    uvicorn.run(
        "scholartrace.api.rest:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        timeout_graceful_shutdown=settings.shutdown_timeout_seconds,
    )


def run_mcp():
    """Entry point for the MCP server."""
    from scholartrace.api.mcp_server import create_mcp_sse_app, mcp

    settings = get_settings()
    validate_runtime_settings(settings, service="mcp")
    if settings.mcp_transport == "stdio":
        mcp.run(transport="stdio")
        return

    uvicorn.run(
        create_mcp_sse_app(settings),
        host=settings.mcp_host,
        port=settings.mcp_port,
        reload=False,
        timeout_graceful_shutdown=settings.shutdown_timeout_seconds,
    )


if __name__ == "__main__":
    run_mcp()
