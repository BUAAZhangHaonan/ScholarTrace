"""CLI entry points for ScholarTrace."""

from scholartrace.config import get_settings


def run_api():
    """Run the FastAPI API server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "scholartrace.api.rest:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


def run_mcp():
    """Entry point for the MCP server (SSE transport for LAN access)."""
    from scholartrace.api.mcp_server import mcp

    mcp.run(transport="sse")


if __name__ == "__main__":
    run_mcp()
