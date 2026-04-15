import asyncio

from scholartrace.config import get_settings


def create_app():
    """Create and return the FastAPI application."""
    from fastapi import FastAPI

    app = FastAPI(
        title="ScholarTrace",
        version="0.1.0",
        description="Multi-source scholarly literature discovery, ranking, and tracing",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def run_api():
    """Run the FastAPI API server."""
    import uvicorn

    settings = get_settings()
    app = create_app()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


def run_mcp():
    """Entry point for the MCP server (stdio transport)."""
    from scholartrace.api.mcp_server import mcp

    mcp.run()


if __name__ == "__main__":
    run_mcp()
