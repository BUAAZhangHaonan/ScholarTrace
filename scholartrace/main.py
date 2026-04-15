import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import InitializationOptions, NotificationOptions

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


async def _run_mcp_server():
    """Create and run the MCP server with stdio transport."""
    server = Server("scholartrace")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scholartrace",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def run_mcp():
    """Entry point for the MCP server."""
    asyncio.run(_run_mcp_server())
