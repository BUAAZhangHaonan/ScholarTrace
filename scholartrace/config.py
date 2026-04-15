from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    All settings can be overridden by setting environment variables
    with the prefix SCHOLARTRACE_, e.g. SCHOLARTRACE_API_PORT=9000.
    """

    # --- Paths ---
    data_dir: Path = Path("data")
    db_path: Path = Path("data/scholartrace.db")

    # --- API keys ---
    semantic_scholar_api_key: str = ""
    openalex_mailto: str = ""
    crossref_mailto: str = ""

    # --- Query limits ---
    max_results_per_source_per_query: int = 200
    target_candidate_pool: int = 500
    max_fulltext_downloads: int = 50

    # --- Ranking weights ---
    weight_relevance: float = 0.35
    weight_recency: float = 0.20
    weight_influence: float = 0.20
    weight_venue: float = 0.10
    weight_fulltext: float = 0.10
    weight_source_agreement: float = 0.05

    # --- Server settings ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8001

    model_config = {
        "env_prefix": "SCHOLARTRACE_",
    }


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
