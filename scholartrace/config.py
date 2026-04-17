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
    semantic_scholar_api_keys: str = ""  # Comma-separated for multi-key rotation
    openalex_mailto: str = ""
    crossref_mailto: str = ""

    # --- Query limits ---
    max_results_per_source_per_query: int = 200
    target_candidate_pool: int = 500
    max_fulltext_downloads: int = 50
    agent_candidate_limit: int = 100
    final_limit: int = 20

    # --- Ranking weights ---
    weight_relevance: float = 0.35
    weight_recency: float = 0.20
    weight_influence: float = 0.20
    weight_venue: float = 0.10
    weight_fulltext: float = 0.10
    weight_source_agreement: float = 0.05

    # --- DeepXiv settings ---
    deepxiv_tokens: str = ""  # Comma-separated DeepXiv API tokens
    deepxiv_pool_size: int = 3
    deepxiv_auto_register: bool = False
    deepxiv_register_sdk_secret: str = ""

    # --- BigModel GLM settings ---
    bigmodel_api_key: str = ""
    bigmodel_base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
    bigmodel_model: str = "glm-5-turbo"

    # --- Server settings ---
    api_host: str = "127.0.0.1"
    api_port: int = 9000
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8001
    mcp_transport: str = "stdio"
    remote_access_enabled: bool = False
    access_token: str = ""

    model_config = {
        "env_prefix": "SCHOLARTRACE_",
    }


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
