from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All runtime configuration. DocuMind-specific vars use the DOCUMIND_ prefix;
    LLM provider keys use their conventional names (ANTHROPIC_API_KEY, ...)."""

    model_config = SettingsConfigDict(env_prefix="DOCUMIND_", env_file=".env", extra="ignore")

    app_name: str = "DocuMind"
    version: str = "0.1.0"

    data_dir: Path = BACKEND_DIR / "data"
    database_url: str | None = None  # defaults to sqlite in data_dir
    samples_dir: Path = BACKEND_DIR.parent / "samples"
    seed_library: bool = True
    cors_origins: list[str] = ["http://localhost:4200", "http://localhost:4300"]

    # Models (fastembed / ONNX). Small enough for a 512 MB instance.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_quantized: bool = True
    model_threads: int | None = None  # ONNX threads; 1-2 keeps memory low on small hosts
    model_cache_dir: Path | None = None
    model_mem_arena: bool = False

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Public demo limits
    max_upload_mb: float = 5
    max_pages: int = 40
    max_docs_per_workspace: int = 5
    workspace_ttl_hours: float = 24
    chat_per_minute: int = 20
    uploads_per_hour: int = 5
    eval_runs_per_hour: int = 3

    # LLM. Provider is auto-picked from whichever key is set unless forced here
    # (anthropic | google | groq | openai | azure | ollama | extractive).
    llm_provider: str | None = None
    llm_model: str | None = None
    ollama_url: str = "http://localhost:11434/v1"
    llm_timeout_seconds: float = 45.0
    llm_max_tokens: int = 700

    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    azure_openai_api_key: str | None = Field(default=None, validation_alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str | None = Field(default=None, validation_alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str | None = Field(default=None, validation_alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(default="2024-10-21", validation_alias="AZURE_OPENAI_API_VERSION")

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'documind.db'}"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
