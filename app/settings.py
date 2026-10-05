from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_mode: Literal["demo", "production"] = "demo"
    model_provider: Literal["demo", "dashscope"] = "demo"
    business_db_url: str = "sqlite://data/business.sqlite3"
    vector_db_url: str = "sqlite://data/vectors.sqlite3"
    jwt_secret: SecretStr = SecretStr("local-demo-only-change-me-32-characters")
    jwt_issuer: str = "rag-reproduction"
    token_ttl_minutes: int = Field(120, ge=5, le=1440)
    dashscope_api_key: SecretStr = SecretStr("")
    dashscope_http_base_url: str = ""
    dashscope_chat_base_url: str = ""
    generation_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v4"
    rerank_model: str = "gte-rerank-v2"
    embedding_dimension: Literal[1024] = 1024
    model_timeout_seconds: float = Field(60, ge=1, le=300)
    model_concurrency: int = Field(4, ge=1, le=32)
    model_daily_dispatch_limit: int = Field(300, ge=1, le=100_000)
    model_max_request_bytes: int = Field(100_000, ge=1000, le=1_000_000)
    model_max_output_tokens: int = Field(2048, ge=256, le=8192)
    check_source_window_projection: bool = False
    upload_dir: Path = Path("data/uploads")
    upload_max_bytes: int = Field(20 * 1024 * 1024, ge=1)
    parsed_max_chars: int = Field(2_000_000, ge=1000)
    sparse_max_chunks: int = Field(100_000, ge=100)
    worker_lease_seconds: int = Field(600, ge=30)
    worker_poll_seconds: float = Field(1.0, ge=0.1, le=10)
    worker_shutdown_seconds: float = Field(60, ge=1, le=300)
    worker_health_file: Path = Path("data/worker-health.json")
    chat_run_timeout_seconds: int = Field(240, ge=10, le=600)
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    enable_ocr: bool = False
    enable_registration: bool = True
    enable_api_workers: bool | None = None

    @model_validator(mode="after")
    def production_checks(self):
        if self.app_mode == "production":
            secret = self.jwt_secret.get_secret_value()
            if len(secret) < 32 or "demo" in secret or "change" in secret:
                raise ValueError(
                    "Production requires a unique JWT_SECRET of at least 32 characters"
                )
            if not self.business_db_url.startswith("mysql://"):
                raise ValueError("Production BUSINESS_DB_URL must be mysql://")
            if not self.vector_db_url.startswith(("postgres://", "postgresql://")):
                raise ValueError("Production VECTOR_DB_URL must be postgres://")
        if self.model_provider == "dashscope":
            if not self.dashscope_api_key.get_secret_value():
                raise ValueError("DASHSCOPE_API_KEY is required")
            for url in (self.dashscope_http_base_url, self.dashscope_chat_base_url):
                if not url.startswith("https://") or "{" in url:
                    raise ValueError("Set valid regional DashScope HTTP and chat base URLs")
        return self

    @property
    def embedding_fingerprint(self) -> str:
        return f"{self.model_provider}:{self.embedding_model}:{self.embedding_dimension}"


def get_settings() -> Settings:
    return Settings()
