"""
Application settings and environment configuration.
Built using Pydantic Settings for strict schema validation.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Relational Database (PostgreSQL) ---
    POSTGRES_USER: str = "sentinel"
    POSTGRES_PASSWORD: str = "sentinel_secret"
    POSTGRES_DB: str = "sentinel_db"
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: Optional[str] = None
    DATABASE_SYNC_URL: Optional[str] = None

    # --- Distributed Message Broker (Redis) ---
    REDIS_URL: str = "redis://127.0.0.1:6379/0"

    # --- Vector Engine (Qdrant) ---
    QDRANT_HOST: str = "127.0.0.1"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_COLLECTION: str = "sentinel_knowledge_base"

    # --- Security & JWT Identity ---
    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "insecure-default-key")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    WEBHOOK_SECRET: str = os.environ.get("WEBHOOK_SECRET", "insecure-default-key")

    # --- Embeddings & Re-Ranking ---
    EMBEDDING_PROVIDER: str = "sentence_transformers"  # sentence_transformers | fastembed | openai | mock
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_ENABLED: bool = True
    HF_HOME: str = "./models"

    # --- LLM Grounded Generation ---
    LLM_PROVIDER: str = "openai_compatible"  # openai_compatible | mock
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "nvidia/nemotron-3.5-lightning:free"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096

    # --- Service Ports & Environment ---
    API_GATEWAY_URL: str = "http://127.0.0.1:8000"
    API_PORT: int = 8000
    UI_PORT: int = 8501
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    def get_database_url(self) -> str:
        """Returns the async database URL, defaulting to local PostgreSQL if unspecified."""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    def get_database_sync_url(self) -> str:
        """Returns the synchronous database URL for worker and migrations."""
        if self.DATABASE_SYNC_URL:
            return self.DATABASE_SYNC_URL
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()

if settings.HF_HOME:
    abs_hf_home = os.path.abspath(settings.HF_HOME)
    os.environ["HF_HOME"] = abs_hf_home
    os.environ["TRANSFORMERS_CACHE"] = abs_hf_home
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = abs_hf_home
    os.environ["HF_HUB_OFFLINE"] = "1"
