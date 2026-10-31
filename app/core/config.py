"""Typed application configuration module using Pydantic Settings."""

from functools import lru_cache
from typing import Any, List, Optional, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Core Application
    APP_NAME: str = Field(default="ExamRAG", description="Application name")
    APP_ENV: str = Field(default="development", description="Environment: development, staging, production, test")
    API_V1_PREFIX: str = Field(default="/api/v1", description="Prefix for API v1 routes")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # OpenRouter LLM & Embeddings
    OPENROUTER_API_KEY: str = Field(default="", description="API key for OpenRouter")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter base URL")
    CHAT_MODEL: str = Field(default="openai/gpt-4o-mini", description="Chat model identifier")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Embedding model identifier")

    # Qdrant Vector Store
    QDRANT_URL: str = Field(default="", description="Qdrant cluster URL")
    QDRANT_API_KEY: str = Field(default="", description="Qdrant Cloud API Key")
    QDRANT_COLLECTION_NAME: str = Field(default="examrag_documents", description="Qdrant collection name")

    # Ingestion & Chunking
    CHUNK_SIZE: int = Field(default=500, description="Chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=100, description="Chunk overlap in characters")
    RETRIEVAL_TOP_K: int = Field(default=3, description="Default top-k chunks to retrieve")
    RELEVANCE_THRESHOLD: Optional[float] = Field(default=None, description="Optional minimum similarity score threshold")
    ABSTAIN_THRESHOLD: float = Field(default=0.65, description="Cosine similarity threshold below which query triggers abstention")
    MAX_UPLOAD_SIZE_MB: int = Field(default=20, description="Max allowed PDF upload size in megabytes")
    EMBEDDING_BATCH_SIZE: int = Field(default=32, description="Batch size for embedding generation")

    # Resilience & Timeouts
    LLM_TIMEOUT_SECONDS: float = Field(default=45.0, description="Timeout for LLM and embedding calls in seconds")
    LLM_MAX_RETRIES: int = Field(default=2, description="Max retries for transient provider failures")
    MAX_TOKENS: int = Field(default=1000, description="Maximum completion tokens requested from LLM")

    # Security & Access Control
    CORS_ORIGINS: Union[List[str], str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins"
    )
    INGESTION_API_KEY: str = Field(default="", description="Secret API key required for ingestion and deletion")
    PUBLIC_CHAT_ENABLED: bool = Field(default=True, description="Whether public chat query is permitted without API key")

    @field_validator("CORS_ORIGINS", mode="after")
    @classmethod
    def parse_cors_origins(cls, value: Union[List[str], str]) -> List[str]:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("[") and value.endswith("]"):
                import json
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    pass
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    @field_validator("MAX_UPLOAD_SIZE_MB", "RETRIEVAL_TOP_K", "EMBEDDING_BATCH_SIZE")
    @classmethod
    def validate_positive_integers(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer greater than zero.")
        return value

    @field_validator("LLM_TIMEOUT_SECONDS")
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than zero.")
        return value

    @field_validator("LLM_MAX_RETRIES")
    @classmethod
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("LLM_MAX_RETRIES cannot be negative.")
        return value

    @field_validator("RELEVANCE_THRESHOLD", mode="before")
    @classmethod
    def validate_relevance_threshold(cls, value: Any) -> Optional[float]:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        float_val = float(value)
        if not (0.0 <= float_val <= 1.0):
            raise ValueError("RELEVANCE_THRESHOLD must be between 0.0 and 1.0.")
        return float_val

    @model_validator(mode="after")
    def validate_chunking_configuration(self) -> "Settings":
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero.")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")
        if self.CHUNK_SIZE <= self.CHUNK_OVERLAP:
            raise ValueError(
                f"CHUNK_SIZE ({self.CHUNK_SIZE}) must be strictly greater than CHUNK_OVERLAP ({self.CHUNK_OVERLAP})."
            )
        return self

    def validate_production_requirements(self) -> None:
        """Validate critical configuration required for production deployment."""
        if self.APP_ENV.lower() == "production":
            missing = []
            if not self.OPENROUTER_API_KEY or "your-" in self.OPENROUTER_API_KEY:
                missing.append("OPENROUTER_API_KEY")
            if not self.QDRANT_URL or "your-" in self.QDRANT_URL:
                missing.append("QDRANT_URL")
            if not self.QDRANT_API_KEY or "your-" in self.QDRANT_API_KEY:
                missing.append("QDRANT_API_KEY")
            if not self.INGESTION_API_KEY or "your-" in self.INGESTION_API_KEY:
                missing.append("INGESTION_API_KEY")

            if missing:
                raise ConfigurationError(
                    f"Production startup failed: Required environment variables are missing or default: {', '.join(missing)}"
                )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
