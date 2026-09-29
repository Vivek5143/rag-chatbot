"""Typed, environment-backed configuration for the application foundation."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when a later runtime phase needs configuration that is missing."""


class Settings(BaseSettings):
    """Application settings loaded from environment variables or an optional .env file.

    Provider secrets are intentionally optional at import time so the API
    foundation and its offline tests can start without external credentials.
    Call :meth:`require_runtime_configuration` before a phase contacts OpenAI
    or Pinecone.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "agentic-ai-index"
    pinecone_namespace: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # Optional OpenAI-compatible base URL. Left empty to use the official OpenAI
    # endpoint (the documented default). Set explicitly to redirect the OpenAI
    # SDK at an alternate OpenAI-compatible provider; this never changes the
    # default provider and is not treated as a configuration requirement.
    openai_base_url: str = ""
    openai_timeout_seconds: float = Field(default=20.0, gt=0.0, le=120.0)
    embedding_dimension: int = Field(default=1536, ge=1)
    embedding_batch_size: int = Field(default=64, ge=1, le=2048)
    pinecone_upsert_batch_size: int = Field(default=100, ge=1, le=1000)
    min_evidence_support_score: float = Field(default=0.8, ge=0.0, le=1.0)
    min_evidence_entailment_score: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence_grader_timeout_seconds: float = Field(default=20.0, gt=0.0, le=120.0)
    top_k: int = Field(default=4, ge=3, le=5)
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)

    @field_validator("openai_api_key", "pinecone_api_key", mode="before")
    @classmethod
    def normalize_optional_secrets(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator(
        "pinecone_index_name",
        "pinecone_namespace",
        "openai_model",
        "openai_embedding_model",
        mode="before",
    )
    @classmethod
    def normalize_text_settings(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip()

    @field_validator("pinecone_index_name", "openai_model", "openai_embedding_model")
    @classmethod
    def require_non_empty_text_settings(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    def missing_runtime_configuration(self) -> tuple[str, ...]:
        """Return provider values that must exist before external calls are made."""

        missing: list[str] = []
        if self.openai_api_key is None:
            missing.append("OPENAI_API_KEY")
        if self.pinecone_api_key is None:
            missing.append("PINECONE_API_KEY")
        if not self.pinecone_namespace:
            missing.append("PINECONE_NAMESPACE")
        return tuple(missing)

    def require_runtime_configuration(self) -> None:
        """Raise a safe, actionable error if a provider-backed phase is not configured."""

        missing = self.missing_runtime_configuration()
        if missing:
            names = ", ".join(missing)
            raise ConfigurationError(f"Missing required runtime configuration: {names}")

    def require_indexing_configuration(self) -> None:
        """Validate configuration required by the explicit embedding/indexing command."""

        self.require_runtime_configuration()

    def require_retrieval_configuration(self) -> None:
        """Validate configuration required by retrieval and evidence evaluation wiring."""

        self.require_runtime_configuration()


@lru_cache
def get_settings() -> Settings:
    """Return one immutable settings instance per process."""

    return Settings()


def build_openai_client(settings: Settings):
    """Construct an OpenAI SDK client honoring an optional base URL.

    ``OPENAI_BASE_URL`` is empty by default, so the client targets the official
    OpenAI endpoint. Setting it explicitly redirects every OpenAI SDK call to
    an alternate OpenAI-compatible provider. This is an explicit operator
    choice, never a silent default, and it does not change the assignment's
    documented default provider.
    """

    if settings.openai_api_key is None:
        raise ConfigurationError("OPENAI_API_KEY must be configured before building an OpenAI client")
    kwargs: dict[str, object] = {
        "api_key": settings.openai_api_key.get_secret_value(),
        "timeout": settings.openai_timeout_seconds,
    }
    base_url = settings.openai_base_url.strip()
    if base_url:
        kwargs["base_url"] = base_url
    from openai import OpenAI

    return OpenAI(**kwargs)
