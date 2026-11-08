"""Typed application settings, loaded from environment variables and `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "AI Backend"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_json: bool | None = None  # None -> JSON everywhere except local
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- LLM (Anthropic) ---
    # Leave unset to let the SDK resolve credentials from its usual sources.
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"
    llm_effort: Effort = "medium"
    llm_max_tokens: int = Field(default=16000, ge=1, le=128000)
    llm_stream_max_tokens: int = Field(default=64000, ge=1, le=128000)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    # Server-side refusal fallback: re-runs a declined request on Anthropic's
    # recommended fallback model inside the same call.
    llm_refusal_fallback: bool = True

    # --- Vector store (optional, used by RAG modules) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None

    @field_validator("anthropic_api_key", "qdrant_api_key", mode="before")
    @classmethod
    def _empty_secret_is_none(cls, value: object) -> object:
        return None if value in ("", None) else value

    @property
    def use_json_logs(self) -> bool:
        if self.log_json is not None:
            return self.log_json
        return self.environment != "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
