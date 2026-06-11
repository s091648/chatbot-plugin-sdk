"""Configuration for the toolbox.

All settings are read from environment variables with the `CHATBOT_` prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


class ChatbotSettings(BaseSettings):
    """Toolbox settings."""

    # Vector dimensions (must match embedding model output)
    embedding_dimension: int = 1024
    sparse_dimension: int = 250002

    # Search / RRF
    rrf_k: int = 60
    search_candidates: int = 50
    max_context_chunks: int = 10

    # LLM (optional — only needed for chat)
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6-20250514"

    # Gemini (fallback if Anthropic is unavailable)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    model_config = {"env_prefix": "CHATBOT_"}


settings = ChatbotSettings()


@dataclass
class DatabaseConfig:
    """Holds async SQLAlchemy engine + session factory."""

    engine: "AsyncEngine"
    session_factory: "async_sessionmaker[AsyncSession]"


@dataclass
class EmbeddingModelConfig:
    """Holds httpx client pointing at embedding microservice."""

    base_url: str
    api_key: str | None

    def endpoint(self, path: str) -> str:
        """Return absolute URL for a given path on the embedding service."""
        return urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))

    def build_client(self) -> httpx.AsyncClient:
        """Build an httpx AsyncClient for the embedding service."""
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers=headers,
            timeout=60,
        )
