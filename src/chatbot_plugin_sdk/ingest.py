"""RagArticleProcessor — handles text ingestion (normalize → chunk → embed → save).

Inherits :class:`BaseRagProcessor` to gain DB config, search, chat, and embedding
support. Adds sync ``configure()``, default ``_normalize_full_text()``, and
async ``ingest()`` pipeline.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from chatbot_plugin_sdk.base import BaseRagProcessor
from chatbot_plugin_sdk.chunking import _chunk_text
from chatbot_plugin_sdk.exceptions import NotConfiguredError, DatabaseError

if TYPE_CHECKING:
    from chatbot_plugin_sdk.config import DatabaseConfig, EmbeddingModelConfig


class RagArticleProcessor(BaseRagProcessor):
    """SDK variant responsible for writing data.

    Extends :class:`BaseRagProcessor` with:

    - sync :meth:`configure` — set DB credentials *and* embedding endpoint
    - :meth:`_normalize_full_text` — default text sanitisation
    - async :meth:`ingest` — full pipeline from raw article text to DB
    """

    # ── Configuration ──

    def configure(
        self,
        dbname: str,
        user: str,
        password: str,
        embedding_model_api: str | None = None,
        embedding_model_api_key: str | None = None,
        *,
        host: str = "localhost",
        port: int = 5432,
    ) -> None:
        """Configure database and optional embedding endpoint.

        Args:
            dbname: PostgreSQL database name.
            user: PostgreSQL username.
            password: PostgreSQL password.
            embedding_model_api: Base URL of the embedding microservice.
                If **None**, embedding-related methods will raise
                :class:`~chatbot_plugin.sdk.exceptions.NotConfiguredError`.
            embedding_model_api_key: Optional API key for the embedding service.
            host: PostgreSQL host (default: ``localhost``).
            port: PostgreSQL port (default: ``5432``).
        """
        self._configure_database(dbname, user, password, host=host, port=port)
        if embedding_model_api:
            self._configure_embedding_model(embedding_model_api, api_key=embedding_model_api_key)

    # ── Normalisation ──

    def _normalize_full_text(self, text: str, **kwargs) -> str:
        """Default text normalisation / sanitisation.

        Steps (best effort, safe on any input):

        1. Unicode normalisation (NFC)
        2. Strip leading / trailing whitespace
        3. Collapse runs of whitespace (spaces, tabs, newlines)
        4. Optional BOM removal

        Users may subclass and override this method, or pass a *callable*
        to :meth:`ingest` via ``normalization``.

        Keyword arguments are accepted for forward compatibility but
        currently ignored.
        """
        # NFC unicode normalisation
        text = unicodedata.normalize("NFC", text)
        # Strip BOM if present
        text = text.lstrip("\ufeff")
        # Strip leading / trailing whitespace
        text = text.strip()
        # Collapse all runs of whitespace to a single space
        text = re.sub(r"\s+", " ", text)
        return text

    # ── Ingest pipeline ──

    async def ingest(
        self,
        full_text: str,
        normalization: str | Callable[[str], str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Ingest raw article text: normalise → chunk → embed → save.

        The embedding model **must** be configured in
        :meth:`configure`; otherwise a
        :class:`~chatbot_plugin.sdk.exceptions.NotConfiguredError`
        is raised.

        Args:
            full_text: Raw article text.
            normalization:
                - ``None`` or ``"default"`` — uses :meth:`_normalize_full_text`
                - ``callable`` — custom normaliser ``fn(text) -> str``
            metadata: Arbitrary article metadata dict. Must contain at
                least ``url`` (used for article_id generation if not
                provided). Other common keys: ``title``, ``source``.
        """
        if self._db_config is None:
            raise NotConfiguredError("Database not configured. Call configure() first.")
        if self._embed_config is None:
            raise NotConfiguredError(
                "Embedding model not configured. "
                "Pass embedding_model_api when calling configure()."
            )

        metadata = metadata or {}

        # 1. Normalise
        if normalization is None or normalization == "default":
            normalised = self._normalize_full_text(full_text)
        elif callable(normalization):
            normalised = normalization(full_text)
        else:
            raise DatabaseError(
                f"Invalid normalization value: {normalization!r}. "
                "Expected None, 'default', or a callable."
            )

        # 2. Chunk
        chunks = _chunk_text(normalised)
        if not chunks:
            raise DatabaseError("No chunks produced — input text may be empty.")

        # 3. Embed
        dense_vectors, sparse_vectors = await self._embed_texts(chunks)

        if not dense_vectors or len(dense_vectors) != len(chunks):
            raise DatabaseError(
                f"Embedding returned {len(dense_vectors)} vectors but "
                f"{len(chunks)} chunks were expected."
            )

        # 4. Save
        # Derive article_id from url if present, otherwise generate a random UUID
        url = metadata.get("url", "")
        article_id = uuid.uuid5(uuid.NAMESPACE_URL, url) if url else uuid.uuid4()

        chunks_data = [
            {
                "chunk_index": i,
                "content": chunk_text,
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
            }
            for i, chunk_text in enumerate(chunks)
        ]

        await self._save_article_and_chunks(
            article_id=article_id,
            metadata=metadata,
            chunks_data=chunks_data,
        )
