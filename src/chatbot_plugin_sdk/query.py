"""RagQueryProcessor — read-only variant for RAG queries.

Inherits :class:`BaseRagProcessor` and delegates read-only operations to the
base class via ``super()`` calls.  Does **not** implement ingest or embedding.

Intended usage::

    sdk = RagQueryProcessor()
    sdk.configure(dbname="chatbot_plugin", user="postgres", password="...")
    response = await sdk.query("What is RAG?")
"""

from __future__ import annotations

from typing import Any

from chatbot_plugin_sdk.contracts import ChatResponse
from chatbot_plugin_sdk.base import BaseRagProcessor
from chatbot_plugin_sdk.exceptions import NotConfiguredError


class RagQueryProcessor(BaseRagProcessor):
    """SDK variant responsible for read-only RAG queries.

    Extends :class:`BaseRagProcessor` with:

    - sync :meth:`configure` — set DB credentials and optional embedding endpoint
    - async :meth:`query` — full RAG pipeline (retrieval + generation)

    Embedding is required for :meth:`query` because it embeds the user prompt
    before searching.  If no embedding endpoint is configured,
    :class:`~chatbot_plugin.sdk.exceptions.NotConfiguredError` is raised.
    """

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
                Required for :meth:`query`; otherwise
                :class:`~chatbot_plugin.sdk.exceptions.NotConfiguredError`.
            embedding_model_api_key: Optional API key for the embedding service.
            host: PostgreSQL host (default: ``localhost``).
            port: PostgreSQL port (default: ``5432``).
        """
        self._configure_database(dbname, user, password, host=host, port=port)
        if embedding_model_api:
            self._configure_embedding_model(embedding_model_api, api_key=embedding_model_api_key)

    async def query(
        self,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """RAG query — retrieve relevant chunks and generate a response.

        This is a thin wrapper around :meth:`BaseRagProcessor.chat` that
        uses ``prompt`` as the search query and user message.

        Args:
            prompt: User question / prompt.
            metadata: Currently unused (reserved for future filters).

        Returns:
            A :class:`~chatbot_plugin.contracts.ChatResponse` with the
            generated reply, article citations, and retrieved chunks.
        """
        if self._db_config is None:
            raise NotConfiguredError("Database not configured. Call configure() first.")

        # Pass through to the base class chat() method — the generation
        # and retrieval logic is shared.
        return await self.chat(message=prompt)
