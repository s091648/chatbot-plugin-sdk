"""chatbot_plugin_sdk — Python SDK for the vector storage toolbox.

Provides two main entry points:

- :class:`RagArticleProcessor` — ingestion pipeline (normalise → chunk → embed → save)
- :class:`RagQueryProcessor` — read-only RAG queries (retrieval + generation)

Usage example::

    # Pipeline: write article to DB
    from chatbot_plugin_sdk import RagArticleProcessor

    ingest = RagArticleProcessor()
    ingest.configure(
        dbname="chatbot_plugin",
        user="postgres",
        password="postgres",
        embedding_model_api="http://localhost:8080",
    )
    await ingest.ingest(
        full_text="Retrieval augmented generation is ...",
        metadata={"url": "https://example.com/article", "title": "RAG 101"},
    )

    # Backend service: query with RAG
    from chatbot_plugin_sdk import RagQueryProcessor

    query = RagQueryProcessor()
    query.configure(
        dbname="chatbot_plugin",
        user="postgres",
        password="postgres",
        embedding_model_api="http://localhost:8080",
    )
    resp = await query.query("What is RAG?")
    print(resp.reply)
"""

from chatbot_plugin_sdk.config import DatabaseConfig, EmbeddingModelConfig
from chatbot_plugin_sdk.ingest import RagArticleProcessor
from chatbot_plugin_sdk.query import RagQueryProcessor
from chatbot_plugin_sdk.exceptions import (
    ChunkingError,
    DatabaseError,
    EmbeddingError,
    LLMError,
    NotConfiguredError,
    ToolboxError,
)

__all__ = [
    "RagArticleProcessor",
    "RagQueryProcessor",
    "DatabaseConfig",
    "EmbeddingModelConfig",
    "ToolboxError",
    "NotConfiguredError",
    "DatabaseError",
    "EmbeddingError",
    "ChunkingError",
    "LLMError",
]
