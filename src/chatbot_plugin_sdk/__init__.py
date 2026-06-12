"""chatbot_plugin_sdk — RAG ingest & retrieval SDK.

Quick start::

    from chatbot_plugin_sdk import (
        IngestProcessor, RetrieveProcessor,
        AsyncPgBackend, SyncPgBackend,  # choose one
        EndpointProvider, LocalProvider,
        DatabaseConfig,
    )

    # ThreadPoolExecutor scenario — use SyncPgBackend:
    backend = SyncPgBackend(DatabaseConfig(dbname="mydb", user="u", password="p"))

    # FastAPI / native asyncio — use AsyncPgBackend:
    backend = AsyncPgBackend(DatabaseConfig(dbname="mydb", user="u", password="p"))

    processor = IngestProcessor()
    processor.configure(
        backend=backend,
        dense=EndpointProvider(url="http://embed:8080", dimension=768),
    )
    await processor.ingest(full_text="...", metadata={"url": "https://...", "title": "..."})
"""

from chatbot_plugin_sdk.processors.ingest import IngestProcessor
from chatbot_plugin_sdk.processors.retrieve import RetrieveProcessor
from chatbot_plugin_sdk.backends.async_pg import AsyncPgBackend
from chatbot_plugin_sdk.backends.sync_pg import SyncPgBackend
from chatbot_plugin_sdk.backends.base import DatabaseBackend, SearchRow
from chatbot_plugin_sdk.providers.endpoint import EndpointProvider
from chatbot_plugin_sdk.providers.local import LocalProvider
from chatbot_plugin_sdk.config import DatabaseConfig
from chatbot_plugin_sdk.contracts.responses import SearchResponse, ChunkResult
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider
from chatbot_plugin_sdk.exceptions import (
    ToolboxError,
    NotConfiguredError,
    DatabaseError,
    EmbeddingError,
    ChunkingError,
)
from chatbot_plugin_sdk.rate_limit import (
    RateLimitStrategy,
    SlidingWindowStrategy,
    RateLimitExhausted,
)

__all__ = [
    # Processors
    "IngestProcessor",
    "RetrieveProcessor",
    # Backends
    "AsyncPgBackend",
    "SyncPgBackend",
    "DatabaseBackend",
    "SearchRow",
    # Providers
    "EndpointProvider",
    "LocalProvider",
    # Config
    "DatabaseConfig",
    # Responses
    "SearchResponse",
    "ChunkResult",
    # Protocols
    "DenseEmbeddingProvider",
    "SparseEmbeddingProvider",
    # Exceptions
    "ToolboxError",
    "NotConfiguredError",
    "DatabaseError",
    "EmbeddingError",
    "ChunkingError",
    # Rate limiting
    "RateLimitStrategy",
    "SlidingWindowStrategy",
    "RateLimitExhausted",
]
