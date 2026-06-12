"""chatbot_plugin_sdk — RAG ingest & retrieval SDK.

Entry points:

- :class:`IngestProcessor` — normalize → chunk → embed → store
- :class:`RetrieveProcessor` — embed query → vector search → :class:`SearchResponse`

Providers (inject into ``configure()``):

- :class:`EndpointProvider` — HTTP embedding (external API or sidecar)
- :class:`LocalProvider` — in-process callable (fastembed, etc.)

Protocols (for custom provider implementations):

- :class:`DenseEmbeddingProvider`
- :class:`SparseEmbeddingProvider`
"""

from chatbot_plugin_sdk.processors.ingest import IngestProcessor
from chatbot_plugin_sdk.processors.retrieve import RetrieveProcessor
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

__all__ = [
    "IngestProcessor",
    "RetrieveProcessor",
    "EndpointProvider",
    "LocalProvider",
    "DatabaseConfig",
    "SearchResponse",
    "ChunkResult",
    "DenseEmbeddingProvider",
    "SparseEmbeddingProvider",
    "ToolboxError",
    "NotConfiguredError",
    "DatabaseError",
    "EmbeddingError",
    "ChunkingError",
]
