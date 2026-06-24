"""Backend Protocol and shared result types for the backends subpackage."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class SearchRow:
    """Single search result row returned by DatabaseBackend.search_dense()."""
    chunk_id: str
    article_id: str
    chunk_index: int
    content: str
    distance: float   # raw cosine distance (0 = identical, 2 = opposite)
    article_metadata: dict[str, Any]  # article-level columns (title, url, etc.)


@runtime_checkable
class DatabaseBackend(Protocol):
    """Storage backend abstraction.

    Two implementations are provided:

    - :class:`AsyncPgBackend` — native asyncpg, binds to the creating event loop.
      Use with FastAPI or any long-running async application.

    - :class:`SyncPgBackend` — psycopg2 wrapped in ``run_in_executor``, not bound
      to any event loop.  Use when callers use ``asyncio.run()`` from a
      ``ThreadPoolExecutor``.

    The processors always ``await`` backend methods, so the async signature is
    mandatory for both implementations.
    """

    schema: str

    async def setup(self, dense_dim: int | None, sparse_dim: int | None = None) -> None:
        """Idempotent.  Creates schema + tables if missing; validates dimension if existing.

        Args:
            dense_dim: VECTOR(N) dimension.  ``None`` → default VECTOR(768) placeholder.
            sparse_dim: SPARSEVEC(N) vocabulary size (e.g. 30522 for BERT SPLADE).
                        ``None`` → column stored as JSONB (sparse provider not configured).

        Called by :class:`IngestProcessor` on first use.
        """
        ...

    async def validate(self, dense_dim: int | None, sparse_dim: int | None = None) -> None:
        """Read-only validation: tables must already exist.  Raises if missing.

        Args:
            dense_dim: Expected VECTOR dimension from the dense provider.
            sparse_dim: Expected SPARSEVEC dimension from the sparse provider.

        Called by :class:`RetrieveProcessor` on first use.
        """
        ...

    async def upsert(
        self,
        article_id: uuid.UUID,
        metadata: dict,
        chunks: list[str],
        dense_vectors: list[list[float]] | None,
        sparse_vectors: list[dict[str, float]] | None,
        articles_column_values: dict[str, Any] | None = None,
    ) -> None:
        """Insert-or-replace article + chunks inside a single transaction.

        ``url`` must be present in ``articles_column_values`` — it is used
        by :class:`IngestProcessor` to derive a deterministic ``article_id``
        via ``uuid.uuid5(NAMESPACE_URL, url)``.

        Args:
            metadata: Opaque JSONB blob — the SDK never interprets its keys.
            articles_column_values: SQL column values for the articles table
                                    (url, title, source, public_article_id,
                                    topic_id).  Keys must be in the known
                                    article column set.  ``url`` is required.

        On success: commits.  On any error: rolls back, raises :exc:`DatabaseError`.
        """
        ...

    async def search_dense(
        self,
        query_vec: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchRow]:
        """Cosine similarity search on the dense_vector column.

        Args:
            filters: Column-level filters on the articles table, e.g.
                     ``{"topic_id": "uuid"}``. Keys must be known article columns.
        """
        ...

    async def search_sparse(
        self,
        query_vec: dict[str, float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchRow]:
        """Maximum inner product search on the sparse_vector column (<#> operator).

        ``distance`` in returned rows is the *negative* inner product (lower = more similar),
        matching the same convention as ``search_dense`` so :func:`_rrf_merge` can treat both
        lists uniformly by rank.
        """
        ...

    async def close(self) -> None:
        """Dispose connection pool.  Call on application shutdown."""
        ...
