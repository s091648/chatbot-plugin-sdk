"""Backend Protocol — the only interface IngestProcessor / RetrieveProcessor depend on."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class SearchRow:
    """Single search result row returned by DatabaseBackend.search_dense()."""
    chunk_id: str
    article_id: str
    chunk_index: int
    content: str
    title: str | None
    url: str | None
    distance: float   # raw cosine distance (0 = identical, 2 = opposite)


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

    async def setup(self, dense_dim: int | None) -> None:
        """Idempotent.  Creates schema + tables if missing; validates dimension if existing.

        Called by :class:`IngestProcessor` on first use.
        """
        ...

    async def validate(self, dense_dim: int | None) -> None:
        """Read-only validation: tables must already exist.  Raises if missing.

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
    ) -> None:
        """Insert-or-replace article + chunks inside a single transaction.

        On success: commits.  On any error: rolls back, raises :exc:`DatabaseError`.
        """
        ...

    async def search_dense(
        self,
        query_vec: list[float],
        top_k: int,
    ) -> list[SearchRow]:
        """Cosine similarity search on the dense_vector column."""
        ...

    async def close(self) -> None:
        """Dispose connection pool.  Call on application shutdown."""
        ...
