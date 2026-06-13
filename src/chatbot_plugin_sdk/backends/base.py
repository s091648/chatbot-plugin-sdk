"""Backend Protocol + shared PostgreSQL DDL helpers.

Both AsyncPgBackend and SyncPgBackend import the constants and helpers below so
that the DDL strings and the dimension-check logic live in exactly one place.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from chatbot_plugin_sdk.exceptions import DatabaseError


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


# ── Shared PostgreSQL DDL helpers ─────────────────────────────────────────────
#
# Why raw SQL (text()) instead of SQLAlchemy DDL API or MetaData.create_all()?
#
#   1. CREATE EXTENSION — not part of SQLAlchemy's portable DDL API.
#   2. CREATE SCHEMA IF NOT EXISTS with a runtime name cannot use MetaData.
#   3. MetaData.create_all() does not reliably handle cross-schema FK references
#      when the schema name is determined at runtime (not at import time).
#   4. VECTOR(dim) requires baking the dimension into the DDL string; the ORM
#      model is defined without a fixed dimension, so the column type is only
#      concrete at setup time.
#
# DML (upsert) uses the SQLAlchemy ORM (session.add / select) because ORM
# handles object identity, auto-flush, and transaction rollback cleanly.
#
# search_dense uses SQLAlchemy Core expressions so that cosine_distance() —
# registered on the Vector column by the pgvector SQLAlchemy integration —
# can be called as a column method rather than embedded raw SQL.
#
# Dimension introspection uses SQLAlchemy reflect (sa_inspect) because the ORM
# model does not record the exact VECTOR dimension at runtime.

_DDL_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"
_DDL_CREATE_SCHEMA    = "CREATE SCHEMA IF NOT EXISTS {schema}"
_DDL_TABLE_EXISTS     = (
    "SELECT 1 FROM information_schema.tables "
    "WHERE table_schema = :s AND table_name = 'article_chunks'"
)
_DDL_CREATE_ARTICLES = """\
CREATE TABLE IF NOT EXISTS {schema}.articles (
    id         UUID PRIMARY KEY,
    url        TEXT NOT NULL UNIQUE,
    title      TEXT,
    source     TEXT,
    metadata   JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)"""
_DDL_CREATE_CHUNKS = """\
CREATE TABLE IF NOT EXISTS {schema}.article_chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id    UUID NOT NULL
                  REFERENCES {schema}.articles(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    {dense_col},
    sparse_vector JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_article_chunk_idx UNIQUE (article_id, chunk_index)
)"""
_DDL_IDX_URL    = "CREATE INDEX IF NOT EXISTS idx_articles_url    ON {schema}.articles(url)"
_DDL_IDX_SOURCE = "CREATE INDEX IF NOT EXISTS idx_articles_source ON {schema}.articles(source)"


def _dense_col_ddl(dense_dim: int | None) -> str:
    """Return the DDL fragment for the dense_vector column."""
    return f"dense_vector VECTOR({dense_dim})" if dense_dim else "dense_vector VECTOR(768)"


def _check_dim_from_cols(cols: list[dict], dense_dim: int) -> None:
    """Raise DatabaseError if the stored VECTOR dimension doesn't match dense_dim.

    Args:
        cols: Column dicts from ``sqlalchemy.inspect(...).get_columns()``.
        dense_dim: Expected dimension from the embedding provider.
    """
    for col in cols:
        if col["name"] == "dense_vector":
            db_dim = getattr(col["type"], "dim", None)
            if db_dim is not None and db_dim != dense_dim:
                raise DatabaseError(
                    f"Dimension mismatch: DB has VECTOR({db_dim}) but "
                    f"provider.dimension={dense_dim}. "
                    "Use the same embedding model that created the table."
                )
            break
