"""PostgreSQL DDL helpers shared by AsyncPgBackend and SyncPgBackend.

Why raw SQL (text()) instead of SQLAlchemy DDL API or MetaData.create_all()?

  1. CREATE EXTENSION is not part of SQLAlchemy's portable DDL API.
  2. CREATE SCHEMA IF NOT EXISTS with a runtime name cannot use MetaData.
  3. MetaData.create_all() does not reliably handle cross-schema FK references
     when the schema name is determined at runtime (not at import time).
  4. VECTOR(dim) requires baking the dimension into the DDL string; the ORM
     model is defined without a fixed dimension, so the column type is only
     concrete at setup time.

DML (upsert) uses the SQLAlchemy ORM (session.add / select) because ORM
handles object identity, auto-flush, and transaction rollback cleanly.

search_dense uses SQLAlchemy Core expressions so that cosine_distance() —
registered on the Vector column by the pgvector SQLAlchemy integration —
can be called as a column method rather than embedded in raw SQL.

Dimension introspection uses SQLAlchemy reflect (sa_inspect) because the ORM
model does not record the exact VECTOR dimension at runtime.
"""
from __future__ import annotations

from chatbot_plugin_sdk.exceptions import DatabaseError

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
    {sparse_col},
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_article_chunk_idx UNIQUE (article_id, chunk_index)
)"""
_DDL_IDX_URL    = "CREATE INDEX IF NOT EXISTS idx_articles_url    ON {schema}.articles(url)"
_DDL_IDX_SOURCE = "CREATE INDEX IF NOT EXISTS idx_articles_source ON {schema}.articles(source)"


def _dense_col_ddl(dense_dim: int | None) -> str:
    """Return the DDL fragment for the dense_vector column."""
    return f"dense_vector VECTOR({dense_dim})" if dense_dim else "dense_vector VECTOR(768)"


def _sparse_col_ddl(sparse_dim: int | None) -> str:
    """Return the DDL fragment for the sparse_vector column.

    Uses SPARSEVEC({dim}) when a sparse provider dimension is configured (pgvector >= 0.7.0).
    Falls back to JSONB when no sparse provider is active so the column is always present.
    """
    if sparse_dim:
        return f"sparse_vector SPARSEVEC({sparse_dim})"
    return "sparse_vector JSONB"


def _to_sparsevec_string(d: dict[str, float], dim: int) -> str:
    """Convert a {str_index: weight} dict to the PostgreSQL SPARSEVEC wire format.

    Example: {0: 0.5, 1: 0.3}, dim=30522  →  "{0:0.5,1:0.3}/30522"
    Zero-weight entries are omitted (they carry no information and waste storage).
    """
    items = ",".join(
        f"{int(k)}:{v}"
        for k, v in sorted(d.items(), key=lambda x: int(x[0]))
        if v != 0
    )
    return f"{{{items}}}/{dim}"


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


def _check_sparse_dim_from_cols(cols: list[dict], sparse_dim: int) -> None:
    """Raise DatabaseError if the stored SPARSEVEC dimension doesn't match sparse_dim.

    Args:
        cols: Column dicts from ``sqlalchemy.inspect(...).get_columns()``.
        sparse_dim: Expected vocabulary size from the sparse embedding provider.
    """
    for col in cols:
        if col["name"] == "sparse_vector":
            db_dim = getattr(col["type"], "dim", None)
            if db_dim is not None and db_dim != sparse_dim:
                raise DatabaseError(
                    f"Sparse dimension mismatch: DB has SPARSEVEC({db_dim}) but "
                    f"provider.dimension={sparse_dim}. "
                    "Use the same SPLADE model that created the table."
                )
            break
