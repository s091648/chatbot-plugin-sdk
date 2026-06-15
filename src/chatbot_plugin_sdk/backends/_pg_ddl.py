"""PostgreSQL DDL helpers shared by AsyncPgBackend and SyncPgBackend.

Why raw SQL (text()) instead of SQLAlchemy DDL API or MetaData.create_all()?

  1. CREATE EXTENSION is not part of SQLAlchemy's portable DDL API.
  2. CREATE SCHEMA IF NOT EXISTS with a runtime name cannot use MetaData.
  3. MetaData.create_all() does not reliably handle cross-schema FK references
     when the schema name is determined at runtime (not at import time).
  4. VECTOR(dim) requires baking the dimension into the DDL string; the ORM
     model is defined without a fixed dimension, so the column type is only
     concrete at setup time.

All DML and DQL also use raw SQL so that table names remain fully configurable
at runtime without needing dynamic ORM model classes.
"""
from __future__ import annotations

from chatbot_plugin_sdk.exceptions import DatabaseError

_DDL_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"
_DDL_CREATE_SCHEMA    = "CREATE SCHEMA IF NOT EXISTS {schema}"

# :s = schema, :t = chunks_table
_DDL_TABLE_EXISTS = (
    "SELECT 1 FROM information_schema.tables "
    "WHERE table_schema = :s AND table_name = :t"
)

_DDL_CREATE_ARTICLES = """\
CREATE TABLE IF NOT EXISTS {schema}.{articles_table} (
    id         UUID PRIMARY KEY,
    url        TEXT NOT NULL UNIQUE,
    title      TEXT,
    source     TEXT,
    metadata   JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)"""

_DDL_CREATE_CHUNKS = """\
CREATE TABLE IF NOT EXISTS {schema}.{chunks_table} (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id    UUID NOT NULL
                  REFERENCES {schema}.{articles_table}(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    {dense_col},
    {sparse_col},
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_{chunks_table}_article_chunk_idx UNIQUE (article_id, chunk_index)
)"""

_DDL_IDX_URL    = "CREATE INDEX IF NOT EXISTS idx_{articles_table}_url    ON {schema}.{articles_table}(url)"
_DDL_IDX_SOURCE = "CREATE INDEX IF NOT EXISTS idx_{articles_table}_source ON {schema}.{articles_table}(source)"

# ── DML (upsert) ────────────────────────────────────────────────────────────

_DML_UPSERT_ARTICLE = """\
INSERT INTO {schema}.{articles_table} (id, url, title, source, metadata)
VALUES (:id, :url, :title, :source, CAST(:metadata AS JSONB))
ON CONFLICT (id) DO UPDATE SET
    url        = EXCLUDED.url,
    title      = EXCLUDED.title,
    source     = EXCLUDED.source,
    metadata   = EXCLUDED.metadata,
    updated_at = now()"""

_DML_DELETE_CHUNKS = "DELETE FROM {schema}.{chunks_table} WHERE article_id = :article_id"

_DML_INSERT_CHUNK = """\
INSERT INTO {schema}.{chunks_table}
    (article_id, chunk_index, content, dense_vector, sparse_vector)
VALUES
    (:article_id, :chunk_index, :content,
     CAST(:dense_vector AS vector),
     CAST(:sparse_vector AS sparsevec))"""

# ── DQL (search) ────────────────────────────────────────────────────────────

_DQL_SEARCH_DENSE = """\
SELECT
    ac.id          AS chunk_id,
    ac.article_id,
    ac.chunk_index,
    ac.content,
    a.title,
    a.url,
    ac.dense_vector <=> CAST(:query_vec AS vector) AS distance
FROM {schema}.{chunks_table} ac
JOIN {schema}.{articles_table} a ON ac.article_id = a.id
WHERE ac.dense_vector IS NOT NULL
ORDER BY distance
LIMIT :top_k"""

_DQL_SEARCH_SPARSE = """\
SELECT
    ac.id          AS chunk_id,
    ac.article_id,
    ac.chunk_index,
    ac.content,
    a.title,
    a.url,
    (ac.sparse_vector <#> CAST(:query_vec AS sparsevec)) AS distance
FROM {schema}.{chunks_table} ac
JOIN {schema}.{articles_table} a ON ac.article_id = a.id
WHERE ac.sparse_vector IS NOT NULL
ORDER BY distance
LIMIT :top_k"""


def _dense_col_ddl(dense_dim: int | None) -> str:
    return f"dense_vector VECTOR({dense_dim})" if dense_dim else "dense_vector VECTOR(768)"


def _sparse_col_ddl(sparse_dim: int | None) -> str:
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


def _dense_vec_str(vec: list[float]) -> str:
    """Convert a Python float list to the PostgreSQL vector wire format: '[0.1,0.2,...]'."""
    return "[" + ",".join(str(x) for x in vec) + "]"


def _check_dim_from_cols(cols: list[dict], dense_dim: int) -> None:
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
