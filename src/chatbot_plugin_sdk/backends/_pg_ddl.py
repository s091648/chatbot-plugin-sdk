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

import re
import uuid as _uuid_mod
from typing import Any
from collections.abc import Mapping

from chatbot_plugin_sdk.exceptions import DatabaseError

_SAFE_COLUMN_NAME = re.compile(r'^[a-z][a-z0-9_]*$')


def _validate_column_name(name: str) -> None:
    if not _SAFE_COLUMN_NAME.match(name):
        raise DatabaseError(
            f"Column name {name!r} is invalid — only lowercase letters, "
            "digits, and underscores are allowed, starting with a letter."
        )


def _is_uuid_value(value: Any) -> bool:
    """Return True if value is parseable as a UUID string."""
    if value is None:
        return False
    try:
        _uuid_mod.UUID(str(value))
        return True
    except (ValueError, AttributeError):
        return False


def _prepare_upsert_params(
    metadata: dict | None = None,
    articles_column_values: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict | None]:
    """Validate column names and split into SQL column params + JSONB metadata blob.

    Column names in ``articles_column_values`` must match ``^[a-z][a-z0-9_]*$``
    to prevent SQL injection.  Whether those columns exist in the target table
    is the caller's responsibility.
    """
    col_params: dict[str, Any] = {}
    if articles_column_values:
        for key, value in articles_column_values.items():
            _validate_column_name(key)
            col_params[key] = value
    jsonb_metadata = dict(metadata) if metadata else None
    return col_params, jsonb_metadata


_CHUNK_ROW_KEYS = frozenset({"chunk_id", "article_id", "chunk_index", "content", "distance"})


def _extract_article_metadata(row_mapping: Mapping) -> dict[str, Any]:
    """Return all article-level values from a SQL result row.

    Excludes the chunk-level fields (chunk_id, article_id, chunk_index, content,
    distance).  All remaining columns — whatever the caller's migration defined —
    are returned as-is.
    """
    return {k: v for k, v in row_mapping.items() if k not in _CHUNK_ROW_KEYS and v is not None}


def _build_upsert_article_sql(
    schema: str,
    articles_table: str,
    col_params: dict[str, Any],
) -> str:
    """Build parameterized INSERT … ON CONFLICT (id) DO UPDATE for the articles table.

    ``id`` (UUID derived from uuid5) is the PK and the conflict target.
    All keys in ``col_params`` become INSERT columns.  UUID-valued entries
    receive an explicit ``CAST(… AS UUID)``.  Column existence is the caller's
    responsibility.
    """
    cols = sorted(col_params.keys())
    insert_cols = ["id"] + cols + ["metadata"]

    cast_placeholders = ["CAST(:id AS UUID)"]
    for c in cols:
        if _is_uuid_value(col_params.get(c)):
            cast_placeholders.append(f"CAST(:{c} AS UUID)")
        else:
            cast_placeholders.append(f":{c}")
    cast_placeholders.append("CAST(:metadata AS JSONB)")

    update_sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
    update_sets += ", metadata = EXCLUDED.metadata, updated_at = now()"

    return (
        f"INSERT INTO {schema}.{articles_table} ({', '.join(insert_cols)})\n"
        f"VALUES ({', '.join(cast_placeholders)})\n"
        f"ON CONFLICT (id) DO UPDATE SET\n    {update_sets}"
    )


def _build_search_where(
    filters: dict[str, Any] | None,
    table_alias: str = "a",
) -> tuple[str, dict[str, Any]]:
    """Build a parameterized WHERE clause fragment from a filters dict.

    Column names are validated against ``_SAFE_COLUMN_NAME`` to prevent SQL
    injection.  Whether those columns exist in the target table is the caller's
    responsibility.  UUID-valued entries receive an explicit ``CAST(… AS UUID)``.
    """
    if not filters:
        return "", {}

    fragments = []
    params: dict[str, Any] = {}

    for col, value in filters.items():
        _validate_column_name(col)
        param_name = f"_f_{col}"
        if _is_uuid_value(value):
            fragments.append(f"({table_alias}.{col} = CAST(:{param_name} AS UUID))")
        else:
            fragments.append(f"({table_alias}.{col} = :{param_name})")
        params[param_name] = value

    where = " AND " + " AND ".join(fragments)
    return where, params


def _build_search_dense_sql(
    schema: str,
    articles_table: str,
    chunks_table: str,
    filters: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    where_frag, filter_params = _build_search_where(filters)
    sql = (
        f"SELECT\n"
        f"    ac.id          AS chunk_id,\n"
        f"    ac.article_id,\n"
        f"    ac.chunk_index,\n"
        f"    ac.content,\n"
        f"    a.*,\n"
        f"    ac.dense_vector <=> CAST(:query_vec AS vector) AS distance\n"
        f"FROM {schema}.{chunks_table} ac\n"
        f"JOIN {schema}.{articles_table} a ON ac.article_id = a.id\n"
        f"WHERE ac.dense_vector IS NOT NULL\n"
        f"{where_frag}\n"
        f"ORDER BY distance\n"
        f"LIMIT :top_k"
    )
    return sql, filter_params


def _build_search_sparse_sql(
    schema: str,
    articles_table: str,
    chunks_table: str,
    filters: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    where_frag, filter_params = _build_search_where(filters)
    sql = (
        f"SELECT\n"
        f"    ac.id          AS chunk_id,\n"
        f"    ac.article_id,\n"
        f"    ac.chunk_index,\n"
        f"    ac.content,\n"
        f"    a.*,\n"
        f"    (ac.sparse_vector <#> CAST(:query_vec AS sparsevec)) AS distance\n"
        f"FROM {schema}.{chunks_table} ac\n"
        f"JOIN {schema}.{articles_table} a ON ac.article_id = a.id\n"
        f"WHERE ac.sparse_vector IS NOT NULL\n"
        f"{where_frag}\n"
        f"ORDER BY distance\n"
        f"LIMIT :top_k"
    )
    return sql, filter_params


_DDL_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"
_DDL_CREATE_SCHEMA    = "CREATE SCHEMA IF NOT EXISTS {schema}"

# :s = schema, :t = chunks_table
_DDL_TABLE_EXISTS = (
    "SELECT 1 FROM information_schema.tables "
    "WHERE table_schema = :s AND table_name = :t"
)

# Minimal SDK schema — callers add business columns via their own migration.
# id = uuid5(NAMESPACE_URL, url), url is the dedup key used to derive it.
_DDL_CREATE_ARTICLES = """\
CREATE TABLE IF NOT EXISTS {schema}.{articles_table} (
    id         UUID PRIMARY KEY,
    url        TEXT NOT NULL UNIQUE,
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

_DDL_IDX_URL = "CREATE INDEX IF NOT EXISTS idx_{articles_table}_url ON {schema}.{articles_table}(url)"

# ── DML (delete / insert chunks) ─────────────────────────────────────────────

_DML_DELETE_CHUNKS = "DELETE FROM {schema}.{chunks_table} WHERE article_id = :article_id"

_DML_INSERT_CHUNK = """\
INSERT INTO {schema}.{chunks_table}
    (article_id, chunk_index, content, dense_vector, sparse_vector)
VALUES
    (:article_id, :chunk_index, :content,
     CAST(:dense_vector AS vector),
     CAST(:sparse_vector AS sparsevec))"""


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
