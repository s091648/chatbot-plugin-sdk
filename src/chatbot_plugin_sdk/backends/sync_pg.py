"""SyncPgBackend — psycopg2 + sync SQLAlchemy, wrapped in run_in_executor.

The engine is NOT bound to any asyncio event loop, so a single instance can be
shared safely across multiple ``asyncio.run()`` calls from a ``ThreadPoolExecutor``.

Requires: psycopg2-binary  (pip install "chatbot-plugin-sdk[sync]")

Usage::

    backend = SyncPgBackend(DatabaseConfig(dbname="mydb", user="u", password="p"))
    processor.configure(backend=backend, dense=dense_provider)

    # ThreadPoolExecutor pattern (safe — engine is not loop-bound):
    def ingest_in_thread(article):
        asyncio.run(processor.ingest(article.content, metadata={"url": article.url}))

    with ThreadPoolExecutor(max_workers=4) as ex:
        ex.map(ingest_in_thread, articles)

    # Shutdown (any thread, any time after all work is done):
    import asyncio; asyncio.run(backend.close())
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from sqlalchemy import create_engine, inspect as sa_inspect, text
from sqlalchemy.orm import sessionmaker
import pgvector.sqlalchemy  # noqa: F401 — registers vector/sparsevec types with SQLAlchemy

import logging

from chatbot_plugin_sdk.backends.base import SearchRow
from chatbot_plugin_sdk.backends._pg_ddl import (
    _build_search_dense_sql,
    _build_search_sparse_sql,
    _build_upsert_article_sql,
    _DDL_CREATE_CHUNKS,
    _DDL_CREATE_ARTICLES,
    _DDL_CREATE_EXTENSION,
    _DDL_CREATE_SCHEMA,
    _DDL_IDX_URL,
    _DDL_TABLE_EXISTS,
    _DML_DELETE_CHUNKS,
    _DML_INSERT_CHUNK,
    _extract_article_metadata,
    _prepare_upsert_params,
    _check_dim_from_cols,
    _check_sparse_dim_from_cols,
    _dense_col_ddl,
    _dense_vec_str,
    _sparse_col_ddl,
    _to_sparsevec_string,
)
from chatbot_plugin_sdk.config import DatabaseConfig
from chatbot_plugin_sdk.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class SyncPgBackend:
    """Thread-safe, event-loop-independent database backend using psycopg2.

    All sync SQLAlchemy operations run in ``asyncio.get_running_loop().run_in_executor``
    so they don't block the calling coroutine's event loop, yet the engine itself
    carries no asyncio state and can be shared across threads.

    Thread safety:
      - The SQLAlchemy connection pool uses internal locks; sharing ``_engine``
        across threads is safe.
      - Each DB operation opens its own session + connection from the pool.
      - Sessions are never shared between calls.

    Multi-process note:
      - Do NOT share this backend across ``fork()`` boundaries.  Create a new
        instance in each child process.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        url = (
            f"postgresql+psycopg2://{config.user}:{config.password}"
            f"@{config.host}:{config.port}/{config.dbname}"
        )
        self._engine = create_engine(url, pool_size=5, max_overflow=10, future=True)
        self._Session = sessionmaker(self._engine)
        self.schema = config.schema
        self.articles_table = config.articles_table
        self.chunks_table = config.chunks_table
        self._sparse_dim: int | None = None

    # ── Async wrappers (schedule sync work onto the event loop's thread pool) ──

    async def setup(self, dense_dim: int | None, sparse_dim: int | None = None) -> None:
        self._sparse_dim = sparse_dim
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._setup_sync, dense_dim, sparse_dim)

    async def validate(self, dense_dim: int | None, sparse_dim: int | None = None) -> None:
        self._sparse_dim = sparse_dim
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._validate_sync, dense_dim, sparse_dim)

    async def upsert(
        self,
        article_id: uuid.UUID,
        metadata: dict,
        chunks: list[str],
        dense_vectors: list[list[float]] | None,
        sparse_vectors: list[dict[str, float]] | None,
        articles_column_values: dict[str, Any] | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._upsert_sync,
            article_id, metadata, chunks, dense_vectors, sparse_vectors, articles_column_values,
        )

    async def search_dense(self, query_vec: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[SearchRow]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_dense_sync, query_vec, top_k, filters)

    async def search_sparse(self, query_vec: dict[str, float], top_k: int, filters: dict[str, Any] | None = None) -> list[SearchRow]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sparse_sync, query_vec, top_k, filters)

    async def close(self) -> None:
        self._engine.dispose()

    # ── Sync implementations ────────────────────────────────────────────────

    def _setup_sync(self, dense_dim: int | None, sparse_dim: int | None) -> None:
        schema = self.schema
        at = self.articles_table
        ct = self.chunks_table
        with self._engine.begin() as conn:
            conn.execute(text(_DDL_CREATE_EXTENSION))
            conn.execute(text(_DDL_CREATE_SCHEMA.format(schema=schema)))
            r = conn.execute(text(_DDL_TABLE_EXISTS), {"s": schema, "t": ct})
            table_existed = r.fetchone() is not None
            if not table_existed:
                dense_col = _dense_col_ddl(dense_dim)
                sparse_col = _sparse_col_ddl(sparse_dim)
                conn.execute(text(_DDL_CREATE_ARTICLES.format(schema=schema, articles_table=at)))
                conn.execute(text(_DDL_CREATE_CHUNKS.format(
                    schema=schema, articles_table=at, chunks_table=ct,
                    dense_col=dense_col, sparse_col=sparse_col,
                )))
                conn.execute(text(_DDL_IDX_URL.format(schema=schema, articles_table=at)))
                logger.info(
                    "vector_tables_created",
                    extra={"schema": schema, "articles_table": at, "chunks_table": ct,
                           "dense_dim": dense_dim, "sparse_dim": sparse_dim},
                )
            else:
                logger.debug("vector_tables_exist", extra={"schema": schema, "chunks_table": ct})

        if table_existed and (dense_dim is not None or sparse_dim is not None):
            self._check_dim(dense_dim, sparse_dim)

    def _validate_sync(self, dense_dim: int | None, sparse_dim: int | None) -> None:
        schema = self.schema
        ct = self.chunks_table
        with self._engine.connect() as conn:
            r = conn.execute(text(_DDL_TABLE_EXISTS), {"s": schema, "t": ct})
            if r.fetchone() is None:
                raise DatabaseError(
                    f"Table {schema}.{ct} not found. "
                    "Run IngestProcessor first to create the schema."
                )
        if dense_dim is not None or sparse_dim is not None:
            self._check_dim(dense_dim, sparse_dim)

    def _check_dim(self, dense_dim: int | None, sparse_dim: int | None) -> None:
        try:
            cols = sa_inspect(self._engine).get_columns(self.chunks_table, schema=self.schema)
        except Exception:
            return
        if dense_dim is not None:
            _check_dim_from_cols(cols, dense_dim)
        if sparse_dim is not None:
            _check_sparse_dim_from_cols(cols, sparse_dim)

    def _upsert_sync(
        self,
        article_id: uuid.UUID,
        metadata: dict,
        chunks: list[str],
        dense_vectors: list[list[float]] | None,
        sparse_vectors: list[dict[str, float]] | None,
        articles_column_values: dict[str, Any] | None = None,
    ) -> None:
        schema = self.schema
        at = self.articles_table
        ct = self.chunks_table
        col_params, jsonb_metadata = _prepare_upsert_params(metadata, articles_column_values)
        try:
            with self._engine.begin() as conn:
                sql = _build_upsert_article_sql(schema, at, col_params)
                params = {"id": str(article_id)}
                params.update(col_params)
                params["metadata"] = json.dumps(jsonb_metadata) if jsonb_metadata is not None else None
                conn.execute(text(sql), params)

                conn.execute(
                    text(_DML_DELETE_CHUNKS.format(schema=schema, chunks_table=ct)),
                    {"article_id": str(article_id)},
                )
                for i, content in enumerate(chunks):
                    dense_val = _dense_vec_str(dense_vectors[i]) if dense_vectors else None
                    sparse_val = (
                        _to_sparsevec_string(sparse_vectors[i], self._sparse_dim)
                        if sparse_vectors is not None and self._sparse_dim
                        else None
                    )
                    conn.execute(
                        text(_DML_INSERT_CHUNK.format(schema=schema, chunks_table=ct)),
                        {
                            "article_id": str(article_id),
                            "chunk_index": i,
                            "content": content,
                            "dense_vector": dense_val,
                            "sparse_vector": sparse_val,
                        },
                    )
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Upsert failed for article_id={article_id}: {exc}") from exc

    def _search_dense_sync(self, query_vec: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[SearchRow]:
        schema = self.schema
        at = self.articles_table
        ct = self.chunks_table
        sql, filter_params = _build_search_dense_sql(schema, at, ct, filters)
        params = {"query_vec": _dense_vec_str(query_vec), "top_k": top_k}
        params.update(filter_params)
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).all()
        return [
            SearchRow(
                chunk_id=str(r.chunk_id),
                article_id=str(r.article_id),
                chunk_index=r.chunk_index,
                content=r.content,
                distance=float(r.distance),
                article_metadata=_extract_article_metadata(r._mapping),
            )
            for r in rows
        ]

    def _search_sparse_sync(self, query_vec: dict[str, float], top_k: int, filters: dict[str, Any] | None = None) -> list[SearchRow]:
        if not self._sparse_dim:
            return []
        schema = self.schema
        at = self.articles_table
        ct = self.chunks_table
        sql, filter_params = _build_search_sparse_sql(schema, at, ct, filters)
        sv_str = _to_sparsevec_string(query_vec, self._sparse_dim)
        params = {"query_vec": sv_str, "top_k": top_k}
        params.update(filter_params)
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).all()
        return [
            SearchRow(
                chunk_id=str(r.chunk_id),
                article_id=str(r.article_id),
                chunk_index=r.chunk_index,
                content=r.content,
                distance=float(r.distance),
                article_metadata=_extract_article_metadata(r._mapping),
            )
            for r in rows
        ]
