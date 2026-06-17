"""AsyncPgBackend — native asyncpg + async SQLAlchemy.

Binds to the event loop that constructs it.  One instance per process / event-loop.
Suitable for FastAPI and long-running asyncio applications.

Usage::

    backend = AsyncPgBackend(DatabaseConfig(dbname="mydb", user="u", password="p"))
    processor.configure(backend=backend, dense=dense_provider)
    # on app shutdown:
    await backend.close()
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import inspect as sa_inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
import pgvector.sqlalchemy  # noqa: F401 — registers vector/sparsevec types with SQLAlchemy

from chatbot_plugin_sdk.backends.base import SearchRow
from chatbot_plugin_sdk.backends._pg_ddl import (
    _build_search_dense_sql,
    _build_search_sparse_sql,
    _build_upsert_article_sql,
    _DDL_CREATE_CHUNKS,
    _DDL_CREATE_ARTICLES,
    _DDL_CREATE_EXTENSION,
    _DDL_CREATE_SCHEMA,
    _DDL_IDX_SOURCE,
    _DDL_IDX_URL,
    _DDL_TABLE_EXISTS,
    _DML_DELETE_CHUNKS,
    _DML_INSERT_CHUNK,
    _split_article_fields,
    _check_dim_from_cols,
    _check_sparse_dim_from_cols,
    _dense_col_ddl,
    _dense_vec_str,
    _sparse_col_ddl,
    _to_sparsevec_string,
)
from chatbot_plugin_sdk.config import DatabaseConfig
from chatbot_plugin_sdk.exceptions import DatabaseError


class AsyncPgBackend:
    """Native asyncpg backend.  Not safe to share across multiple ``asyncio.run()`` calls
    in different threads — use :class:`SyncPgBackend` for ThreadPoolExecutor scenarios.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        url = (
            f"postgresql+asyncpg://{config.user}:{config.password}"
            f"@{config.host}:{config.port}/{config.dbname}"
        )
        self._engine = create_async_engine(url, pool_size=5, max_overflow=10, future=True)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self.schema = config.schema
        self.articles_table = config.articles_table
        self.chunks_table = config.chunks_table
        self._sparse_dim: int | None = None

    # ── Setup / validation ─────────────────────────────────────────────────

    async def setup(self, dense_dim: int | None, sparse_dim: int | None = None) -> None:
        self._sparse_dim = sparse_dim
        await self._setup_ddl(dense_dim, sparse_dim)

    async def validate(self, dense_dim: int | None, sparse_dim: int | None = None) -> None:
        self._sparse_dim = sparse_dim
        schema = self.schema
        ct = self.chunks_table
        async with self._engine.connect() as conn:
            r = await conn.execute(text(_DDL_TABLE_EXISTS), {"s": schema, "t": ct})
            if r.fetchone() is None:
                raise DatabaseError(
                    f"Table {schema}.{ct} not found. "
                    "Run IngestProcessor first to create the schema."
                )
        if dense_dim is not None or sparse_dim is not None:
            await self._check_dim(dense_dim, sparse_dim)

    # ── Write ──────────────────────────────────────────────────────────────

    async def upsert(
        self,
        article_id: uuid.UUID,
        metadata: dict,
        chunks: list[str],
        dense_vectors: list[list[float]] | None,
        sparse_vectors: list[dict[str, float]] | None,
        article_columns: dict[str, Any] | None = None,
    ) -> None:
        schema = self.schema
        at = self.articles_table
        ct = self.chunks_table
        col_params, jsonb_metadata = _split_article_fields(metadata, article_columns)
        try:
            async with self._engine.begin() as conn:
                sql = _build_upsert_article_sql(schema, at, col_params)
                params = {"id": str(article_id)}
                params.update(col_params)
                params["metadata"] = json.dumps(jsonb_metadata) if jsonb_metadata is not None else None
                await conn.execute(text(sql), params)

                await conn.execute(
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
                    await conn.execute(
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
            raise DatabaseError(f"Upsert failed for article {article_id}: {exc}") from exc

    # ── Read ───────────────────────────────────────────────────────────────

    async def search_dense(
        self,
        query_vec: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchRow]:
        schema = self.schema
        at = self.articles_table
        ct = self.chunks_table
        sql, filter_params = _build_search_dense_sql(schema, at, ct, filters)
        params = {"query_vec": _dense_vec_str(query_vec), "top_k": top_k}
        params.update(filter_params)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), params)).all()
        return [
            SearchRow(
                chunk_id=str(r.chunk_id),
                article_id=str(r.article_id),
                chunk_index=r.chunk_index,
                content=r.content,
                title=r.title,
                url=r.url,
                distance=float(r.distance),
                public_article_id=str(r.public_article_id) if r.public_article_id else None,
            )
            for r in rows
        ]

    async def search_sparse(
        self,
        query_vec: dict[str, float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchRow]:
        if not self._sparse_dim:
            return []
        schema = self.schema
        at = self.articles_table
        ct = self.chunks_table
        sql, filter_params = _build_search_sparse_sql(schema, at, ct, filters)
        sv_str = _to_sparsevec_string(query_vec, self._sparse_dim)
        params = {"query_vec": sv_str, "top_k": top_k}
        params.update(filter_params)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), params)).all()
        return [
            SearchRow(
                chunk_id=str(r.chunk_id),
                article_id=str(r.article_id),
                chunk_index=r.chunk_index,
                content=r.content,
                title=r.title,
                url=r.url,
                distance=float(r.distance),
                public_article_id=str(r.public_article_id) if r.public_article_id else None,
            )
            for r in rows
        ]

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._engine.dispose()

    # ── Private helpers ────────────────────────────────────────────────────

    async def _setup_ddl(self, dense_dim: int | None, sparse_dim: int | None) -> None:
        schema = self.schema
        at = self.articles_table
        ct = self.chunks_table
        async with self._engine.begin() as conn:
            await conn.execute(text(_DDL_CREATE_EXTENSION))
            await conn.execute(text(_DDL_CREATE_SCHEMA.format(schema=schema)))
            r = await conn.execute(text(_DDL_TABLE_EXISTS), {"s": schema, "t": ct})
            table_existed = r.fetchone() is not None
            if not table_existed:
                dense_col = _dense_col_ddl(dense_dim)
                sparse_col = _sparse_col_ddl(sparse_dim)
                await conn.execute(text(_DDL_CREATE_ARTICLES.format(schema=schema, articles_table=at)))
                await conn.execute(text(_DDL_CREATE_CHUNKS.format(
                    schema=schema, articles_table=at, chunks_table=ct,
                    dense_col=dense_col, sparse_col=sparse_col,
                )))
                await conn.execute(text(_DDL_IDX_URL.format(schema=schema, articles_table=at)))
                await conn.execute(text(_DDL_IDX_SOURCE.format(schema=schema, articles_table=at)))

        if table_existed and (dense_dim is not None or sparse_dim is not None):
            await self._check_dim(dense_dim, sparse_dim)

    async def _check_dim(self, dense_dim: int | None, sparse_dim: int | None) -> None:
        ct = self.chunks_table
        async with self._engine.connect() as conn:
            cols = await conn.run_sync(
                lambda sync_conn: sa_inspect(sync_conn).get_columns(ct, schema=self.schema)
            )
        if dense_dim is not None:
            _check_dim_from_cols(cols, dense_dim)
        if sparse_dim is not None:
            _check_sparse_dim_from_cols(cols, sparse_dim)
