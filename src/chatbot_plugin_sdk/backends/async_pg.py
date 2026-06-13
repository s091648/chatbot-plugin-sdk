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

import uuid

from sqlalchemy import delete, inspect as sa_inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from chatbot_plugin_sdk.backends.base import SearchRow
from chatbot_plugin_sdk.backends._pg_ddl import (
    _DDL_CREATE_CHUNKS,
    _DDL_CREATE_ARTICLES,
    _DDL_CREATE_EXTENSION,
    _DDL_CREATE_SCHEMA,
    _DDL_IDX_SOURCE,
    _DDL_IDX_URL,
    _DDL_TABLE_EXISTS,
    _check_dim_from_cols,
    _check_sparse_dim_from_cols,
    _dense_col_ddl,
    _sparse_col_ddl,
    _to_sparsevec_string,
)
from chatbot_plugin_sdk.config import DatabaseConfig
from chatbot_plugin_sdk.exceptions import DatabaseError
from chatbot_plugin_sdk.models.article import Article
from chatbot_plugin_sdk.models.chunk import ArticleChunk


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
        self._sparse_dim: int | None = None

    # ── Setup / validation ─────────────────────────────────────────────────

    async def setup(self, dense_dim: int | None, sparse_dim: int | None = None) -> None:
        self._sparse_dim = sparse_dim
        await self._setup_ddl(dense_dim, sparse_dim)

    async def validate(self, dense_dim: int | None, sparse_dim: int | None = None) -> None:
        self._sparse_dim = sparse_dim
        schema = self.schema
        async with self._engine.connect() as conn:
            r = await conn.execute(text(_DDL_TABLE_EXISTS), {"s": schema})
            if r.fetchone() is None:
                raise DatabaseError(
                    f"Table {schema}.article_chunks not found. "
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
    ) -> None:
        session = self._session_factory()
        try:
            async with session.begin():
                existing = (await session.execute(
                    select(Article).where(Article.id == article_id)
                )).scalar_one_or_none()

                if existing is not None:
                    existing.url = metadata.get("url", "")
                    existing.title = metadata.get("title")
                    existing.source = metadata.get("source")
                    existing.metadata_ = metadata.get("metadata")
                    await session.execute(
                        delete(ArticleChunk).where(ArticleChunk.article_id == article_id)
                    )
                else:
                    session.add(Article(
                        id=article_id,
                        url=metadata.get("url", ""),
                        title=metadata.get("title"),
                        source=metadata.get("source"),
                        metadata_=metadata.get("metadata"),
                    ))

                for i, content in enumerate(chunks):
                    sparse_val = None
                    if sparse_vectors is not None and self._sparse_dim:
                        sparse_val = _to_sparsevec_string(sparse_vectors[i], self._sparse_dim)
                    session.add(ArticleChunk(
                        article_id=article_id,
                        chunk_index=i,
                        content=content,
                        dense_vector=dense_vectors[i] if dense_vectors else None,
                        sparse_vector=sparse_val,
                    ))
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Upsert failed for article {article_id}: {exc}") from exc
        finally:
            await session.close()

    async def search_sparse(self, query_vec: dict[str, float], top_k: int) -> list[SearchRow]:
        """Inner-product search on the SPARSEVEC column using the <#> operator."""
        if not self._sparse_dim:
            return []
        sv_str = _to_sparsevec_string(query_vec, self._sparse_dim)
        async with self._session_factory() as db:
            stmt = (
                select(
                    ArticleChunk.id.label("chunk_id"),
                    ArticleChunk.article_id,
                    ArticleChunk.chunk_index,
                    ArticleChunk.content,
                    Article.title,
                    Article.url,
                    ArticleChunk.sparse_vector.max_inner_product(sv_str).label("distance"),
                )
                .join(Article, ArticleChunk.article_id == Article.id)
                .where(ArticleChunk.sparse_vector.isnot(None))
                .order_by("distance")  # <#> returns negative inner product; ASC = most similar first
                .limit(top_k)
            )
            rows = (await db.execute(stmt)).all()

        return [
            SearchRow(
                chunk_id=str(r.chunk_id),
                article_id=str(r.article_id),
                chunk_index=r.chunk_index,
                content=r.content,
                title=r.title,
                url=r.url,
                distance=float(r.distance),
            )
            for r in rows
        ]

    # ── Read ───────────────────────────────────────────────────────────────

    async def search_dense(self, query_vec: list[float], top_k: int) -> list[SearchRow]:
        async with self._session_factory() as db:
            stmt = (
                select(
                    ArticleChunk.id.label("chunk_id"),
                    ArticleChunk.article_id,
                    ArticleChunk.chunk_index,
                    ArticleChunk.content,
                    Article.title,
                    Article.url,
                    ArticleChunk.dense_vector.cosine_distance(query_vec).label("distance"),
                )
                .join(Article, ArticleChunk.article_id == Article.id)
                .where(ArticleChunk.dense_vector.isnot(None))
                .order_by("distance")
                .limit(top_k)
            )
            rows = (await db.execute(stmt)).all()

        return [
            SearchRow(
                chunk_id=str(r.chunk_id),
                article_id=str(r.article_id),
                chunk_index=r.chunk_index,
                content=r.content,
                title=r.title,
                url=r.url,
                distance=float(r.distance),
            )
            for r in rows
        ]

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Dispose all pooled connections.  Call on application shutdown."""
        await self._engine.dispose()

    # ── Private helpers ────────────────────────────────────────────────────

    async def _setup_ddl(self, dense_dim: int | None, sparse_dim: int | None) -> None:
        """Create schema + tables if missing; validate dimensions if table already existed."""
        schema = self.schema
        async with self._engine.begin() as conn:
            await conn.execute(text(_DDL_CREATE_EXTENSION))
            await conn.execute(text(_DDL_CREATE_SCHEMA.format(schema=schema)))
            r = await conn.execute(text(_DDL_TABLE_EXISTS), {"s": schema})
            table_existed = r.fetchone() is not None
            if not table_existed:
                dense_col = _dense_col_ddl(dense_dim)
                sparse_col = _sparse_col_ddl(sparse_dim)
                await conn.execute(text(_DDL_CREATE_ARTICLES.format(schema=schema)))
                await conn.execute(text(_DDL_CREATE_CHUNKS.format(
                    schema=schema, dense_col=dense_col, sparse_col=sparse_col,
                )))
                await conn.execute(text(_DDL_IDX_URL.format(schema=schema)))
                await conn.execute(text(_DDL_IDX_SOURCE.format(schema=schema)))

        if table_existed and (dense_dim is not None or sparse_dim is not None):
            await self._check_dim(dense_dim, sparse_dim)

    async def _check_dim(self, dense_dim: int | None, sparse_dim: int | None) -> None:
        """Validate that existing column dimensions match the configured providers.

        Uses run_sync because SQLAlchemy's async engine requires bridging to the
        sync inspector API to introspect column types.
        """
        async with self._engine.connect() as conn:
            cols = await conn.run_sync(
                lambda sync_conn: sa_inspect(sync_conn).get_columns(
                    "article_chunks", schema=self.schema
                )
            )
        if dense_dim is not None:
            _check_dim_from_cols(cols, dense_dim)
        if sparse_dim is not None:
            _check_sparse_dim_from_cols(cols, sparse_dim)
