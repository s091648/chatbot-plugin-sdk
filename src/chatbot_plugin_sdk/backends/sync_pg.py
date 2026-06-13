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
import uuid

from sqlalchemy import create_engine, delete, inspect as sa_inspect, select, text
from sqlalchemy.orm import sessionmaker

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
    _dense_col_ddl,
)
from chatbot_plugin_sdk.config import DatabaseConfig
from chatbot_plugin_sdk.exceptions import DatabaseError
from chatbot_plugin_sdk.models.article import Article
from chatbot_plugin_sdk.models.chunk import ArticleChunk


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

    # ── Async wrappers (schedule sync work onto the event loop's thread pool) ──

    async def setup(self, dense_dim: int | None) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._setup_sync, dense_dim)

    async def validate(self, dense_dim: int | None) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._validate_sync, dense_dim)

    async def upsert(
        self,
        article_id: uuid.UUID,
        metadata: dict,
        chunks: list[str],
        dense_vectors: list[list[float]] | None,
        sparse_vectors: list[dict[str, float]] | None,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._upsert_sync,
            article_id, metadata, chunks, dense_vectors, sparse_vectors,
        )

    async def search_dense(self, query_vec: list[float], top_k: int) -> list[SearchRow]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_dense_sync, query_vec, top_k)

    async def close(self) -> None:
        self._engine.dispose()

    # ── Sync implementations ────────────────────────────────────────────────

    def _setup_sync(self, dense_dim: int | None) -> None:
        """Create schema + tables if missing; validate dimension if table already existed."""
        schema = self.schema
        with self._engine.begin() as conn:
            conn.execute(text(_DDL_CREATE_EXTENSION))
            conn.execute(text(_DDL_CREATE_SCHEMA.format(schema=schema)))
            r = conn.execute(text(_DDL_TABLE_EXISTS), {"s": schema})
            table_existed = r.fetchone() is not None
            if not table_existed:
                dense_col = _dense_col_ddl(dense_dim)
                conn.execute(text(_DDL_CREATE_ARTICLES.format(schema=schema)))
                conn.execute(text(_DDL_CREATE_CHUNKS.format(schema=schema, dense_col=dense_col)))
                conn.execute(text(_DDL_IDX_URL.format(schema=schema)))
                conn.execute(text(_DDL_IDX_SOURCE.format(schema=schema)))

        # Validate dimension outside the DDL transaction (separate connection)
        if table_existed and dense_dim is not None:
            self._check_dim(dense_dim)

    def _validate_sync(self, dense_dim: int | None) -> None:
        schema = self.schema
        with self._engine.connect() as conn:
            r = conn.execute(text(_DDL_TABLE_EXISTS), {"s": schema})
            if r.fetchone() is None:
                raise DatabaseError(
                    f"Table {schema}.article_chunks not found. "
                    "Run IngestProcessor first to create the schema."
                )
        if dense_dim is not None:
            self._check_dim(dense_dim)

    def _check_dim(self, dense_dim: int) -> None:
        """Validate that the existing VECTOR column dimension matches dense_dim."""
        try:
            cols = sa_inspect(self._engine).get_columns("article_chunks", schema=self.schema)
        except Exception:
            return  # table may not exist yet (race condition at startup — harmless)
        _check_dim_from_cols(cols, dense_dim)

    def _upsert_sync(
        self,
        article_id: uuid.UUID,
        metadata: dict,
        chunks: list[str],
        dense_vectors: list[list[float]] | None,
        sparse_vectors: list[dict[str, float]] | None,
    ) -> None:
        try:
            with self._Session() as session:
                with session.begin():
                    existing = session.execute(
                        select(Article).where(Article.id == article_id)
                    ).scalar_one_or_none()

                    if existing is not None:
                        existing.url = metadata.get("url", "")
                        existing.title = metadata.get("title")
                        existing.source = metadata.get("source")
                        existing.metadata_ = metadata.get("metadata")
                        session.execute(
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
                        session.add(ArticleChunk(
                            article_id=article_id,
                            chunk_index=i,
                            content=content,
                            dense_vector=dense_vectors[i] if dense_vectors else None,
                            sparse_vector=sparse_vectors[i] if sparse_vectors else None,
                        ))
                    # session.begin() auto-commits on clean exit, rolls back on exception
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Upsert failed for article {article_id}: {exc}") from exc

    def _search_dense_sync(self, query_vec: list[float], top_k: int) -> list[SearchRow]:
        with self._Session() as db:
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
            rows = db.execute(stmt).all()

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
