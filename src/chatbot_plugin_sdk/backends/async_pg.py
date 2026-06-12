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

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from chatbot_plugin_sdk.backends.base import SearchRow
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

    # ── Setup / validation ─────────────────────────────────────────────────

    async def setup(self, dense_dim: int | None) -> None:
        schema = self.schema
        table_existed = await self._ensure_tables(dense_dim)
        if table_existed and dense_dim is not None:
            await self._check_dim_compat(dense_dim)

    async def _ensure_tables(self, dense_dim: int | None) -> bool:
        """Create schema + tables if missing.  Returns True if table already existed."""
        schema = self.schema
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            r = await conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema=:s AND table_name='article_chunks'"
            ), {"s": schema})
            if r.fetchone() is not None:
                return True  # table exists — skip creation

            dense_col = f"dense_vector VECTOR({dense_dim})" if dense_dim else "dense_vector VECTOR(768)"
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {schema}.articles (
                    id         UUID PRIMARY KEY,
                    url        TEXT NOT NULL UNIQUE,
                    title      TEXT,
                    source     TEXT,
                    metadata   JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))
            await conn.execute(text(f"""
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
                )
            """))
            await conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_articles_url ON {schema}.articles(url)"
            ))
            await conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_articles_source ON {schema}.articles(source)"
            ))
            return False

    async def _check_dim_compat(self, dense_dim: int) -> None:
        async with self._engine.connect() as conn:
            def _get_dim(sync_conn):
                from sqlalchemy import inspect as sa_inspect
                inspector = sa_inspect(sync_conn)
                cols = inspector.get_columns("article_chunks", schema=self.schema)
                for col in cols:
                    if col["name"] == "dense_vector":
                        return getattr(col["type"], "dim", None)
                return None

            db_dim = await conn.run_sync(_get_dim)
        if db_dim is not None and db_dim != dense_dim:
            raise DatabaseError(
                f"Dimension mismatch: DB has VECTOR({db_dim}) "
                f"but provider.dimension={dense_dim}. "
                "Use the same embedding model that created the table."
            )

    async def validate(self, dense_dim: int | None) -> None:
        schema = self.schema
        async with self._engine.connect() as conn:
            r = await conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema=:s AND table_name='article_chunks'"
            ), {"s": schema})
            if r.fetchone() is None:
                raise DatabaseError(
                    f"Table {schema}.article_chunks not found. "
                    "Run IngestProcessor first to create the schema."
                )
            if dense_dim is not None:
                def _get_dim(sync_conn):
                    from sqlalchemy import inspect as sa_inspect
                    inspector = sa_inspect(sync_conn)
                    cols = inspector.get_columns("article_chunks", schema=schema)
                    for col in cols:
                        if col["name"] == "dense_vector":
                            return getattr(col["type"], "dim", None)
                    return None

                db_dim = await conn.run_sync(_get_dim)
                if db_dim is not None and db_dim != dense_dim:
                    raise DatabaseError(
                        f"Dimension mismatch: DB has VECTOR({db_dim}) "
                        f"but provider.dimension={dense_dim}."
                    )

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
                    session.add(ArticleChunk(
                        article_id=article_id,
                        chunk_index=i,
                        content=content,
                        dense_vector=dense_vectors[i] if dense_vectors else None,
                        sparse_vector=sparse_vectors[i] if sparse_vectors else None,
                    ))
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Upsert failed for article {article_id}: {exc}") from exc
        finally:
            await session.close()

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
