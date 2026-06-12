from __future__ import annotations
import re
import unicodedata
import uuid
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from chatbot_plugin_sdk.chunking import _chunk_text
from chatbot_plugin_sdk.config import DatabaseConfig, _RuntimeDatabase
from chatbot_plugin_sdk.exceptions import DatabaseError, NotConfiguredError
from chatbot_plugin_sdk.models.article import Article
from chatbot_plugin_sdk.models.chunk import ArticleChunk
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider


class IngestProcessor:
    """文章向量化寫入處理器。

    Pipeline: normalize → chunk → embed（dense/sparse） → upsert to DB

    Usage::

        processor = IngestProcessor()
        processor.configure(
            db=DatabaseConfig(dbname="mydb", user="u", password="p"),
            dense=EndpointProvider(url="http://embed:8080", dimension=768),
        )
        await processor.ingest(
            full_text="...",
            metadata={"url": "https://example.com/article", "title": "My Article"},
        )
    """

    def __init__(self) -> None:
        self._db_config: DatabaseConfig | None = None
        self._runtime: _RuntimeDatabase | None = None
        self._dense: DenseEmbeddingProvider | None = None
        self._sparse: SparseEmbeddingProvider | None = None
        self._ready: bool = False

    def configure(
        self,
        db: DatabaseConfig,
        dense: DenseEmbeddingProvider | None = None,
        sparse: SparseEmbeddingProvider | None = None,
    ) -> None:
        """設定 DB 連線與 embedding providers。純同步，不做任何 IO。"""
        if dense is None and sparse is None:
            raise NotConfiguredError(
                "至少需要配置 dense 或 sparse 其中一種 embedding provider。"
            )
        self._db_config = db
        self._dense = dense
        self._sparse = sparse
        self._ready = False

    def _build_runtime(self) -> _RuntimeDatabase:
        assert self._db_config is not None
        db = self._db_config
        url = f"postgresql+asyncpg://{db.user}:{db.password}@{db.host}:{db.port}/{db.dbname}"
        engine = create_async_engine(url, echo=False, future=True)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        return _RuntimeDatabase(engine=engine, session_factory=factory, schema=db.schema)

    async def ensure_ready(self) -> None:
        """冪等：首次呼叫時建立 schema/table（若不存在）或驗證已存在的 schema 相容性。

        - 表不存在 → 建立 vectors schema + articles + article_chunks
        - 表存在   → 讀取 DB 中 dense_vector 的實際維度，與 provider.dimension 比對
        """
        if self._ready:
            return
        if self._db_config is None:
            raise NotConfiguredError("尚未呼叫 configure()。")

        if self._runtime is None:
            self._runtime = self._build_runtime()

        schema = self._runtime.schema
        async with self._runtime.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            result = await conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = :schema AND table_name = 'article_chunks'"
            ), {"schema": schema})
            table_exists = result.fetchone() is not None

        if not table_exists:
            await self._create_tables()
        else:
            await self._validate_schema_compatibility()

        self._ready = True

    async def _create_tables(self) -> None:
        assert self._runtime is not None
        schema = self._runtime.schema
        dense_dim = self._dense.dimension if self._dense else None

        async with self._runtime.engine.begin() as conn:
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {schema}.articles (
                    id          UUID PRIMARY KEY,
                    url         TEXT NOT NULL UNIQUE,
                    title       TEXT,
                    source      TEXT,
                    metadata    JSONB,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))
            dense_col = (
                f"dense_vector VECTOR({dense_dim})"
                if dense_dim
                else "dense_vector VECTOR(768)"
            )
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
                f"CREATE INDEX IF NOT EXISTS idx_articles_url "
                f"ON {schema}.articles(url)"
            ))
            await conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_articles_source "
                f"ON {schema}.articles(source)"
            ))

    async def _validate_schema_compatibility(self) -> None:
        assert self._runtime is not None
        if self._dense is None:
            return

        schema = self._runtime.schema
        async with self._runtime.engine.connect() as conn:
            def _get_dim(sync_conn):
                from sqlalchemy import inspect as sa_inspect
                inspector = sa_inspect(sync_conn)
                cols = inspector.get_columns("article_chunks", schema=schema)
                for col in cols:
                    if col["name"] == "dense_vector":
                        return getattr(col["type"], "dim", None)
                return None
            db_dim = await conn.run_sync(_get_dim)

        if db_dim is not None and db_dim != self._dense.dimension:
            raise DatabaseError(
                f"Provider dimension mismatch: DB has VECTOR({db_dim}) "
                f"but provider.dimension={self._dense.dimension}. "
                "Use the same embedding model as when the table was created."
            )

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.lstrip("﻿").strip()
        return re.sub(r"\s+", " ", text)

    async def ingest(
        self,
        full_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """完整 ingest pipeline：normalize → chunk → embed → upsert。

        Args:
            full_text: 文章全文（PDF 解析後、HTML stripped 或純文字）。
            metadata: 至少含 ``url``（str）；建議也提供 ``title``、``source``。
                      article_id 由 url 透過 uuid5 推導，確保冪等性。
        """
        await self.ensure_ready()
        assert self._runtime is not None

        metadata = metadata or {}
        url = metadata.get("url", "")
        if not url:
            raise DatabaseError("metadata must contain 'url' to ensure idempotent ingest.")

        normalized = self._normalize(full_text)
        if not normalized:
            raise DatabaseError("Empty text after normalization.")

        chunks = _chunk_text(normalized)
        if not chunks:
            raise DatabaseError("No chunks produced — input text may be too short.")

        dense_vectors: list[list[float]] | None = None
        sparse_vectors: list[dict[str, float]] | None = None

        if self._dense is not None:
            dense_vectors = await self._dense.embed(chunks)
            if len(dense_vectors) != len(chunks):
                raise DatabaseError(
                    f"Dense embedding returned {len(dense_vectors)} vectors "
                    f"but {len(chunks)} chunks expected."
                )

        if self._sparse is not None:
            sparse_vectors = await self._sparse.embed(chunks)
            if len(sparse_vectors) != len(chunks):
                raise DatabaseError(
                    f"Sparse embedding returned {len(sparse_vectors)} vectors "
                    f"but {len(chunks)} chunks expected."
                )

        article_id = uuid.uuid5(uuid.NAMESPACE_URL, url)
        await self._upsert(
            article_id=article_id,
            metadata=metadata,
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
        )

    async def _upsert(
        self,
        article_id: uuid.UUID,
        metadata: dict[str, Any],
        chunks: list[str],
        dense_vectors: list[list[float]] | None,
        sparse_vectors: list[dict[str, float]] | None,
    ) -> None:
        assert self._runtime is not None
        session = self._runtime.session_factory()
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
        except Exception as exc:
            raise DatabaseError(f"Failed to upsert article {article_id}: {exc}") from exc
        finally:
            await session.close()
