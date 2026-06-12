from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from chatbot_plugin_sdk.config import DatabaseConfig, _RuntimeDatabase
from chatbot_plugin_sdk.contracts.responses import ChunkResult, SearchResponse
from chatbot_plugin_sdk.exceptions import DatabaseError, NotConfiguredError
from chatbot_plugin_sdk.models.article import Article
from chatbot_plugin_sdk.models.chunk import ArticleChunk
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider


class RetrieveProcessor:
    """向量語意搜尋處理器（read-only）。

    Pipeline: embed query → pgvector similarity search → return SearchResponse

    SDK 在回傳 SearchResponse（chunks）後即完成職責；LLM 生成由 caller 負責。

    Usage::

        retriever = RetrieveProcessor()
        retriever.configure(
            db=DatabaseConfig(dbname="mydb", user="u", password="p"),
            dense=EndpointProvider(url="http://embed:8080", dimension=768),
        )
        result = await retriever.search("What is RAG?")
        # result.chunks 是 list[ChunkResult]，交給 LLM 生成回答
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
        """設定 DB 連線與 embedding providers。"""
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
        """驗證 DB 中的 schema 與 provider 相容。不建表，只驗證表已存在。"""
        if self._ready:
            return
        if self._db_config is None:
            raise NotConfiguredError("尚未呼叫 configure()。")
        if self._runtime is None:
            self._runtime = self._build_runtime()

        schema = self._runtime.schema
        async with self._runtime.engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = :schema AND table_name = 'article_chunks'"
            ), {"schema": schema})
            if result.fetchone() is None:
                raise DatabaseError(
                    f"Table {schema}.article_chunks does not exist. "
                    "Run IngestProcessor.ensure_ready() first to create the schema."
                )

        if self._dense is not None:
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
                    f"but provider.dimension={self._dense.dimension}."
                )

        self._ready = True

    async def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """語意搜尋。

        若只有 dense provider → dense cosine search。
        若只有 sparse provider → sparse search（需要 pgvector sparsevec，待實作）。
        若兩者都有 → hybrid RRF fusion（待實作）。

        Returns:
            SearchResponse 包含按相關度排序的 ChunkResult 列表。
        """
        await self.ensure_ready()
        assert self._runtime is not None

        if self._dense is not None:
            return await self._dense_search(query, top_k)

        raise NotConfiguredError(
            "Dense provider is required for search in this version. "
            "Sparse-only and hybrid search are not yet implemented."
        )

    async def _dense_search(self, query: str, top_k: int) -> SearchResponse:
        assert self._runtime is not None and self._dense is not None

        dense_vecs = await self._dense.embed([query])
        query_vec = dense_vecs[0]

        async with self._runtime.session_factory() as db:
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

        chunks = [
            ChunkResult(
                chunk_id=str(row.chunk_id),
                article_id=str(row.article_id),
                article_title=row.title,
                article_url=row.url,
                chunk_index=row.chunk_index,
                content=row.content,
                score=round(1.0 - float(row.distance), 6),
            )
            for row in rows
        ]
        return SearchResponse(chunks=chunks)
