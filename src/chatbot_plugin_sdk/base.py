"""BaseRagProcessor — contains search, chat, and LLM fallback chain.

This is the base class for all toolbox SDK variants.
It manages database configuration and provides common operations:
- Hybrid dense + sparse search with RRF fusion
- Chat with RAG context (retrieval + LLM generation)
- LLM fallback chain (Anthropic → Gemini → raw context)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING
from uuid import UUID

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from chatbot_plugin_sdk.config import settings
from chatbot_plugin_sdk.contracts import (
    ArticleCitation,
    ChatResponse,
    ChunkResult,
    SearchResponse,
    StoreChunksResponse,
)
from chatbot_plugin_sdk.models import Article, ArticleChunk
from chatbot_plugin_sdk.chunking import _chunk_text
from chatbot_plugin_sdk.config import DatabaseConfig, EmbeddingModelConfig
from chatbot_plugin_sdk.exceptions import (
    DatabaseError,
    EmbeddingError,
    LLMError,
    NotConfiguredError,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class BaseRagProcessor:
    """Base SDK class for database-backed vector toolbox.

    Manages an async SQLAlchemy engine + session factory and shared
    database operations. Subclasses add ingestion or query behaviour.

    This class is NOT public — use :class:`RagArticleProcessor` or
    :class:`RagQueryProcessor` instead.
    """

    def __init__(self) -> None:
        self._db_config: DatabaseConfig | None = None
        self._embed_config: EmbeddingModelConfig | None = None
        self._tables_created: bool = False

    # ── Configuration helpers ──

    def _configure_database(
        self,
        dbname: str,
        user: str,
        password: str,
        host: str = "localhost",
        port: int = 5432,
    ) -> DatabaseConfig:
        """Build an async SQLAlchemy engine and session factory.

        Returns a :class:`DatabaseConfig` containing ``engine`` and
        ``session_factory``.  Tables are **not** created here — that
        happens lazily via :meth:`_ensure_tables` with proper async
        context.
        """
        database_url = (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"
        )
        engine = create_async_engine(database_url, echo=False, future=True)
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._db_config = DatabaseConfig(engine=engine, session_factory=session_factory)
        return self._db_config

    def _configure_embedding_model(self, api_url: str, api_key: str | None = None) -> EmbeddingModelConfig:
        """Configure the HTTP embedding model endpoint.

        Args:
            api_url: Base URL of the embedding microservice (e.g. ``http://localhost:8080``).
            api_key: Optional API key forwarded as *Authorization: Bearer <token>*.

        Returns:
            An :class:`EmbeddingModelConfig` instance.
        """
        self._embed_config = EmbeddingModelConfig(base_url=api_url, api_key=api_key)
        return self._embed_config

    async def _ensure_tables(self) -> None:
        """Create DB extensions and tables if they don't exist yet.

        Safe to call multiple times — guarded by ``_tables_created``.
        """
        if self._tables_created or self._db_config is None:
            return

        from chatbot_plugin_sdk.models import Base

        async with self._db_config.engine.begin() as conn:
            # pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS sparsevec"))
            await conn.run_sync(Base.metadata.create_all)
        self._tables_created = True

    async def _get_session(self) -> AsyncSession:
        """Yield a fresh async DB session with table-creation side effect."""
        await self._ensure_tables()
        if self._db_config is None:
            raise NotConfiguredError("Database not configured. Call configure() or _configure_database() first.")
        return self._db_config.session_factory()

    # ── Embedding support ──

    async def _embed_query(self, query: str) -> tuple[list[float], dict[int, float]]:
        """Embed a query string via the configured HTTP embedding service.

        POSTs ``{"texts": [query]}`` to the embedding service and
        expects ``{"dense": [[...]], "sparse": [{...}]}``.
        """
        if self._embed_config is None:
            raise NotConfiguredError(
                "Embedding model not configured. "
                "Pass embedding_model_api when calling configure()."
            )
        import httpx
        async with self._embed_config.build_client() as client:
            payload = {"texts": [query]}
            try:
                resp = await client.post("/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc

        dense = data.get("dense", [])
        sparse_raw = data.get("sparse", [])
        if not dense or not sparse_raw:
            raise EmbeddingError("Embedding response missing dense or sparse vectors")

        sparse_vec: dict[int, float] = {int(k): float(v) for k, v in sparse_raw[0].items()}
        return dense[0], sparse_vec

    async def _embed_texts(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        """Embed a batch of texts via the configured HTTP embedding service.

        POSTs ``{"texts": texts}`` to the embedding service and
        expects ``{"dense": [...], "sparse": [...]}``.
        """
        if self._embed_config is None:
            raise NotConfiguredError(
                "Embedding model not configured. "
                "Pass embedding_model_api when calling configure()."
            )
        import httpx
        async with self._embed_config.build_client() as client:
            payload = {"texts": texts}
            try:
                resp = await client.post("/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc

        dense_list: list[list[float]] = data.get("dense", [])
        sparse_raw_list: list[dict] = data.get("sparse", [])
        if not dense_list or not sparse_raw_list:
            raise EmbeddingError("Embedding response missing dense or sparse vectors")

        sparse_list: list[dict[int, float]] = []
        for raw in sparse_raw_list:
            sparse_list.append({int(k): float(v) for k, v in raw.items()})
        return dense_list, sparse_list

    # ── Search ──

    async def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """Hybrid dense + sparse search with RRF fusion.

        Embeds ``query`` via :meth:`_embed_query`, then queries the
        database for dense and sparse candidates and fuses rankings.
        """
        from pgvector import SparseVector

        dense_vec, sparse_weights = await self._embed_query(query)
        sparse_vec = SparseVector(sparse_weights, settings.sparse_dimension)

        candidates = settings.search_candidates
        k = settings.rrf_k

        async with await self._get_session() as db:
            # Dense candidates
            dense_stmt = (
                select(
                    ArticleChunk.id.label("chunk_id"),
                    ArticleChunk.article_id,
                    ArticleChunk.chunk_index,
                    ArticleChunk.content,
                    Article.title,
                    Article.url,
                )
                .join(Article, ArticleChunk.article_id == Article.id)
                .where(ArticleChunk.dense_vector.isnot(None))
                .order_by(ArticleChunk.dense_vector.cosine_distance(dense_vec))
                .limit(candidates)
            )
            dense_result = await db.execute(dense_stmt)
            dense_rows = dense_result.all()

            # Sparse candidates
            sparse_stmt = (
                select(
                    ArticleChunk.id.label("chunk_id"),
                    ArticleChunk.article_id,
                    ArticleChunk.chunk_index,
                    ArticleChunk.content,
                    Article.title,
                    Article.url,
                )
                .join(Article, ArticleChunk.article_id == Article.id)
                .where(ArticleChunk.sparse_vector.isnot(None))
                .order_by(ArticleChunk.sparse_vector.max_inner_product(sparse_vec))
                .limit(candidates)
            )
            sparse_result = await db.execute(sparse_stmt)
            sparse_rows = sparse_result.all()

        # RRF fusion
        chunk_scores: dict[str, tuple[float, Any]] = {}

        for rank, row in enumerate(dense_rows, start=1):
            chunk_id = str(row.chunk_id)
            chunk_scores[chunk_id] = (1.0 / (k + rank), row)

        for rank, row in enumerate(sparse_rows, start=1):
            chunk_id = str(row.chunk_id)
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] = (
                    chunk_scores[chunk_id][0] + 1.0 / (k + rank),
                    chunk_scores[chunk_id][1],
                )
            else:
                chunk_scores[chunk_id] = (1.0 / (k + rank), row)

        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )[:top_k]

        chunks = [
            ChunkResult(
                chunk_id=chunk_id,
                article_id=str(row.article_id),
                article_title=row.title,
                article_url=row.url,
                chunk_index=row.chunk_index,
                content=row.content,
                score=round(score, 6),
            )
            for chunk_id, (score, row) in sorted_chunks
        ]

        return SearchResponse(chunks=chunks)

    # ── Chat ──

    async def chat(
        self,
        message: str,
        top_k: int | None = None,
    ) -> ChatResponse:
        """Chat with RAG context.

        1. Search for relevant chunks using ``message`` as query.
        2. Assemble context from top chunks.
        3. Call LLM with system prompt + context + user message.
        """
        if top_k is None:
            top_k = settings.max_context_chunks

        search_result = await self.search(message, top_k=top_k)

        if not search_result.chunks:
            return ChatResponse(
                reply="I couldn't find any relevant context to answer your question.",
                articles_used=[],
                chunks=[],
            )

        # Assemble context
        context_parts = []
        for chunk in search_result.chunks:
            source = chunk.article_title or "Unknown source"
            context_parts.append(f"[source: {source}]\n{chunk.content}")
        context = "\n\n".join(context_parts)

        reply = await self._call_llm(context, message)

        # Deduplicate articles_used
        seen_ids: set[str] = set()
        articles_used: list[ArticleCitation] = []
        for chunk in search_result.chunks:
            if chunk.article_id not in seen_ids:
                seen_ids.add(chunk.article_id)
                articles_used.append(
                    ArticleCitation(
                        id=chunk.article_id,
                        title=chunk.article_title,
                        url=chunk.article_url,
                    )
                )

        return ChatResponse(
            reply=reply,
            articles_used=articles_used,
            chunks=search_result.chunks,
        )

    async def _call_llm(self, context: str, question: str) -> str:
        """Call LLM for chat. Tries Anthropic first, falls back to Gemini."""
        system = (
            "You are a helpful research assistant. Answer the user's question "
            "using only the provided context. Cite sources using the [source: Title] "
            "annotations already present in the context. If the context does not "
            "contain enough information, say so clearly."
        )
        user_prompt = f"{context}\n\nQuestion: {question}"

        # Try Anthropic first
        if settings.llm_api_key:
            try:
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=settings.llm_api_key)
                response = await client.messages.create(
                    model=settings.llm_model,
                    max_tokens=2048,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text
            except Exception:
                pass  # Fallback to Gemini

        # Fallback to Gemini, or return raw context if no LLM keys configured
        if settings.gemini_api_key:
            try:
                return await self._call_gemini(system, user_prompt)
            except Exception:
                pass  # Gemini failed

        return (
            "[No LLM configured — returning raw retrieved context]\n\n"
            f"{user_prompt}\n\n"
            "[Set CHATBOT_LLM_API_KEY (Anthropic) or CHATBOT_GEMINI_API_KEY "
            "to enable LLM-generated responses.]"
        )

    async def _call_gemini(self, system: str, prompt: str) -> str:
        """Call Google Gemini REST API."""
        import httpx

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_model}:generateContent"
        )
        params = {"key": settings.gemini_api_key}
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "generationConfig": {"maxOutputTokens": 2048},
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, params=params, json=payload)
            if resp.status_code != 200:
                raise LLMError(
                    f"Gemini API error: {resp.status_code} {resp.text}"
                )
            data = resp.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError) as e:
                raise LLMError(f"Unexpected Gemini response: {data}") from e

    # ── Save (used by routers / RagArticleProcessor) ──

    async def _save_article_and_chunks(
        self,
        article_id: UUID,
        metadata: dict[str, Any],
        chunks_data: list[dict[str, Any]],
    ) -> StoreChunksResponse:
        """Upsert an article and store its chunks.

        This is the internal persistence method shared by both the
        ``/tools/chunks`` endpoint and :meth:`RagArticleProcessor.ingest`.

        Args:
            article_id: Article UUID.
            metadata: Article metadata dict with ``url``, ``title``, ``source``, etc.
            chunks_data: List of chunk dicts with ``chunk_index``, ``content``,
                ``dense_vector``, and optional ``sparse_vector``.

        Returns:
            A :class:`StoreChunksResponse` with ``stored`` count and ``article_id``.
        """
        from pgvector import SparseVector

        expected_dim = settings.embedding_dimension
        for chunk in chunks_data:
            if len(chunk["dense_vector"]) != expected_dim:
                raise DatabaseError(
                    f"Dense vector dimension mismatch at chunk_index={chunk['chunk_index']}: "
                    f"expected {expected_dim}, got {len(chunk['dense_vector'])}"
                )

        session = await self._get_session()
        try:
            async with session.begin():
                # Check for existing article
                result = await session.execute(
                    select(Article).where(Article.id == article_id)
                )
                existing = result.scalar_one_or_none()

                if existing is not None:
                    # Update metadata
                    existing.url = metadata.get("url", "")
                    existing.title = metadata.get("title")
                    existing.source = metadata.get("source")
                    existing.metadata_ = metadata.get("metadata")

                    # Delete old chunks
                    await session.execute(
                        delete(ArticleChunk).where(ArticleChunk.article_id == article_id)
                    )
                else:
                    # Create new article
                    new_article = Article(
                        id=article_id,
                        url=metadata.get("url", ""),
                        title=metadata.get("title"),
                        source=metadata.get("source"),
                        metadata_=metadata.get("metadata"),
                    )
                    session.add(new_article)

                # Insert new chunks
                for chunk in chunks_data:
                    sparse = None
                    if chunk.get("sparse_vector"):
                        sparse = SparseVector(chunk["sparse_vector"], settings.sparse_dimension)

                    article_chunk = ArticleChunk(
                        article_id=article_id,
                        chunk_index=chunk["chunk_index"],
                        content=chunk["content"],
                        dense_vector=chunk["dense_vector"],
                        sparse_vector=sparse,
                    )
                    session.add(article_chunk)

            return StoreChunksResponse(
                stored=len(chunks_data),
                article_id=str(article_id),
            )
        except Exception as exc:
            raise DatabaseError(f"Failed to save article and chunks: {exc}") from exc
        finally:
            await session.close()
