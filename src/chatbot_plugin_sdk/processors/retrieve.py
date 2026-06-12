from __future__ import annotations

from chatbot_plugin_sdk.backends.base import DatabaseBackend
from chatbot_plugin_sdk.contracts.responses import ChunkResult, SearchResponse
from chatbot_plugin_sdk.exceptions import NotConfiguredError
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider


class RetrieveProcessor:
    """向量語意搜尋處理器（read-only）。

    Pipeline: embed query → backend.search_dense() → SearchResponse

    SDK 在回傳 SearchResponse（chunks）後即完成職責；LLM 生成由 caller 負責。

    Usage::

        retriever = RetrieveProcessor()
        retriever.configure(
            backend=AsyncPgBackend(config),   # or SyncPgBackend
            dense=EndpointProvider(url="http://embed:8080", dimension=768),
        )
        result = await retriever.search("What is RAG?")
        # result.chunks: list[ChunkResult] — pass to your LLM

    Note: the backend must use the same schema as IngestProcessor.
    """

    def __init__(self) -> None:
        self._backend: DatabaseBackend | None = None
        self._dense: DenseEmbeddingProvider | None = None
        self._sparse: SparseEmbeddingProvider | None = None
        self._ready: bool = False

    def configure(
        self,
        backend: DatabaseBackend,
        dense: DenseEmbeddingProvider | None = None,
        sparse: SparseEmbeddingProvider | None = None,
    ) -> None:
        if dense is None and sparse is None:
            raise NotConfiguredError(
                "至少需要配置 dense 或 sparse 其中一種 embedding provider。"
            )
        self._backend = backend
        self._dense = dense
        self._sparse = sparse
        self._ready = False

    async def ensure_ready(self) -> None:
        """Validate tables exist + dimension compatible — delegates to backend.validate()."""
        if self._ready:
            return
        if self._backend is None:
            raise NotConfiguredError("尚未呼叫 configure()。")
        dense_dim = self._dense.dimension if self._dense else None
        await self._backend.validate(dense_dim)
        self._ready = True

    async def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """Semantic search.

        Currently supports dense-only search.  Sparse / hybrid pending.

        Returns:
            :class:`SearchResponse` with chunks ordered by descending similarity.
        """
        await self.ensure_ready()

        if self._dense is not None:
            dense_vecs = await self._dense.embed([query])
            rows = await self._backend.search_dense(dense_vecs[0], top_k)
            chunks = [
                ChunkResult(
                    chunk_id=r.chunk_id,
                    article_id=r.article_id,
                    article_title=r.title,
                    article_url=r.url,
                    chunk_index=r.chunk_index,
                    content=r.content,
                    score=round(1.0 - r.distance, 6),
                )
                for r in rows
            ]
            return SearchResponse(chunks=chunks)

        raise NotConfiguredError(
            "Dense provider is required for search in this version. "
            "Sparse-only and hybrid search are not yet implemented."
        )
