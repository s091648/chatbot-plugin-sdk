from __future__ import annotations

import logging
from typing import Any

from chatbot_plugin_sdk.backends.base import DatabaseBackend, SearchRow
from chatbot_plugin_sdk.contracts.responses import ChunkResult, SearchResponse
from chatbot_plugin_sdk.exceptions import NotConfiguredError
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider
from chatbot_plugin_sdk.rerankers.base import Reranker

logger = logging.getLogger(__name__)


def _rrf_merge(
    dense_rows: list[SearchRow],
    sparse_rows: list[SearchRow],
    k: int = 60,
) -> list[tuple[SearchRow, float]]:
    """Reciprocal Rank Fusion of two ranked lists.

    Score for each chunk: ``sum(1 / (k + rank + 1))`` across both lists.
    Higher score = appears higher in more lists = more relevant.

    Args:
        dense_rows:  Rows sorted by dense cosine distance ASC (most similar first).
        sparse_rows: Rows sorted by sparse inner-product distance ASC (most similar first).
        k:           RRF constant, typically 60.  Dampens the impact of top ranks.

    Returns:
        ``(SearchRow, rrf_score)`` pairs sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    rows: dict[str, SearchRow] = {}

    for rank, row in enumerate(dense_rows):
        key = row.chunk_id
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        rows[key] = row

    for rank, row in enumerate(sparse_rows):
        key = row.chunk_id
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        rows.setdefault(key, row)  # prefer the dense row when both sources have the chunk

    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [(rows[c], scores[c]) for c in ordered]


class RetrieveProcessor:
    """向量語意搜尋處理器（read-only）。

    Supports three retrieval modes depending on configured providers:

    - **Dense-only**: embed query → ``backend.search_dense()`` → cosine similarity score
    - **Sparse-only**: embed query → ``backend.search_sparse()`` → inner-product score
    - **Hybrid** (dense + sparse): both searches → RRF merge → optional reranker

    When a reranker is configured, ``top_k * 3`` candidates are fetched and re-scored
    before returning the final ``top_k`` results.

    SDK 在回傳 SearchResponse（chunks）後即完成職責；LLM 生成由 caller 負責。

    Usage::

        retriever = RetrieveProcessor()
        retriever.configure(
            backend=AsyncPgBackend(config),
            dense=EndpointProvider(url="http://embed:8080", dimension=768),
            sparse=LocalProvider(fn=splade_fn, dimension=30522),
            reranker=FastEmbedReranker(),
        )
        result = await retriever.retrieve("What is RAG?")
        # result.chunks: list[ChunkResult] — pass to your LLM
    """

    def __init__(self) -> None:
        self._backend: DatabaseBackend | None = None
        self._dense: DenseEmbeddingProvider | None = None
        self._sparse: SparseEmbeddingProvider | None = None
        self._reranker: Reranker | None = None
        self._ready: bool = False

    def configure(
        self,
        backend: DatabaseBackend,
        dense: DenseEmbeddingProvider | None = None,
        sparse: SparseEmbeddingProvider | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if dense is None and sparse is None:
            raise NotConfiguredError(
                "至少需要配置 dense 或 sparse 其中一種 embedding provider。"
            )
        self._backend = backend
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._ready = False

    async def _ensure_ready(self) -> None:
        """Validate tables exist + dimension compatible — delegates to backend.validate()."""
        if self._ready:
            return
        if self._backend is None:
            raise NotConfiguredError("尚未呼叫 configure()。")
        dense_dim = self._dense.dimension if self._dense else None
        sparse_dim = self._sparse.dimension if self._sparse else None
        logger.debug("retrieve_validate", extra={"dense_dim": dense_dim, "sparse_dim": sparse_dim})
        await self._backend.validate(dense_dim, sparse_dim)
        self._ready = True
        logger.info("retrieve_ready", extra={"dense_dim": dense_dim, "sparse_dim": sparse_dim})

    async def retrieve(self, query: str, top_k: int = 10, min_score: float = 0.0, min_rerank_score: float = 0.0, filters: dict[str, Any] | None = None) -> SearchResponse:
        """Retrieve the top-k chunks most semantically similar to the query.

        Hybrid mode (dense + sparse): fetches ``top_k * 3`` candidates from each source,
        merges with RRF, then optionally re-ranks with the cross-encoder.

        Returns:
            :class:`SearchResponse` with chunks ordered by descending similarity score.
        """
        await self._ensure_ready()

        candidate_k = top_k * 3 if self._reranker is not None else top_k
        logger.debug(
            "retrieve_start",
            extra={"query_len": len(query), "top_k": top_k, "candidate_k": candidate_k,
                   "mode": "hybrid" if (self._dense and self._sparse) else "dense" if self._dense else "sparse",
                   "reranker": type(self._reranker).__name__ if self._reranker else None,
                   "filters": filters},
        )

        # ── Retrieval ─────────────────────────────────────────────────────────
        if self._dense is not None and self._sparse is not None:
            dense_vecs, sparse_vecs = await _gather(
                self._dense.embed([query]),
                self._sparse.embed([query]),
            )
            dense_rows, sparse_rows = await _gather(
                self._backend.search_dense(dense_vecs[0], candidate_k, filters=filters),
                self._backend.search_sparse(sparse_vecs[0], candidate_k, filters=filters),
            )
            merged = _rrf_merge(dense_rows, sparse_rows)[:candidate_k]
            ranked_rows = [r for r, _ in merged]
            rrf_scores = {r.chunk_id: s for r, s in merged}
            if min_score > 0:
                ranked_rows = [r for r in ranked_rows if rrf_scores.get(r.chunk_id, 0) >= min_score]

        elif self._dense is not None:
            dense_vecs = await self._dense.embed([query])
            ranked_rows = await self._backend.search_dense(dense_vecs[0], candidate_k, filters=filters)
            rrf_scores = {}
            if min_score > 0:
                ranked_rows = [r for r in ranked_rows if (1.0 - r.distance) >= min_score]

        else:
            sparse_vecs = await self._sparse.embed([query])  # type: ignore[union-attr]
            ranked_rows = await self._backend.search_sparse(sparse_vecs[0], candidate_k, filters=filters)
            rrf_scores = {}
            if min_score > 0:
                ranked_rows = [r for r in ranked_rows if (-r.distance) >= min_score]

        # ── Re-rank ───────────────────────────────────────────────────────────
        if self._reranker is not None:
            reranked = await self._reranker.rerank(query, ranked_rows)
            if min_rerank_score > 0:
                reranked = [(r, s) for r, s in reranked if s >= min_rerank_score]
            result = SearchResponse(chunks=[
                ChunkResult(
                    chunk_id=r.chunk_id,
                    article_id=r.article_id,
                    article_title=r.title,
                    article_url=r.url,
                    public_article_id=r.public_article_id,
                    chunk_index=r.chunk_index,
                    content=r.content,
                    score=round(s, 6),
                )
                for r, s in reranked[:top_k]
            ])
            logger.info("retrieve_complete", extra={"chunk_count": len(result.chunks), "reranked": True})
            return result

        # ── Score ─────────────────────────────────────────────────────────────
        if rrf_scores:
            result = SearchResponse(chunks=[
                ChunkResult(
                    chunk_id=r.chunk_id,
                    article_id=r.article_id,
                    article_title=r.title,
                    article_url=r.url,
                    public_article_id=r.public_article_id,
                    chunk_index=r.chunk_index,
                    content=r.content,
                    score=round(rrf_scores[r.chunk_id], 6),
                )
                for r in ranked_rows[:top_k]
            ])
            logger.info("retrieve_complete", extra={"chunk_count": len(result.chunks), "mode": "hybrid_rrf"})
            return result

        # Dense-only or sparse-only: 1 − distance (cosine) or −distance (inner product)
        result = SearchResponse(chunks=[
            ChunkResult(
                chunk_id=r.chunk_id,
                article_id=r.article_id,
                article_title=r.title,
                article_url=r.url,
                public_article_id=r.public_article_id,
                chunk_index=r.chunk_index,
                content=r.content,
                score=round(1.0 - r.distance, 6),
            )
            for r in ranked_rows[:top_k]
        ])
        logger.info("retrieve_complete", extra={"chunk_count": len(result.chunks), "mode": "single"})
        return result


async def _gather(coro_a, coro_b):
    """Run two coroutines concurrently and return both results."""
    import asyncio
    return await asyncio.gather(coro_a, coro_b)