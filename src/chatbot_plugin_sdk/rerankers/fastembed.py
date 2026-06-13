"""FastEmbedReranker — cross-encoder reranker via fastembed TextCrossEncoder."""
from __future__ import annotations

import asyncio
import math

from chatbot_plugin_sdk.backends.base import SearchRow


class FastEmbedReranker:
    """Cross-encoder reranker backed by fastembed.

    Uses ``BAAI/bge-reranker-v2-m3`` by default — a multilingual model that
    covers both English and Chinese without any additional configuration.

    The sync TextCrossEncoder is offloaded to ``run_in_executor`` so it does
    not block the event loop.

    Requires::

        pip install "chatbot-plugin-sdk[fastembed]"

    Usage::

        reranker = FastEmbedReranker()
        retriever.configure(backend=backend, dense=provider, reranker=reranker)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for FastEmbedReranker. "
                "Install with: pip install 'chatbot-plugin-sdk[fastembed]'"
            ) from exc
        self._model = TextCrossEncoder(model_name=model_name)

    async def rerank(
        self,
        query: str,
        rows: list[SearchRow],
    ) -> list[tuple[SearchRow, float]]:
        """Score rows against the query and return them sorted by descending relevance.

        Args:
            query: The user's search query.
            rows:  Candidate chunks from dense / sparse / hybrid retrieval.

        Returns:
            ``(SearchRow, score)`` pairs sorted by descending score.
            Score is sigmoid-normalized: ``1 / (1 + exp(-raw))`` → (0, 1).
        """
        if not rows:
            return []

        docs = [r.content for r in rows]
        loop = asyncio.get_running_loop()
        raw_scores: list[float] = await loop.run_in_executor(
            None,
            lambda: list(self._model.rerank(query, docs)),
        )

        scored = [
            (row, 1.0 / (1.0 + math.exp(-raw)))
            for row, raw in zip(rows, raw_scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
