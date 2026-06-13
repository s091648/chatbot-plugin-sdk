"""Reranker Protocol — implemented by FastEmbedReranker and any custom reranker."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from chatbot_plugin_sdk.backends.base import SearchRow


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder that re-scores candidate chunks against the query.

    Usage::

        reranker = FastEmbedReranker()
        ranked = await reranker.rerank(query, candidate_rows)
        top5 = ranked[:5]  # (SearchRow, score) pairs, descending

    Implement this Protocol to plug in any cross-encoder model.
    """

    async def rerank(
        self,
        query: str,
        rows: list[SearchRow],
    ) -> list[tuple[SearchRow, float]]:
        """Score each row against the query.

        Returns:
            Pairs of ``(SearchRow, score)`` sorted by descending relevance.
            Score is in ``(0, 1)`` — sigmoid-normalized raw model output.
        """
        ...
