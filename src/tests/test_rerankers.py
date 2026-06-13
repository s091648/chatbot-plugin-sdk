"""Tests for rerankers subpackage."""

import math
from unittest.mock import MagicMock, patch

import pytest

from chatbot_plugin_sdk.backends.base import SearchRow
from chatbot_plugin_sdk.rerankers.base import Reranker
from chatbot_plugin_sdk.rerankers.fastembed import FastEmbedReranker


def _row(chunk_id: str, content: str = "text") -> SearchRow:
    return SearchRow(
        chunk_id=chunk_id, article_id="a1", chunk_index=0,
        content=content, title="T", url="https://x.com", distance=0.1,
    )


# ── Protocol conformance ───────────────────────────────────────────────────────

class TestRerankerProtocol:
    def test_fastembed_reranker_satisfies_protocol(self):
        with patch("chatbot_plugin_sdk.rerankers.fastembed.FastEmbedReranker.__init__", return_value=None):
            instance = FastEmbedReranker.__new__(FastEmbedReranker)
        assert isinstance(instance, Reranker)


# ── FastEmbedReranker ──────────────────────────────────────────────────────────

class TestFastEmbedReranker:
    def _make_reranker(self, raw_scores: list[float]) -> FastEmbedReranker:
        """Return a FastEmbedReranker with a mocked TextCrossEncoder."""
        mock_model = MagicMock()
        mock_model.rerank.return_value = raw_scores
        with patch(
            "chatbot_plugin_sdk.rerankers.fastembed.FastEmbedReranker.__init__",
            return_value=None,
        ):
            reranker = FastEmbedReranker.__new__(FastEmbedReranker)
            reranker._model = mock_model
        return reranker

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_rows(self):
        reranker = self._make_reranker([])
        result = await reranker.rerank("q", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_scores_are_sigmoid_normalized(self):
        # raw score 0.0 → sigmoid(0) = 0.5
        reranker = self._make_reranker([0.0])
        rows = [_row("c1")]
        result = await reranker.rerank("q", rows)
        assert len(result) == 1
        _, score = result[0]
        assert score == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.asyncio
    async def test_rows_sorted_descending_by_score(self):
        # raw: c1 = -1.0 → low score; c2 = 2.0 → high score → c2 should be first
        reranker = self._make_reranker([-1.0, 2.0])
        rows = [_row("c1"), _row("c2")]
        result = await reranker.rerank("q", rows)
        assert result[0][0].chunk_id == "c2"
        assert result[1][0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_score_monotone_with_raw(self):
        # Higher raw score must yield higher sigmoid score
        reranker = self._make_reranker([-2.0, 0.0, 3.0])
        rows = [_row("c1"), _row("c2"), _row("c3")]
        result = await reranker.rerank("q", rows)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_import_error_when_fastembed_missing(self):
        with patch.dict("sys.modules", {"fastembed": None, "fastembed.rerank": None, "fastembed.rerank.cross_encoder": None}):
            with pytest.raises(ImportError, match="fastembed"):
                FastEmbedReranker()
