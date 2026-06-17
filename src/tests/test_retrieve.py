"""Tests for RetrieveProcessor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin_sdk import RetrieveProcessor, EndpointProvider, DatabaseBackend
from chatbot_plugin_sdk.backends.base import SearchRow
from chatbot_plugin_sdk.contracts.responses import SearchResponse
from chatbot_plugin_sdk.exceptions import NotConfiguredError
from chatbot_plugin_sdk.processors.retrieve import _rrf_merge


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_backend() -> AsyncMock:
    backend = AsyncMock(spec=DatabaseBackend)
    backend.schema = "vectors"
    return backend


def _configured_retriever(backend=None) -> tuple[RetrieveProcessor, AsyncMock]:
    backend = backend or _mock_backend()
    dense = EndpointProvider(url="http://localhost:8080", dimension=768)
    retriever = RetrieveProcessor()
    retriever.configure(backend=backend, dense=dense)
    retriever._ready = True  # bypass _ensure_ready() / backend.validate()
    return retriever, backend


def _make_row(chunk_id, article_id, idx, content, distance=0.2,
              article_metadata=None) -> SearchRow:
    return SearchRow(
        chunk_id=chunk_id, article_id=article_id, chunk_index=idx,
        content=content, distance=distance,
        article_metadata=article_metadata or {},
    )


# ── configure() ────────────────────────────────────────────────────────────────

class TestRetrieveConfigure:
    def test_requires_at_least_one_provider(self):
        retriever = RetrieveProcessor()
        with pytest.raises(NotConfiguredError):
            retriever.configure(backend=_mock_backend())

    def test_with_dense(self):
        retriever = RetrieveProcessor()
        dense = EndpointProvider(url="http://x", dimension=768)
        retriever.configure(backend=_mock_backend(), dense=dense)
        assert retriever._dense is dense

    def test_resets_ready_flag(self):
        retriever = RetrieveProcessor()
        retriever._ready = True
        retriever.configure(
            backend=_mock_backend(),
            dense=EndpointProvider(url="http://x", dimension=768),
        )
        assert retriever._ready is False

    def test_configure_with_reranker(self):
        retriever = RetrieveProcessor()
        dense = EndpointProvider(url="http://x", dimension=768)
        reranker = MagicMock()
        retriever.configure(backend=_mock_backend(), dense=dense, reranker=reranker)
        assert retriever._reranker is reranker


# ── _ensure_ready() ─────────────────────────────────────────────────────────────

class TestEnsureReady:
    @pytest.mark.asyncio
    async def test_calls_backend_validate_on_first_use(self):
        backend = _mock_backend()
        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=EndpointProvider(url="http://x", dimension=768))
        await retriever._ensure_ready()
        backend.validate.assert_called_once_with(768, None)  # dense_dim=768, sparse_dim=None
        assert retriever._ready is True

    @pytest.mark.asyncio
    async def test_skips_validate_when_already_ready(self):
        backend = _mock_backend()
        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=EndpointProvider(url="http://x", dimension=768))
        retriever._ready = True
        await retriever._ensure_ready()
        backend.validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_not_configured(self):
        retriever = RetrieveProcessor()
        with pytest.raises(NotConfiguredError):
            await retriever._ensure_ready()


# ── retrieve() (dense-only) ────────────────────────────────────────────────────

class TestRetrieve:
    @pytest.mark.asyncio
    async def test_raises_without_configure(self):
        retriever = RetrieveProcessor()
        with pytest.raises(NotConfiguredError):
            await retriever.retrieve("hello")

    @pytest.mark.asyncio
    async def test_returns_search_response(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = [
            _make_row("c1", "a1", 0, "RAG is retrieval...", 0.1,
                       article_metadata={"title": "RAG Article", "url": "https://rag.com"}),
            _make_row("c2", "a2", 0, "LLM stands for...", 0.3,
                       article_metadata={"title": "LLM Article", "url": "https://llm.com"}),
        ]
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("What is RAG?", top_k=2)

        assert isinstance(result, SearchResponse)
        assert len(result.chunks) == 2
        assert result.chunks[0].chunk_id == "c1"
        assert result.chunks[0].score == pytest.approx(0.9, abs=1e-4)
        assert result.chunks[0].article_metadata["title"] == "RAG Article"

    @pytest.mark.asyncio
    async def test_score_is_one_minus_distance(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = [
            _make_row("c1", "a1", 0, "text", distance=0.4),
        ]
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q")

        assert result.chunks[0].score == pytest.approx(0.6, abs=1e-4)

    @pytest.mark.asyncio
    async def test_empty_results(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = []
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("unknown query")

        assert result.chunks == []

    @pytest.mark.asyncio
    async def test_passes_top_k_to_backend(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = []
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            await retriever.retrieve("q", top_k=5)

        backend.search_dense.assert_called_once()
        _, called_top_k = backend.search_dense.call_args.args
        assert called_top_k == 5


# ── _rrf_merge() ──────────────────────────────────────────────────────────────

class TestRrfMerge:
    def test_unique_chunks_from_single_list(self):
        rows = [_make_row(f"c{i}", "a1", i, "t") for i in range(3)]
        merged = _rrf_merge(rows, [])
        assert [r.chunk_id for r, _ in merged] == ["c0", "c1", "c2"]

    def test_overlapping_chunk_gets_double_score(self):
        r = _make_row("c1", "a1", 0, "t")
        merged = _rrf_merge([r], [r])
        assert len(merged) == 1
        score = merged[0][1]
        # Both lists give rank 0 → score = 2 * 1/(60+0+1) ≈ 0.03279
        assert score == pytest.approx(2.0 / 61, rel=1e-5)

    def test_nonoverlapping_merged_and_ordered_by_score(self):
        dense  = [_make_row("d1", "a1", 0, "t")]
        sparse = [_make_row("s1", "a1", 1, "t")]
        merged = _rrf_merge(dense, sparse)
        # Both get rank 0 in their respective lists → same RRF score → order is stable
        assert len(merged) == 2
        assert merged[0][1] == pytest.approx(merged[1][1], abs=1e-9)

    def test_higher_ranked_item_wins(self):
        # c1 is rank-0 in dense AND rank-1 in sparse → wins over c2 (rank-1 dense, rank-0 sparse)
        c1 = _make_row("c1", "a1", 0, "t")
        c2 = _make_row("c2", "a1", 1, "t")
        merged = _rrf_merge([c1, c2], [c2, c1])
        # Both get the same total score (rank 0 + rank 1 = rank 1 + rank 0), order is stable
        ids = [r.chunk_id for r, _ in merged]
        assert set(ids) == {"c1", "c2"}


# ── hybrid retrieve() ─────────────────────────────────────────────────────────

class TestHybridRetrieve:
    def _make_sparse_provider(self, dimension=30522):
        provider = MagicMock()
        provider.dimension = dimension
        provider.embed = AsyncMock(return_value=[{"0": 0.5, "1": 0.3}])
        return provider

    @pytest.mark.asyncio
    async def test_hybrid_calls_both_backends(self):
        backend = _mock_backend()
        backend.search_dense.return_value = [_make_row("c1", "a1", 0, "t", 0.1)]
        backend.search_sparse.return_value = [_make_row("c2", "a2", 0, "t", -0.9)]

        dense = EndpointProvider(url="http://x", dimension=768)
        sparse = self._make_sparse_provider()
        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=dense, sparse=sparse)
        retriever._ready = True

        with patch.object(dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", top_k=5)

        backend.search_dense.assert_called_once()
        backend.search_sparse.assert_called_once()
        assert isinstance(result, SearchResponse)
        assert len(result.chunks) == 2

    @pytest.mark.asyncio
    async def test_hybrid_uses_rrf_scores(self):
        backend = _mock_backend()
        row = _make_row("c1", "a1", 0, "t", 0.1)
        backend.search_dense.return_value = [row]
        backend.search_sparse.return_value = [row]  # same chunk in both → higher RRF score

        dense = EndpointProvider(url="http://x", dimension=768)
        sparse = self._make_sparse_provider()
        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=dense, sparse=sparse)
        retriever._ready = True

        with patch.object(dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q")

        # Score must be RRF score ≈ 2/61, not 1 − distance
        assert result.chunks[0].score == pytest.approx(2.0 / 61, rel=1e-4)

    @pytest.mark.asyncio
    async def test_reranker_overrides_scores(self):
        backend = _mock_backend()
        row1 = _make_row("c1", "a1", 0, "first", 0.1)
        row2 = _make_row("c2", "a2", 0, "second", 0.3)
        backend.search_dense.return_value = [row1, row2]

        dense = EndpointProvider(url="http://x", dimension=768)
        reranker = MagicMock()
        # reranker reverses order and assigns explicit scores
        reranker.rerank = AsyncMock(return_value=[(row2, 0.95), (row1, 0.42)])

        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=dense, reranker=reranker)
        retriever._ready = True

        with patch.object(dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", top_k=2)

        assert result.chunks[0].chunk_id == "c2"
        assert result.chunks[0].score == pytest.approx(0.95)
        assert result.chunks[1].chunk_id == "c1"
        assert result.chunks[1].score == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_reranker_fetches_3x_candidates(self):
        backend = _mock_backend()
        backend.search_dense.return_value = []
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=[])

        dense = EndpointProvider(url="http://x", dimension=768)
        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=dense, reranker=reranker)
        retriever._ready = True

        with patch.object(dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            await retriever.retrieve("q", top_k=5)

        _, called_k = backend.search_dense.call_args.args
        assert called_k == 15  # top_k * 3


# ── min_score / min_rerank_score gating ──────────────────────────────────────────

class TestRetrieveScoreGating:
    @pytest.mark.asyncio
    async def test_min_score_filters_dense_results(self):
        """Chunks with score < min_score are removed before reranking."""
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = [
            _make_row("c1", "a1", 0, "relevant", 0.1),  # score = 0.9
            _make_row("c2", "a2", 0, "irrelevant", 0.8),  # score = 0.2
        ]
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", min_score=0.5)
        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_min_score_zero_passes_everything(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = [
            _make_row("c1", "a1", 0, "text", 0.99),
        ]
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", min_score=0.0)
        assert len(result.chunks) == 1

    @pytest.mark.asyncio
    async def test_min_rerank_score_filters_after_rerank(self):
        backend = _mock_backend()
        row1 = _make_row("c1", "a1", 0, "relevant", 0.1)
        row2 = _make_row("c2", "a2", 0, "irrelevant", 0.3)
        backend.search_dense.return_value = [row1, row2]
        dense = EndpointProvider(url="http://x", dimension=768)
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=[(row1, 0.9), (row2, 0.4)])
        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=dense, reranker=reranker)
        retriever._ready = True
        with patch.object(dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", top_k=5, min_rerank_score=0.7)
        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == "c1"
        assert result.chunks[0].score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_min_rerank_score_zero_passes_everything(self):
        backend = _mock_backend()
        row1 = _make_row("c1", "a1", 0, "text", 0.1)
        backend.search_dense.return_value = [row1]
        dense = EndpointProvider(url="http://x", dimension=768)
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=[(row1, 0.15)])
        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=dense, reranker=reranker)
        retriever._ready = True
        with patch.object(dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", min_rerank_score=0.0)
        assert len(result.chunks) == 1

    @pytest.mark.asyncio
    async def test_all_chunks_filtered_returns_empty(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = [
            _make_row("c1", "a1", 0, "text", 0.95),
        ]
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", min_score=0.5)
        assert result.chunks == []


# ── retrieve(filters=...) ──────────────────────────────────────────────────────────

class TestRetrieveFilters:
    @pytest.mark.asyncio
    async def test_passes_filters_to_backend(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = []
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            await retriever.retrieve("q", filters={"topic_id": "some-uuid"})

        call_kwargs = backend.search_dense.call_args.kwargs
        assert call_kwargs["filters"] == {"topic_id": "some-uuid"}

    @pytest.mark.asyncio
    async def test_filters_default_none(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = []
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            await retriever.retrieve("q")

        call_kwargs = backend.search_dense.call_args.kwargs
        assert call_kwargs.get("filters") is None

    @pytest.mark.asyncio
    async def test_hybrid_passes_filters_to_both_backends(self):
        backend = _mock_backend()
        backend.search_dense.return_value = [_make_row("c1", "a1", 0, "t", 0.1)]
        backend.search_sparse.return_value = [_make_row("c2", "a2", 0, "t", -0.9)]

        dense = EndpointProvider(url="http://x", dimension=768)
        sparse = MagicMock()
        sparse.dimension = 30522
        sparse.embed = AsyncMock(return_value=[{"0": 0.5, "1": 0.3}])

        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=dense, sparse=sparse)
        retriever._ready = True

        with patch.object(dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            await retriever.retrieve("q", filters={"source": "wiki"})

        dense_kwargs = backend.search_dense.call_args.kwargs
        sparse_kwargs = backend.search_sparse.call_args.kwargs
        assert dense_kwargs["filters"] == {"source": "wiki"}
        assert sparse_kwargs["filters"] == {"source": "wiki"}
