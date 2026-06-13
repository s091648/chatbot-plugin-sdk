"""Tests for RetrieveProcessor."""

from unittest.mock import AsyncMock, patch

import pytest

from chatbot_plugin_sdk import RetrieveProcessor, EndpointProvider, DatabaseBackend
from chatbot_plugin_sdk.backends.base import SearchRow
from chatbot_plugin_sdk.contracts.responses import SearchResponse
from chatbot_plugin_sdk.exceptions import NotConfiguredError


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


def _make_row(chunk_id, article_id, idx, content, title, url, distance=0.2) -> SearchRow:
    return SearchRow(
        chunk_id=chunk_id, article_id=article_id, chunk_index=idx,
        content=content, title=title, url=url, distance=distance,
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


# ── _ensure_ready() ─────────────────────────────────────────────────────────────

class TestEnsureReady:
    @pytest.mark.asyncio
    async def test_calls_backend_validate_on_first_use(self):
        backend = _mock_backend()
        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=EndpointProvider(url="http://x", dimension=768))
        await retriever._ensure_ready()
        backend.validate.assert_called_once_with(768)
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


# ── search() ───────────────────────────────────────────────────────────────────

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
            _make_row("c1", "a1", 0, "RAG is retrieval...", "RAG Article", "https://rag.com", 0.1),
            _make_row("c2", "a2", 0, "LLM stands for...", "LLM Article", "https://llm.com", 0.3),
        ]
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("What is RAG?", top_k=2)

        assert isinstance(result, SearchResponse)
        assert len(result.chunks) == 2
        assert result.chunks[0].chunk_id == "c1"
        assert result.chunks[0].score == pytest.approx(0.9, abs=1e-4)
        assert result.chunks[0].article_title == "RAG Article"

    @pytest.mark.asyncio
    async def test_score_is_one_minus_distance(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = [
            _make_row("c1", "a1", 0, "text", "T", "https://x.com", distance=0.4),
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
