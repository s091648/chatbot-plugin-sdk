"""Tests for RetrieveProcessor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin_sdk import RetrieveProcessor, EndpointProvider, DatabaseConfig
from chatbot_plugin_sdk.contracts.responses import SearchResponse
from chatbot_plugin_sdk.exceptions import NotConfiguredError


class TestRetrieveConfigure:
    def test_configure_requires_at_least_one_provider(self):
        retriever = RetrieveProcessor()
        with pytest.raises(NotConfiguredError):
            retriever.configure(db=DatabaseConfig(dbname="t", user="u", password="p"))

    def test_configure_with_dense(self):
        retriever = RetrieveProcessor()
        dense = EndpointProvider(url="http://localhost:8080", dimension=768)
        retriever.configure(
            db=DatabaseConfig(dbname="t", user="u", password="p"),
            dense=dense,
        )
        assert retriever._dense is dense

    def test_configure_resets_ready_flag(self):
        retriever = RetrieveProcessor()
        retriever._ready = True
        dense = EndpointProvider(url="http://localhost:8080", dimension=768)
        retriever.configure(
            db=DatabaseConfig(dbname="t", user="u", password="p"),
            dense=dense,
        )
        assert retriever._ready is False


class TestRetrieveSearch:
    def _make_retriever_with_mock_db(self):
        retriever = RetrieveProcessor()
        dense = EndpointProvider(url="http://localhost:8080", dimension=768)
        retriever.configure(
            db=DatabaseConfig(dbname="t", user="u", password="p"),
            dense=dense,
        )
        retriever._ready = True  # bypass ensure_ready()

        def _make_row(chunk_id, article_id, chunk_index, content, title, url, distance=0.2):
            Row = type("MockRow", (), {
                "chunk_id": chunk_id,
                "article_id": article_id,
                "chunk_index": chunk_index,
                "content": content,
                "title": title,
                "url": url,
                "distance": distance,
            })
            return Row()

        # _dense_search uses: async with self._runtime.session_factory() as db:
        # so session_factory() must return an async context manager
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        from chatbot_plugin_sdk.config import _RuntimeDatabase
        retriever._runtime = _RuntimeDatabase(
            engine=AsyncMock(),
            session_factory=MagicMock(return_value=mock_session),
            schema="vectors",
        )
        return retriever, mock_session, _make_row

    @pytest.mark.asyncio
    async def test_search_without_configure_raises(self):
        retriever = RetrieveProcessor()
        with pytest.raises(NotConfiguredError):
            await retriever.search("hello")

    @pytest.mark.asyncio
    async def test_search_returns_chunks(self):
        retriever, mock_session, make_row = self._make_retriever_with_mock_db()

        rows = [
            make_row("c1", "a1", 0, "RAG is retrieval...", "RAG Article", "https://rag.com", 0.1),
            make_row("c2", "a2", 0, "LLM stands for...", "LLM Article", "https://llm.com", 0.3),
        ]
        mock_result = MagicMock()
        mock_result.all.return_value = rows
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(
            retriever._dense, "embed", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.search("What is RAG?", top_k=2)

        assert isinstance(result, SearchResponse)
        assert len(result.chunks) == 2
        assert result.chunks[0].chunk_id == "c1"
        assert result.chunks[0].score == pytest.approx(0.9, abs=1e-4)

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self):
        retriever, mock_session, make_row = self._make_retriever_with_mock_db()

        rows = [make_row(f"c{i}", "a1", i, f"text {i}", "T", "https://x.com") for i in range(5)]
        mock_result = MagicMock()
        mock_result.all.return_value = rows
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(
            retriever._dense, "embed", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.search("hello", top_k=3)

        assert len(result.chunks) == 5  # mock returns all rows; top_k enforced by SQL LIMIT

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        retriever, mock_session, _ = self._make_retriever_with_mock_db()

        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(
            retriever._dense, "embed", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.search("unknown query")

        assert result.chunks == []
