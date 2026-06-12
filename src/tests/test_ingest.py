"""Tests for IngestProcessor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin_sdk import IngestProcessor, EndpointProvider, DatabaseConfig
from chatbot_plugin_sdk.exceptions import NotConfiguredError, DatabaseError
from chatbot_plugin_sdk.chunking import _chunk_text


class TestNormalization:
    def test_normalise_collapses_whitespace(self):
        result = IngestProcessor._normalize("  Hello   world\n\t  foo  ")
        assert result == "Hello world foo"

    def test_normalise_nfc_unicode(self):
        # e + combining acute = é
        result = IngestProcessor._normalize("café")
        assert result == "café"

    def test_normalise_strips_bom(self):
        result = IngestProcessor._normalize("﻿Hello")
        assert result == "Hello"


class TestChunking:
    def test_chunk_text_basic(self):
        text = "Hello world this is a test of chunking. " * 20
        chunks = _chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) > 0
        assert all(len(c) <= 50 for c in chunks)

    def test_chunk_text_empty(self):
        chunks = _chunk_text("   ")
        assert chunks == []


class TestIngestConfigure:
    def test_configure_requires_at_least_one_provider(self):
        processor = IngestProcessor()
        with pytest.raises(NotConfiguredError):
            processor.configure(db=DatabaseConfig(dbname="t", user="u", password="p"))

    def test_configure_with_dense_only(self):
        processor = IngestProcessor()
        dense = EndpointProvider(url="http://localhost:8080", dimension=768)
        processor.configure(
            db=DatabaseConfig(dbname="t", user="u", password="p"),
            dense=dense,
        )
        assert processor._dense is dense
        assert processor._sparse is None

    def test_configure_with_sparse_only(self):
        processor = IngestProcessor()
        sparse = EndpointProvider(url="http://localhost:8080", response_key="sparse")
        processor.configure(
            db=DatabaseConfig(dbname="t", user="u", password="p"),
            sparse=sparse,
        )
        assert processor._dense is None
        assert processor._sparse is sparse

    def test_configure_resets_ready_flag(self):
        processor = IngestProcessor()
        processor._ready = True
        dense = EndpointProvider(url="http://localhost:8080", dimension=768)
        processor.configure(
            db=DatabaseConfig(dbname="t", user="u", password="p"),
            dense=dense,
        )
        assert processor._ready is False


class TestIngestPipeline:
    def _make_processor_with_mock_db(self):
        """Build IngestProcessor with _ready=True and mocked session factory."""
        processor = IngestProcessor()
        dense = EndpointProvider(url="http://localhost:8080", dimension=768)
        processor.configure(
            db=DatabaseConfig(dbname="t", user="u", password="p"),
            dense=dense,
        )
        processor._ready = True  # bypass ensure_ready()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_session.begin = MagicMock(return_value=mock_session)
        mock_session.close = AsyncMock()

        from chatbot_plugin_sdk.config import _RuntimeDatabase
        processor._runtime = _RuntimeDatabase(
            engine=AsyncMock(),
            session_factory=MagicMock(return_value=mock_session),
            schema="vectors",
        )
        return processor, mock_session

    @pytest.mark.asyncio
    async def test_ingest_without_configure_raises(self):
        processor = IngestProcessor()
        with pytest.raises(NotConfiguredError):
            await processor.ingest("hello", metadata={"url": "https://example.com"})

    @pytest.mark.asyncio
    async def test_ingest_empty_text_raises(self):
        processor, _ = self._make_processor_with_mock_db()
        with pytest.raises(DatabaseError):
            await processor.ingest("   ", metadata={"url": "https://example.com"})

    @pytest.mark.asyncio
    async def test_ingest_missing_url_raises(self):
        processor, _ = self._make_processor_with_mock_db()
        with pytest.raises(DatabaseError, match="url"):
            await processor.ingest("some text content", metadata={})

    @pytest.mark.asyncio
    async def test_ingest_dense_pipeline(self):
        processor, mock_session = self._make_processor_with_mock_db()

        with patch.object(
            processor._dense, "embed", new_callable=AsyncMock
        ) as mock_embed:
            # side_effect returns exactly as many vectors as chunks received
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                metadata={"url": "https://example.com/article", "title": "Test"},
            )
            mock_embed.assert_called_once()
            assert mock_session.execute.called
            assert mock_session.add.called

    @pytest.mark.asyncio
    async def test_ingest_dense_vector_count_mismatch_raises(self):
        processor, _ = self._make_processor_with_mock_db()

        with patch.object(
            processor._dense, "embed", new_callable=AsyncMock
        ) as mock_embed:
            # Return only 1 vector for many chunks
            mock_embed.return_value = [[0.1] * 768]
            with pytest.raises(DatabaseError, match="Dense embedding returned"):
                await processor.ingest(
                    "Hello world. " * 100,
                    metadata={"url": "https://example.com/article"},
                )
