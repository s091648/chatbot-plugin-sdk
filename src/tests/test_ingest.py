"""Tests for RagArticleProcessor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin_sdk import RagArticleProcessor
from chatbot_plugin_sdk.exceptions import NotConfiguredError, DatabaseError
from chatbot_plugin_sdk.chunking import _chunk_text


class TestNormalization:
    def test_normalise_collapses_whitespace(self):
        sdk = RagArticleProcessor()
        text = "  Hello   world\n\t  foo  "
        result = sdk._normalize_full_text(text)
        assert result == "Hello world foo"

    def test_normalise_nfc_unicode(self):
        sdk = RagArticleProcessor()
        # e + combining acute = é
        text = "cafe\u0301"
        result = sdk._normalize_full_text(text)
        assert result == "caf\u00e9"

    def test_normalise_strips_bom(self):
        sdk = RagArticleProcessor()
        text = "\ufeffHello"
        result = sdk._normalize_full_text(text)
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


class TestIngestErrors:
    @pytest.mark.asyncio
    async def test_ingest_without_db_config_raises(self):
        sdk = RagArticleProcessor()
        with pytest.raises(NotConfiguredError):
            await sdk.ingest("hello", metadata={"url": "https://example.com"})

    @pytest.mark.asyncio
    async def test_ingest_without_embedding_config_raises(self):
        sdk = RagArticleProcessor()
        sdk.configure(dbname="test", user="test", password="test")
        with pytest.raises(NotConfiguredError):
            await sdk.ingest("hello", metadata={"url": "https://example.com"})

    @pytest.mark.asyncio
    async def test_ingest_empty_text_raises(self):
        sdk = RagArticleProcessor()
        sdk.configure(
            dbname="test", user="test", password="test",
            embedding_model_api="http://localhost:8080",
        )
        with pytest.raises(DatabaseError):
            await sdk.ingest("   ", metadata={"url": "https://example.com"})


class TestIngestPipeline:
    def _mock_session(self):
        """Build a MagicMock session that works with SQLAlchemy async patterns."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        # begin() returns self for use as async context manager
        mock_session.begin = MagicMock(return_value=mock_session)
        mock_session.close = AsyncMock()
        return mock_session

    @pytest.mark.asyncio
    async def test_ingest_full_pipeline(self):
        sdk = RagArticleProcessor()
        sdk.configure(
            dbname="test", user="test", password="test",
            embedding_model_api="http://localhost:8080",
        )

        # Mock the engine so no real DB connection is attempted
        sdk._db_config.engine = AsyncMock()
        sdk._tables_created = True

        mock_session = self._mock_session()
        sdk._db_config.session_factory = MagicMock(return_value=mock_session)

        with patch.object(
            sdk, "_embed_texts",
            new_callable=AsyncMock,
        ) as mock_embed:
            mock_embed.return_value = (
                [[0.1] * 1024 for _ in range(3)],
                [{i: 0.1} for i in range(3)],
            )
            await sdk.ingest(
                "Hello world. " * 100,
                metadata={"url": "https://example.com/article", "title": "Test"},
            )
            mock_embed.assert_called_once()
            # Verify save was called
            assert mock_session.execute.called

    @pytest.mark.asyncio
    async def test_ingest_with_custom_normalization(self):
        sdk = RagArticleProcessor()
        sdk.configure(
            dbname="test", user="test", password="test",
            embedding_model_api="http://localhost:8080",
        )

        # Mock the engine so no real DB connection is attempted
        sdk._db_config.engine = AsyncMock()
        sdk._tables_created = True

        mock_session = self._mock_session()
        sdk._db_config.session_factory = MagicMock(return_value=mock_session)

        custom_called = []

        def custom_norm(text):
            custom_called.append(text)
            return text.upper()

        with patch.object(
            sdk, "_embed_texts",
            new_callable=AsyncMock,
        ) as mock_embed:
            mock_embed.return_value = (
                [[0.1] * 1024],
                [{1: 0.1}],
            )
            await sdk.ingest(
                "hello world",
                normalization=custom_norm,
                metadata={"url": "https://example.com/article"},
            )
            assert len(custom_called) == 1
            assert custom_called[0] == "hello world"
