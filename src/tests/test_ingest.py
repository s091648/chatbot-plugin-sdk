"""Tests for IngestProcessor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import uuid

from chatbot_plugin_sdk import (
    IngestProcessor,
    EndpointProvider,
    DatabaseBackend,
)
from chatbot_plugin_sdk.exceptions import NotConfiguredError, DatabaseError
from chatbot_plugin_sdk.chunking import _chunk_text


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_backend(schema: str = "vectors") -> AsyncMock:
    """Return a fully-mocked DatabaseBackend."""
    backend = AsyncMock(spec=DatabaseBackend)
    backend.schema = schema
    return backend


def _configured_processor(backend=None) -> tuple[IngestProcessor, AsyncMock]:
    backend = backend or _mock_backend()
    dense = EndpointProvider(url="http://localhost:8080", dimension=768)
    processor = IngestProcessor()
    processor.configure(backend=backend, dense=dense)
    processor._ready = True  # bypass _ensure_ready() / backend.setup()
    return processor, backend


# ── Normalisation ──────────────────────────────────────────────────────────────

class TestNormalization:
    def test_normalise_collapses_whitespace(self):
        assert IngestProcessor._normalize("  Hello   world\n\t  foo  ") == "Hello world foo"

    def test_normalise_nfc_unicode(self):
        assert IngestProcessor._normalize("café") == "café"

    def test_normalise_strips_bom(self):
        assert IngestProcessor._normalize("﻿Hello") == "Hello"


# ── Chunking ───────────────────────────────────────────────────────────────────

class TestChunking:
    def test_basic_chunking(self):
        chunks = _chunk_text("Hello world. " * 20, chunk_size=50, overlap=10)
        assert len(chunks) > 0
        assert all(len(c) <= 50 for c in chunks)

    def test_empty_text_returns_empty(self):
        assert _chunk_text("   ") == []


# ── configure() ────────────────────────────────────────────────────────────────

class TestIngestConfigure:
    def test_requires_at_least_one_provider(self):
        processor = IngestProcessor()
        with pytest.raises(NotConfiguredError):
            processor.configure(backend=_mock_backend())

    def test_with_dense_only(self):
        processor = IngestProcessor()
        dense = EndpointProvider(url="http://x", dimension=768)
        processor.configure(backend=_mock_backend(), dense=dense)
        assert processor._dense is dense
        assert processor._sparse is None

    def test_with_sparse_only(self):
        processor = IngestProcessor()
        sparse = EndpointProvider(url="http://x", response_key="sparse")
        processor.configure(backend=_mock_backend(), sparse=sparse)
        assert processor._dense is None
        assert processor._sparse is sparse

    def test_resets_ready_flag(self):
        processor = IngestProcessor()
        processor._ready = True
        processor.configure(
            backend=_mock_backend(),
            dense=EndpointProvider(url="http://x", dimension=768),
        )
        assert processor._ready is False


# ── _ensure_ready() ─────────────────────────────────────────────────────────────

class TestEnsureReady:
    @pytest.mark.asyncio
    async def test_calls_backend_setup_on_first_use(self):
        backend = _mock_backend()
        processor = IngestProcessor()
        processor.configure(backend=backend, dense=EndpointProvider(url="http://x", dimension=768))
        await processor._ensure_ready()
        backend.setup.assert_called_once_with(768, None)  # dense_dim=768, sparse_dim=None
        assert processor._ready is True

    @pytest.mark.asyncio
    async def test_skips_setup_when_already_ready(self):
        backend = _mock_backend()
        processor = IngestProcessor()
        processor.configure(backend=backend, dense=EndpointProvider(url="http://x", dimension=768))
        processor._ready = True
        await processor._ensure_ready()
        backend.setup.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_not_configured(self):
        processor = IngestProcessor()
        with pytest.raises(NotConfiguredError):
            await processor._ensure_ready()


# ── ingest() ───────────────────────────────────────────────────────────────────

class TestIngestPipeline:
    @pytest.mark.asyncio
    async def test_raises_without_configure(self):
        processor = IngestProcessor()
        with pytest.raises(NotConfiguredError):
            await processor.ingest("hello", article_id="550e8400-e29b-41d4-a716-446655440000")

    @pytest.mark.asyncio
    async def test_raises_on_empty_text(self):
        processor, _ = _configured_processor()
        with pytest.raises(DatabaseError):
            await processor.ingest("   ", article_id="550e8400-e29b-41d4-a716-446655440000")

    @pytest.mark.asyncio
    async def test_calls_backend_upsert(self):
        processor, backend = _configured_processor()
        article_id = "550e8400-e29b-41d4-a716-446655440000"
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                article_id=article_id,
                article_columns={"url": "https://example.com/article", "title": "Test"},
            )
        mock_embed.assert_called_once()
        backend.upsert.assert_called_once()
        call_args = backend.upsert.call_args.args
        assert call_args[0] == uuid.UUID(article_id)
        assert call_args[4] is None  # no sparse provider

    @pytest.mark.asyncio
    async def test_article_id_passed_through_to_backend(self):
        processor, backend = _configured_processor()
        article_id = uuid.uuid4()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest("text " * 200, article_id=article_id)

        actual_id = backend.upsert.call_args.args[0]
        assert actual_id == article_id

    @pytest.mark.asyncio
    async def test_article_id_string_converted_to_uuid(self):
        processor, backend = _configured_processor()
        article_id_str = "550e8400-e29b-41d4-a716-446655440000"
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest("text " * 200, article_id=article_id_str)

        actual_id = backend.upsert.call_args.args[0]
        assert actual_id == uuid.UUID(article_id_str)

    @pytest.mark.asyncio
    async def test_raises_on_dense_vector_count_mismatch(self):
        processor, _ = _configured_processor()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]  # only 1 vector for many chunks
            with pytest.raises(DatabaseError, match="Dense embedding returned"):
                await processor.ingest(
                    "Hello world. " * 100,
                    article_id="550e8400-e29b-41d4-a716-446655440000",
                )

    @pytest.mark.asyncio
    async def test_metadata_not_promoted_to_sql_columns(self):
        """metadata keys should never be auto-promoted to SQL columns."""
        processor, backend = _configured_processor()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                article_id="550e8400-e29b-41d4-a716-446655440000",
                metadata={"url": "https://example.com/a", "title": "Test"},
            )

        # metadata should be passed through as-is (opaque JSONB)
        call_args = backend.upsert.call_args.args
        metadata_arg = call_args[1]
        assert metadata_arg["url"] == "https://example.com/a"
        assert metadata_arg["title"] == "Test"


# ── ingest(article_columns=...) ──────────────────────────────────────────────────────

class TestIngestArticleColumns:
    @pytest.mark.asyncio
    async def test_passes_article_columns_to_backend(self):
        processor, backend = _configured_processor()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                article_id="550e8400-e29b-41d4-a716-446655440000",
                article_columns={"url": "https://example.com/a", "title": "Test", "topic_id": "some-uuid"},
            )

        call_kwargs = backend.upsert.call_args.kwargs
        assert call_kwargs.get("article_columns") == {"url": "https://example.com/a", "title": "Test", "topic_id": "some-uuid"}

    @pytest.mark.asyncio
    async def test_article_columns_default_none(self):
        processor, backend = _configured_processor()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                article_id="550e8400-e29b-41d4-a716-446655440000",
            )

        call_kwargs = backend.upsert.call_args.kwargs
        assert call_kwargs.get("article_columns") is None
