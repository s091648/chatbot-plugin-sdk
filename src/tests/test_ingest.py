"""Tests for IngestProcessor."""

import asyncio
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

    def test_custom_chunk_size_and_overlap_stored(self):
        processor = IngestProcessor()
        processor.configure(
            backend=_mock_backend(),
            dense=EndpointProvider(url="http://x", dimension=768),
            chunk_size=1500,
            chunk_overlap=150,
        )
        assert processor._chunk_size == 1500
        assert processor._chunk_overlap == 150

    def test_default_chunk_size_and_overlap(self):
        from chatbot_plugin_sdk.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
        processor = IngestProcessor()
        processor.configure(
            backend=_mock_backend(),
            dense=EndpointProvider(url="http://x", dimension=768),
        )
        assert processor._chunk_size == DEFAULT_CHUNK_SIZE
        assert processor._chunk_overlap == DEFAULT_CHUNK_OVERLAP

    def test_builds_dense_coordinator_when_dense_configured(self):
        from chatbot_plugin_sdk import EmbeddingBatchCoordinator
        processor = IngestProcessor()
        processor.configure(backend=_mock_backend(), dense=EndpointProvider(url="http://x", dimension=768))
        assert isinstance(processor._dense_coordinator, EmbeddingBatchCoordinator)

    def test_no_dense_coordinator_when_sparse_only(self):
        processor = IngestProcessor()
        processor.configure(backend=_mock_backend(), sparse=EndpointProvider(url="http://x", response_key="sparse"))
        assert processor._dense_coordinator is None

    def test_custom_embed_queue_factory_reaches_coordinator(self):
        import asyncio
        created = []

        def factory():
            q = asyncio.Queue()
            created.append(q)
            return q

        processor = IngestProcessor()
        processor.configure(
            backend=_mock_backend(),
            dense=EndpointProvider(url="http://x", dimension=768),
            embed_queue_factory=factory,
        )
        assert processor._dense_coordinator._queue_factory is factory


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
            await processor.ingest("hello")

    @pytest.mark.asyncio
    async def test_raises_on_empty_text(self):
        processor, _ = _configured_processor()
        with pytest.raises(DatabaseError):
            await processor.ingest(
                "   ", articles_column_values={"url": "https://example.com/article"}
            )

    @pytest.mark.asyncio
    async def test_raises_when_url_missing(self):
        processor, _ = _configured_processor()
        with pytest.raises(DatabaseError, match="url"):
            await processor.ingest("Hello world. " * 100)

    @pytest.mark.asyncio
    async def test_calls_backend_upsert(self):
        processor, backend = _configured_processor()
        url = "https://example.com/article"
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                articles_column_values={"url": url, "title": "Test"},
            )
        mock_embed.assert_called_once()
        backend.upsert.assert_called_once()
        call_args = backend.upsert.call_args.args
        assert call_args[0] == uuid.uuid5(uuid.NAMESPACE_URL, url)
        assert call_args[4] is None  # no sparse provider

    @pytest.mark.asyncio
    async def test_article_id_derived_deterministically_from_url(self):
        processor, backend = _configured_processor()
        url = "https://example.com/deterministic-article"
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest("text " * 200, articles_column_values={"url": url})

        actual_id = backend.upsert.call_args.args[0]
        assert actual_id == uuid.uuid5(uuid.NAMESPACE_URL, url)

    @pytest.mark.asyncio
    async def test_raises_on_dense_vector_count_mismatch(self):
        """A batch-level vector/text count mismatch is now caught inside
        EmbeddingBatchCoordinator (see test_batching.py's dedicated regression
        test) before _embed_in_batches_dense() can ever return a short list —
        so ingest()'s own post-hoc DatabaseError check (still present as a
        defensive backstop) is no longer reachable for the dense path with a
        single-batch mismatch; the failure now surfaces as EmbeddingError."""
        from chatbot_plugin_sdk.exceptions import EmbeddingError

        processor, _ = _configured_processor()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]  # only 1 vector for many chunks
            with pytest.raises(EmbeddingError, match="vectors"):
                await processor.ingest(
                    "Hello world. " * 100,
                    articles_column_values={"url": "https://example.com/article"},
                )

    @pytest.mark.asyncio
    async def test_metadata_not_promoted_to_sql_columns(self):
        """metadata keys should never be auto-promoted to SQL columns."""
        processor, backend = _configured_processor()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                articles_column_values={"url": "https://example.com/a"},
                metadata={"url": "https://example.com/a", "title": "Test"},
            )

        # metadata should be passed through as-is (opaque JSONB)
        call_args = backend.upsert.call_args.args
        metadata_arg = call_args[1]
        assert metadata_arg["url"] == "https://example.com/a"
        assert metadata_arg["title"] == "Test"


# ── ingest(articles_column_values=...) ────────────────────────────────────────────────

class TestIngestArticleColumns:
    @pytest.mark.asyncio
    async def test_passes_article_columns_to_backend(self):
        processor, backend = _configured_processor()
        columns = {"url": "https://example.com/a", "title": "Test", "topic_id": "some-uuid"}
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                articles_column_values=columns,
            )

        call_kwargs = backend.upsert.call_args.kwargs
        assert call_kwargs.get("articles_column_values") == columns


# ── aclose() ───────────────────────────────────────────────────────────────────

class TestAclose:
    @pytest.mark.asyncio
    async def test_aclose_before_configure_is_a_noop(self):
        processor = IngestProcessor()
        await processor.aclose()  # must not raise

    @pytest.mark.asyncio
    async def test_aclose_cancels_dense_coordinator_worker(self):
        processor, backend = _configured_processor()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                articles_column_values={"url": "https://example.com/article"},
            )
        worker = processor._dense_coordinator._worker_task
        await processor.aclose()
        assert worker.cancelled() or worker.done()


# ── get_embed_queue() / set_embed_queue() ───────────────────────────────────────

class TestGetSetEmbedQueue:
    def test_get_embed_queue_returns_none_when_dense_not_configured(self):
        processor = IngestProcessor()
        processor.configure(backend=_mock_backend(), sparse=EndpointProvider(url="http://x", response_key="sparse"))
        assert processor.get_embed_queue() is None

    def test_get_embed_queue_returns_none_before_first_use(self):
        processor, _ = _configured_processor()
        assert processor.get_embed_queue() is None

    @pytest.mark.asyncio
    async def test_get_embed_queue_returns_current_queue_after_first_use(self):
        processor, _ = _configured_processor()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                articles_column_values={"url": "https://example.com/article"},
            )
        assert processor.get_embed_queue() is processor._dense_coordinator._queue
        await processor.aclose()

    @pytest.mark.asyncio
    async def test_set_embed_queue_raises_when_dense_not_configured(self):
        processor = IngestProcessor()
        processor.configure(backend=_mock_backend(), sparse=EndpointProvider(url="http://x", response_key="sparse"))
        with pytest.raises(NotConfiguredError):
            await processor.set_embed_queue(asyncio.Queue())

    @pytest.mark.asyncio
    async def test_set_embed_queue_swaps_the_coordinators_queue(self):
        processor, _ = _configured_processor()
        custom_queue = asyncio.Queue()
        await processor.set_embed_queue(custom_queue)
        assert processor.get_embed_queue() is custom_queue

        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                articles_column_values={"url": "https://example.com/article"},
            )
        assert processor.get_embed_queue() is custom_queue
        await processor.aclose()
