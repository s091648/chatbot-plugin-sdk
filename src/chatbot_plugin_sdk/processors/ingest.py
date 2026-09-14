from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
import uuid
from typing import Any

from chatbot_plugin_sdk.backends.base import DatabaseBackend
from chatbot_plugin_sdk.batching import EmbeddingBatchCoordinator, EmbedWorkItem, QueueFactory
from chatbot_plugin_sdk.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, _chunk_text
from chatbot_plugin_sdk.exceptions import DatabaseError, NotConfiguredError
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider

logger = logging.getLogger(__name__)


class IngestProcessor:
    """文章向量化寫入處理器。

    Pipeline: normalize → chunk → embed (dense / sparse) → upsert via backend

    Usage::

        # ThreadPoolExecutor (sync psycopg2):
        backend = SyncPgBackend(DatabaseConfig(...))

        # FastAPI / native async (asyncpg):
        backend = AsyncPgBackend(DatabaseConfig(...))

        processor = IngestProcessor()
        processor.configure(
            backend=backend,
            dense=EndpointProvider(url="http://embed:8080", dimension=768),
        )
        await processor.ingest(
            full_text="...",
            articles_column_values={
                "url": "https://example.com/article",  # required — used as idempotent key
                "title": "My Article",
            },
        )

    Thread-safety notes:
        - The processor itself holds no per-call mutable state after ``configure()``.
        - ``_ready`` may be set concurrently by multiple threads during startup; the
          worst case is ``backend.setup()`` being called twice, which is idempotent.
        - Use :class:`SyncPgBackend` for ``ThreadPoolExecutor`` + ``asyncio.run()``
          patterns.  :class:`AsyncPgBackend` must live inside a single event loop.
    """

    def __init__(self) -> None:
        self._backend: DatabaseBackend | None = None
        self._dense: DenseEmbeddingProvider | None = None
        self._sparse: SparseEmbeddingProvider | None = None
        self._ready: bool = False
        self._ready_lock: asyncio.Lock = asyncio.Lock()
        self._embed_batch_size: int = 16
        self._chunk_size: int = DEFAULT_CHUNK_SIZE
        self._chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
        self._dense_coordinator: EmbeddingBatchCoordinator | None = None
        self._sparse_coordinator: EmbeddingBatchCoordinator | None = None

    def configure(
        self,
        backend: DatabaseBackend,
        dense: DenseEmbeddingProvider | None = None,
        sparse: SparseEmbeddingProvider | None = None,
        embed_batch_size: int = 16,
        embed_queue_factory: QueueFactory | None = None,
        sparse_embed_queue_factory: QueueFactory | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Bind backend + providers.  Pure sync, no I/O.

        Args:
            embed_batch_size: Max chunks sent to each provider's ``embed()`` per
                              call. Smaller values reduce peak memory when using
                              local ONNX models (e.g. SPLADE). Default: 16.
            embed_queue_factory: Optional factory for the dense-embedding
                              coordinator's internal queue (see
                              ``EmbeddingBatchCoordinator``) — inject a custom
                              ``asyncio.Queue`` subclass (priority ordering,
                              instrumentation, etc.) here. Defaults to a plain
                              ``asyncio.Queue()``. Ignored when ``dense`` is None.
            sparse_embed_queue_factory: Same seam as ``embed_queue_factory``,
                              for the sparse-embedding coordinator's own,
                              independent queue. Ignored when ``sparse`` is None.
            chunk_size: Maximum characters per chunk. Default: 500.
            chunk_overlap: Overlap characters between consecutive chunks. Default: 50.
        """
        if dense is None and sparse is None:
            raise NotConfiguredError(
                "至少需要配置 dense 或 sparse 其中一種 embedding provider。"
            )
        self._backend = backend
        self._dense = dense
        self._sparse = sparse
        self._embed_batch_size = embed_batch_size
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._ready = False
        # Dense and sparse each get their own coordinator — independent queue,
        # independent worker task, independent rate-limit headroom awareness —
        # so ingest() can run both concurrently (see ingest() below) without
        # one embedding kind's batching decisions interfering with the
        # other's, even though both may be draining chunks from the same
        # article at the same time.
        self._dense_coordinator = (
            EmbeddingBatchCoordinator(
                provider=dense, embed_batch_size=embed_batch_size, queue_factory=embed_queue_factory,
            )
            if dense is not None else None
        )
        self._sparse_coordinator = (
            EmbeddingBatchCoordinator(
                provider=sparse, embed_batch_size=embed_batch_size, queue_factory=sparse_embed_queue_factory,
            )
            if sparse is not None else None
        )

    async def _ensure_ready(self) -> None:
        """Idempotent first-use initialisation — delegates to backend.setup().

        Serialised by ``_ready_lock`` so a burst of concurrent first-time
        ``ingest()`` calls runs ``backend.setup()`` once, not once per call —
        each ``setup()`` opens its own connection, so the unsynchronised
        version turned the first fan-out into a cold-connect stampede.
        """
        if self._ready:
            return
        if self._backend is None:
            raise NotConfiguredError("尚未呼叫 configure()。")
        async with self._ready_lock:
            if self._ready:
                return
            dense_dim = self._dense.dimension if self._dense else None
            sparse_dim = self._sparse.dimension if self._sparse else None
            logger.debug("vector_store_setup", extra={"dense_dim": dense_dim, "sparse_dim": sparse_dim})
            await self._backend.setup(dense_dim, sparse_dim)
            self._ready = True
            logger.info("vector_store_ready", extra={"dense_dim": dense_dim, "sparse_dim": sparse_dim})

    async def prewarm(self, connections: int | None = None) -> None:
        """Run first-use setup and pre-open a batch of DB connections now,
        instead of on the first (usually concurrent) ``ingest()`` calls.

        Call once, after ``configure()``, from a host that fans out many
        concurrent ``ingest()`` calls: it collapses the initial ``setup()`` +
        cold-connect burst — whose per-connection DNS lookups otherwise
        stampede asyncio's default executor and time out — into one
        sequential warm-up. No-op-safe if the backend exposes no ``prewarm``.
        """
        await self._ensure_ready()
        backend_prewarm = getattr(self._backend, "prewarm", None)
        if backend_prewarm is not None:
            await backend_prewarm(connections)

    async def _embed_in_batches_dense(self, chunks: list[str]) -> list[list[float]]:
        assert self._dense_coordinator is not None
        return await self._dense_coordinator.embed_many(chunks)

    async def _maybe_embed_dense(self, chunks: list[str]) -> list[list[float]] | None:
        """None (a no-op) when dense isn't configured; otherwise embeds and
        validates the result — split out from ingest() so it can be run
        concurrently with _maybe_embed_sparse via asyncio.gather()."""
        if self._dense is None:
            return None
        vectors = await self._embed_in_batches_dense(chunks)
        if len(vectors) != len(chunks):
            raise DatabaseError(
                f"Dense embedding returned {len(vectors)} vectors but {len(chunks)} chunks expected."
            )
        return vectors

    async def _maybe_embed_sparse(self, chunks: list[str]) -> list[dict[str, float]] | None:
        """Sparse counterpart of _maybe_embed_dense — see its docstring."""
        if self._sparse is None:
            return None
        vectors = await self._embed_in_batches_sparse(chunks)
        if len(vectors) != len(chunks):
            raise DatabaseError(
                f"Sparse embedding returned {len(vectors)} vectors but {len(chunks)} chunks expected."
            )
        return vectors

    async def aclose(self) -> None:
        """Release background resources — cancels the dense- and sparse-
        embedding coordinators' worker tasks, for whichever were ever
        started. Idempotent; safe to call even if ``configure()`` was never
        called or neither ``dense`` nor ``sparse`` is configured."""
        if self._dense_coordinator is not None:
            await self._dense_coordinator.aclose()
        if self._sparse_coordinator is not None:
            await self._sparse_coordinator.aclose()

    def get_embed_queue(self) -> "asyncio.Queue[EmbedWorkItem] | None":
        """Return the dense-embedding coordinator's current queue, or None if
        dense embedding isn't configured, or no work has been submitted yet
        and set_embed_queue() was never called. See
        EmbeddingBatchCoordinator.get_queue()."""
        if self._dense_coordinator is None:
            return None
        return self._dense_coordinator.get_queue()

    async def set_embed_queue(self, queue: "asyncio.Queue[EmbedWorkItem]") -> None:
        """Replace the dense-embedding coordinator's queue — see
        EmbeddingBatchCoordinator.set_queue() for the safe-swap semantics
        (migrates not-yet-claimed items, stops and restarts the worker).

        Raises NotConfiguredError if dense embedding isn't configured."""
        if self._dense_coordinator is None:
            raise NotConfiguredError("Dense embedding isn't configured — nothing to set a queue on.")
        await self._dense_coordinator.set_queue(queue)

    def get_embed_sparse_queue(self) -> "asyncio.Queue[EmbedWorkItem] | None":
        """Sparse-embedding counterpart of get_embed_queue() — returns the
        sparse coordinator's own, independent queue, or None if sparse
        embedding isn't configured or no work has been submitted yet."""
        if self._sparse_coordinator is None:
            return None
        return self._sparse_coordinator.get_queue()

    async def set_embed_sparse_queue(self, queue: "asyncio.Queue[EmbedWorkItem]") -> None:
        """Sparse-embedding counterpart of set_embed_queue() — see
        EmbeddingBatchCoordinator.set_queue() for the safe-swap semantics.

        Raises NotConfiguredError if sparse embedding isn't configured."""
        if self._sparse_coordinator is None:
            raise NotConfiguredError("Sparse embedding isn't configured — nothing to set a queue on.")
        await self._sparse_coordinator.set_queue(queue)

    async def _embed_in_batches_sparse(self, chunks: list[str]) -> list[dict[str, float]]:
        assert self._sparse_coordinator is not None
        return await self._sparse_coordinator.embed_many(chunks)

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.lstrip("﻿").strip()
        return re.sub(r"\s+", " ", text)

    async def ingest(
        self,
        full_text: str,
        articles_column_values: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Full ingest pipeline: normalize → chunk → embed → upsert.

        Args:
            full_text: Raw article text (HTML-stripped or plain).
            articles_column_values: SQL column values for the articles table.
                                    Must include ``url`` — it is used to derive
                                    a deterministic ``article_id`` via
                                    ``uuid.uuid5(NAMESPACE_URL, url)`` for
                                    idempotent upserts.  Any other keys become
                                    INSERT columns; column existence is the
                                    caller's responsibility.
            metadata: Opaque JSONB metadata — the SDK never interprets its keys.
        """
        await self._ensure_ready()

        url = (articles_column_values or {}).get("url") or ""
        if not url:
            raise DatabaseError(
                "'url' is required in articles_column_values — "
                "it is used to derive the idempotent article_id via uuid5."
            )
        article_id = uuid.uuid5(uuid.NAMESPACE_URL, url)

        normalized = self._normalize(full_text)
        if not normalized:
            raise DatabaseError("Empty text after normalization.")

        chunks = _chunk_text(normalized, chunk_size=self._chunk_size, overlap=self._chunk_overlap)
        if not chunks:
            raise DatabaseError("No chunks produced — input text may be too short.")

        # Dense and sparse embed the same chunks independently — neither needs
        # the other's result — so run them concurrently via separate
        # coordinators (see configure()) instead of sequentially. If either
        # fails, cancel the other rather than leaving it to keep enqueuing/
        # running in the background after this ingest() call has already
        # raised for this article.
        dense_task = asyncio.ensure_future(self._maybe_embed_dense(chunks))
        sparse_task = asyncio.ensure_future(self._maybe_embed_sparse(chunks))
        try:
            dense_vectors, sparse_vectors = await asyncio.gather(dense_task, sparse_task)
        except BaseException:
            dense_task.cancel()
            sparse_task.cancel()
            raise

        logger.debug(
            "ingest_upserting",
            extra={"url": url, "chunk_count": len(chunks), "embed_batch_size": self._embed_batch_size},
        )
        await self._backend.upsert(
            article_id,
            metadata or {},
            chunks,
            dense_vectors,
            sparse_vectors,
            articles_column_values=articles_column_values,
        )
        logger.info(
            "ingest_complete",
            extra={"url": url, "chunk_count": len(chunks), "has_sparse": sparse_vectors is not None},
        )
