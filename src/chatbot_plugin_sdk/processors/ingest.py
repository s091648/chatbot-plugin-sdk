from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from typing import Any

from chatbot_plugin_sdk.backends.base import DatabaseBackend
from chatbot_plugin_sdk.chunking import _chunk_text
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
        self._embed_batch_size: int = 16

    def configure(
        self,
        backend: DatabaseBackend,
        dense: DenseEmbeddingProvider | None = None,
        sparse: SparseEmbeddingProvider | None = None,
        embed_batch_size: int = 16,
    ) -> None:
        """Bind backend + providers.  Pure sync, no I/O.

        Args:
            embed_batch_size: Max chunks sent to each provider's ``embed()`` per
                              call. Smaller values reduce peak memory when using
                              local ONNX models (e.g. SPLADE). Default: 16.
        """
        if dense is None and sparse is None:
            raise NotConfiguredError(
                "至少需要配置 dense 或 sparse 其中一種 embedding provider。"
            )
        self._backend = backend
        self._dense = dense
        self._sparse = sparse
        self._embed_batch_size = embed_batch_size
        self._ready = False

    async def _ensure_ready(self) -> None:
        """Idempotent first-use initialisation — delegates to backend.setup()."""
        if self._ready:
            return
        if self._backend is None:
            raise NotConfiguredError("尚未呼叫 configure()。")
        dense_dim = self._dense.dimension if self._dense else None
        sparse_dim = self._sparse.dimension if self._sparse else None
        logger.debug("vector_store_setup", extra={"dense_dim": dense_dim, "sparse_dim": sparse_dim})
        await self._backend.setup(dense_dim, sparse_dim)
        self._ready = True
        logger.info("vector_store_ready", extra={"dense_dim": dense_dim, "sparse_dim": sparse_dim})

    async def _embed_in_batches_dense(self, chunks: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(chunks), self._embed_batch_size):
            batch = chunks[i : i + self._embed_batch_size]
            results.extend(await self._dense.embed(batch))  # type: ignore[union-attr]
        return results

    async def _embed_in_batches_sparse(self, chunks: list[str]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        for i in range(0, len(chunks), self._embed_batch_size):
            batch = chunks[i : i + self._embed_batch_size]
            results.extend(await self._sparse.embed(batch))  # type: ignore[union-attr]
        return results

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

        chunks = _chunk_text(normalized)
        if not chunks:
            raise DatabaseError("No chunks produced — input text may be too short.")

        dense_vectors: list[list[float]] | None = None
        sparse_vectors: list[dict[str, float]] | None = None

        if self._dense is not None:
            dense_vectors = await self._embed_in_batches_dense(chunks)
            if len(dense_vectors) != len(chunks):
                raise DatabaseError(
                    f"Dense embedding returned {len(dense_vectors)} vectors "
                    f"but {len(chunks)} chunks expected."
                )

        if self._sparse is not None:
            sparse_vectors = await self._embed_in_batches_sparse(chunks)
            if len(sparse_vectors) != len(chunks):
                raise DatabaseError(
                    f"Sparse embedding returned {len(sparse_vectors)} vectors "
                    f"but {len(chunks)} chunks expected."
                )

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
