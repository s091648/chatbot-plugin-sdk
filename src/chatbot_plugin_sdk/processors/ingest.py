from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Any

from chatbot_plugin_sdk.backends.base import DatabaseBackend
from chatbot_plugin_sdk.chunking import _chunk_text
from chatbot_plugin_sdk.exceptions import DatabaseError, NotConfiguredError
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider


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
            metadata={"url": "https://example.com/article", "title": "My Article"},
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

    def configure(
        self,
        backend: DatabaseBackend,
        dense: DenseEmbeddingProvider | None = None,
        sparse: SparseEmbeddingProvider | None = None,
    ) -> None:
        """Bind backend + providers.  Pure sync, no I/O."""
        if dense is None and sparse is None:
            raise NotConfiguredError(
                "至少需要配置 dense 或 sparse 其中一種 embedding provider。"
            )
        self._backend = backend
        self._dense = dense
        self._sparse = sparse
        self._ready = False

    async def ensure_ready(self) -> None:
        """Idempotent first-use initialisation — delegates to backend.setup()."""
        if self._ready:
            return
        if self._backend is None:
            raise NotConfiguredError("尚未呼叫 configure()。")
        dense_dim = self._dense.dimension if self._dense else None
        await self._backend.setup(dense_dim)
        self._ready = True

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.lstrip("﻿").strip()
        return re.sub(r"\s+", " ", text)

    async def ingest(
        self,
        full_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Full ingest pipeline: normalize → chunk → embed → upsert.

        Args:
            full_text: Raw article text (HTML-stripped or plain).
            metadata:  Must contain ``url`` (str) for idempotent upsert keying.
                       Also accepts ``title`` and ``source``.
        """
        await self.ensure_ready()

        metadata = metadata or {}
        url = metadata.get("url", "")
        if not url:
            raise DatabaseError("metadata must contain 'url' to ensure idempotent ingest.")

        normalized = self._normalize(full_text)
        if not normalized:
            raise DatabaseError("Empty text after normalization.")

        chunks = _chunk_text(normalized)
        if not chunks:
            raise DatabaseError("No chunks produced — input text may be too short.")

        dense_vectors: list[list[float]] | None = None
        sparse_vectors: list[dict[str, float]] | None = None

        if self._dense is not None:
            dense_vectors = await self._dense.embed(chunks)
            if len(dense_vectors) != len(chunks):
                raise DatabaseError(
                    f"Dense embedding returned {len(dense_vectors)} vectors "
                    f"but {len(chunks)} chunks expected."
                )

        if self._sparse is not None:
            sparse_vectors = await self._sparse.embed(chunks)
            if len(sparse_vectors) != len(chunks):
                raise DatabaseError(
                    f"Sparse embedding returned {len(sparse_vectors)} vectors "
                    f"but {len(chunks)} chunks expected."
                )

        article_id = uuid.uuid5(uuid.NAMESPACE_URL, url)
        await self._backend.upsert(article_id, metadata, chunks, dense_vectors, sparse_vectors)
