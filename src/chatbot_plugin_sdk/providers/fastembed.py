from __future__ import annotations

import os

from chatbot_plugin_sdk.providers.local import LocalProvider


class FastEmbedDenseProvider:
    """Dense embedding provider backed by fastembed TextEmbedding (in-process).

    fastembed is loaded lazily; ``ImportError`` propagates to the caller if the
    ``fastembed`` optional dependency is not installed.

    Respects ``FASTEMBED_CACHE_PATH`` env var for model storage.
    """

    def __init__(self, model: str, dimension: int) -> None:
        from fastembed import TextEmbedding
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or None
        _model = TextEmbedding(model, cache_dir=cache_dir)
        self._provider = LocalProvider(
            fn=lambda texts: [v.tolist() for v in _model.embed(texts)],
            dimension=dimension,
        )
        self.dimension: int = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._provider.embed(texts)


class FastEmbedSparseProvider:
    """Sparse embedding provider backed by fastembed SparseTextEmbedding (in-process).

    Respects ``FASTEMBED_CACHE_PATH`` env var for model storage.
    """

    def __init__(self, model: str, dimension: int) -> None:
        from fastembed import SparseTextEmbedding
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or None
        _model = SparseTextEmbedding(model, cache_dir=cache_dir)
        self._provider = LocalProvider(
            fn=lambda texts: [
                {str(idx): float(weight) for idx, weight in zip(v.indices, v.values)}
                for v in _model.embed(texts)
            ],
            dimension=dimension,
        )
        self.dimension: int = dimension

    async def embed(self, texts: list[str]) -> list[dict[str, float]]:
        return await self._provider.embed(texts)
