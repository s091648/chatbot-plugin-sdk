from __future__ import annotations

from chatbot_plugin_sdk.config import FASTEMBED_CACHE_PATH
from chatbot_plugin_sdk.providers.local import LocalProvider


class FastEmbedDenseProvider:
    """Dense embedding provider backed by fastembed TextEmbedding (in-process).

    fastembed is loaded lazily; ``ImportError`` propagates to the caller if the
    ``fastembed`` optional dependency is not installed.

    Respects ``FASTEMBED_CACHE_PATH`` env var for model storage (read via config).

    Args:
        batch_size: Number of texts per ONNX inference call. Lower values reduce
                    peak memory at the cost of throughput. Default: 32.
    """

    def __init__(self, model: str, dimension: int, batch_size: int = 32) -> None:
        from fastembed import TextEmbedding
        cache_dir = FASTEMBED_CACHE_PATH
        _model = TextEmbedding(model, cache_dir=cache_dir)
        _batch_size = batch_size
        self._provider = LocalProvider(
            fn=lambda texts: [v.tolist() for v in _model.embed(texts, batch_size=_batch_size)],
            dimension=dimension,
        )
        self.dimension: int = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._provider.embed(texts)


class FastEmbedSparseProvider:
    """Sparse embedding provider backed by fastembed SparseTextEmbedding (in-process).

    Respects ``FASTEMBED_CACHE_PATH`` env var for model storage.

    Args:
        batch_size: Number of texts per ONNX inference call. SPLADE models hold
                    the full vocabulary (~30k) in activations, so smaller values
                    (4–8) significantly cut peak memory. Default: 8.
    """

    def __init__(self, model: str, dimension: int, batch_size: int = 8) -> None:
        from fastembed import SparseTextEmbedding
        cache_dir = FASTEMBED_CACHE_PATH
        _model = SparseTextEmbedding(model, cache_dir=cache_dir)
        _batch_size = batch_size
        self._provider = LocalProvider(
            fn=lambda texts: [
                {str(idx): float(weight) for idx, weight in zip(v.indices, v.values)}
                for v in _model.embed(texts, batch_size=_batch_size)
            ],
            dimension=dimension,
        )
        self.dimension: int = dimension

    async def embed(self, texts: list[str]) -> list[dict[str, float]]:
        return await self._provider.embed(texts)
