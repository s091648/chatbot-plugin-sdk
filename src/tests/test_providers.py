"""Tests for EndpointProvider and LocalProvider."""

import asyncio
import pytest

from chatbot_plugin_sdk import EndpointProvider, LocalProvider
from chatbot_plugin_sdk.exceptions import EmbeddingError
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider


class TestEndpointProvider:
    def test_dense_requires_dimension(self):
        with pytest.raises(ValueError, match="dimension is required"):
            EndpointProvider(url="http://localhost:8080", response_key="dense")

    def test_sparse_does_not_require_dimension(self):
        p = EndpointProvider(url="http://localhost:8080", response_key="sparse")
        assert p.dimension == 0

    def test_dense_sets_dimension(self):
        p = EndpointProvider(url="http://localhost:8080", dimension=768)
        assert p.dimension == 768

    def test_dense_satisfies_protocol(self):
        p = EndpointProvider(url="http://localhost:8080", dimension=768)
        assert isinstance(p, DenseEmbeddingProvider)

    def test_sparse_satisfies_protocol(self):
        p = EndpointProvider(url="http://localhost:8080", response_key="sparse")
        assert isinstance(p, SparseEmbeddingProvider)


class TestLocalProvider:
    def test_requires_callable(self):
        with pytest.raises(TypeError, match="callable"):
            LocalProvider(fn="not a function")  # type: ignore

    def test_sets_dimension(self):
        p = LocalProvider(fn=lambda t: [[0.1]], dimension=384)
        assert p.dimension == 384

    def test_satisfies_dense_protocol(self):
        p = LocalProvider(fn=lambda t: [[0.1]], dimension=384)
        assert isinstance(p, DenseEmbeddingProvider)

    def test_satisfies_sparse_protocol(self):
        p = LocalProvider(fn=lambda t: [{"1": 0.5}])
        assert isinstance(p, SparseEmbeddingProvider)

    @pytest.mark.asyncio
    async def test_wraps_sync_callable(self):
        called_with = []

        def sync_fn(texts):
            called_with.extend(texts)
            return [[0.1, 0.2, 0.3] for _ in texts]

        p = LocalProvider(fn=sync_fn, dimension=3)
        result = await p.embed(["hello", "world"])
        assert called_with == ["hello", "world"]
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_wraps_async_callable(self):
        async def async_fn(texts):
            return [[0.9] * 4 for _ in texts]

        p = LocalProvider(fn=async_fn, dimension=4)
        result = await p.embed(["test"])
        assert result == [[0.9, 0.9, 0.9, 0.9]]

    @pytest.mark.asyncio
    async def test_raises_embedding_error_on_failure(self):
        def bad_fn(texts):
            raise RuntimeError("model crashed")

        p = LocalProvider(fn=bad_fn, dimension=8)
        with pytest.raises(EmbeddingError, match="model crashed"):
            await p.embed(["test"])
