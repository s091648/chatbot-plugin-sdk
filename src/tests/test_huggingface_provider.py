"""Tests for HuggingFaceDenseProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin_sdk import HuggingFaceDenseProvider, SlidingWindowStrategy
from chatbot_plugin_sdk.exceptions import EmbeddingError
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider


def _make_mock_response(status_code: int, json_data=None, headers=None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_data or {})
    resp.headers = headers or {}
    resp.text = str(json_data)
    return resp


def _make_mock_client(response: MagicMock) -> MagicMock:
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


class TestHuggingFaceDenseProviderInit:
    def test_sets_dimension(self):
        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="BAAI/bge-m3", dimension=1024)
        assert p.dimension == 1024

    def test_satisfies_dense_protocol(self):
        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="BAAI/bge-m3", dimension=1024)
        assert isinstance(p, DenseEmbeddingProvider)

    def test_url_built_from_model(self):
        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="BAAI/bge-m3", dimension=1024)
        assert p._url == "https://api-inference.huggingface.co/models/BAAI/bge-m3"


class TestHuggingFaceDenseProviderEmbed:
    @pytest.mark.asyncio
    async def test_returns_vectors_on_200(self):
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        resp = _make_mock_response(200, vectors)
        client = _make_mock_client(resp)

        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="BAAI/bge-m3", dimension=3)
        with patch.object(p, "_build_client", return_value=client):
            result = await p.embed(["hello", "world"])

        assert result == vectors

    @pytest.mark.asyncio
    async def test_sends_inputs_and_wait_for_model(self):
        resp = _make_mock_response(200, [[0.1]])
        client = _make_mock_client(resp)

        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="BAAI/bge-m3", dimension=1)
        with patch.object(p, "_build_client", return_value=client):
            await p.embed(["test"])

        call_kwargs = client.post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["inputs"] == ["test"]
        assert payload["options"]["wait_for_model"] is True

    @pytest.mark.asyncio
    async def test_raises_on_unexpected_response_shape(self):
        resp = _make_mock_response(200, {"error": "unexpected"})
        client = _make_mock_client(resp)

        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="BAAI/bge-m3", dimension=1)
        with patch.object(p, "_build_client", return_value=client):
            with pytest.raises(EmbeddingError, match="Unexpected HuggingFace response shape"):
                await p.embed(["test"])

    @pytest.mark.asyncio
    async def test_raises_embedding_error_on_non_retryable_status(self):
        resp = _make_mock_response(401, {"error": "Unauthorized"})
        client = _make_mock_client(resp)

        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="BAAI/bge-m3", dimension=1)
        with patch.object(p, "_build_client", return_value=client):
            with pytest.raises(EmbeddingError, match="HTTP 401"):
                await p.embed(["test"])

    @pytest.mark.asyncio
    async def test_retries_on_503_and_succeeds(self):
        vectors = [[0.1, 0.2]]
        resp_503 = _make_mock_response(503, {"estimated_time": 0.01})
        resp_200 = _make_mock_response(200, vectors)

        client = AsyncMock()
        client.post = AsyncMock(side_effect=[resp_503, resp_200])
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="m", dimension=2, max_retries=3)
        with patch.object(p, "_build_client", return_value=client):
            result = await p.embed(["hello"])

        assert result == vectors
        assert client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_503_retries(self):
        resp_503 = _make_mock_response(503, {"estimated_time": 0.01})

        client = AsyncMock()
        client.post = AsyncMock(return_value=resp_503)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="m", dimension=1, max_retries=2)
        with patch.object(p, "_build_client", return_value=client):
            with pytest.raises(EmbeddingError, match="unavailable"):
                await p.embed(["test"])

    @pytest.mark.asyncio
    async def test_retries_on_429_and_succeeds(self):
        vectors = [[0.9]]
        resp_429 = _make_mock_response(429, {}, headers={"Retry-After": "0.01"})
        resp_200 = _make_mock_response(200, vectors)

        client = AsyncMock()
        client.post = AsyncMock(side_effect=[resp_429, resp_200])
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="m", dimension=1, max_retries=3)
        with patch.object(p, "_build_client", return_value=client):
            result = await p.embed(["test"])

        assert result == vectors

    @pytest.mark.asyncio
    async def test_does_not_retry_long_503_delay(self):
        # estimated_time > 300 seconds — treat as infra issue, don't retry
        resp_503 = _make_mock_response(503, {"estimated_time": 301.0})
        client = _make_mock_client(resp_503)

        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="m", dimension=1, max_retries=5)
        with patch.object(p, "_build_client", return_value=client):
            with pytest.raises(EmbeddingError, match="unavailable"):
                await p.embed(["test"])

        assert client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_raises_embedding_error_on_network_failure(self):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=Exception("connection refused"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="m", dimension=1)
        with patch.object(p, "_build_client", return_value=client):
            with pytest.raises(EmbeddingError, match="request failed"):
                await p.embed(["test"])


class TestHuggingFaceDenseProviderRateLimit:
    @pytest.mark.asyncio
    async def test_acquire_called_before_request(self):
        strategy = MagicMock(spec=SlidingWindowStrategy)
        strategy.acquire = AsyncMock()

        vectors = [[0.1]]
        resp = _make_mock_response(200, vectors)
        client = _make_mock_client(resp)

        p = HuggingFaceDenseProvider(
            api_token="hf_xxx", model="m", dimension=1, rate_limit=strategy
        )
        with patch.object(p, "_build_client", return_value=client):
            await p.embed(["hello"])

        strategy.acquire.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_rate_limit_when_none(self):
        vectors = [[0.1]]
        resp = _make_mock_response(200, vectors)
        client = _make_mock_client(resp)

        p = HuggingFaceDenseProvider(api_token="hf_xxx", model="m", dimension=1)
        with patch.object(p, "_build_client", return_value=client):
            result = await p.embed(["hello"])

        assert result == vectors
