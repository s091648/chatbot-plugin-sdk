"""Tests for LLM call paths (_call_llm and _call_gemini) on BaseRagProcessor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin_sdk.config import settings
from chatbot_plugin_sdk.base import BaseRagProcessor
from chatbot_plugin_sdk.exceptions import LLMError


def _mock_httpx_client(http_response: MagicMock) -> AsyncMock:
    """Build a mock httpx.AsyncClient that returns the given response."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=http_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@pytest.mark.asyncio
async def test_call_gemini_success():
    """_call_gemini returns text when Gemini responds 200 OK."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Hi from Gemini"}]}}]
    }

    mock_client = _mock_httpx_client(mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with patch.object(settings, "gemini_api_key", "fake-key"):
            with patch.object(settings, "gemini_model", "gemini-test"):
                sdk = BaseRagProcessor()
                result = await sdk._call_gemini("sys", "prompt")
                assert result == "Hi from Gemini"


@pytest.mark.asyncio
async def test_call_gemini_api_error():
    """_call_gemini raises LLMError when Gemini returns non-200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "rate limited"

    mock_client = _mock_httpx_client(mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with patch.object(settings, "gemini_api_key", "fake-key"):
            sdk = BaseRagProcessor()
            with pytest.raises(LLMError) as exc:
                await sdk._call_gemini("sys", "prompt")
            assert "429" in str(exc.value)


@pytest.mark.asyncio
async def test_call_gemini_malformed_response():
    """_call_gemini raises LLMError on unexpected response shape."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"error": "whoops"}

    mock_client = _mock_httpx_client(mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with patch.object(settings, "gemini_api_key", "fake-key"):
            sdk = BaseRagProcessor()
            with pytest.raises(LLMError) as exc:
                await sdk._call_gemini("sys", "prompt")
            assert "Unexpected Gemini response" in str(exc.value)


@pytest.mark.asyncio
async def test_call_llm_anthropic_success():
    """_call_llm returns Anthropic response when key is set and works."""
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Claude says hello")]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        with patch.object(settings, "llm_api_key", "sk-real"):
            with patch.object(settings, "llm_model", "claude-test"):
                sdk = BaseRagProcessor()
                result = await sdk._call_llm("some context", "question?")
                assert result == "Claude says hello"


@pytest.mark.asyncio
async def test_call_llm_anthropic_fallback_to_gemini():
    """_call_llm falls back to Gemini when Anthropic fails."""
    mock_anthropic_client = MagicMock()
    mock_anthropic_client.messages = MagicMock()
    mock_anthropic_client.messages.create = MagicMock(side_effect=Exception("boom"))

    mock_gemini_resp = MagicMock()
    mock_gemini_resp.status_code = 200
    mock_gemini_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Gemini fallback"}]}}]
    }

    mock_httpx_client = _mock_httpx_client(mock_gemini_resp)

    with patch("anthropic.AsyncAnthropic", return_value=mock_anthropic_client):
        with patch("httpx.AsyncClient", return_value=mock_httpx_client):
            with patch.object(settings, "llm_api_key", "sk-real", create=True):
                with patch.object(settings, "gemini_api_key", "fake-key", create=True):
                    with patch.object(settings, "llm_model", "claude-test"):
                        with patch.object(settings, "gemini_model", "gemini-test"):
                            sdk = BaseRagProcessor()
                            result = await sdk._call_llm("ctx", "q?")
                            assert result == "Gemini fallback"
