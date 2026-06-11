"""Tests for RagQueryProcessor."""

from unittest.mock import AsyncMock, patch

import pytest

from chatbot_plugin_sdk.config import settings
from chatbot_plugin_sdk.contracts import ChatResponse
from chatbot_plugin_sdk import RagQueryProcessor
from chatbot_plugin_sdk.exceptions import NotConfiguredError


class TestQuerySDKConfigure:
    def test_configure_sets_db(self):
        sdk = RagQueryProcessor()
        sdk.configure(dbname="test", user="test", password="test")
        assert sdk._db_config is not None
        assert sdk._embed_config is None  # not provided

    def test_configure_sets_embed(self):
        sdk = RagQueryProcessor()
        sdk.configure(
            dbname="test", user="test", password="test",
            embedding_model_api="http://localhost:8080",
        )
        assert sdk._db_config is not None
        assert sdk._embed_config is not None

    def test_configure_with_kwargs(self):
        sdk = RagQueryProcessor()
        sdk.configure(
            dbname="test", user="test", password="test",
            host="db.example.com", port=5433,
        )
        assert sdk._db_config is not None


class TestQuerySDKQuery:
    @pytest.mark.asyncio
    async def test_query_delegates_to_chat(self):
        sdk = RagQueryProcessor()
        sdk.configure(
            dbname="test", user="test", password="test",
            embedding_model_api="http://localhost:8080",
        )
        with patch.object(
            sdk, "chat",
            new_callable=AsyncMock,
        ) as mock_chat:
            mock_chat.return_value = ChatResponse(
                reply="RAG is...",
                articles_used=[],
                chunks=[],
            )
            resp = await sdk.query("What is RAG?")
            assert resp.reply == "RAG is..."
            mock_chat.assert_called_once_with(message="What is RAG?")

    @pytest.mark.asyncio
    async def test_query_without_config_raises(self):
        sdk = RagQueryProcessor()
        with pytest.raises(NotConfiguredError):
            await sdk.query("What is RAG?")
