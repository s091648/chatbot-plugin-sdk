"""Unit tests for BaseRagProcessor (successor to ToolboxService)."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin_sdk.config import settings
from chatbot_plugin_sdk.contracts import (
    ArticleInfo,
    ChatRequest,
    ChunkData,
    SearchRequest,
    StoreChunksRequest,
    StoreChunksResponse,
)
from chatbot_plugin_sdk.models import Article, ArticleChunk
from chatbot_plugin_sdk.base import BaseRagProcessor
from chatbot_plugin_sdk.exceptions import DatabaseError


def _make_mock_db_config():
    """Build a mock DatabaseConfig with engine and session_factory."""
    mock_factory = MagicMock()
    mock_session = AsyncMock()
    mock_factory.return_value = mock_session
    mock_session.begin = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()

    mock_cfg = MagicMock()
    mock_cfg.engine = mock_engine
    mock_cfg.session_factory = mock_factory
    return mock_cfg, mock_session


@pytest.fixture
def sdk() -> BaseRagProcessor:
    """BaseRagProcessor with mocked DB config."""
    cfg, session = _make_mock_db_config()
    sdk = BaseRagProcessor()
    sdk._db_config = cfg
    sdk._tables_created = True
    sdk._embed_config = MagicMock()
    return sdk


@pytest.fixture
def make_search_row():
    """Factory to create mock rows from search query results."""

    def _make(chunk_id, article_id, chunk_index, content, title, url):
        Row = type(
            "MockRow",
            (),
            {
                "chunk_id": chunk_id,
                "article_id": article_id,
                "chunk_index": chunk_index,
                "content": content,
                "title": title,
                "url": url,
            },
        )
        return Row()

    return _make


# ── search ──


@pytest.mark.asyncio
@patch("chatbot_plugin_sdk.base.BaseRagProcessor._embed_query")
async def test_search_fuses_dense_and_sparse(mock_embed, sdk, make_search_row):
    """Dense and sparse candidates are RRF-fused and top_k returned."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})

    # Dense returns chunk-a, chunk-b; Sparse returns chunk-b, chunk-c
    row_a = make_search_row("chunk-a", "art-1", 0, "text a", "Title A", "https://a.com")
    row_b = make_search_row("chunk-b", "art-1", 1, "text b", "Title A", "https://a.com")
    row_c = make_search_row("chunk-c", "art-2", 0, "text c", "Title C", "https://c.com")

    # Two sequential execute() calls: dense then sparse
    mock_session = sdk._db_config.session_factory.return_value = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    dense_result = MagicMock()
    dense_result.all.return_value = [row_a, row_b]
    sparse_result = MagicMock()
    sparse_result.all.return_value = [row_b, row_c]

    mock_session.execute.side_effect = [dense_result, sparse_result]

    resp = await sdk.search("hello", top_k=2)

    assert len(resp.chunks) == 2
    # chunk-b appears in both lists → highest fused score
    assert resp.chunks[0].chunk_id == "chunk-b"
    mock_embed.assert_called_once_with("hello")


@pytest.mark.asyncio
@patch("chatbot_plugin_sdk.base.BaseRagProcessor._embed_query")
async def test_search_only_dense_candidates(mock_embed, sdk, make_search_row):
    """When sparse has no results, dense-only scores still work."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})

    row_a = make_search_row("chunk-a", "art-1", 0, "text a", "Title A", "https://a.com")

    mock_session = sdk._db_config.session_factory.return_value = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    dense_result = MagicMock()
    dense_result.all.return_value = [row_a]
    sparse_result = MagicMock()
    sparse_result.all.return_value = []

    mock_session.execute.side_effect = [dense_result, sparse_result]

    resp = await sdk.search("hello")
    assert len(resp.chunks) == 1
    assert resp.chunks[0].chunk_id == "chunk-a"


@pytest.mark.asyncio
@patch("chatbot_plugin_sdk.base.BaseRagProcessor._embed_query")
async def test_search_respects_top_k(mock_embed, sdk, make_search_row):
    """top_k limits the number of results returned."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})

    rows = [
        make_search_row(f"chunk-{i}", "art-1", i, f"text {i}", "T", "https://x.com")
        for i in range(5)
    ]

    mock_session = sdk._db_config.session_factory.return_value = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    dense_result = MagicMock()
    dense_result.all.return_value = rows
    sparse_result = MagicMock()
    sparse_result.all.return_value = []

    mock_session.execute.side_effect = [dense_result, sparse_result]

    resp = await sdk.search("hello", top_k=3)
    assert len(resp.chunks) == 3


# ── chat ──


@pytest.mark.asyncio
@patch.object(BaseRagProcessor, "_call_llm")
@patch("chatbot_plugin_sdk.base.BaseRagProcessor._embed_query")
async def test_chat_returns_reply(mock_embed, mock_llm, sdk, make_search_row):
    """Chat searches for context, calls LLM, and returns reply + citations."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})
    mock_llm.return_value = "Yes, RAG is..."

    row_a = make_search_row("chunk-a", "art-1", 0, "RAG is retrieval...", "RAG Article", "https://rag.com")

    mock_session = sdk._db_config.session_factory.return_value = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    dense_result = MagicMock()
    dense_result.all.return_value = [row_a]
    sparse_result = MagicMock()
    sparse_result.all.return_value = []

    mock_session.execute.side_effect = [dense_result, sparse_result]

    resp = await sdk.chat("What is RAG?")

    assert resp.reply == "Yes, RAG is..."
    assert len(resp.articles_used) == 1
    assert resp.articles_used[0].id == "art-1"
    assert resp.articles_used[0].title == "RAG Article"
    assert len(resp.chunks) == 1
    mock_llm.assert_called_once()


@pytest.mark.asyncio
@patch.object(settings, "llm_api_key", "")
@patch.object(settings, "gemini_api_key", "")
@patch("chatbot_plugin_sdk.base.BaseRagProcessor._embed_query")
async def test_chat_no_llm_key_returns_raw_context(mock_embed, sdk, make_search_row):
    """When no LLM key is configured, chat returns raw context + question."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})

    row_a = make_search_row("chunk-a", "art-1", 0, "RAG is retrieval...", "RAG Article", "https://rag.com")

    mock_session = sdk._db_config.session_factory.return_value = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    dense_result = MagicMock()
    dense_result.all.return_value = [row_a]
    sparse_result = MagicMock()
    sparse_result.all.return_value = []

    mock_session.execute.side_effect = [dense_result, sparse_result]

    resp = await sdk.chat("What is RAG?")

    assert "[No LLM configured" in resp.reply
    assert "Question: What is RAG?" in resp.reply
    assert len(resp.articles_used) == 1
    assert len(resp.chunks) == 1


@pytest.mark.asyncio
@patch("chatbot_plugin_sdk.base.BaseRagProcessor._embed_query")
async def test_chat_no_results(mock_embed, sdk):
    """When search returns no chunks, chat responds with a fallback message."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})

    mock_session = sdk._db_config.session_factory.return_value = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    dense_result = MagicMock()
    dense_result.all.return_value = []
    sparse_result = MagicMock()
    sparse_result.all.return_value = []

    mock_session.execute.side_effect = [dense_result, sparse_result]

    resp = await sdk.chat("What is RAG?")

    assert "couldn't find any relevant context" in resp.reply
    assert resp.articles_used == []
    assert resp.chunks == []
