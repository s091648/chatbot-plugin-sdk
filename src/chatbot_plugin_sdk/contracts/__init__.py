"""Pydantic contracts — structured versions of specs/toolbox-api.md.

These models are the source of truth for request/response validation.
They must stay in sync with specs/toolbox-api.md. If the spec changes,
update the corresponding model here first.
"""

from chatbot_plugin_sdk.contracts.requests import ArticleInfo, ChunkData, StoreChunksRequest, SearchRequest, ChatRequest
from chatbot_plugin_sdk.contracts.responses import (
    StoreChunksResponse,
    SearchResponse,
    ChatResponse,
    ChunkResult,
    ArticleCitation,
)

__all__ = [
    "ArticleInfo",
    "ArticleCitation",
    "ChunkData",
    "ChunkResult",
    "StoreChunksRequest",
    "StoreChunksResponse",
    "SearchRequest",
    "SearchResponse",
    "ChatRequest",
    "ChatResponse",
]
