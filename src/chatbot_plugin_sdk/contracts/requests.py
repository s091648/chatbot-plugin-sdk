"""Request contracts — mirrors specs/toolbox-api.md request bodies."""

from typing import Any

from pydantic import BaseModel, Field


class ArticleInfo(BaseModel):
    """Article metadata included in a store-chunks request."""

    id: str = Field(..., description="Article UUID (upsert key)")
    url: str | None = Field(default=None, description="Source URL — an article_columns value, not a first-class SDK concept")
    title: str | None = Field(default=None, description="Article title")
    source: str | None = Field(default=None, description="Source domain / feed name")
    metadata: dict | None = Field(default=None, description="Arbitrary JSON metadata")


class ChunkData(BaseModel):
    """Single pre-chunked, pre-embedded data fragment."""

    chunk_index: int = Field(..., ge=0, description="Position within article (0-based)")
    content: str = Field(..., description="Chunk text content")
    dense_vector: list[float] = Field(..., description="Dense embedding vector")
    sparse_vector: dict[str, float] | None = Field(
        default=None, description="Lexical weights as {token_index: weight}"
    )


class StoreChunksRequest(BaseModel):
    """POST /tools/chunks request.

    Spec: specs/toolbox-api.md — POST /tools/chunks
    """

    article: ArticleInfo = Field(..., description="Article metadata")
    chunks: list[ChunkData] = Field(..., min_length=1, description="Pre-chunked pre-embedded data fragments")


class SearchRequest(BaseModel):
    """POST /tools/search request.

    Spec: specs/toolbox-api.md — POST /tools/search
    """

    query: str = Field(..., min_length=1, description="Raw query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top chunks to return")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Column-level filters on the articles table, e.g. {\"topic_id\": \"uuid\"}",
    )


class ChatRequest(BaseModel):
    """POST /tools/chat request.

    Spec: specs/toolbox-api.md — POST /tools/chat
    """

    message: str = Field(..., min_length=1, description="User message")
