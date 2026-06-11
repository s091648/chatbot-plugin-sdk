"""Response contracts — mirrors specs/toolbox-api.md response bodies."""

from pydantic import BaseModel, Field


class StoreChunksResponse(BaseModel):
    """POST /tools/chunks response.

    Spec: specs/toolbox-api.md — POST /tools/chunks — 201
    """

    stored: int = Field(..., ge=0, description="Number of chunks stored")
    article_id: str = Field(..., description="The article UUID (echoed back)")


class ChunkResult(BaseModel):
    """A single matching chunk in search/chat results."""

    chunk_id: str = Field(..., description="Chunk UUID")
    article_id: str = Field(..., description="Parent article UUID")
    article_title: str | None = Field(default=None, description="Article title")
    article_url: str | None = Field(default=None, description="Source URL")
    chunk_index: int = Field(..., description="Position within article")
    content: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="RRF fusion score (higher is better)")


class SearchResponse(BaseModel):
    """POST /tools/search response.

    Spec: specs/toolbox-api.md — POST /tools/search — 200
    """

    chunks: list[ChunkResult] = Field(default_factory=list, description="Matching chunks ordered by score")


class ArticleCitation(BaseModel):
    """Article citation used in chat response."""

    id: str = Field(..., description="Article UUID")
    title: str | None = Field(default=None, description="Article title")
    url: str | None = Field(default=None, description="Source URL")


class ChatResponse(BaseModel):
    """POST /tools/chat response.

    Spec: specs/toolbox-api.md — POST /tools/chat — 200
    """

    reply: str = Field(..., description="LLM-generated answer")
    articles_used: list[ArticleCitation] = Field(default_factory=list, description="Unique articles contributing to context")
    chunks: list[ChunkResult] = Field(default_factory=list, description="All chunks used in the context")
