from typing import Any

from pydantic import BaseModel, Field


class StoreChunksResponse(BaseModel):
    stored: int = Field(..., ge=0)
    article_id: str


class ChunkResult(BaseModel):
    chunk_id: str
    article_id: str
    article_metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int
    content: str
    score: float = Field(..., description="Cosine similarity score (0-1, higher is better)")


class SearchResponse(BaseModel):
    chunks: list[ChunkResult] = Field(default_factory=list)
