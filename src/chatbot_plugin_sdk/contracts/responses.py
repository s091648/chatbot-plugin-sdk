from pydantic import BaseModel, Field


class StoreChunksResponse(BaseModel):
    stored: int = Field(..., ge=0)
    article_id: str


class ChunkResult(BaseModel):
    chunk_id: str
    article_id: str
    article_title: str | None = None
    article_url: str | None = None
    public_article_id: str | None = None
    chunk_index: int
    content: str
    score: float = Field(..., description="Cosine similarity score (0-1, higher is better)")


class SearchResponse(BaseModel):
    chunks: list[ChunkResult] = Field(default_factory=list)
