from sqlalchemy import Column, ForeignKey, Integer, Text, DateTime, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import SPARSEVEC, Vector

from chatbot_plugin_sdk.models.article import Base


class ArticleChunk(Base):
    __tablename__ = "article_chunks"
    __table_args__ = (
        UniqueConstraint("article_id", "chunk_index", name="uq_article_chunk_idx"),
        {"schema": "vectors"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    article_id = Column(
        UUID(as_uuid=True),
        ForeignKey("vectors.articles.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    # ORM placeholder — actual dimension determined by DDL at setup() time.
    # Vector(768): BERT-class default; SparseVector(30522): BERT vocab size (SPLADE).
    dense_vector = Column(Vector(768), nullable=True)
    sparse_vector = Column(SPARSEVEC(30522), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    article = relationship("Article", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<ArticleChunk(id={self.id}, article_id={self.article_id}, index={self.chunk_index})>"
