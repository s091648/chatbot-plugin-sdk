from sqlalchemy import Column, ForeignKey, Integer, Text, DateTime, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

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
    # nullable: 只配置 dense 時 sparse 為 NULL，反之亦然
    # Vector(768) 是 ORM placeholder；實際維度由 IngestProcessor._create_tables() 動態決定
    dense_vector = Column(Vector(768), nullable=True)
    sparse_vector = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    article = relationship("Article", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<ArticleChunk(id={self.id}, article_id={self.article_id}, index={self.chunk_index})>"
