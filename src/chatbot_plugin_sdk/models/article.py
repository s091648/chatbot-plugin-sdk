from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (
        Index("idx_articles_source", "source"),
        Index("idx_articles_url", "url"),
        {"schema": "vectors"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True)
    url = Column(String, nullable=False, unique=True)
    title = Column(String, nullable=True)
    source = Column(String, nullable=True)
    public_article_id = Column(UUID(as_uuid=True), nullable=True)
    topic_id = Column(UUID(as_uuid=True), nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        server_onupdate=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )
    chunks = relationship("ArticleChunk", back_populates="article", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title={self.title!r})>"
