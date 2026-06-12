# chatbot-plugin-sdk Refactoring Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 將現有 SDK 重構為職責清晰、無環境變數依賴、符合 Strategy Pattern 的 RAG ingest/retrieve 工具庫。

**Architecture:** 移除 God Class `BaseRagProcessor`，以 `IngestProcessor` 和 `RetrieveProcessor` 取代，各自透過依賴注入接收 `EndpointProvider` 或 `LocalProvider`。SDK 邊界到「回傳 chunks」為止，LLM 生成完全移出。Provider 透過 `DenseEmbeddingProvider` / `SparseEmbeddingProvider` Protocol 定義介面，方便使用者自帶實作。

**Tech Stack:** Python 3.11, SQLAlchemy 2.x asyncio, pgvector, httpx, asyncio, typing.Protocol

---

## File Map

### 新增
```
src/chatbot_plugin_sdk/
├── protocols.py                    ← DenseEmbeddingProvider, SparseEmbeddingProvider Protocols
├── processors/
│   ├── __init__.py
│   ├── ingest.py                   ← IngestProcessor
│   └── retrieve.py                 ← RetrieveProcessor
└── providers/
    ├── __init__.py
    ├── endpoint.py                 ← EndpointProvider (HTTP, covers external API + sidecar)
    └── local.py                    ← LocalProvider (in-process callable wrapper)
```

### 修改
```
src/chatbot_plugin_sdk/
├── __init__.py                     ← 更新 public exports
├── config.py                       ← 只保留 DatabaseConfig dataclass，移除 ChatbotSettings
├── models/chunk.py                 ← 加 sparse_vector JSONB 欄位、schema="vectors"
├── models/article.py               ← 加 schema="vectors"
├── exceptions.py                   ← 移除 LLMError
└── contracts/responses.py          ← 移除 ChatResponse，保留 SearchResponse, ChunkResult
```

### 刪除
```
src/chatbot_plugin_sdk/
├── base.py                         ← God Class，全部移除
├── ingest.py                       ← 邏輯移至 processors/ingest.py
└── query.py                        ← 邏輯移至 processors/retrieve.py
```

---

## Task 1：更新 config.py — 移除環境變數，只留 DatabaseConfig

**Files:**
- Modify: `src/chatbot_plugin_sdk/config.py`

- [x] **Step 1: 將 config.py 改寫為純 dataclass，無 pydantic-settings 依賴**

```python
# src/chatbot_plugin_sdk/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, AsyncSession


@dataclass
class DatabaseConfig:
    dbname: str
    user: str
    password: str
    host: str = "localhost"
    port: int = 5432
    schema: str = "vectors"


@dataclass
class _RuntimeDatabase:
    """Internal: holds live SQLAlchemy engine + session factory after configure()."""
    engine: "AsyncEngine"
    session_factory: "async_sessionmaker[AsyncSession]"
    schema: str
```

- [x] **Step 2: 從 pyproject.toml 移除 pydantic-settings 依賴（若僅 config.py 使用）**

確認 `pydantic-settings` 是否還有其他地方使用。若無，從 `[project] dependencies` 移除：
```toml
# 移除這行（若已確認無其他使用者）
# "pydantic-settings>=2.0",
```

- [x] **Step 3: 驗證**

```bash
cd /path/to/chatbot-plugin-sdk
python -c "from chatbot_plugin_sdk.config import DatabaseConfig; print(DatabaseConfig('mydb','user','pw'))"
```
Expected: `DatabaseConfig(dbname='mydb', user='user', password='pw', host='localhost', port=5432, schema='vectors')`

---

## Task 2：定義 Protocol 介面

**Files:**
- Create: `src/chatbot_plugin_sdk/protocols.py`

- [x] **Step 1: 建立 protocols.py**

```python
# src/chatbot_plugin_sdk/protocols.py
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class DenseEmbeddingProvider(Protocol):
    """HTTP endpoint 或 in-process callable，輸出 dense 向量。
    
    dimension 屬性供 ensure_ready() 在首次建表時決定 VECTOR(N) 的 N。
    """
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """將文字批次轉換為 dense 向量列表。"""
        ...


@runtime_checkable
class SparseEmbeddingProvider(Protocol):
    """HTTP endpoint 或 in-process callable，輸出 sparse 向量（token_id → weight）。"""

    async def embed(self, texts: list[str]) -> list[dict[str, float]]:
        """將文字批次轉換為 sparse 向量列表（每個向量是 {token_id_str: weight} dict）。"""
        ...
```

- [x] **Step 2: 驗證 Protocol 結構正確**

```bash
python -c "
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider

class MyDense:
    dimension = 768
    async def embed(self, texts): return [[0.1] * 768]

print(isinstance(MyDense(), DenseEmbeddingProvider))  # True
"
```
Expected: `True`

---

## Task 3：更新 ORM models — 加 vectors schema 與 sparse_vector 欄位

**Files:**
- Modify: `src/chatbot_plugin_sdk/models/article.py`
- Modify: `src/chatbot_plugin_sdk/models/chunk.py`

- [x] **Step 1: 更新 article.py，加 schema="vectors"**

```python
# src/chatbot_plugin_sdk/models/article.py
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
```

- [x] **Step 2: 更新 chunk.py — 加 sparse_vector、移除對 settings 的依賴、加 schema="vectors"**

```python
# src/chatbot_plugin_sdk/models/chunk.py
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
    dense_vector = Column(Vector(768), nullable=True)   # dimension 在 ensure_ready() 中動態建立
    sparse_vector = Column(JSONB, nullable=True)        # {token_id_str: weight}
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    article = relationship("Article", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<ArticleChunk(id={self.id}, article_id={self.article_id}, index={self.chunk_index})>"
```

> **注意：** `dense_vector = Column(Vector(768), ...)` 中的 768 是 DDL 建立時用的 placeholder。`ensure_ready()` 實際上會用 `CREATE TABLE ... dense_vector VECTOR(:dim)` 的方式動態建立，不依賴這個 model 定義裡的數字。See Task 6。

---

## Task 4：實作 EndpointProvider

**Files:**
- Create: `src/chatbot_plugin_sdk/providers/__init__.py`
- Create: `src/chatbot_plugin_sdk/providers/endpoint.py`

- [x] **Step 1: 建立 providers/__init__.py（空）**

```python
# src/chatbot_plugin_sdk/providers/__init__.py
```

- [x] **Step 2: 建立 endpoint.py**

```python
# src/chatbot_plugin_sdk/providers/endpoint.py
from __future__ import annotations
import httpx
from chatbot_plugin_sdk.exceptions import EmbeddingError


class EndpointProvider:
    """HTTP embedding provider。適用於外部 API（如 Google AI Studio）
    與內部 sidecar microservice（如自架 fastembed HTTP service），
    兩者呼叫方式相同，差別只在 URL。

    Args:
        url: embedding service 的 base URL（如 "http://localhost:8080"）。
        response_key: API response 中存放向量的 key（dense 用 "dense"，sparse 用 "sparse"）。
        api_key: 若 service 需要 Bearer token，在此傳入。
        dimension: dense 向量的維度。使用 response_key="dense" 時必填；
                   response_key="sparse" 時可省略（sparse 無固定維度）。
        timeout: HTTP 請求 timeout（秒），預設 60。
    
    Usage:
        # Dense（搭配 Google AI Studio 或 HuggingFace TEI）
        dense = EndpointProvider(url="http://embed:8080", response_key="dense",
                                 api_key="...", dimension=768)
        
        # Sparse（搭配自架 SPLADE sidecar）
        sparse = EndpointProvider(url="http://embed:8080", response_key="sparse")
    """

    def __init__(
        self,
        url: str,
        response_key: str = "dense",
        api_key: str | None = None,
        dimension: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        if response_key == "dense" and dimension is None:
            raise ValueError("dimension is required when response_key='dense'")
        self._url = url.rstrip("/")
        self._response_key = response_key
        self._api_key = api_key
        self._timeout = timeout
        # dimension 屬性：DenseEmbeddingProvider Protocol 要求
        self.dimension: int = dimension or 0  # sparse 時為 0（未使用）

    def _build_client(self) -> httpx.AsyncClient:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return httpx.AsyncClient(base_url=self._url, headers=headers, timeout=self._timeout)

    async def embed(self, texts: list[str]) -> list:
        """送出 POST /embed 請求，回傳對應 response_key 的向量列表。

        Request body: {"texts": ["text1", "text2", ...]}
        Expected response: {"dense": [[...], ...], "sparse": [{...}, ...]}
        """
        async with self._build_client() as client:
            try:
                resp = await client.post("/embed", json={"texts": texts})
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(f"Embedding endpoint returned {exc.response.status_code}: {exc.response.text}") from exc
            except Exception as exc:
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc

        result = data.get(self._response_key)
        if not result:
            raise EmbeddingError(
                f"Embedding response missing key '{self._response_key}'. "
                f"Available keys: {list(data.keys())}"
            )
        return result
```

- [x] **Step 3: 驗證型別相容性**

```python
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider
from chatbot_plugin_sdk.providers.endpoint import EndpointProvider

dense_p = EndpointProvider(url="http://localhost:8080", response_key="dense", dimension=768)
sparse_p = EndpointProvider(url="http://localhost:8080", response_key="sparse")

print(isinstance(dense_p, DenseEmbeddingProvider))   # True（有 dimension + embed）
print(isinstance(sparse_p, SparseEmbeddingProvider)) # True（有 embed）
```
Expected: `True True`

---

## Task 5：實作 LocalProvider

**Files:**
- Create: `src/chatbot_plugin_sdk/providers/local.py`

- [x] **Step 1: 建立 local.py**

```python
# src/chatbot_plugin_sdk/providers/local.py
from __future__ import annotations
import asyncio
from collections.abc import Callable
from chatbot_plugin_sdk.exceptions import EmbeddingError


class LocalProvider:
    """In-process embedding provider，接受 sync 或 async callable。

    適用於在同一個 Python process 內直接呼叫 embedding function，
    例如自行初始化的 fastembed 模型。
    Sync callable 會被包進 asyncio executor 以避免 block event loop。

    Args:
        fn: 接受 list[str] 並回傳向量列表的 callable。
            Dense: fn(texts) -> list[list[float]]
            Sparse: fn(texts) -> list[dict[str, float]]
        dimension: dense 向量維度。用於 dense 場景時必填；sparse 可省略。
    
    Usage:
        from fastembed import TextEmbedding
        model = TextEmbedding("BAAI/bge-small-en")
        
        dense = LocalProvider(
            fn=lambda texts: [v.tolist() for v in model.embed(texts)],
            dimension=384,
        )
    """

    def __init__(
        self,
        fn: Callable[[list[str]], list],
        dimension: int | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn)}")
        self._fn = fn
        self.dimension: int = dimension or 0  # sparse 時為 0（未使用）

    async def embed(self, texts: list[str]) -> list:
        """呼叫 fn(texts)，自動處理 sync/async 差異。"""
        try:
            if asyncio.iscoroutinefunction(self._fn):
                return await self._fn(texts)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._fn, texts)
        except Exception as exc:
            raise EmbeddingError(f"Local embedding function failed: {exc}") from exc
```

---

## Task 6：實作 IngestProcessor

**Files:**
- Create: `src/chatbot_plugin_sdk/processors/__init__.py`
- Create: `src/chatbot_plugin_sdk/processors/ingest.py`

- [x] **Step 1: 建立 processors/__init__.py（空）**

```python
# src/chatbot_plugin_sdk/processors/__init__.py
```

- [x] **Step 2: 建立 ingest.py**

```python
# src/chatbot_plugin_sdk/processors/ingest.py
from __future__ import annotations
import re
import unicodedata
import uuid
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from chatbot_plugin_sdk.chunking import _chunk_text
from chatbot_plugin_sdk.config import DatabaseConfig, _RuntimeDatabase
from chatbot_plugin_sdk.exceptions import DatabaseError, NotConfiguredError
from chatbot_plugin_sdk.models.article import Article
from chatbot_plugin_sdk.models.chunk import ArticleChunk
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider


class IngestProcessor:
    """文章向量化寫入處理器。

    Pipeline: normalize → chunk → embed（dense/sparse） → upsert to DB
    
    Usage:
        processor = IngestProcessor()
        processor.configure(
            db=DatabaseConfig(dbname="mydb", user="u", password="p"),
            dense=EndpointProvider(url="http://embed:8080", dimension=768),
        )
        await processor.ingest(full_text="...", metadata={"url": "https://...", "title": "..."})
    """

    def __init__(self) -> None:
        self._db_config: DatabaseConfig | None = None
        self._runtime: _RuntimeDatabase | None = None
        self._dense: DenseEmbeddingProvider | None = None
        self._sparse: SparseEmbeddingProvider | None = None
        self._ready: bool = False

    # ── Configuration ──

    def configure(
        self,
        db: DatabaseConfig,
        dense: DenseEmbeddingProvider | None = None,
        sparse: SparseEmbeddingProvider | None = None,
    ) -> None:
        """設定 DB 連線與 embedding providers。純同步，不做任何 IO。"""
        if dense is None and sparse is None:
            raise NotConfiguredError("至少需要配置 dense 或 sparse 其中一種 embedding provider。")
        self._db_config = db
        self._dense = dense
        self._sparse = sparse
        self._ready = False  # reset，讓下次 ingest 重新 ensure_ready

    # ── Table management ──

    def _build_runtime(self) -> _RuntimeDatabase:
        assert self._db_config is not None
        db = self._db_config
        url = f"postgresql+asyncpg://{db.user}:{db.password}@{db.host}:{db.port}/{db.dbname}"
        engine = create_async_engine(url, echo=False, future=True)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        return _RuntimeDatabase(engine=engine, session_factory=factory, schema=db.schema)

    async def ensure_ready(self) -> None:
        """冪等：首次呼叫時建立 schema/table（若不存在）或驗證已存在的 schema 相容性。
        
        - 表不存在 → 建立 vectors schema + articles + article_chunks（使用 provider.dimension）
        - 表存在   → 讀取 DB 中 dense_vector 的實際維度，與 provider.dimension 比對
        """
        if self._ready:
            return
        if self._db_config is None:
            raise NotConfiguredError("尚未呼叫 configure()。")

        if self._runtime is None:
            self._runtime = self._build_runtime()

        schema = self._runtime.schema
        async with self._runtime.engine.begin() as conn:
            # 確保 pgvector extension 存在
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # 確保 schema 存在
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

            # 確認 article_chunks 是否已存在
            result = await conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = :schema AND table_name = 'article_chunks'"
            ), {"schema": schema})
            table_exists = result.fetchone() is not None

        if not table_exists:
            await self._create_tables()
        else:
            await self._validate_schema_compatibility()

        self._ready = True

    async def _create_tables(self) -> None:
        """首次建立 articles + article_chunks table。"""
        assert self._runtime is not None
        schema = self._runtime.schema
        dense_dim = self._dense.dimension if self._dense else None

        async with self._runtime.engine.begin() as conn:
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {schema}.articles (
                    id          UUID PRIMARY KEY,
                    url         TEXT NOT NULL UNIQUE,
                    title       TEXT,
                    source      TEXT,
                    metadata    JSONB,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))
            dense_col = f"dense_vector VECTOR({dense_dim})" if dense_dim else "dense_vector VECTOR(768)"
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {schema}.article_chunks (
                    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    article_id   UUID NOT NULL REFERENCES {schema}.articles(id) ON DELETE CASCADE,
                    chunk_index  INTEGER NOT NULL,
                    content      TEXT NOT NULL,
                    {dense_col},
                    sparse_vector JSONB,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT uq_article_chunk_idx UNIQUE (article_id, chunk_index)
                )
            """))
            await conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_articles_url ON {schema}.articles(url)"
            ))
            await conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_articles_source ON {schema}.articles(source)"
            ))

    async def _validate_schema_compatibility(self) -> None:
        """驗證已存在的 schema 與目前 provider 的 dimension 相符。"""
        assert self._runtime is not None
        if self._dense is None:
            return  # 沒有 dense provider 就不需要驗證維度

        schema = self._runtime.schema
        async with self._runtime.engine.connect() as conn:
            def _get_dim(sync_conn):
                from sqlalchemy import inspect as sa_inspect
                inspector = sa_inspect(sync_conn)
                cols = inspector.get_columns("article_chunks", schema=schema)
                for col in cols:
                    if col["name"] == "dense_vector":
                        return getattr(col["type"], "dim", None)
                return None
            db_dim = await conn.run_sync(_get_dim)

        if db_dim is not None and db_dim != self._dense.dimension:
            raise DatabaseError(
                f"Provider dimension mismatch: DB has VECTOR({db_dim}) "
                f"but provider.dimension={self._dense.dimension}. "
                "Use the same embedding model as when the table was created."
            )

    # ── Normalisation ──

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.lstrip("﻿").strip()
        return re.sub(r"\s+", " ", text)

    # ── Ingest pipeline ──

    async def ingest(
        self,
        full_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """完整 ingest pipeline：normalize → chunk → embed → upsert。

        Args:
            full_text: 文章全文（PDF 解析後、HTML stripped 或純文字）。
            metadata: 至少含 "url"（str）；建議也提供 "title"、"source"。
                      article_id 由 url 透過 uuid5 推導，確保冪等性。
        """
        await self.ensure_ready()
        assert self._runtime is not None

        metadata = metadata or {}
        url = metadata.get("url", "")
        if not url:
            raise DatabaseError("metadata must contain 'url' to ensure idempotent ingest.")

        # 1. Normalize
        normalized = self._normalize(full_text)
        if not normalized:
            raise DatabaseError("Empty text after normalization.")

        # 2. Chunk
        chunks = _chunk_text(normalized)
        if not chunks:
            raise DatabaseError("No chunks produced — input text may be too short.")

        # 3. Embed
        dense_vectors: list[list[float]] | None = None
        sparse_vectors: list[dict[str, float]] | None = None

        if self._dense is not None:
            dense_vectors = await self._dense.embed(chunks)
            if len(dense_vectors) != len(chunks):
                raise DatabaseError(
                    f"Dense embedding returned {len(dense_vectors)} vectors "
                    f"but {len(chunks)} chunks expected."
                )

        if self._sparse is not None:
            sparse_vectors = await self._sparse.embed(chunks)
            if len(sparse_vectors) != len(chunks):
                raise DatabaseError(
                    f"Sparse embedding returned {len(sparse_vectors)} vectors "
                    f"but {len(chunks)} chunks expected."
                )

        # 4. Upsert
        article_id = uuid.uuid5(uuid.NAMESPACE_URL, url)
        await self._upsert(
            article_id=article_id,
            metadata=metadata,
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
        )

    async def _upsert(
        self,
        article_id: uuid.UUID,
        metadata: dict[str, Any],
        chunks: list[str],
        dense_vectors: list[list[float]] | None,
        sparse_vectors: list[dict[str, float]] | None,
    ) -> None:
        assert self._runtime is not None
        session = self._runtime.session_factory()
        try:
            async with session.begin():
                existing = (await session.execute(
                    select(Article).where(Article.id == article_id)
                )).scalar_one_or_none()

                if existing is not None:
                    existing.url = metadata.get("url", "")
                    existing.title = metadata.get("title")
                    existing.source = metadata.get("source")
                    existing.metadata_ = metadata.get("metadata")
                    await session.execute(
                        delete(ArticleChunk).where(ArticleChunk.article_id == article_id)
                    )
                else:
                    session.add(Article(
                        id=article_id,
                        url=metadata.get("url", ""),
                        title=metadata.get("title"),
                        source=metadata.get("source"),
                        metadata_=metadata.get("metadata"),
                    ))

                for i, content in enumerate(chunks):
                    session.add(ArticleChunk(
                        article_id=article_id,
                        chunk_index=i,
                        content=content,
                        dense_vector=dense_vectors[i] if dense_vectors else None,
                        sparse_vector=sparse_vectors[i] if sparse_vectors else None,
                    ))
        except Exception as exc:
            raise DatabaseError(f"Failed to upsert article {article_id}: {exc}") from exc
        finally:
            await session.close()
```

---

## Task 7：實作 RetrieveProcessor

**Files:**
- Create: `src/chatbot_plugin_sdk/processors/retrieve.py`

- [x] **Step 1: 建立 retrieve.py**

```python
# src/chatbot_plugin_sdk/processors/retrieve.py
from __future__ import annotations
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from chatbot_plugin_sdk.config import DatabaseConfig, _RuntimeDatabase
from chatbot_plugin_sdk.contracts.responses import ChunkResult, SearchResponse
from chatbot_plugin_sdk.exceptions import DatabaseError, NotConfiguredError
from chatbot_plugin_sdk.models.article import Article
from chatbot_plugin_sdk.models.chunk import ArticleChunk
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider


class RetrieveProcessor:
    """向量語意搜尋處理器（read-only）。

    Pipeline: embed query → pgvector similarity search → return SearchResponse
    SDK 在回傳 SearchResponse（chunks）後即完成職責；LLM 生成由 caller 負責。

    Usage:
        retriever = RetrieveProcessor()
        retriever.configure(
            db=DatabaseConfig(dbname="mydb", user="u", password="p"),
            dense=EndpointProvider(url="http://embed:8080", dimension=768),
        )
        result = await retriever.search("What is RAG?")
        # result.chunks 是 list[ChunkResult]，交給 LLM 生成回答
    """

    def __init__(self) -> None:
        self._db_config: DatabaseConfig | None = None
        self._runtime: _RuntimeDatabase | None = None
        self._dense: DenseEmbeddingProvider | None = None
        self._sparse: SparseEmbeddingProvider | None = None
        self._ready: bool = False

    # ── Configuration ──

    def configure(
        self,
        db: DatabaseConfig,
        dense: DenseEmbeddingProvider | None = None,
        sparse: SparseEmbeddingProvider | None = None,
    ) -> None:
        """設定 DB 連線與 embedding providers。"""
        if dense is None and sparse is None:
            raise NotConfiguredError("至少需要配置 dense 或 sparse 其中一種 embedding provider。")
        self._db_config = db
        self._dense = dense
        self._sparse = sparse
        self._ready = False

    # ── Readiness ──

    def _build_runtime(self) -> _RuntimeDatabase:
        assert self._db_config is not None
        db = self._db_config
        url = f"postgresql+asyncpg://{db.user}:{db.password}@{db.host}:{db.port}/{db.dbname}"
        engine = create_async_engine(url, echo=False, future=True)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        return _RuntimeDatabase(engine=engine, session_factory=factory, schema=db.schema)

    async def ensure_ready(self) -> None:
        """驗證 DB 中的 schema 與 provider 相容。RetrieveProcessor 不建表，只驗證。"""
        if self._ready:
            return
        if self._db_config is None:
            raise NotConfiguredError("尚未呼叫 configure()。")
        if self._runtime is None:
            self._runtime = self._build_runtime()

        schema = self._runtime.schema
        async with self._runtime.engine.connect() as conn:
            from sqlalchemy import text
            result = await conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = :schema AND table_name = 'article_chunks'"
            ), {"schema": schema})
            if result.fetchone() is None:
                raise DatabaseError(
                    f"Table {schema}.article_chunks does not exist. "
                    "Run IngestProcessor.ensure_ready() first to create the schema."
                )

        # 驗證 dense dimension 相符
        if self._dense is not None:
            async with self._runtime.engine.connect() as conn:
                def _get_dim(sync_conn):
                    from sqlalchemy import inspect as sa_inspect
                    inspector = sa_inspect(sync_conn)
                    cols = inspector.get_columns("article_chunks", schema=schema)
                    for col in cols:
                        if col["name"] == "dense_vector":
                            return getattr(col["type"], "dim", None)
                    return None
                db_dim = await conn.run_sync(_get_dim)

            if db_dim is not None and db_dim != self._dense.dimension:
                raise DatabaseError(
                    f"Provider dimension mismatch: DB has VECTOR({db_dim}) "
                    f"but provider.dimension={self._dense.dimension}."
                )

        self._ready = True

    # ── Search ──

    async def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """語意搜尋。

        優先使用 dense（cosine similarity）。若兩種 provider 都存在則做 hybrid（RRF 融合）。
        目前版本：只有 dense 時走 dense；只有 sparse 時 raise NotImplementedError（待補）。

        Returns:
            SearchResponse 包含按相關度排序的 ChunkResult 列表。
        """
        await self.ensure_ready()
        assert self._runtime is not None

        if self._dense is not None:
            return await self._dense_search(query, top_k)
        raise NotConfiguredError("Dense provider is required for search in this version.")

    async def _dense_search(self, query: str, top_k: int) -> SearchResponse:
        assert self._runtime is not None and self._dense is not None

        dense_vecs = await self._dense.embed([query])
        query_vec = dense_vecs[0]

        async with self._runtime.session_factory() as db:
            stmt = (
                select(
                    ArticleChunk.id.label("chunk_id"),
                    ArticleChunk.article_id,
                    ArticleChunk.chunk_index,
                    ArticleChunk.content,
                    Article.title,
                    Article.url,
                    ArticleChunk.dense_vector.cosine_distance(query_vec).label("distance"),
                )
                .join(Article, ArticleChunk.article_id == Article.id)
                .where(ArticleChunk.dense_vector.isnot(None))
                .order_by("distance")
                .limit(top_k)
            )
            rows = (await db.execute(stmt)).all()

        chunks = [
            ChunkResult(
                chunk_id=str(row.chunk_id),
                article_id=str(row.article_id),
                article_title=row.title,
                article_url=row.url,
                chunk_index=row.chunk_index,
                content=row.content,
                score=1.0 - float(row.distance),  # cosine distance → similarity
            )
            for row in rows
        ]
        return SearchResponse(chunks=chunks)
```

---

## Task 8：清理移除的 code

**Files:**
- Delete: `src/chatbot_plugin_sdk/base.py`
- Delete: `src/chatbot_plugin_sdk/ingest.py`
- Delete: `src/chatbot_plugin_sdk/query.py`
- Modify: `src/chatbot_plugin_sdk/exceptions.py`
- Modify: `src/chatbot_plugin_sdk/contracts/responses.py`

- [x] **Step 1: 移除 base.py、ingest.py、query.py**

```bash
rm src/chatbot_plugin_sdk/base.py
rm src/chatbot_plugin_sdk/ingest.py
rm src/chatbot_plugin_sdk/query.py
```

- [x] **Step 2: 更新 exceptions.py — 移除 LLMError**

```python
# src/chatbot_plugin_sdk/exceptions.py
from __future__ import annotations


class ToolboxError(Exception):
    """Base exception for all SDK errors."""


class NotConfiguredError(ToolboxError):
    """Raised when a required SDK setting is missing."""


class DatabaseError(ToolboxError):
    """Raised when a database operation fails."""


class EmbeddingError(ToolboxError):
    """Raised when embedding model call fails."""


class ChunkingError(ToolboxError):
    """Raised when text chunking fails."""
```

- [x] **Step 3: 更新 contracts/responses.py — 移除 ChatResponse**

```python
# src/chatbot_plugin_sdk/contracts/responses.py
from pydantic import BaseModel, Field


class StoreChunksResponse(BaseModel):
    stored: int = Field(..., ge=0)
    article_id: str


class ChunkResult(BaseModel):
    chunk_id: str
    article_id: str
    article_title: str | None = None
    article_url: str | None = None
    chunk_index: int
    content: str
    score: float = Field(..., description="Cosine similarity score (0-1, higher is better)")


class SearchResponse(BaseModel):
    chunks: list[ChunkResult] = Field(default_factory=list)
```

---

## Task 9：更新 `__init__.py` — 新的 public exports

**Files:**
- Modify: `src/chatbot_plugin_sdk/__init__.py`

- [x] **Step 1: 更新 __init__.py**

```python
# src/chatbot_plugin_sdk/__init__.py
"""chatbot_plugin_sdk — RAG ingest & retrieval SDK.

Entry points:
- IngestProcessor  — normalize → chunk → embed → store
- RetrieveProcessor — embed query → vector search → SearchResponse

Providers (inject into configure()):
- EndpointProvider — HTTP embedding (external API or sidecar)
- LocalProvider    — in-process callable (fastembed, etc.)

Protocols (for custom provider implementations):
- DenseEmbeddingProvider
- SparseEmbeddingProvider
"""

from chatbot_plugin_sdk.processors.ingest import IngestProcessor
from chatbot_plugin_sdk.processors.retrieve import RetrieveProcessor
from chatbot_plugin_sdk.providers.endpoint import EndpointProvider
from chatbot_plugin_sdk.providers.local import LocalProvider
from chatbot_plugin_sdk.config import DatabaseConfig
from chatbot_plugin_sdk.contracts.responses import SearchResponse, ChunkResult
from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider, SparseEmbeddingProvider
from chatbot_plugin_sdk.exceptions import (
    ToolboxError,
    NotConfiguredError,
    DatabaseError,
    EmbeddingError,
    ChunkingError,
)

__all__ = [
    "IngestProcessor",
    "RetrieveProcessor",
    "EndpointProvider",
    "LocalProvider",
    "DatabaseConfig",
    "SearchResponse",
    "ChunkResult",
    "DenseEmbeddingProvider",
    "SparseEmbeddingProvider",
    "ToolboxError",
    "NotConfiguredError",
    "DatabaseError",
    "EmbeddingError",
    "ChunkingError",
]
```

- [x] **Step 2: 執行 import 驗證**

```bash
python -c "
from chatbot_plugin_sdk import (
    IngestProcessor, RetrieveProcessor,
    EndpointProvider, LocalProvider,
    DatabaseConfig, SearchResponse,
)
print('All imports OK')
"
```
Expected: `All imports OK`

---

## Task 10：更新 pyproject.toml 依賴

**Files:**
- Modify: `pyproject.toml`

- [x] **Step 1: 移除不再需要的 pydantic-settings，確認 asyncpg 存在**

```toml
[project]
name = "chatbot-plugin-sdk"
version = "0.4.0"  # bump version
dependencies = [
    "pydantic>=2.0",
    # "pydantic-settings>=2.0",  ← 移除，config 改用 dataclass
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg",
    "pgvector",
    "httpx",
]

[project.optional-dependencies]
fastembed = ["fastembed>=0.3"]   # LocalProvider 需要時 pip install chatbot-plugin-sdk[fastembed]
```

---

## Task 11：執行既有 tests，修正失敗項目

**Files:**
- Modify: `src/tests/test_ingest.py`
- Modify: `src/tests/test_query.py`
- Modify: `src/tests/test_service.py`
- Modify: `src/tests/test_llm_caller.py`

- [x] **Step 1: 執行 tests，觀察哪些失敗**

```bash
cd /path/to/chatbot-plugin-sdk
uv run pytest src/tests/ -v 2>&1 | head -60
```

- [x] **Step 2: 更新 test_ingest.py — 改用新 IngestProcessor**

舊的 `RagArticleProcessor` 替換為 `IngestProcessor`，移除 LLM 相關 mock：
```python
from chatbot_plugin_sdk import IngestProcessor, EndpointProvider, DatabaseConfig

# 舊: processor = RagArticleProcessor()
# 新:
processor = IngestProcessor()
processor.configure(
    db=DatabaseConfig(dbname="test", user="u", password="p"),
    dense=EndpointProvider(url="http://mock:8080", dimension=768),
)
```

- [x] **Step 3: 刪除 test_llm_caller.py（LLM 功能已從 SDK 移除）**

```bash
rm src/tests/test_llm_caller.py
```

- [x] **Step 4: 更新 test_query.py — 改用新 RetrieveProcessor**

```python
from chatbot_plugin_sdk import RetrieveProcessor, EndpointProvider, DatabaseConfig

retriever = RetrieveProcessor()
retriever.configure(
    db=DatabaseConfig(dbname="test", user="u", password="p"),
    dense=EndpointProvider(url="http://mock:8080", dimension=768),
)
```

- [x] **Step 5: 執行 tests，確認全部通過**

```bash
uv run pytest src/tests/ -v
```
Expected: 全部 PASS，無 `FAILED` 或 `ERROR`

---

## Task 12：更新 scrape-analyzer 的 integration code

> **注意：** 這個 task 在 `scrape-analyzer` repo 中執行，不在 `chatbot-plugin-sdk`。

**Files:**
- Modify: `src/infrastructure/vector_store/rag_sdk_vector_store_impl.py`

- [x] **Step 1: 更新 import 與 metadata keys**

```python
# src/infrastructure/vector_store/rag_sdk_vector_store_impl.py
from chatbot_plugin_sdk import IngestProcessor   # was: from rag_sdk import VectorizingProcessor
from src.modules.articles.domain.services.vector_store_service import VectorStoreService
from src.shared.logging import get_logger

logger = get_logger(__name__)


class RagSdkVectorStoreService(VectorStoreService):
    def __init__(self, processor: IngestProcessor) -> None:
        self._processor = processor

    def ingest(self, article) -> None:
        self._processor.ingest(
            full_text=article.content,
            metadata={
                "url": str(article.url),       # SDK 用 url 推導 article_id（uuid5）
                "title": getattr(article, "title", None),
                "source": getattr(article, "source", None),
            },
        )
        logger.info("article_vectorized", article_id=str(article.id))
```

- [x] **Step 2: 驗證 scrape-analyzer 單元測試仍通過**

```bash
# 在 scrape-analyzer 目錄
uv run pytest src/tests/unit/ -v -k "vector"
```

---

## Self-Review Checklist

### Spec Coverage

| 需求 | 對應 Task |
|------|----------|
| 移除 `BaseRagProcessor` God Class | Task 8 (delete base.py) |
| `IngestProcessor` + `RetrieveProcessor` | Task 6, 7 |
| Protocol: `DenseEmbeddingProvider`, `SparseEmbeddingProvider` | Task 2 |
| `EndpointProvider` (HTTP, API + sidecar) | Task 4 |
| `LocalProvider` (in-process callable) | Task 5 |
| `configure()` 不讀環境變數 | Task 1 (DatabaseConfig), Task 6, 7 |
| `ensure_ready()` 建表/驗表邏輯 | Task 6, 7 |
| Dense/Sparse 可獨立運作，nullable columns | Task 3, 6 |
| `vectors` schema 隔離 | Task 3, 6 |
| 移除 LLM (`chat`, `_call_llm`, `ChatResponse`) | Task 8 |
| `dimension` 從 provider 讀取 | Task 2 (Protocol), Task 6 |
| scrape-analyzer metadata key 修正 | Task 12 |
| 修正 ingest.py bug (tuple unpack) | Task 6 (重寫，bug 自動消除) |
| Public exports 更新 | Task 9 |

### 已知限制（本 Plan 範疇外）
- Sparse-only search 尚未實作（Task 7 有 `raise NotImplementedError` 佔位）
- Hybrid search (dense + sparse RRF fusion) 未在本 plan 實作
- ColBERT / multi-vector 未納入
