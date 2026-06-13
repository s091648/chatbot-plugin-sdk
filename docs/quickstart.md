# Quick Start

## Prerequisites

- Python 3.11+
- PostgreSQL with the `pgvector` extension enabled
- An embedding service (HTTP endpoint or local model)

### Enable pgvector in PostgreSQL

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Installation

```bash
pip install chatbot-plugin-sdk

# For ThreadPoolExecutor / sync usage:
pip install "chatbot-plugin-sdk[sync]"
```

## FastAPI Application

This is the recommended pattern when you control your own asyncio event loop.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from chatbot_plugin_sdk import (
    AsyncPgBackend, DatabaseConfig,
    EndpointProvider,
    IngestProcessor, RetrieveProcessor,
)

config = DatabaseConfig(
    dbname="mydb",
    user="postgres",
    password="secret",
    host="localhost",
    port=5432,
    schema="vectors",      # pgvector tables will live here
)
backend = AsyncPgBackend(config)
provider = EndpointProvider(
    url="http://embedding-service:8080",
    dimension=768,
)

ingestor = IngestProcessor()
ingestor.configure(backend=backend, dense=provider)

retriever = RetrieveProcessor()
retriever.configure(backend=backend, dense=provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await backend.close()   # dispose connection pool on shutdown

app = FastAPI(lifespan=lifespan)


@app.post("/ingest")
async def ingest(url: str, text: str, title: str = ""):
    await ingestor.ingest(
        full_text=text,
        metadata={"url": url, "title": title},
    )
    return {"status": "ok"}


@app.get("/search")
async def search(q: str, top_k: int = 5):
    result = await retriever.retrieve(q, top_k=top_k)
    return result
```

## ThreadPoolExecutor (Scraper / Background Worker)

Use `SyncPgBackend` when you call `asyncio.run()` from multiple threads — for example inside a `ThreadPoolExecutor` or a task queue worker.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from chatbot_plugin_sdk import (
    SyncPgBackend, DatabaseConfig,
    EndpointProvider,
    IngestProcessor,
)
from chatbot_plugin_sdk.rate_limit import SlidingWindowStrategy

# SyncPgBackend + one shared EndpointProvider with rate limiting
backend = SyncPgBackend(DatabaseConfig(
    dbname="mydb", user="postgres", password="secret",
))
provider = EndpointProvider(
    url="https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent",
    dimension=768,
    api_key="AIza...",
    rate_limit=SlidingWindowStrategy(rpm=10, tpm=40_000, rpd=1_500),
)

processor = IngestProcessor()
processor.configure(backend=backend, dense=provider)


def ingest_article(url: str, text: str) -> None:
    """Called from a thread — each call gets its own event loop."""
    asyncio.run(processor.ingest(
        full_text=text,
        metadata={"url": url, "title": ""},
    ))


with ThreadPoolExecutor(max_workers=4) as pool:
    articles = [
        ("https://example.com/a", "Article A text..."),
        ("https://example.com/b", "Article B text..."),
    ]
    list(pool.map(lambda a: ingest_article(*a), articles))
```

!!! warning "SyncPgBackend for ThreadPoolExecutor"
    Do **not** use `AsyncPgBackend` here — its async engine is bound to the event loop
    created in the first `asyncio.run()` call and cannot be reused across threads.
    See [Choosing a Backend](guide/backends.md) for the full explanation.

## Local Embedding Model (fastembed)

No external HTTP service needed:

```python
from chatbot_plugin_sdk import LocalProvider, IngestProcessor, AsyncPgBackend, DatabaseConfig

try:
    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed_fn(texts):
        return [v.tolist() for v in model.embed(texts)]

    provider = LocalProvider(fn=embed_fn, dimension=384)
except ImportError:
    raise SystemExit("Run: pip install chatbot-plugin-sdk[fastembed]")

backend = AsyncPgBackend(DatabaseConfig(dbname="mydb", user="u", password="p"))
processor = IngestProcessor()
processor.configure(backend=backend, dense=provider)
```

## Semantic Search

Once articles are ingested:

```python
result = await retriever.retrieve("What is retrieval-augmented generation?", top_k=5)

for chunk in result.chunks:
    print(f"[{chunk.score:.3f}] {chunk.article_title} — {chunk.content[:120]}")
```

`score` is `1 − cosine_distance` — the closer to `1.0`, the more similar to the query.

## Next Steps

- [Choosing a Backend](guide/backends.md) — when to use `AsyncPgBackend` vs `SyncPgBackend`
- [Embedding Providers](guide/providers.md) — `EndpointProvider`, `LocalProvider`, rate limiting
- [API Reference](api/processors.md) — full method signatures and options
