# chatbot-plugin-sdk

Python SDK for vector-based RAG — article ingestion and semantic search over **PostgreSQL + pgvector**.

## What It Does

```
┌─────────────┐   IngestProcessor.ingest()   ┌──────────────────┐
│   Article   │ ──────────────────────────► │  PostgreSQL DB   │
│   (text)    │  normalize → chunk → embed   │  (pgvector)      │
└─────────────┘  → upsert (idempotent)       └────────┬─────────┘
                                                       │
                                              cosine similarity
                                                       │
┌─────────────┐  RetrieveProcessor.search()  ┌────────▼─────────┐
│ User query  │ ◄────────────────────────── │  top-k chunks    │
└─────────────┘  embed query → rank results  └──────────────────┘
```

## Key Features

| Feature | Detail |
|---------|--------|
| **Idempotent ingest** | Article ID is `uuid5(NAMESPACE_URL, url)` — re-ingesting the same URL replaces existing chunks |
| **Two backends** | `AsyncPgBackend` for FastAPI / native asyncio; `SyncPgBackend` for `ThreadPoolExecutor` |
| **Flexible providers** | `EndpointProvider` for any HTTP embedding API; `LocalProvider` for in-process callables |
| **Rate limiting** | Optional `SlidingWindowStrategy` (RPM / TPM / RPD) for external APIs like Google AI Studio |
| **Protocol-based DI** | Swap any component via `DenseEmbeddingProvider` / `SparseEmbeddingProvider` / `DatabaseBackend` |

## Installation

```bash
# Core (async PostgreSQL)
pip install chatbot-plugin-sdk

# + sync PostgreSQL for ThreadPoolExecutor
pip install "chatbot-plugin-sdk[sync]"

# + local fastembed models
pip install "chatbot-plugin-sdk[fastembed]"
```

## 30-Second Example

```python
import asyncio
from chatbot_plugin_sdk import (
    IngestProcessor, RetrieveProcessor,
    AsyncPgBackend, DatabaseConfig, EndpointProvider,
)

config = DatabaseConfig(dbname="mydb", user="u", password="p")
backend = AsyncPgBackend(config)
provider = EndpointProvider(url="http://embed:8080", dimension=768)

# --- Ingest ---
ingestor = IngestProcessor()
ingestor.configure(backend=backend, dense=provider)

await ingestor.ingest(
    full_text="Retrieval-augmented generation (RAG) is ...",
    metadata={"url": "https://example.com/rag-intro", "title": "RAG 101"},
)

# --- Retrieve ---
retriever = RetrieveProcessor()
retriever.configure(backend=backend, dense=provider)

result = await retriever.search("What is RAG?", top_k=5)
for chunk in result.chunks:
    print(chunk.score, chunk.content[:80])
```

→ [Quick Start](quickstart.md) for a full runnable example including Docker setup.
