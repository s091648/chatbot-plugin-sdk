# chatbot-plugin-sdk

Python SDK for vector-based RAG — article ingestion and semantic search over PostgreSQL + pgvector.

## Installation

```bash
pip install chatbot-plugin-sdk

# Sync backend (ThreadPoolExecutor):
pip install "chatbot-plugin-sdk[sync]"

# Local fastembed models:
pip install "chatbot-plugin-sdk[fastembed]"
```

## Quick Example

```python
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

# --- Search ---
retriever = RetrieveProcessor()
retriever.configure(backend=backend, dense=provider)

result = await retriever.search("What is RAG?", top_k=5)
for chunk in result.chunks:
    print(chunk.score, chunk.content[:80])
```

## Key Classes

| Class | Description |
|-------|-------------|
| `IngestProcessor` | Write pipeline — normalize → chunk → embed → upsert |
| `RetrieveProcessor` | Read pipeline — embed query → cosine search → ranked chunks |
| `AsyncPgBackend` | asyncpg backend; binds to one event loop (FastAPI / asyncio apps) |
| `SyncPgBackend` | psycopg2 backend; thread-safe, for `ThreadPoolExecutor` / `asyncio.run()` |
| `EndpointProvider` | HTTP embedding provider (external APIs or internal sidecars) |
| `LocalProvider` | In-process embedding via sync or async callable |
| `SlidingWindowStrategy` | Optional rate limiting: RPM / TPM / RPD sliding window |

## Documentation

```bash
# Preview docs locally
uv run --group docs mkdocs serve
```
