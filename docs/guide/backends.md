# Choosing a Backend

Both `AsyncPgBackend` and `SyncPgBackend` implement the same `DatabaseBackend` Protocol, so `IngestProcessor` and `RetrieveProcessor` work identically with either. The choice is purely about your application's concurrency model.

## Summary

| | `AsyncPgBackend` | `SyncPgBackend` |
|---|---|---|
| **Driver** | asyncpg | psycopg2 |
| **SQLAlchemy** | async (`AsyncEngine`) | sync (`Engine`) |
| **Thread-safe?** | No — event-loop-bound | Yes — uses `threading.Lock` |
| **Use with** | FastAPI, asyncio apps | `ThreadPoolExecutor`, `asyncio.run()` per thread |
| **Extra dep** | _(bundled)_ | `pip install chatbot-plugin-sdk[sync]` |

## AsyncPgBackend

`AsyncPgBackend` creates an `asyncpg`-backed `AsyncEngine` on `__init__`.  The engine is bound to the event loop that was running **at creation time**.

```python
from chatbot_plugin_sdk import AsyncPgBackend, DatabaseConfig

backend = AsyncPgBackend(DatabaseConfig(
    dbname="mydb", user="postgres", password="secret",
))
```

**Good for:**

- FastAPI applications — the engine is created once at startup inside one event loop.
- Any long-running async service where one event loop owns the backend for its lifetime.

**Bad for:**

- `asyncio.run()` called from multiple threads — each `asyncio.run()` creates a new event loop, and the engine created in thread A is unusable in thread B.

## SyncPgBackend

`SyncPgBackend` uses psycopg2 and a sync `Engine`.  All async method signatures (`async def setup`, `async def upsert`, …) wrap synchronous DB calls via `loop.run_in_executor()`, so they are awaitable without blocking the event loop.

```python
from chatbot_plugin_sdk import SyncPgBackend, DatabaseConfig

backend = SyncPgBackend(DatabaseConfig(
    dbname="mydb", user="postgres", password="secret",
))
```

**Good for:**

- Scraper pipelines that run `asyncio.run(processor.ingest(...))` from a `ThreadPoolExecutor`.
- Celery / RQ tasks where each task creates its own event loop.
- Any context where `asyncio.run()` is called from multiple threads sharing one backend.

**Bad for:**

- High-throughput async services — the `run_in_executor` overhead is unnecessary when you already have a proper event loop.

!!! info "Connection pool is always thread-safe"
    The SQLAlchemy `Engine` pool itself is thread-safe. Each database call checks out a connection, uses it, and returns it. Connections are never shared between concurrent calls.

## Shutdown

Always call `backend.close()` on application shutdown to cleanly dispose the connection pool:

```python
# FastAPI lifespan
@asynccontextmanager
async def lifespan(app):
    yield
    await backend.close()

# Script / one-shot
try:
    await processor.ingest(...)
finally:
    await backend.close()
```

## Multi-Process Safety

Both backends are safe to use **after** a `fork()` if you create the backend **inside each worker process** (never in the parent and shared across children).

```python
# Celery worker: create per-process, not module-level
@worker_process_init.connect
def setup_backend(sender, **kwargs):
    app.backend = SyncPgBackend(config)
```

## Custom Backend

You can implement your own storage by satisfying the `DatabaseBackend` Protocol:

```python
from chatbot_plugin_sdk import DatabaseBackend  # Protocol

class MyCustomBackend:
    schema = "vectors"

    async def setup(self, dense_dim): ...
    async def validate(self, dense_dim): ...
    async def upsert(self, article_id, metadata, chunks, dense_vectors, sparse_vectors): ...
    async def search_dense(self, query_vec, top_k): ...
    async def close(self): ...
```

The processors accept any object satisfying this protocol — no inheritance required.
