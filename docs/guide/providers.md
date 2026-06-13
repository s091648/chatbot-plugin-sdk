# Embedding Providers

Providers convert text into vectors.  Both processors accept `dense` and `sparse` providers via `configure()`.

## EndpointProvider

Sends a `POST /embed` request to any HTTP embedding service — external APIs or internal sidecars — with the same interface.

```python
from chatbot_plugin_sdk import EndpointProvider

# Dense (Google AI Studio, HuggingFace TEI, custom sidecar, ...)
dense = EndpointProvider(
    url="http://embedding-service:8080",
    response_key="dense",    # default
    dimension=768,
    api_key="...",           # added as Authorization: Bearer
    timeout=60.0,            # seconds
)

# Sparse (SPLADE or similar)
sparse = EndpointProvider(
    url="http://embedding-service:8080",
    response_key="sparse",
    # dimension not required for sparse
)
```

**Expected request / response format:**

```
POST /embed
{"texts": ["sentence one", "sentence two"]}

200 OK
{"dense": [[0.1, 0.2, ...], [0.3, 0.4, ...]], "sparse": [{...}, {...}]}
```

## LocalProvider

Wraps a Python callable (sync or async) for in-process embedding — no HTTP overhead.

```python
from chatbot_plugin_sdk import LocalProvider

# Sync callable — automatically wrapped in run_in_executor
def my_embed(texts: list[str]) -> list[list[float]]:
    ...

provider = LocalProvider(fn=my_embed, dimension=384)

# Async callable — called directly
async def my_async_embed(texts):
    ...

provider = LocalProvider(fn=my_async_embed, dimension=384)
```

### Using fastembed

```bash
pip install "chatbot-plugin-sdk[fastembed]"
```

```python
from fastembed import TextEmbedding
from chatbot_plugin_sdk import LocalProvider

model = TextEmbedding("BAAI/bge-small-en-v1.5")

provider = LocalProvider(
    fn=lambda texts: [v.tolist() for v in model.embed(texts)],
    dimension=384,
)
```

## Rate Limiting

When using external APIs (e.g., Google AI Studio), you will hit rate limits.  Pass a `SlidingWindowStrategy` to `EndpointProvider` to enforce your quota before requests are sent.

```python
from chatbot_plugin_sdk import EndpointProvider
from chatbot_plugin_sdk.rate_limit import SlidingWindowStrategy

strategy = SlidingWindowStrategy(
    rpm=10,       # max requests per minute (0 = no limit)
    tpm=40_000,   # max tokens per minute (0 = no limit)
    rpd=1_500,    # max requests per day — raises RateLimitExhausted when reached
)

provider = EndpointProvider(
    url="https://generativelanguage.googleapis.com/...",
    dimension=768,
    api_key="AIza...",
    rate_limit=strategy,
)
```

The strategy:

1. **Before each `embed()` call** — calls `strategy.acquire(estimated_tokens)`.
   - If under limit: claims the slot immediately.
   - If RPM or TPM window is full: `await asyncio.sleep(wait_seconds)` until a slot opens.
   - If RPD is reached: raises `RateLimitExhausted` immediately.
2. **After a successful call** — calls `strategy.record_usage(estimated_tokens)` to finalize the token count in the sliding window.

Token estimation: `max(1, total_chars // 4)` — roughly 4 characters per token.

!!! tip "Thread safety"
    A single `SlidingWindowStrategy` instance can be shared across threads (e.g., inside a `ThreadPoolExecutor`) because state is guarded by `threading.Lock`.  The `asyncio.sleep` call in `acquire()` is non-blocking — it yields the event loop without freezing other coroutines.

### Handling RateLimitExhausted

`RateLimitExhausted` is raised when the **daily** cap (`rpd`) is reached.  Unlike RPM/TPM (which sleep until the window clears), the daily cap is a hard stop — no retry makes sense until the next day.

```python
from chatbot_plugin_sdk import RateLimitExhausted

try:
    await processor.ingest(full_text=text, metadata={"url": url})
except RateLimitExhausted:
    # Switch to a fallback provider, queue for tomorrow, or alert
    logger.warning("Daily embedding quota exhausted — skipping article")
```

## Custom Provider

Implement either Protocol to use your own embedding logic:

```python
from chatbot_plugin_sdk import DenseEmbeddingProvider

class MyDenseProvider:
    dimension = 512

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # ... call your model ...
        return vectors

processor.configure(backend=backend, dense=MyDenseProvider())
```
