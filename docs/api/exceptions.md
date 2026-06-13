# Exceptions

All SDK exceptions inherit from `ToolboxError`.

## Exception Hierarchy

```
ToolboxError
├── NotConfiguredError   — configure() not called, or missing required args
├── DatabaseError        — DB operation failed (upsert, schema mismatch, ...)
├── EmbeddingError       — HTTP call to embedding endpoint failed
└── ChunkingError        — text chunking produced an unexpected result
```

`RateLimitExhausted` (in `chatbot_plugin_sdk.rate_limit`) inherits directly from `Exception`
so callers can catch it independently of storage errors.

---

::: chatbot_plugin_sdk.exceptions.ToolboxError

::: chatbot_plugin_sdk.exceptions.NotConfiguredError

::: chatbot_plugin_sdk.exceptions.DatabaseError

::: chatbot_plugin_sdk.exceptions.EmbeddingError

::: chatbot_plugin_sdk.exceptions.ChunkingError

::: chatbot_plugin_sdk.rate_limit.RateLimitExhausted
