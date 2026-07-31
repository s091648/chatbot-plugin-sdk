# Exceptions

All SDK exceptions inherit from `ToolboxError`.

## Exception Hierarchy

```
ToolboxError
├── NotConfiguredError        — configure() not called (caller/integration misuse)
├── ChunkingError             — text chunking produced an unexpected result
└── ExternalDependencyError   — shared category: infra the SDK depends on but
    │                           does not control failed. Catch this one type to
    │                           handle "something upstream broke" generically.
    ├── DatabaseError         — DB operation failed (upsert, schema mismatch, ...)
    ├── EmbeddingError        — embedding provider call failed (not rate-limited)
    └── RateLimitExhausted    — embedding provider rate/quota limit reached
                                 (chatbot_plugin_sdk.rate_limit)
```

This SDK has no HTTP layer of its own, so it does not map exceptions to status
codes — that's the embedding application's job. `ExternalDependencyError`
exists so an application can, at its own API boundary, catch one type and
translate it into its own external-dependency-failure category (e.g. a
502/503) without enumerating every leaf exception below it.

Implementers of a custom `DenseEmbeddingProvider`/`SparseEmbeddingProvider`
should follow the same contract: wrap failures as `EmbeddingError` (or
`RateLimitExhausted` for a rate/quota limit), never let a provider-internal
exception propagate unwrapped — see `chatbot_plugin_sdk.protocols`.

---

::: chatbot_plugin_sdk.exceptions.ToolboxError

::: chatbot_plugin_sdk.exceptions.NotConfiguredError

::: chatbot_plugin_sdk.exceptions.ChunkingError

::: chatbot_plugin_sdk.exceptions.ExternalDependencyError

::: chatbot_plugin_sdk.exceptions.DatabaseError

::: chatbot_plugin_sdk.exceptions.EmbeddingError

::: chatbot_plugin_sdk.rate_limit.RateLimitExhausted
