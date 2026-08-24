# Providers

## EndpointProvider

::: chatbot_plugin_sdk.providers.endpoint.EndpointProvider
    options:
      members:
        - __init__
        - embed

## LocalProvider

::: chatbot_plugin_sdk.providers.local.LocalProvider
    options:
      members:
        - __init__
        - embed

## GeminiDenseProvider

::: chatbot_plugin_sdk.providers.gemini.GeminiDenseProvider
    options:
      members:
        - __init__
        - embed

## Protocols

::: chatbot_plugin_sdk.protocols.DenseEmbeddingProvider

::: chatbot_plugin_sdk.protocols.SparseEmbeddingProvider

## Tracing

Optional, dependency-free tracing hook: `GeminiDenseProvider` and `RetrieveProcessor`
both accept a `tracer` constructor argument satisfying the `Tracer` protocol below.
Omitted, they resolve one themselves via `default_tracer()` — real OpenTelemetry if
`opentelemetry-api` is installed (an optional `otel` extra, not a hard dependency of
this SDK), else the built-in no-op implementations.

::: chatbot_plugin_sdk.protocols.Tracer

::: chatbot_plugin_sdk.protocols.Span

::: chatbot_plugin_sdk.protocols.default_tracer

::: chatbot_plugin_sdk.protocols.NoOpTracer

::: chatbot_plugin_sdk.protocols.NoOpSpan
