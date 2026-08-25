# Processors

The two processors are the main entry points for the SDK.  Configure once, then call `ingest()` or `search()` as many times as needed.

## IngestProcessor

!!! tip "Call `aclose()` when you're done"
    Dense-embedding calls are coordinated through a background worker (see
    [Batching](batching.md)) that starts lazily on first `ingest()` call and
    otherwise runs forever. Call `await processor.aclose()` once you're done
    issuing `ingest()` calls for a given `configure()`-d processor — e.g. at
    the end of a batch job or pipeline run — to stop it cleanly.

::: chatbot_plugin_sdk.processors.ingest.IngestProcessor
    options:
      members:
        - __init__
        - configure
        - ingest
        - aclose
        - get_embed_queue
        - set_embed_queue

## RetrieveProcessor

::: chatbot_plugin_sdk.processors.retrieve.RetrieveProcessor
    options:
      members:
        - __init__
        - configure
        - retrieve
