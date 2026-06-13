# Processors

The two processors are the main entry points for the SDK.  Configure once, then call `ingest()` or `search()` as many times as needed.

## IngestProcessor

::: chatbot_plugin_sdk.processors.ingest.IngestProcessor
    options:
      members:
        - configure
        - ensure_ready
        - ingest

## RetrieveProcessor

::: chatbot_plugin_sdk.processors.retrieve.RetrieveProcessor
    options:
      members:
        - configure
        - ensure_ready
        - search
