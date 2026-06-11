# chatbot-plugin-sdk

Python SDK for the vector storage toolbox — RAG article ingestion and querying.

## Installation

```bash
pip install chatbot-plugin-sdk
```

## Usage

### ArticleProcessor — write articles to vector DB

```python
from chatbot_plugin_sdk import RagArticleProcessor

processor = RagArticleProcessor()
processor.configure(
    dbname="chatbot_plugin",
    user="postgres",
    password="postgres",
    embedding_model_api="http://localhost:8080",
)

await processor.ingest(
    full_text="Retrieval augmented generation is ...",
    metadata={"url": "https://example.com/article", "title": "RAG 101"},
)
```

### QueryProcessor — RAG query

```python
from chatbot_plugin_sdk import RagQueryProcessor

processor = RagQueryProcessor()
processor.configure(
    dbname="chatbot_plugin",
    user="postgres",
    password="postgres",
    embedding_model_api="http://localhost:8080",
)

response = await processor.query("What is RAG?")
print(response.reply)
```

## Class Hierarchy

| Class | Description |
|-------|-------------|
| `BaseRagProcessor` | Internal base class — DB, search, chat, LLM fallback |
| `RagArticleProcessor` | Write — configure + ingest pipeline |
| `RagQueryProcessor` | Read — configure + query (wraps chat) |
