from __future__ import annotations
from typing import Protocol, runtime_checkable

# Exception contract for implementers of either protocol below: a failed
# embed() call must raise chatbot_plugin_sdk.exceptions.EmbeddingError (or a
# more specific subclass — e.g. RateLimitExhausted for a rate/quota limit),
# never let a provider-internal exception (an HTTP client error, a raw SDK
# exception from the underlying model API, etc.) propagate unwrapped. This
# lets IngestProcessor/RetrieveProcessor callers catch one SDK-owned
# exception family regardless of which provider is configured. See the
# built-in providers (gemini.py, huggingface.py, endpoint.py, local.py) for
# the pattern.


@runtime_checkable
class DenseEmbeddingProvider(Protocol):
    """HTTP endpoint 或 in-process callable，輸出 dense 向量。

    dimension 屬性供 ensure_ready() 在首次建表時決定 VECTOR(N) 的 N。
    """
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@runtime_checkable
class SparseEmbeddingProvider(Protocol):
    """HTTP endpoint 或 in-process callable，輸出 sparse 向量（token_id → weight）。

    dimension 屬性為詞彙表大小（SPLADE / BERT: 30522），供 setup() 建立 SPARSEVEC(N) 欄位使用。
    """
    dimension: int

    async def embed(self, texts: list[str]) -> list[dict[str, float]]:
        ...
