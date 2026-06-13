from __future__ import annotations
from typing import Protocol, runtime_checkable


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
