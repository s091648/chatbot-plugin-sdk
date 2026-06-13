from __future__ import annotations
import asyncio
import logging
from collections.abc import Callable
from chatbot_plugin_sdk.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class LocalProvider:
    """In-process embedding provider，接受 sync 或 async callable。

    適用於在同一個 Python process 內直接呼叫 embedding function，
    例如自行初始化的 fastembed 模型。
    Sync callable 會被包進 asyncio executor 以避免 block event loop。

    Args:
        fn: 接受 list[str] 並回傳向量列表的 callable。
            Dense: fn(texts) -> list[list[float]]
            Sparse: fn(texts) -> list[dict[str, float]]
        dimension: dense 向量維度。用於 dense 場景時必填；sparse 可省略。

    Usage::

        from fastembed import TextEmbedding
        model = TextEmbedding("BAAI/bge-small-en")

        dense = LocalProvider(
            fn=lambda texts: [v.tolist() for v in model.embed(texts)],
            dimension=384,
        )
    """

    def __init__(
        self,
        fn: Callable[[list[str]], list],
        dimension: int | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn)}")
        self._fn = fn
        self.dimension: int = dimension or 0  # sparse 時為 0（不使用）

    async def embed(self, texts: list[str]) -> list:
        """呼叫 fn(texts)，自動處理 sync/async 差異。"""
        logger.debug("local_embedding_called", extra={"text_count": len(texts), "dimension": self.dimension})
        try:
            if asyncio.iscoroutinefunction(self._fn):
                result = await self._fn(texts)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._fn, texts)
            logger.debug("local_embedding_done", extra={"vector_count": len(result)})
            return result
        except Exception as exc:
            logger.warning("local_embedding_failed", extra={"error": str(exc)})
            raise EmbeddingError(f"Local embedding function failed: {exc}") from exc
