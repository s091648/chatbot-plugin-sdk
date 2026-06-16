from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatbot_plugin_sdk.rate_limit import RateLimitStrategy


class GeminiDenseProvider:
    """Dense embedding provider backed by Google Gemini (google-genai).

    ``google-genai`` is loaded lazily; ``ImportError`` propagates to the caller
    if the ``gemini`` optional dependency group is not installed.

    Args:
        api_key: Gemini API key.
        model: Embedding model name (default: ``gemini-embedding-001``).
        dimension: Output vector dimension (default: 768).
        rate_limit: Optional rate-limiting strategy (e.g. ``SlidingWindowStrategy``).
                    Construct it in the caller; use ``build_dense_provider`` for the
                    standard ``rpm / tpm / rpd`` → strategy conversion.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
        rate_limit: "RateLimitStrategy | None" = None,
    ) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.dimension: int = dimension
        self._rate_limit = rate_limit

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config={"task_type": "CLASSIFICATION", "output_dimensionality": self.dimension},
        )
        return [list(e.values) for e in response.embeddings]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._rate_limit is not None:
            estimated_tokens = max(1, sum(len(t) for t in texts) // 4)
            await self._rate_limit.acquire(estimated_tokens)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, texts)
