from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx

from chatbot_plugin_sdk.exceptions import EmbeddingError

if TYPE_CHECKING:
    from chatbot_plugin_sdk.rate_limit import RateLimitStrategy

logger = logging.getLogger(__name__)

_HF_INFERENCE_BASE = "https://api-inference.huggingface.co/models"
_MAX_RETRYABLE_DELAY_SECS = 300.0


class HuggingFaceDenseProvider:
    """Dense embedding provider backed by HuggingFace Serverless Inference API.

    Calls ``POST https://api-inference.huggingface.co/models/{model}``
    with ``{"inputs": [...], "options": {"wait_for_model": true}}``.
    Response is ``[[float, ...], ...]`` (list of vectors).

    Retries on:
    - HTTP 503 (model cold-starting) — waits ``estimated_time`` seconds from response body
    - HTTP 429 (rate limit) — waits ``Retry-After`` header value

    Delays longer than 5 minutes are not retried (daily quota or infra issue).

    Args:
        api_token: HuggingFace API token (``hf_xxx``).
        model: Model repo ID, e.g. ``"BAAI/bge-m3"``.
        dimension: Output vector dimension.
        rate_limit: Optional rate-limiting strategy.
        max_retries: Retry count on 503/429 (default: 5).
        timeout: HTTP timeout in seconds (default: 60).
    """

    def __init__(
        self,
        api_token: str,
        model: str,
        dimension: int,
        rate_limit: "RateLimitStrategy | None" = None,
        max_retries: int = 5,
        timeout: float = 60.0,
    ) -> None:
        self._api_token = api_token
        self._model = model
        self.dimension = dimension
        self._rate_limit = rate_limit
        self._max_retries = max_retries
        self._timeout = timeout
        self._url = f"{_HF_INFERENCE_BASE}/{model}"

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._api_token}"},
            timeout=self._timeout,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._rate_limit is not None:
            estimated_tokens = max(1, sum(len(t) for t in texts) // 4)
            await self._rate_limit.acquire(estimated_tokens)

        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        for attempt in range(self._max_retries):
            async with self._build_client() as client:
                try:
                    resp = await client.post(self._url, json=payload)
                except Exception as exc:
                    logger.warning(
                        "hf_request_failed",
                        extra={"model": self._model, "error": str(exc)},
                    )
                    raise EmbeddingError(f"HuggingFace request failed: {exc}") from exc

                if resp.status_code == 200:
                    data = resp.json()
                    if not isinstance(data, list) or not data or not isinstance(data[0], list):
                        raise EmbeddingError(
                            f"Unexpected HuggingFace response shape for model {self._model!r}: "
                            f"expected list[list[float]], got {type(data).__name__}"
                        )
                    return data

                if resp.status_code == 503:
                    try:
                        wait = float(resp.json().get("estimated_time", 20.0))
                    except Exception:
                        wait = 20.0
                    if wait > _MAX_RETRYABLE_DELAY_SECS or attempt >= self._max_retries - 1:
                        raise EmbeddingError(
                            f"HuggingFace model {self._model!r} unavailable "
                            f"after {attempt + 1} attempt(s)"
                        )
                    logger.warning(
                        "hf_model_loading_retrying",
                        extra={"model": self._model, "wait": wait, "attempt": attempt + 1},
                    )
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code == 429:
                    try:
                        wait = float(resp.headers.get("Retry-After", 60))
                    except Exception:
                        wait = 60.0
                    if wait > _MAX_RETRYABLE_DELAY_SECS or attempt >= self._max_retries - 1:
                        raise EmbeddingError(
                            f"HuggingFace rate limit for {self._model!r} exceeded "
                            f"after {attempt + 1} attempt(s)"
                        )
                    logger.warning(
                        "hf_rate_limited_retrying",
                        extra={"model": self._model, "wait": wait, "attempt": attempt + 1},
                    )
                    await asyncio.sleep(wait)
                    continue

                raise EmbeddingError(
                    f"HuggingFace API returned HTTP {resp.status_code} "
                    f"for model {self._model!r}: {resp.text}"
                )

        raise EmbeddingError(
            f"HuggingFace embedding failed for {self._model!r} "
            f"after {self._max_retries} attempt(s)"
        )
