from __future__ import annotations
import asyncio
import logging
from typing import TYPE_CHECKING
import httpx
from chatbot_plugin_sdk.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from chatbot_plugin_sdk.rate_limit import RateLimitStrategy


class EndpointProvider:
    """HTTP embedding provider。適用於外部 API（如 Google AI Studio）
    與內部 sidecar microservice（如自架 fastembed HTTP service），
    兩者呼叫方式相同，差別只在 URL。

    Args:
        url: embedding service 的 base URL（如 "http://localhost:8080"）。
        response_key: API response 中存放向量的 key（dense 用 "dense"，sparse 用 "sparse"）。
        api_key: 若 service 需要 Bearer token，在此傳入。
        dimension: dense 向量的維度。使用 response_key="dense" 時必填；
                   response_key="sparse" 時可省略（sparse 無固定維度）。
        timeout: HTTP 請求 timeout（秒），預設 60。
        rate_limit: 選填的 rate limiting 策略（如 ``SlidingWindowStrategy``）。
                    使用外部 API（如 Google AI Studio）時建議設定；
                    內部 service 或自架 model 可省略（傳 ``None``）。

    Usage::

        from chatbot_plugin_sdk import EndpointProvider
        from chatbot_plugin_sdk.rate_limit import SlidingWindowStrategy

        # 搭配 Google AI Studio（有 rpm/tpm/rpd 限制）
        dense = EndpointProvider(
            url="https://generativelanguage.googleapis.com/...",
            dimension=768,
            api_key="AIza...",
            rate_limit=SlidingWindowStrategy(rpm=10, tpm=40_000, rpd=1_500),
        )

        # 內部 sidecar（無限流）
        sparse = EndpointProvider(url="http://embed:8080", response_key="sparse")
    """

    def __init__(
        self,
        url: str,
        response_key: str = "dense",
        api_key: str | None = None,
        dimension: int | None = None,
        timeout: float = 60.0,
        rate_limit: "RateLimitStrategy | None" = None,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        if response_key == "dense" and dimension is None:
            raise ValueError("dimension is required when response_key='dense'")
        self._url = url.rstrip("/")
        self._response_key = response_key
        self._api_key = api_key
        self._timeout = timeout
        self._rate_limit = rate_limit
        self._retries = retries
        self._retry_delay = retry_delay
        self.dimension: int = dimension or 0  # sparse 時為 0（不使用）

    def _build_client(self) -> httpx.AsyncClient:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return httpx.AsyncClient(base_url=self._url, headers=headers, timeout=self._timeout)

    async def embed(self, texts: list[str]) -> list:
        """送出 POST /embed 請求，回傳對應 response_key 的向量列表。

        Request body: ``{"texts": ["text1", ...]}``
        Expected response: ``{"dense": [[...], ...], "sparse": [{...}, ...]}``

        Retries on connection errors (e.g. serverless cold start) with
        exponential backoff. HTTP status errors are not retried.
        """
        # Estimate tokens: rough approximation (4 chars ≈ 1 token)
        _estimated_tokens = 0
        if self._rate_limit is not None:
            _estimated_tokens = max(1, sum(len(t) for t in texts) // 4)
            await self._rate_limit.acquire(_estimated_tokens)

        logger.debug(
            "embedding_request",
            extra={"url": self._url, "response_key": self._response_key, "text_count": len(texts)},
        )

        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            async with self._build_client() as client:
                try:
                    resp = await client.post("/embed", json={"texts": texts})
                    resp.raise_for_status()
                    data = resp.json()
                    break  # success
                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "embedding_http_error",
                        extra={"url": self._url, "status": exc.response.status_code},
                    )
                    raise EmbeddingError(
                        f"Embedding endpoint returned {exc.response.status_code}: {exc.response.text}"
                    ) from exc
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    last_exc = exc
                    if attempt < self._retries:
                        wait = self._retry_delay * (2 ** (attempt - 1))
                        logger.info(
                            "embedding_retry",
                            extra={"url": self._url, "attempt": attempt, "retry_after": wait, "error": str(exc)},
                        )
                        await asyncio.sleep(wait)
                except Exception as exc:
                    logger.warning("embedding_request_failed", extra={"url": self._url, "error": str(exc)})
                    raise EmbeddingError(f"Embedding request failed: {exc}") from exc
        else:
            raise EmbeddingError(
                f"Embedding request failed after {self._retries} attempts: {last_exc}"
            ) from last_exc

        result = data.get(self._response_key)
        if not result:
            raise EmbeddingError(
                f"Embedding response missing key '{self._response_key}'. "
                f"Available keys: {list(data.keys())}"
            )

        if self._rate_limit is not None:
            self._rate_limit.record_usage(_estimated_tokens)

        return result
