from __future__ import annotations
import httpx
from chatbot_plugin_sdk.exceptions import EmbeddingError


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

    Usage::

        # Dense（搭配 Google AI Studio 或 HuggingFace TEI）
        dense = EndpointProvider(url="http://embed:8080", response_key="dense",
                                 api_key="...", dimension=768)

        # Sparse（搭配自架 SPLADE sidecar）
        sparse = EndpointProvider(url="http://embed:8080", response_key="sparse")
    """

    def __init__(
        self,
        url: str,
        response_key: str = "dense",
        api_key: str | None = None,
        dimension: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        if response_key == "dense" and dimension is None:
            raise ValueError("dimension is required when response_key='dense'")
        self._url = url.rstrip("/")
        self._response_key = response_key
        self._api_key = api_key
        self._timeout = timeout
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
        """
        async with self._build_client() as client:
            try:
                resp = await client.post("/embed", json={"texts": texts})
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(
                    f"Embedding endpoint returned {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except Exception as exc:
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc

        result = data.get(self._response_key)
        if not result:
            raise EmbeddingError(
                f"Embedding response missing key '{self._response_key}'. "
                f"Available keys: {list(data.keys())}"
            )
        return result
