from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from chatbot_plugin_sdk.rate_limit import RateLimitExhausted

if TYPE_CHECKING:
    from chatbot_plugin_sdk.rate_limit import RateLimitStrategy

logger = logging.getLogger(__name__)

# Retry delays beyond this threshold are treated as a daily quota exhaustion —
# waiting would block the pipeline for hours, so we let the error propagate.
_MAX_RETRYABLE_DELAY_SECS = 300.0


def _parse_retry_delay(exc: Exception) -> float | None:
    """Extract the Google-suggested retry delay (seconds) from a 429 error.

    Scans the exception message for ``'retry in Xs'`` or the ``retryDelay`` field.
    Returns ``None`` when no parseable delay is found.
    """
    try:
        msg = str(exc)
        m = re.search(r'retry(?:\s+in)?\s+(\d+(?:\.\d+)?)s', msg, re.IGNORECASE)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def _is_quota_error(exc: Exception) -> bool:
    return "429" in str(exc) or getattr(exc, "status_code", None) == 429


def _is_daily_quota_error(exc: Exception) -> bool:
    """True when the 429 is Google's daily (RPD) quota, not a per-minute one.

    Google's structured error body (embedded in ``str(exc)`` for
    ``google.genai.errors.APIError``) reports a ``QuotaFailure`` violation
    whose ``quotaId`` contains ``PerDay`` for daily-cap errors. This is more
    reliable than guessing from ``retryDelay`` size — a daily quota error can
    still carry a short suggested delay, which would otherwise be retried.
    """
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg and "PerDay" in msg


class GeminiDenseProvider:
    """Dense embedding provider backed by Google Gemini (google-genai).

    ``google-genai`` is loaded lazily; ``ImportError`` propagates to the caller
    if the ``gemini`` optional dependency group is not installed.

    When Google returns HTTP 429 the provider sleeps for the suggested
    ``retryDelay`` (parsed from the error response) and retries transparently.
    A daily (RPD) quota violation — detected from the ``QuotaFailure`` detail
    in the error body, not the delay's length — raises ``RateLimitExhausted``
    immediately instead of retrying, since it won't recover within the run.
    Delays longer than 5 minutes are treated the same way as a fallback.

    Args:
        api_key: Gemini API key.
        model: Embedding model name (default: ``gemini-embedding-001``).
        dimension: Output vector dimension (default: 768).
        rate_limit: Optional rate-limiting strategy (e.g. ``SlidingWindowStrategy``).
                    Construct it in the caller; use ``build_dense_provider`` for the
                    standard ``rpm / tpm / rpd`` → strategy conversion.
        max_retries: How many times to retry on 429 before giving up (default: 5).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
        rate_limit: "RateLimitStrategy | None" = None,
        max_retries: int = 5,
    ) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.dimension: int = dimension
        self._rate_limit = rate_limit
        self._max_retries = max_retries

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
        for attempt in range(self._max_retries):
            try:
                return await loop.run_in_executor(None, self._embed_sync, texts)
            except Exception as exc:
                if not _is_quota_error(exc):
                    raise

                if _is_daily_quota_error(exc):
                    logger.error(
                        "gemini_daily_quota_exhausted",
                        extra={"model": self._model},
                    )
                    raise RateLimitExhausted(
                        f"Daily quota exceeded for {self._model}"
                    ) from exc

                delay = _parse_retry_delay(exc)
                if delay is None or delay > _MAX_RETRYABLE_DELAY_SECS:
                    logger.error(
                        "gemini_daily_quota_exhausted",
                        extra={"delay": delay, "model": self._model},
                    )
                    raise

                if attempt >= self._max_retries - 1:
                    logger.error(
                        "gemini_rate_limit_max_retries_exceeded",
                        extra={"attempts": self._max_retries, "model": self._model},
                    )
                    raise

                logger.warning(
                    "gemini_rate_limited_retrying",
                    extra={"delay": delay, "attempt": attempt + 1, "max": self._max_retries},
                )
                await asyncio.sleep(delay)

        raise RuntimeError("unreachable")
