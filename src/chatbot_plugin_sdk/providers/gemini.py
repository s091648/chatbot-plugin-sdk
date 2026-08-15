from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from chatbot_plugin_sdk.exceptions import EmbeddingError
from chatbot_plugin_sdk.rate_limit import RateLimitExhausted

if TYPE_CHECKING:
    from chatbot_plugin_sdk.rate_limit import RateLimitStrategy

logger = logging.getLogger(__name__)

# Retry delays beyond this threshold are treated as a daily quota exhaustion —
# waiting would block the pipeline for hours, so we let the error propagate.
_MAX_RETRYABLE_DELAY_SECS = 300.0

# Fallback wait when a 429 is confirmed non-daily (RPM/TPM/unknown) but its
# body carries no parseable delay at all (e.g. Google's response includes
# only a `google.rpc.Help` link, no `RetryInfo`) — retried the same as a
# parsed delay would be, since the only alternative (giving up immediately)
# throws away recoverable requests: a same-process quota this narrow
# typically clears within seconds, not hours.
_DEFAULT_QUOTA_BACKOFF_SECS = 15.0


def _parse_retry_delay(exc: Exception) -> float | None:
    """Extract the Google-suggested retry delay (seconds) from a 429 error.

    Scans the exception message for the structured ``retryDelay`` field as
    rendered in google-genai's ``ClientError`` string form (e.g.
    ``"retryDelay": "13s"``) or the prose form ``'retry in Xs'``. Returns
    ``None`` when no parseable delay is found.
    """
    try:
        msg = str(exc)
        m = re.search(r'retryDelay["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)s', msg, re.IGNORECASE)
        if m:
            return float(m.group(1))
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


def _is_token_quota_error(exc: Exception) -> bool:
    """True when the 429 is a per-minute TOKEN quota (TPM), not RPD or RPM.

    Google's ``QuotaFailure.violations[].quotaId`` names the dimension that
    was exceeded — token-based quotas contain ``Token`` (e.g.
    ``GenerateContentInputTokensPerModelPerMinute-FreeTier``), while
    request-count quotas (RPM) contain ``Requests`` instead. Checked after
    :func:`_is_daily_quota_error` so a daily token cap (which also contains
    ``Token``) is still classified as RPD, not TPM.
    """
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" not in msg or "PerDay" in msg:
        return False
    return "token" in msg.lower()


def _quota_dimension(exc: Exception) -> str:
    """Names which Google quota dimension a 429 violated, for logging.

    Built on the same ``QuotaFailure.violations[].quotaId`` substring checks
    as :func:`_is_daily_quota_error`/:func:`_is_token_quota_error` (kept as
    separate booleans there since each gates different retry behavior) —
    this just labels the result so log lines can show *which* of RPD/TPM/RPM
    was hit instead of only whether a retry delay was parseable. Returns
    ``"unknown"`` for a 429 whose error body doesn't carry a recognizable
    quotaId (e.g. Google changes the format, or a non-``RESOURCE_EXHAUSTED``
    429 reaches here via the plain ``"429"`` substring check in
    :func:`_is_quota_error`).
    """
    if _is_daily_quota_error(exc):
        return "rpd"
    if _is_token_quota_error(exc):
        return "tpm"
    if "RESOURCE_EXHAUSTED" in str(exc):
        return "rpm"
    return "unknown"


class GeminiDenseProvider:
    """Dense embedding provider backed by Google Gemini (google-genai).

    ``google-genai`` is loaded lazily; ``ImportError`` propagates to the caller
    if the ``gemini`` optional dependency group is not installed.

    When Google returns HTTP 429 the provider sleeps for the suggested
    ``retryDelay`` (parsed from the error response) and retries transparently.
    A daily (RPD) quota violation — detected from the ``QuotaFailure`` detail
    in the error body, not the delay's length — raises ``RateLimitExhausted``
    immediately instead of retrying, since it won't recover within the run.
    The instance also latches: once a daily quota 429 is seen, every later
    ``embed()`` call in the same process raises ``RateLimitExhausted``
    immediately without making an API call, since Google's daily cap is
    tracked server-side across the whole account/day — it will not clear
    before this process exits. There is deliberately no cross-process
    persistence (Postgres/Redis) for this: the account-wide reset time isn't
    reliably known, so a stored "still exhausted" flag would have no correct
    time to flip back off. Re-checking once per process (this latch) and
    trusting the next process's first real call to re-probe Google avoids
    that problem entirely.
    A per-minute TOKEN quota (TPM) violation, also detected from the
    ``QuotaFailure`` detail, is retried the same way by default; passing
    ``split_batch_on_tpm=True`` makes it instead wait out the suggested delay
    *and* halve the batch before retrying each half — waiting alone doesn't
    help when the batch itself is the problem, and halving alone doesn't help
    if requests are still fired back-to-back, so the two are combined.
    A 429 confirmed as non-daily (RPM/TPM/unknown) but with no parseable
    ``retryDelay`` (Google's error body doesn't always include one — e.g. it
    may carry only a ``google.rpc.Help`` link) falls back to a fixed
    ``_DEFAULT_QUOTA_BACKOFF_SECS`` wait and retries like any other RPM/TPM
    429, instead of giving up: the only evidence available (``dimension !=
    rpd``) says this is expected to clear on its own, so treating "can't
    parse a delay" as fatal would throw away recoverable requests. A delay
    Google *does* supply that exceeds 5 minutes, or repeated 429s past
    ``max_retries``, are raised as ``RateLimitExhausted`` rather than the raw
    ``google.genai`` exception, so every quota-exhaustion path is catchable
    by callers as one type. Any other failure (network error, malformed
    response, auth failure) is raised as ``EmbeddingError``, never the raw
    SDK/HTTP exception.

    Args:
        api_key: Gemini API key.
        model: Embedding model name (default: ``gemini-embedding-001``).
        dimension: Output vector dimension (default: 768).
        rate_limit: Optional rate-limiting strategy (e.g. ``SlidingWindowStrategy``).
                    Construct it in the caller; use ``build_dense_provider`` for the
                    standard ``rpm / tpm / rpd`` → strategy conversion.
        max_retries: How many times to retry on 429 before giving up (default: 5).
        split_batch_on_tpm: When a 429 is identified as a per-minute TOKEN
                    quota (TPM) and the batch has more than one text, wait out
                    the suggested delay then split the batch in half and retry
                    each half independently (recursing further if still too
                    large) instead of retrying the full batch unchanged.
                    Default ``False`` — opt in only if TPM 429s are observed;
                    the default RPM-style wait-and-retry already recovers once
                    the sliding window resets.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
        rate_limit: "RateLimitStrategy | None" = None,
        max_retries: int = 5,
        split_batch_on_tpm: bool = False,
    ) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.dimension: int = dimension
        self._rate_limit = rate_limit
        self._max_retries = max_retries
        self._split_batch_on_tpm = split_batch_on_tpm
        self._daily_exhausted = False

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config={"task_type": "CLASSIFICATION", "output_dimensionality": self.dimension},
        )
        return [list(e.values) for e in response.embeddings]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._daily_exhausted:
            logger.warning(
                "gemini_daily_quota_skip",
                extra={"model": self._model},
            )
            raise RateLimitExhausted(
                f"Daily quota already exhausted for {self._model} this run"
            )

        if self._rate_limit is not None:
            estimated_tokens = max(1, sum(len(t) for t in texts) // 4)
            await self._rate_limit.acquire(estimated_tokens, request_units=len(texts))

        loop = asyncio.get_event_loop()
        for attempt in range(self._max_retries):
            try:
                return await loop.run_in_executor(None, self._embed_sync, texts)
            except Exception as exc:
                if not _is_quota_error(exc):
                    raise EmbeddingError(
                        f"Gemini embedding request failed: {exc}"
                    ) from exc

                if _is_daily_quota_error(exc):
                    self._daily_exhausted = True
                    logger.error(
                        "gemini_daily_quota_exhausted",
                        extra={"model": self._model, "quota_dimension": "rpd"},
                    )
                    raise RateLimitExhausted(
                        f"Daily quota exceeded for {self._model}"
                    ) from exc

                delay = _parse_retry_delay(exc)
                dimension = _quota_dimension(exc)

                if (
                    self._split_batch_on_tpm
                    and len(texts) > 1
                    and _is_token_quota_error(exc)
                ):
                    logger.warning(
                        "gemini_tpm_quota_split",
                        extra={
                            "batch_size": len(texts),
                            "delay": delay,
                            "model": self._model,
                            "quota_dimension": dimension,
                        },
                    )
                    if delay is not None and delay <= _MAX_RETRYABLE_DELAY_SECS:
                        await asyncio.sleep(delay)
                    mid = len(texts) // 2
                    left = await self.embed(texts[:mid])
                    right = await self.embed(texts[mid:])
                    return left + right

                if delay is not None and delay > _MAX_RETRYABLE_DELAY_SECS:
                    logger.error(
                        "gemini_quota_delay_too_long",
                        extra={"delay": delay, "model": self._model, "quota_dimension": dimension},
                    )
                    raise RateLimitExhausted(
                        f"Quota exceeded for {self._model} with retry delay {delay}s "
                        f"exceeding the {_MAX_RETRYABLE_DELAY_SECS}s threshold "
                        f"(dimension={dimension})"
                    ) from exc

                if delay is None:
                    # Confirmed non-daily (the RPD branch above already returned/raised),
                    # but Google's body carried no parseable delay — assume it's still
                    # recoverable and back off with a fixed wait rather than giving up.
                    delay = _DEFAULT_QUOTA_BACKOFF_SECS
                    logger.warning(
                        "gemini_quota_no_retryable_delay_fallback_backoff",
                        extra={
                            "fallback_delay": delay,
                            "model": self._model,
                            "quota_dimension": dimension,
                        },
                    )

                if attempt >= self._max_retries - 1:
                    logger.error(
                        "gemini_rate_limit_max_retries_exceeded",
                        extra={
                            "attempts": self._max_retries,
                            "model": self._model,
                            "quota_dimension": dimension,
                        },
                    )
                    raise RateLimitExhausted(
                        f"Quota exceeded for {self._model} after {self._max_retries} retries"
                    ) from exc

                logger.warning(
                    "gemini_rate_limited_retrying",
                    extra={
                        "delay": delay,
                        "attempt": attempt + 1,
                        "max": self._max_retries,
                        "quota_dimension": dimension,
                    },
                )
                await asyncio.sleep(delay)

        raise RuntimeError("unreachable")
