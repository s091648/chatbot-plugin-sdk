from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from chatbot_plugin_sdk.exceptions import EmbeddingError
from chatbot_plugin_sdk.protocols import Tracer, default_tracer
from chatbot_plugin_sdk.rate_limit import (
    RateLimitExhausted,
    RpdExhausted,
    RpmExhausted,
    TpmExhausted,
    estimate_tokens,
)

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

# Backoff between retries of a transient 503 ("The model is overloaded,
# please try again later") / 502 — not a quota condition (no RPD/RPM/TPM
# dimension applies), just Google-side capacity that usually clears within
# seconds, so a short fixed wait is enough.
_OVERLOAD_BACKOFF_SECS = 5.0


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
    """True for a 429.

    Prefers the structured ``.code`` attribute that
    ``google.genai.errors.APIError`` actually sets before falling back to a
    substring scan for exceptions that aren't a genai ``APIError`` at all.
    Previously checked ``.status_code`` instead of ``.code`` — no
    google-genai exception has ever exposed that name, so that check was
    dead code, silently masked because the substring check below usually
    also matched.
    """
    if getattr(exc, "code", None) == 429:
        return True
    return "429" in str(exc)


def _is_overloaded_error(exc: Exception) -> bool:
    """True for Gemini's transient 503 ("The model is overloaded, please try
    again later") or a 502 — neither carries a 429 status, so
    :func:`_is_quota_error` never matches them. Before this, such errors fell
    straight through ``GeminiDenseProvider.embed()``'s quota-handling branch
    into a hard, immediate ``EmbeddingError`` with no retry at all — in
    production this looked exactly like "rate limiting stopped working",
    when the real cause was that the error never reached the rate-limit
    machinery in the first place. These aren't a quota condition (no
    RPD/RPM/TPM dimension applies), just transient capacity errors on
    Google's side that usually clear within seconds — handled as a plain
    retry-with-backoff, independent of the 429 quota path below.
    """
    if getattr(exc, "code", None) in (502, 503):
        return True
    msg = str(exc)
    return "503" in msg or "502" in msg or "UNAVAILABLE" in msg


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


def _is_request_quota_error(exc: Exception) -> bool:
    """True when the 429 is a per-minute REQUEST-COUNT quota (RPM), not RPD or TPM.

    Positive counterpart to :func:`_is_token_quota_error` — request-count
    quotas' ``quotaId`` contains ``Requests`` (e.g.
    ``GenerateRequestsPerMinutePerProjectPerModel-FreeTier``), the substring
    :func:`_quota_dimension`'s docstring already documented this module as
    relying on, but which (before this) was never actually checked for —
    "not daily, not token" was silently treated as "must be rpm" instead.
    Checked after :func:`_is_daily_quota_error` so a daily request cap
    (which also contains ``Requests``) stays classified as RPD, not RPM.
    """
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" not in msg or "PerDay" in msg:
        return False
    return "request" in msg.lower()


def _dimension_from_local_headroom(rate_limit: "RateLimitStrategy | None") -> str:
    """Disambiguates RPM vs TPM for a 429 whose body carries no classifiable
    ``QuotaFailure`` detail at all — confirmed in production (scrape-analyzer,
    2026-09-22 RAG ingest incident): some accounts/tiers' 429 responses
    include only a generic ``google.rpc.Help`` link, no ``violations[].quotaId``
    whatsoever, so none of :func:`_is_daily_quota_error`/
    :func:`_is_token_quota_error`/:func:`_is_request_quota_error` can match.
    Blindly assuming "rpm" in that case (the historical behavior) was itself
    the bug: it let a real, repeated TPM exhaustion keep retrying the same
    full-size batch under ``split_batch_on_tpm`` instead of ever shrinking it,
    because ``dimension`` was never "tpm" for that account's error shape.

    Falls back to the provider's own local ``SlidingWindowStrategy.headroom()``
    instead: a limiter configured to mirror the real account quota (the
    intended way to use this SDK — see ``build_dense_provider``'s ``rpm``/
    ``tpm`` args) already knows, independent of what Google's response body
    does or doesn't disclose, which dimension it's currently closer to
    exhausting. Reports whichever of RPM/TPM has proportionally less headroom
    left *locally* — not a guess, but the same live accounting
    :class:`~chatbot_plugin_sdk.batching.EmbeddingBatchCoordinator` already
    trusts to size batches before dispatching them.

    Still falls back to the historical ``"rpm"`` default when no ``rate_limit``
    was given, it exposes no ``headroom()`` (a custom ``RateLimitStrategy``
    predating it), or ``headroom()`` itself raises — no local signal available
    is not a reason to guess differently than before.
    """
    headroom_fn = getattr(rate_limit, "headroom", None) if rate_limit is not None else None
    if headroom_fn is None:
        return "rpm"
    try:
        remaining_units, remaining_tokens = headroom_fn()
    except Exception:
        return "rpm"
    rpm_cap = getattr(rate_limit, "rpm", 0) or 0
    tpm_cap = getattr(rate_limit, "tpm", 0) or 0
    rpm_frac = (remaining_units / rpm_cap) if rpm_cap > 0 else 1.0
    tpm_frac = (remaining_tokens / tpm_cap) if tpm_cap > 0 else 1.0
    return "tpm" if tpm_frac < rpm_frac else "rpm"


def _quota_dimension(exc: Exception, rate_limit: "RateLimitStrategy | None" = None) -> str:
    """Names which Google quota dimension a 429 violated, for logging *and*
    (as of this fix) for gating ``split_batch_on_tpm`` — see ``GeminiDenseProvider.embed()``.

    Built on the same ``QuotaFailure.violations[].quotaId`` substring checks
    as :func:`_is_daily_quota_error`/:func:`_is_token_quota_error`/
    :func:`_is_request_quota_error` (kept as separate booleans there since
    each gates different retry behavior) — this labels the result so log
    lines and callers can see *which* of RPD/TPM/RPM was hit. When none of
    the three match (Google's body carries no recognizable ``quotaId`` at
    all, not merely an unrecognized one) falls back to
    :func:`_dimension_from_local_headroom` rather than assuming "rpm"
    outright. Returns ``"unknown"`` only when the 429 isn't even
    ``RESOURCE_EXHAUSTED``-shaped (e.g. a non-quota 429 reaches here via the
    plain ``"429"`` substring check in :func:`_is_quota_error`).

    ``rate_limit``: the calling provider's own configured rate limiter
    (``self._rate_limit``) — optional, purely to feed the headroom fallback;
    omit it to get the pre-fix behavior for the unclassifiable case.
    """
    if _is_daily_quota_error(exc):
        return "rpd"
    if _is_token_quota_error(exc):
        return "tpm"
    if _is_request_quota_error(exc):
        return "rpm"
    if "RESOURCE_EXHAUSTED" not in str(exc):
        return "unknown"
    return _dimension_from_local_headroom(rate_limit)


_DIMENSION_EXC: dict[str, type[RateLimitExhausted]] = {
    "rpd": RpdExhausted,
    "rpm": RpmExhausted,
    "tpm": TpmExhausted,
}


def _exc_for_dimension(dimension: str) -> "type[RateLimitExhausted]":
    """Map a _quota_dimension() result to the typed exception to raise —
    RateLimitExhausted itself (the base class) for "unknown", so a 429 whose
    dimension can't be determined is still treated conservatively (fails
    just this call, doesn't trip EmbeddingBatchCoordinator's RPD breaker)."""
    return _DIMENSION_EXC.get(dimension, RateLimitExhausted)


class GeminiDenseProvider:
    """Dense embedding provider backed by Google Gemini (google-genai).

    ``google-genai`` is loaded lazily; ``ImportError`` propagates to the caller
    if the ``gemini`` optional dependency group is not installed.

    When Google returns HTTP 429 the provider sleeps for the suggested
    ``retryDelay`` (parsed from the error response) and retries transparently.
    A daily (RPD) quota violation — detected from the ``QuotaFailure`` detail
    in the error body, not the delay's length — raises ``RpdExhausted``
    immediately instead of retrying, since it won't recover within the run.
    The instance also latches: once a daily quota 429 is seen, every later
    ``embed()`` call in the same process raises ``RpdExhausted``
    immediately without making an API call, since Google's daily cap is
    tracked server-side across the whole account/day — it will not clear
    before this process exits. There is deliberately no cross-process
    persistence (Postgres/Redis) for this: the account-wide reset time isn't
    reliably known, so a stored "still exhausted" flag would have no correct
    time to flip back off. Re-checking once per process (this latch) and
    trusting the next process's first real call to re-probe Google avoids
    that problem entirely.
    A per-minute TOKEN quota (TPM) violation is retried the same way by
    default; passing ``split_batch_on_tpm=True`` makes it instead wait out
    the suggested delay *and* halve the batch before retrying each half —
    waiting alone doesn't help when the batch itself is the problem, and
    halving alone doesn't help if requests are still fired back-to-back, so
    the two are combined. TPM detection prefers the ``QuotaFailure`` detail
    in Google's error body when present, but falls back to the provider's
    own local rate-limiter headroom (:func:`_dimension_from_local_headroom`)
    when that detail is absent entirely — some accounts/tiers' 429 responses
    carry only a generic ``google.rpc.Help`` link, no structured quota
    detail, which previously made every such 429 default to "rpm" and never
    engage ``split_batch_on_tpm`` even when the real, repeated cause was TPM.
    A 429 confirmed as non-daily (RPM/TPM/unknown) but with no parseable
    ``retryDelay`` (Google's error body doesn't always include one — e.g. it
    may carry only a ``google.rpc.Help`` link) falls back to a fixed
    ``_DEFAULT_QUOTA_BACKOFF_SECS`` wait and retries like any other RPM/TPM
    429, instead of giving up: the only evidence available (``dimension !=
    rpd``) says this is expected to clear on its own, so treating "can't
    parse a delay" as fatal would throw away recoverable requests. A delay
    Google *does* supply that exceeds 5 minutes, or repeated 429s past
    ``max_retries``, are raised as the ``RateLimitExhausted`` subclass
    matching the violated dimension (``RpdExhausted``/``RpmExhausted``/
    ``TpmExhausted``, or the plain base class if the dimension couldn't be
    determined) rather than the raw ``google.genai`` exception, so every
    quota-exhaustion path is catchable by callers as one type — or as a
    specific dimension, e.g. to circuit-break only on ``RpdExhausted``. A
    transient 503 ("model is overloaded") or 502 — never shaped as a 429, so
    it wouldn't otherwise be recognized as retryable at all — is retried up
    to ``max_retries`` with a fixed short backoff and, if still failing after
    that, raised as ``EmbeddingError`` (not a ``RateLimitExhausted``
    subclass: it isn't a quota condition, no RPD/RPM/TPM dimension applies).
    Any other failure (network error, malformed response, auth failure) is
    raised as ``EmbeddingError`` immediately, never the raw SDK/HTTP
    exception.

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
        tracer: Optional Tracer-protocol implementation to route this
                    provider's spans into your own already-configured
                    OpenTelemetry TracerProvider (or any other Tracer
                    implementation). Omitted, this uses real OTel if
                    opentelemetry-api is installed, else a no-op — see
                    ``chatbot_plugin_sdk.protocols.default_tracer``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
        rate_limit: "RateLimitStrategy | None" = None,
        max_retries: int = 5,
        split_batch_on_tpm: bool = False,
        tracer: Tracer | None = None,
    ) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.dimension: int = dimension
        self._rate_limit = rate_limit
        self._max_retries = max_retries
        self._split_batch_on_tpm = split_batch_on_tpm
        self._daily_exhausted = False
        self._tracer: Tracer = tracer or default_tracer(__name__)

    @property
    def rate_limit(self) -> "RateLimitStrategy | None":
        """Public read access to the configured rate limiter — lets a caller
        like EmbeddingBatchCoordinator query current headroom (via
        ``rate_limit.headroom()``) before forming a batch, without the
        coordinator needing to know this is a Gemini provider specifically."""
        return self._rate_limit

    def _embed_sync(self, texts: list[str]) -> "tuple[list[list[float]], int | None]":
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config={"task_type": "CLASSIFICATION", "output_dimensionality": self.dimension},
        )
        vectors = [list(e.values) for e in response.embeddings]
        return vectors, self._extract_actual_tokens(response)

    @staticmethod
    def _extract_actual_tokens(response) -> "int | None":
        """Sum Google's own real per-chunk token count
        (``EmbedContentResponse.embeddings[i].statistics.token_count``) when
        every embedding in the response carries one, so a successful call can
        feed ``rate_limit.record_usage()`` the real figure instead of leaving
        the TPM window holding ``estimate_tokens()``'s chars/4 approximation
        for that call indefinitely. Returns ``None`` (skip the correction,
        don't guess) if the field is absent on any embedding — API version
        drift or a mocked/partial response must not silently record a
        misleading total."""
        embeddings = getattr(response, "embeddings", None) or []
        if not embeddings:
            return None
        counts: list[int] = []
        for e in embeddings:
            stats = getattr(e, "statistics", None)
            token_count = getattr(stats, "token_count", None) if stats is not None else None
            if token_count is None:
                return None
            counts.append(token_count)
        return sum(counts)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._daily_exhausted:
            logger.warning(
                "gemini_daily_quota_skip",
                extra={"model": self._model},
            )
            raise RpdExhausted(
                f"Daily quota already exhausted for {self._model} this run"
            )

        # embed() is a plain coroutine (no yield to a caller mid-span), so
        # start_as_current_span() as a `with` is safe here — unlike the
        # generator-based spans elsewhere in this codebase (see e.g.
        # chatbot-plugin's chat_service.py), there's no risk of resuming in a
        # different contextvars.Context.
        #
        # _embed_sync() runs the actual Gemini API call via the *synchronous*
        # google-genai client, offloaded to the default ThreadPoolExecutor via
        # run_in_executor — asyncio does NOT propagate the current
        # contextvars.Context into that worker thread, so any span the sync
        # client's own HTTP layer might create is invisible/orphaned from
        # this trace. The "embed_sync_done" event on *this* span is the only
        # place that call's real duration is actually visible.
        with self._tracer.start_as_current_span(
            "gemini_dense.embed", attributes={"model": self._model, "text_count": len(texts)},
        ) as span:
            if self._rate_limit is not None:
                estimated_tokens = estimate_tokens(texts)
                await self._rate_limit.acquire(estimated_tokens, request_units=len(texts))
                span.add_event("rate_limit_acquired")

            loop = asyncio.get_event_loop()
            for attempt in range(self._max_retries):
                try:
                    result, actual_tokens = await loop.run_in_executor(None, self._embed_sync, texts)
                    span.add_event("embed_sync_done")
                    if self._rate_limit is not None and actual_tokens is not None:
                        self._rate_limit.record_usage(actual_tokens)
                    return result
                except Exception as exc:
                    if _is_overloaded_error(exc):
                        if attempt >= self._max_retries - 1:
                            logger.error(
                                "gemini_overloaded_max_retries_exceeded",
                                extra={"attempts": self._max_retries, "model": self._model},
                            )
                            raise EmbeddingError(
                                f"Gemini embedding request failed after "
                                f"{self._max_retries} retries (model overloaded): {exc}"
                            ) from exc
                        logger.warning(
                            "gemini_overloaded_retrying",
                            extra={
                                "attempt": attempt + 1,
                                "max": self._max_retries,
                                "delay": _OVERLOAD_BACKOFF_SECS,
                            },
                        )
                        await asyncio.sleep(_OVERLOAD_BACKOFF_SECS)
                        continue

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
                        raise RpdExhausted(
                            f"Daily quota exceeded for {self._model}"
                        ) from exc

                    delay = _parse_retry_delay(exc)
                    dimension = _quota_dimension(exc, self._rate_limit)

                    if (
                        self._split_batch_on_tpm
                        and len(texts) > 1
                        and dimension == "tpm"
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
                        raise _exc_for_dimension(dimension)(
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
                        raise _exc_for_dimension(dimension)(
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
