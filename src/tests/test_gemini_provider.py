"""Tests for GeminiDenseProvider's 429 classification and retry/split behavior."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin_sdk.exceptions import EmbeddingError
from chatbot_plugin_sdk.providers.gemini import (
    GeminiDenseProvider,
    _is_daily_quota_error,
    _is_overloaded_error,
    _is_quota_error,
    _is_request_quota_error,
    _is_token_quota_error,
    _parse_retry_delay,
    _quota_dimension,
)
from chatbot_plugin_sdk.rate_limit import RateLimitExhausted, RpdExhausted, RpmExhausted, TpmExhausted


def _quota_exc(quota_id: str, delay: float | None = None) -> Exception:
    """Builds a fake exception mimicking google.genai's 429 error string."""
    parts = ["429 RESOURCE_EXHAUSTED.", f"'quotaId': '{quota_id}'"]
    if delay is not None:
        parts.append(f"Please retry in {delay}s.")
    return Exception(" ".join(parts))


DAILY_EXC = _quota_exc("EmbedContentRequestsPerDayPerProjectPerModel-FreeTier")
RPM_EXC = _quota_exc("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", delay=5)
TPM_EXC = _quota_exc("GenerateContentInputTokensPerModelPerMinute-FreeTier", delay=5)

# Real-world shape (scrape-analyzer production incident, 2026-09-22): Google's 429 body
# for this account/tier carries no QuotaFailure/violations/quotaId at all — only a
# generic google.rpc.Help link. None of _is_daily_quota_error/_is_token_quota_error/
# _is_request_quota_error can match this; it's what exercises the headroom fallback.
UNCLASSIFIED_EXC = Exception(
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
    "'You exceeded your current quota, please check your plan and "
    "billing details.', 'status': 'RESOURCE_EXHAUSTED', 'details': "
    "[{'@type': 'type.googleapis.com/google.rpc.Help', 'links': []}]}}"
)


class _FakeHeadroomLimiter:
    """Minimal stand-in for SlidingWindowStrategy — enough surface for
    _dimension_from_local_headroom (rpm/tpm caps + a headroom() reading), plus
    a no-op acquire() so it also satisfies RateLimitStrategy when passed to a
    real GeminiDenseProvider (embed() calls acquire() unconditionally whenever
    rate_limit is not None, before dimension classification is ever reached)."""

    def __init__(self, remaining_units: float, remaining_tokens: float, rpm: int = 100, tpm: int = 30_000):
        self.rpm = rpm
        self.tpm = tpm
        self._remaining_units = remaining_units
        self._remaining_tokens = remaining_tokens

    def headroom(self):
        return self._remaining_units, self._remaining_tokens

    async def acquire(self, estimated_tokens: int = 0, request_units: int = 1) -> None:
        return None


class _FakeApiError(Exception):
    """Minimal stand-in for google.genai.errors.APIError's structured shape —
    carries a real ``.code`` attribute (the name genai actually uses; NOT
    ``.status_code``) independent of its message text, so tests can prove
    _is_quota_error/_is_overloaded_error read ``.code`` and don't merely get
    lucky on a substring match against str(exc)."""

    def __init__(self, code: int, message: str = "boom"):
        self.code = code
        super().__init__(message)


# ── Classification helpers ──────────────────────────────────────────────────

class TestClassification:
    def test_daily_quota_detected(self):
        assert _is_daily_quota_error(DAILY_EXC) is True

    def test_rpm_not_daily(self):
        assert _is_daily_quota_error(RPM_EXC) is False

    def test_tpm_not_daily(self):
        assert _is_daily_quota_error(TPM_EXC) is False

    def test_tpm_detected(self):
        assert _is_token_quota_error(TPM_EXC) is True

    def test_rpm_not_token(self):
        assert _is_token_quota_error(RPM_EXC) is False

    def test_daily_token_cap_still_classified_as_daily_not_tpm(self):
        # A daily quota whose quotaId also happens to mention "Tokens" must
        # stay in the RPD bucket (immediate skip), not fall into TPM (split).
        daily_token_exc = _quota_exc("InputTokensPerDayPerProjectPerModel-FreeTier")
        assert _is_daily_quota_error(daily_token_exc) is True
        assert _is_token_quota_error(daily_token_exc) is False

    def test_request_quota_detected(self):
        assert _is_request_quota_error(RPM_EXC) is True

    def test_tpm_not_request(self):
        assert _is_request_quota_error(TPM_EXC) is False

    def test_daily_request_cap_still_classified_as_daily_not_rpm(self):
        # Mirrors test_daily_token_cap_still_classified_as_daily_not_tpm —
        # DAILY_EXC's own quotaId ("...RequestsPerDay...") contains "Requests"
        # too; the RPD check must still win.
        assert _is_daily_quota_error(DAILY_EXC) is True
        assert _is_request_quota_error(DAILY_EXC) is False

    def test_unclassified_exc_matches_none_of_the_three_structured_checks(self):
        # Sanity check on the fixture itself — UNCLASSIFIED_EXC is only a
        # useful regression fixture if it genuinely can't be classified by
        # any of the structured quotaId checks.
        assert _is_daily_quota_error(UNCLASSIFIED_EXC) is False
        assert _is_token_quota_error(UNCLASSIFIED_EXC) is False
        assert _is_request_quota_error(UNCLASSIFIED_EXC) is False

    def test_parse_retry_delay(self):
        assert _parse_retry_delay(RPM_EXC) == 5.0

    def test_parse_retry_delay_missing(self):
        assert _parse_retry_delay(DAILY_EXC) is None

    def test_parse_retry_delay_structured_retry_delay_field(self):
        # google-genai's ClientError renders its structured error body as a
        # dict literal in str(exc) — e.g. "'retryDelay': '13s'" — which the
        # prose-only "retry in Xs" regex does not match.
        exc = Exception(
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'status': "
            "'RESOURCE_EXHAUSTED', 'details': [{'@type': "
            "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '13s'}]}}"
        )
        assert _parse_retry_delay(exc) == 13.0

    def test_parse_retry_delay_no_retry_info_at_all(self):
        # Real-world case: Google's 429 body carries only a Help link, no
        # RetryInfo and no prose delay — must resolve to None, not raise.
        exc = Exception(
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
            "'You exceeded your current quota, please check your plan and "
            "billing details.', 'status': 'RESOURCE_EXHAUSTED', 'details': "
            "[{'@type': 'type.googleapis.com/google.rpc.Help', 'links': []}]}}"
        )
        assert _parse_retry_delay(exc) is None

    def test_quota_dimension_rpd(self):
        assert _quota_dimension(DAILY_EXC) == "rpd"

    def test_quota_dimension_tpm(self):
        assert _quota_dimension(TPM_EXC) == "tpm"

    def test_quota_dimension_rpm(self):
        assert _quota_dimension(RPM_EXC) == "rpm"

    def test_quota_dimension_falls_back_to_rpm_for_other_resource_exhausted_quota_ids(self):
        # Any RESOURCE_EXHAUSTED quotaId that isn't a Day or Token cap is
        # assumed request-count-based (RPM), mirroring _is_token_quota_error's
        # "request-count quotas contain Requests instead" documented split.
        assert _quota_dimension(_quota_exc("SomeNewQuotaShape")) == "rpm"

    def test_quota_dimension_unknown_for_non_resource_exhausted_429(self):
        assert _quota_dimension(Exception("429 Too Many Requests")) == "unknown"


class TestQuotaDimensionHeadroomFallback:
    """_dimension_from_local_headroom — the fix for the production incident where
    Google's 429 body carries no classifiable QuotaFailure at all (UNCLASSIFIED_EXC)."""

    def test_falls_back_to_rpm_with_no_rate_limit_given(self):
        # Pre-fix behavior, preserved when there's no local signal to consult.
        assert _quota_dimension(UNCLASSIFIED_EXC) == "rpm"

    def test_tight_tpm_headroom_classifies_as_tpm(self):
        # Mirrors the real incident's Google AI Studio reading: RPM 82/100
        # (healthy), TPM 29.6K/30K (nearly exhausted).
        limiter = _FakeHeadroomLimiter(remaining_units=18, remaining_tokens=400, rpm=100, tpm=30_000)
        assert _quota_dimension(UNCLASSIFIED_EXC, limiter) == "tpm"

    def test_tight_rpm_headroom_classifies_as_rpm(self):
        limiter = _FakeHeadroomLimiter(remaining_units=1, remaining_tokens=20_000, rpm=100, tpm=30_000)
        assert _quota_dimension(UNCLASSIFIED_EXC, limiter) == "rpm"

    def test_falls_back_to_rpm_when_rate_limit_has_no_headroom_method(self):
        assert _quota_dimension(UNCLASSIFIED_EXC, object()) == "rpm"

    def test_falls_back_to_rpm_when_headroom_raises(self):
        class _BoomLimiter:
            def headroom(self):
                raise RuntimeError("boom")
        assert _quota_dimension(UNCLASSIFIED_EXC, _BoomLimiter()) == "rpm"

    def test_structured_classification_wins_over_headroom(self):
        # A structured quotaId (RPD/RPM/TPM) must never be overridden by the
        # headroom guess — headroom is a fallback for the unclassifiable case
        # only, even if it would otherwise "disagree".
        limiter = _FakeHeadroomLimiter(remaining_units=100, remaining_tokens=1, rpm=100, tpm=30_000)
        assert _quota_dimension(DAILY_EXC, limiter) == "rpd"
        assert _quota_dimension(RPM_EXC, limiter) == "rpm"
        assert _quota_dimension(TPM_EXC, limiter) == "tpm"


class TestOverloadAndStructuredCode:
    """_is_quota_error's fixed .code check (was dead-code .status_code) and
    the new _is_overloaded_error — the fix for 503/502 "model overloaded"
    errors that previously bypassed all quota/retry handling entirely and
    hit EmbeddingError immediately, which looked like rate limiting having
    silently stopped working."""

    def test_is_quota_error_reads_structured_code_attribute(self):
        assert _is_quota_error(_FakeApiError(429, "boom")) is True

    def test_is_quota_error_false_for_unrelated_structured_code(self):
        assert _is_quota_error(_FakeApiError(500, "boom")) is False

    def test_is_quota_error_still_matches_429_substring_without_code_attr(self):
        assert _is_quota_error(Exception("429 RESOURCE_EXHAUSTED.")) is True

    def test_is_overloaded_error_reads_structured_code_attribute(self):
        assert _is_overloaded_error(_FakeApiError(503, "boom")) is True
        assert _is_overloaded_error(_FakeApiError(502, "boom")) is True

    def test_is_overloaded_error_false_for_quota_code(self):
        assert _is_overloaded_error(_FakeApiError(429, "boom")) is False

    def test_is_overloaded_error_matches_503_substring_without_code_attr(self):
        exc = Exception(
            "503 UNAVAILABLE. {'error': {'code': 503, 'message': "
            "'The model is overloaded. Please try again later.', "
            "'status': 'UNAVAILABLE'}}"
        )
        assert _is_overloaded_error(exc) is True

    def test_is_overloaded_error_false_for_unrelated_error(self):
        assert _is_overloaded_error(ValueError("boom")) is False


# ── embed() behavior ────────────────────────────────────────────────────────

def _make_provider(**kwargs) -> GeminiDenseProvider:
    return GeminiDenseProvider(api_key="fake-key", **kwargs)


class TestEmbedRetryBehavior:
    @pytest.mark.asyncio
    async def test_non_quota_error_raises_embedding_error(self):
        provider = _make_provider()
        with patch.object(provider, "_embed_sync", side_effect=ValueError("boom")):
            with pytest.raises(EmbeddingError):
                await provider.embed(["a"])

    @pytest.mark.asyncio
    async def test_daily_quota_raises_immediately_without_sleep(self):
        provider = _make_provider()
        with patch.object(provider, "_embed_sync", side_effect=DAILY_EXC), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RpdExhausted, match="Daily quota") as exc_info:
                await provider.embed(["a"])
        mock_sleep.assert_not_called()
        assert exc_info.value.dimension == "rpd"

    @pytest.mark.asyncio
    async def test_daily_quota_latches_and_skips_later_calls_without_api_call(self):
        # A real process only ever needs to learn the daily cap is blown once —
        # every later embed() in the same run should fail fast, no HTTP call.
        provider = _make_provider()
        mock_embed_sync = MagicMock(side_effect=DAILY_EXC)
        with patch.object(provider, "_embed_sync", mock_embed_sync):
            with pytest.raises(RpdExhausted, match="Daily quota exceeded"):
                await provider.embed(["a"])

        assert mock_embed_sync.call_count == 1

        with pytest.raises(RpdExhausted, match="already exhausted"):
            await provider.embed(["b", "c"])

        # No further _embed_sync call for the second, already-latched request.
        assert mock_embed_sync.call_count == 1

    @pytest.mark.asyncio
    async def test_daily_quota_latch_skips_rate_limit_acquire_too(self):
        strategy = AsyncMock()
        strategy.acquire = AsyncMock()
        provider = _make_provider(rate_limit=strategy)
        with patch.object(provider, "_embed_sync", side_effect=DAILY_EXC):
            with pytest.raises(RpdExhausted):
                await provider.embed(["a"])
        strategy.acquire.assert_awaited_once()

        strategy.acquire.reset_mock()
        with pytest.raises(RpdExhausted, match="already exhausted"):
            await provider.embed(["b"])
        strategy.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_rpm_quota_waits_then_succeeds(self):
        provider = _make_provider(max_retries=3)
        calls = {"n": 0}

        def fake_embed_sync(texts):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RPM_EXC
            return [[0.1] * 768 for _ in texts], None

        with patch.object(provider, "_embed_sync", side_effect=fake_embed_sync), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await provider.embed(["a"])

        assert result == [[0.1] * 768]
        mock_sleep.assert_awaited_once_with(5.0)

    @pytest.mark.asyncio
    async def test_tpm_without_split_flag_falls_back_to_wait_and_retry(self):
        # split_batch_on_tpm defaults to False — TPM should behave like RPM.
        provider = _make_provider(max_retries=3, split_batch_on_tpm=False)
        calls = {"n": 0}

        def fake_embed_sync(texts):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TPM_EXC
            return [[0.1] * 768 for _ in texts], None

        with patch.object(provider, "_embed_sync", side_effect=fake_embed_sync), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await provider.embed(["a", "b"])

        assert result == [[0.1] * 768, [0.1] * 768]
        mock_sleep.assert_awaited_once_with(5.0)

    @pytest.mark.asyncio
    async def test_tpm_with_split_flag_halves_batch_and_retries_each_half(self):
        provider = _make_provider(max_retries=3, split_batch_on_tpm=True)
        seen_batches: list[list[str]] = []

        def fake_embed_sync(texts):
            seen_batches.append(list(texts))
            if len(texts) > 2:
                raise TPM_EXC
            return [[0.5] * 768 for _ in texts], None

        with patch.object(provider, "_embed_sync", side_effect=fake_embed_sync), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await provider.embed(["a", "b", "c", "d"])

        assert result == [[0.5] * 768] * 4
        # First call is the full batch (fails), then each half succeeds
        # directly — no further splitting needed.
        assert seen_batches[0] == ["a", "b", "c", "d"]
        assert all(len(b) == 2 for b in seen_batches[1:])
        assert len(seen_batches) == 3
        mock_sleep.assert_awaited_once_with(5.0)

    @pytest.mark.asyncio
    async def test_tpm_with_split_flag_but_single_text_falls_back_to_wait(self):
        # Can't split a batch of 1 — must fall through to the generic
        # wait-and-retry path instead of looping forever.
        provider = _make_provider(max_retries=3, split_batch_on_tpm=True)
        calls = {"n": 0}

        def fake_embed_sync(texts):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TPM_EXC
            return [[0.1] * 768], None

        with patch.object(provider, "_embed_sync", side_effect=fake_embed_sync), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await provider.embed(["a"])

        assert result == [[0.1] * 768]
        mock_sleep.assert_awaited_once_with(5.0)

    @pytest.mark.asyncio
    async def test_unclassified_429_with_tight_tpm_headroom_splits_batch(self):
        """Regression test for the production incident (scrape-analyzer,
        2026-09-22): Google's 429 carries no structured quotaId at all
        (UNCLASSIFIED_EXC), but the provider's own rate limiter reports TPM
        nearly exhausted (matching the real Google AI Studio reading). Before
        this fix, dimension always defaulted to "rpm" here, so
        split_batch_on_tpm never engaged and the same full-size batch was
        retried unchanged every time."""
        limiter = _FakeHeadroomLimiter(remaining_units=18, remaining_tokens=400, rpm=100, tpm=30_000)
        provider = _make_provider(max_retries=3, split_batch_on_tpm=True, rate_limit=limiter)
        seen_batches: list[list[str]] = []

        def fake_embed_sync(texts):
            seen_batches.append(list(texts))
            if len(texts) > 2:
                raise UNCLASSIFIED_EXC
            return [[0.5] * 768 for _ in texts], None

        with patch.object(provider, "_embed_sync", side_effect=fake_embed_sync), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock):
            result = await provider.embed(["a", "b", "c", "d"])

        assert result == [[0.5] * 768] * 4
        assert seen_batches[0] == ["a", "b", "c", "d"]
        assert all(len(b) == 2 for b in seen_batches[1:])
        assert len(seen_batches) == 3

    @pytest.mark.asyncio
    async def test_no_parseable_delay_falls_back_to_default_backoff_and_retries(self):
        # A 429 confirmed non-daily but with no parseable retryDelay must not
        # be treated as fatal — it should back off with the fixed default
        # delay and retry, the same posture as a parsed RPM/TPM delay.
        provider = _make_provider(max_retries=3)
        calls = {"n": 0}
        no_delay_exc = _quota_exc("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")

        def fake_embed_sync(texts):
            calls["n"] += 1
            if calls["n"] == 1:
                raise no_delay_exc
            return [[0.1] * 768 for _ in texts], None

        with patch.object(provider, "_embed_sync", side_effect=fake_embed_sync), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await provider.embed(["a"])

        assert result == [[0.1] * 768]
        mock_sleep.assert_awaited_once_with(15.0)

    @pytest.mark.asyncio
    async def test_no_parseable_delay_exhausts_max_retries_then_raises(self):
        provider = _make_provider(max_retries=2)
        no_delay_exc = _quota_exc("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
        with patch.object(provider, "_embed_sync", side_effect=no_delay_exc), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RpmExhausted, match="after 2 retries") as exc_info:
                await provider.embed(["a"])
        mock_sleep.assert_awaited_once_with(15.0)
        assert exc_info.value.dimension == "rpm"

    @pytest.mark.asyncio
    async def test_delay_over_threshold_raises_immediately(self):
        provider = _make_provider()
        long_delay_exc = _quota_exc(
            "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", delay=301
        )
        with patch.object(provider, "_embed_sync", side_effect=long_delay_exc), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RpmExhausted, match="exceeding the 300.0s threshold"):
                await provider.embed(["a"])
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_delay_over_threshold_on_tpm_dimension_raises_tpm_exhausted(self):
        provider = _make_provider()
        long_delay_exc = _quota_exc(
            "GenerateContentInputTokensPerModelPerMinute-FreeTier", delay=301
        )
        with patch.object(provider, "_embed_sync", side_effect=long_delay_exc), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TpmExhausted) as exc_info:
                await provider.embed(["a"])
        assert exc_info.value.dimension == "tpm"

    @pytest.mark.asyncio
    async def test_exhausts_max_retries_then_raises(self):
        provider = _make_provider(max_retries=2)
        with patch.object(provider, "_embed_sync", side_effect=RPM_EXC), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RpmExhausted, match="after 2 retries"):
                await provider.embed(["a"])
        mock_sleep.assert_awaited_once_with(5.0)

    @pytest.mark.asyncio
    async def test_overloaded_503_retries_then_succeeds(self):
        provider = _make_provider(max_retries=3)
        overloaded_exc = _FakeApiError(503, "The model is overloaded. Please try again later.")
        calls = {"n": 0}

        def fake_embed_sync(texts):
            calls["n"] += 1
            if calls["n"] == 1:
                raise overloaded_exc
            return [[0.1] * 768 for _ in texts], None

        with patch.object(provider, "_embed_sync", side_effect=fake_embed_sync), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await provider.embed(["a"])

        assert result == [[0.1] * 768]
        mock_sleep.assert_awaited_once_with(5.0)

    @pytest.mark.asyncio
    async def test_overloaded_503_exhausts_max_retries_then_raises_embedding_error(self):
        # Not a quota condition — must surface as EmbeddingError, never a
        # RateLimitExhausted subclass (no RPD/RPM/TPM dimension applies).
        provider = _make_provider(max_retries=2)
        overloaded_exc = _FakeApiError(503, "The model is overloaded. Please try again later.")
        with patch.object(provider, "_embed_sync", side_effect=overloaded_exc), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(EmbeddingError, match="overloaded"):
                await provider.embed(["a"])
        assert mock_sleep.await_count == 1

    @pytest.mark.asyncio
    async def test_unknown_dimension_raises_plain_base_class(self):
        """A 429 whose quotaId can't be classified at all (_quota_dimension
        returns "unknown") must raise the plain RateLimitExhausted base class
        — not RpdExhausted, which would incorrectly let a caller like
        EmbeddingBatchCoordinator circuit-break the whole run on it."""
        provider = _make_provider(max_retries=1)
        unknown_exc = Exception("429 Too Many Requests")
        with patch.object(provider, "_embed_sync", side_effect=unknown_exc), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RateLimitExhausted) as exc_info:
                await provider.embed(["a"])
        assert type(exc_info.value) is RateLimitExhausted
        assert exc_info.value.dimension == "unknown"


# ── Real token-usage feedback (_extract_actual_tokens / record_usage) ──────────

def _fake_response(token_counts):
    """Builds a fake EmbedContentResponse-shaped object: one embedding per
    entry in token_counts, each carrying that entry as
    embeddings[i].statistics.token_count. Pass None for an entry to simulate
    a missing/absent statistics field on that one embedding."""

    class _Stats:
        def __init__(self, token_count):
            self.token_count = token_count

    class _Embedding:
        def __init__(self, token_count):
            self.values = [0.1, 0.2, 0.3]
            self.statistics = _Stats(token_count) if token_count is not None else None

    class _Response:
        def __init__(self, counts):
            self.embeddings = [_Embedding(c) for c in counts]

    return _Response(token_counts)


class TestExtractActualTokens:
    def test_sums_token_count_across_all_embeddings(self):
        response = _fake_response([10, 20, 5])
        assert GeminiDenseProvider._extract_actual_tokens(response) == 35

    def test_none_when_any_embedding_has_no_statistics(self):
        response = _fake_response([10, None, 5])
        assert GeminiDenseProvider._extract_actual_tokens(response) is None

    def test_none_when_embeddings_list_is_empty(self):
        response = _fake_response([])
        assert GeminiDenseProvider._extract_actual_tokens(response) is None

    def test_none_when_response_has_no_embeddings_attribute(self):
        assert GeminiDenseProvider._extract_actual_tokens(object()) is None


class TestRecordUsageFeedback:
    @pytest.mark.asyncio
    async def test_record_usage_called_with_real_token_count_on_success(self):
        strategy = AsyncMock()
        strategy.acquire = AsyncMock()
        strategy.record_usage = MagicMock()
        provider = _make_provider(rate_limit=strategy)
        with patch.object(provider, "_embed_sync", return_value=([[0.1, 0.2, 0.3]], 42)):
            await provider.embed(["a"])
        strategy.record_usage.assert_called_once_with(42)

    @pytest.mark.asyncio
    async def test_record_usage_skipped_when_actual_tokens_is_none(self):
        # e.g. the response was missing statistics — must not record a guess.
        strategy = AsyncMock()
        strategy.acquire = AsyncMock()
        strategy.record_usage = MagicMock()
        provider = _make_provider(rate_limit=strategy)
        with patch.object(provider, "_embed_sync", return_value=([[0.1, 0.2, 0.3]], None)):
            await provider.embed(["a"])
        strategy.record_usage.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_usage_skipped_when_no_rate_limit_configured(self):
        provider = _make_provider(rate_limit=None)
        with patch.object(provider, "_embed_sync", return_value=([[0.1, 0.2, 0.3]], 42)):
            result = await provider.embed(["a"])
        assert result == [[0.1, 0.2, 0.3]]

    @pytest.mark.asyncio
    async def test_not_called_on_a_failed_attempt_that_is_later_retried(self):
        strategy = AsyncMock()
        strategy.acquire = AsyncMock()
        strategy.record_usage = MagicMock()
        provider = _make_provider(max_retries=3, rate_limit=strategy)
        calls = {"n": 0}

        def fake_embed_sync(texts):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RPM_EXC
            return [[0.1] * 768 for _ in texts], 17

        with patch.object(provider, "_embed_sync", side_effect=fake_embed_sync), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock):
            await provider.embed(["a"])

        # Only the successful (2nd) attempt's real count is recorded — the
        # rejected 1st attempt never charged real tokens, so nothing to record.
        strategy.record_usage.assert_called_once_with(17)
