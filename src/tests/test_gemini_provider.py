"""Tests for GeminiDenseProvider's 429 classification and retry/split behavior."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin_sdk.exceptions import EmbeddingError
from chatbot_plugin_sdk.providers.gemini import (
    GeminiDenseProvider,
    _is_daily_quota_error,
    _is_token_quota_error,
    _parse_retry_delay,
    _quota_dimension,
)
from chatbot_plugin_sdk.rate_limit import RateLimitExhausted


def _quota_exc(quota_id: str, delay: float | None = None) -> Exception:
    """Builds a fake exception mimicking google.genai's 429 error string."""
    parts = ["429 RESOURCE_EXHAUSTED.", f"'quotaId': '{quota_id}'"]
    if delay is not None:
        parts.append(f"Please retry in {delay}s.")
    return Exception(" ".join(parts))


DAILY_EXC = _quota_exc("EmbedContentRequestsPerDayPerProjectPerModel-FreeTier")
RPM_EXC = _quota_exc("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", delay=5)
TPM_EXC = _quota_exc("GenerateContentInputTokensPerModelPerMinute-FreeTier", delay=5)


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
            with pytest.raises(RateLimitExhausted, match="Daily quota"):
                await provider.embed(["a"])
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_daily_quota_latches_and_skips_later_calls_without_api_call(self):
        # A real process only ever needs to learn the daily cap is blown once —
        # every later embed() in the same run should fail fast, no HTTP call.
        provider = _make_provider()
        mock_embed_sync = MagicMock(side_effect=DAILY_EXC)
        with patch.object(provider, "_embed_sync", mock_embed_sync):
            with pytest.raises(RateLimitExhausted, match="Daily quota exceeded"):
                await provider.embed(["a"])

        assert mock_embed_sync.call_count == 1

        with pytest.raises(RateLimitExhausted, match="already exhausted"):
            await provider.embed(["b", "c"])

        # No further _embed_sync call for the second, already-latched request.
        assert mock_embed_sync.call_count == 1

    @pytest.mark.asyncio
    async def test_daily_quota_latch_skips_rate_limit_acquire_too(self):
        strategy = AsyncMock()
        strategy.acquire = AsyncMock()
        provider = _make_provider(rate_limit=strategy)
        with patch.object(provider, "_embed_sync", side_effect=DAILY_EXC):
            with pytest.raises(RateLimitExhausted):
                await provider.embed(["a"])
        strategy.acquire.assert_awaited_once()

        strategy.acquire.reset_mock()
        with pytest.raises(RateLimitExhausted, match="already exhausted"):
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
            return [[0.1] * 768 for _ in texts]

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
            return [[0.1] * 768 for _ in texts]

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
            return [[0.5] * 768 for _ in texts]

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
            return [[0.1] * 768]

        with patch.object(provider, "_embed_sync", side_effect=fake_embed_sync), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await provider.embed(["a"])

        assert result == [[0.1] * 768]
        mock_sleep.assert_awaited_once_with(5.0)

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
            return [[0.1] * 768 for _ in texts]

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
            with pytest.raises(RateLimitExhausted, match="after 2 retries"):
                await provider.embed(["a"])
        mock_sleep.assert_awaited_once_with(15.0)

    @pytest.mark.asyncio
    async def test_delay_over_threshold_raises_immediately(self):
        provider = _make_provider()
        long_delay_exc = _quota_exc(
            "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", delay=301
        )
        with patch.object(provider, "_embed_sync", side_effect=long_delay_exc), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RateLimitExhausted, match="exceeding the 300.0s threshold"):
                await provider.embed(["a"])
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_exhausts_max_retries_then_raises(self):
        provider = _make_provider(max_retries=2)
        with patch.object(provider, "_embed_sync", side_effect=RPM_EXC), \
             patch("chatbot_plugin_sdk.providers.gemini.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RateLimitExhausted, match="after 2 retries"):
                await provider.embed(["a"])
        mock_sleep.assert_awaited_once_with(5.0)
