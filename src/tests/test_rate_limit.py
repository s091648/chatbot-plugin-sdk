"""Tests for SlidingWindowStrategy and RateLimitExhausted."""
from __future__ import annotations

import time
from collections import deque
from unittest.mock import AsyncMock, patch

import pytest

from chatbot_plugin_sdk import RateLimitExhausted, SlidingWindowStrategy
from chatbot_plugin_sdk.rate_limit import RateLimitStrategy, estimate_tokens


# ── Protocol conformance ────────────────────────────────────────────────────────

class TestProtocol:
    def test_sliding_window_satisfies_protocol(self):
        strategy = SlidingWindowStrategy(rpm=10)
        assert isinstance(strategy, RateLimitStrategy)


# ── _compute_wait() — internal slot logic ──────────────────────────────────────

class TestComputeWait:
    def test_first_request_claims_slot(self):
        strategy = SlidingWindowStrategy(rpm=5)
        wait = strategy._compute_wait(0)
        assert wait == 0
        assert len(strategy._rpm_window) == 1
        assert strategy._daily_count == 1

    def test_slot_claimed_within_rpm_limit(self):
        strategy = SlidingWindowStrategy(rpm=3)
        strategy._compute_wait(0)
        strategy._compute_wait(0)
        wait = strategy._compute_wait(0)  # 3rd — still within limit
        assert wait == 0
        assert len(strategy._rpm_window) == 3

    def test_rpm_window_full_returns_positive_wait(self):
        strategy = SlidingWindowStrategy(rpm=2)
        strategy._compute_wait(0)
        strategy._compute_wait(0)
        wait = strategy._compute_wait(0)  # window is full
        assert wait > 0
        # Slot not claimed
        assert len(strategy._rpm_window) == 2
        assert strategy._daily_count == 2

    def test_tpm_window_full_returns_positive_wait(self):
        strategy = SlidingWindowStrategy(rpm=100, tpm=50)
        strategy._compute_wait(30)  # 30 tokens used
        wait = strategy._compute_wait(30)  # 30 + 30 = 60 > 50 → wait
        assert wait > 0

    def test_tpm_within_limit_claims_slot(self):
        strategy = SlidingWindowStrategy(rpm=100, tpm=100)
        strategy._compute_wait(40)  # 40 tokens used
        wait = strategy._compute_wait(40)  # 40 + 40 = 80 ≤ 100 → OK
        assert wait == 0

    def test_rpd_zero_means_no_daily_cap(self):
        strategy = SlidingWindowStrategy(rpd=0)
        for _ in range(1000):
            strategy._daily_count = 999  # simulate many calls
        wait = strategy._compute_wait(0)
        assert wait == 0

    def test_rpd_reached_raises_exhausted(self):
        strategy = SlidingWindowStrategy(rpd=5)
        for _ in range(5):
            strategy._compute_wait(0)
        with pytest.raises(RateLimitExhausted):
            strategy._compute_wait(0)

    def test_stale_rpm_entries_evicted(self):
        strategy = SlidingWindowStrategy(rpm=2)
        # Add two stale entries (older than 60s)
        old_time = time.monotonic() - 61
        strategy._rpm_window = deque([(old_time, 1), (old_time, 1)])
        strategy._daily_count = 2
        wait = strategy._compute_wait(0)  # should evict stale, then claim
        assert wait == 0
        assert len(strategy._rpm_window) == 1  # only the fresh one

    def test_request_units_consumed_per_call(self):
        """A batched call (request_units > 1) should consume that many RPM units at once."""
        strategy = SlidingWindowStrategy(rpm=10)
        wait = strategy._compute_wait(0, request_units=6)
        assert wait == 0
        assert strategy._rpm_window[-1] == pytest.approx((strategy._rpm_window[-1][0], 6))
        assert strategy._daily_count == 6

    def test_oversized_single_batch_admitted_once_then_blocks_window(self):
        """A single batch whose request_units alone exceeds rpm is admitted (empty
        window), but then occupies the whole window until it ages out."""
        strategy = SlidingWindowStrategy(rpm=10)
        wait = strategy._compute_wait(0, request_units=15)
        assert wait == 0  # admitted despite exceeding rpm alone — empty window

        wait2 = strategy._compute_wait(0, request_units=1)
        assert wait2 > 0  # any further request must wait for the window to drain

    def test_stale_tpm_entries_evicted(self):
        strategy = SlidingWindowStrategy(rpm=100, tpm=50)
        old_time = time.monotonic() - 61
        strategy._tpm_window = deque([(old_time, 49)])  # stale: 49 tokens
        strategy._daily_count = 1
        wait = strategy._compute_wait(49)  # 49 tokens, but stale window evicted → OK
        assert wait == 0


# ── record_usage() ─────────────────────────────────────────────────────────────

class TestRecordUsage:
    def test_updates_last_tpm_entry(self):
        strategy = SlidingWindowStrategy(rpm=10, tpm=1000)
        strategy._compute_wait(100)  # estimate: 100 tokens
        strategy.record_usage(80)    # actual: 80 tokens
        assert strategy._tpm_window[-1][1] == 80

    def test_no_op_when_tpm_window_empty(self):
        strategy = SlidingWindowStrategy(rpm=10)
        # Should not raise even with empty window
        strategy.record_usage(50)


# ── acquire() — async interface ────────────────────────────────────────────────

class TestAcquire:
    @pytest.mark.asyncio
    async def test_acquire_under_limit_returns_immediately(self):
        strategy = SlidingWindowStrategy(rpm=10)
        # No sleep should be called for first request
        with patch("chatbot_plugin_sdk.rate_limit.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await strategy.acquire(0)
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_acquire_over_rpm_calls_sleep(self):
        strategy = SlidingWindowStrategy(rpm=1)
        strategy._compute_wait(0)  # claim the only slot

        sleep_calls: list[float] = []

        async def fake_sleep(t: float) -> None:
            sleep_calls.append(t)
            # Manually evict the window so acquire succeeds on next loop iteration
            strategy._rpm_window.clear()

        with patch("chatbot_plugin_sdk.rate_limit.asyncio.sleep", side_effect=fake_sleep):
            await strategy.acquire(0)

        assert len(sleep_calls) == 1
        assert sleep_calls[0] > 0

    @pytest.mark.asyncio
    async def test_acquire_raises_when_rpd_exhausted(self):
        strategy = SlidingWindowStrategy(rpd=2)
        strategy._daily_count = 2  # already at cap

        with pytest.raises(RateLimitExhausted):
            await strategy.acquire(0)


# ── No-limit mode (all zeros) ──────────────────────────────────────────────────

class TestNoLimit:
    @pytest.mark.asyncio
    async def test_zero_limits_never_block(self):
        strategy = SlidingWindowStrategy(rpm=0, tpm=0, rpd=0)
        with patch("chatbot_plugin_sdk.rate_limit.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            for _ in range(100):
                await strategy.acquire(99999)
        mock_sleep.assert_not_called()


# ── estimate_tokens() ────────────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_estimates_four_chars_per_token(self):
        assert estimate_tokens(["a" * 400]) == 100

    def test_sums_across_multiple_texts(self):
        assert estimate_tokens(["a" * 40, "b" * 60]) == 25

    def test_never_returns_zero_for_nonempty_input(self):
        assert estimate_tokens(["a"]) == 1

    def test_empty_list_returns_one(self):
        assert estimate_tokens([]) == 1


# ── headroom() — non-blocking peek, doesn't claim anything ─────────────────────

class TestHeadroom:
    def test_disabled_dimensions_report_infinite_headroom(self):
        strategy = SlidingWindowStrategy(rpm=0, tpm=0)
        remaining_units, remaining_tokens = strategy.headroom()
        assert remaining_units == float("inf")
        assert remaining_tokens == float("inf")

    def test_fresh_window_reports_full_capacity(self):
        strategy = SlidingWindowStrategy(rpm=100, tpm=30_000)
        remaining_units, remaining_tokens = strategy.headroom()
        assert remaining_units == 100
        assert remaining_tokens == 30_000

    def test_reflects_usage_already_claimed_via_acquire(self):
        strategy = SlidingWindowStrategy(rpm=100, tpm=30_000)
        strategy._compute_wait(estimated_tokens=12_000, request_units=96)
        remaining_units, remaining_tokens = strategy.headroom()
        assert remaining_units == 4
        assert remaining_tokens == 18_000

    def test_does_not_claim_any_slot(self):
        """headroom() is a pure peek — calling it repeatedly must not change
        what it reports, unlike acquire()/_compute_wait() which claim a slot
        on success."""
        strategy = SlidingWindowStrategy(rpm=100, tpm=30_000)
        first = strategy.headroom()
        second = strategy.headroom()
        assert first == second == (100, 30_000)

    def test_ignores_rpd_entirely(self):
        """rpd is a whole-run hard cap enforced by acquire() raising
        RateLimitExhausted, not a per-batch sizing signal — headroom() must
        not reflect it even when the daily cap is already exhausted."""
        strategy = SlidingWindowStrategy(rpm=100, tpm=30_000, rpd=1)
        strategy._daily_count = 1  # already at the daily cap
        remaining_units, remaining_tokens = strategy.headroom()
        assert remaining_units == 100
        assert remaining_tokens == 30_000

    def test_evicts_stale_entries_before_reporting(self):
        strategy = SlidingWindowStrategy(rpm=100, tpm=30_000)
        strategy._compute_wait(estimated_tokens=12_000, request_units=96)
        # Backdate the claimed window entries past the 60s sliding window.
        stale_time = time.monotonic() - 61
        strategy._rpm_window = deque([(stale_time, 96)])
        strategy._tpm_window = deque([(stale_time, 12_000)])
        remaining_units, remaining_tokens = strategy.headroom()
        assert remaining_units == 100
        assert remaining_tokens == 30_000
