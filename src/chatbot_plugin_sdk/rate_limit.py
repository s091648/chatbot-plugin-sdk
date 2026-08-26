"""Async-safe sliding-window rate limiter for embedding API calls.

Design notes vs. scrape-analyzer's SlidingWindowStrategy:
  - Uses asyncio.sleep() instead of time.sleep() so it doesn't block the event loop.
  - Uses threading.Lock() (not asyncio.Lock) so state is safe when an EndpointProvider
    is shared across multiple asyncio.run() calls in a ThreadPoolExecutor.
  - RPM/RPD count `request_units` per embed() call (default 1; providers that batch
    multiple texts into one HTTP call pass `len(texts)`) rather than a flat 1 per
    call. Upstream embedding APIs (e.g. Gemini's batchEmbedContents) typically bill
    "requests per minute" per input item, not per HTTP call, so a single embed()
    call carrying 50 texts consumes 50 units of RPM/RPD — the same way TPM already
    scales with `estimated_tokens` instead of being flat per call. A caller that
    doesn't pass `request_units` keeps the old 1-unit-per-call behavior.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Protocol, runtime_checkable

from chatbot_plugin_sdk.exceptions import ExternalDependencyError


def estimate_tokens(texts: list[str]) -> int:
    """Rough token estimate shared by every rate-limited provider (4 chars ≈
    1 token — same approximation SlidingWindowStrategy's docstring already
    assumes). Also used by EmbeddingBatchCoordinator to right-size a batch
    against SlidingWindowStrategy.headroom() before dispatching it, instead
    of sizing batches purely by item count."""
    return max(1, sum(len(t) for t in texts) // 4)


class RateLimitExhausted(ExternalDependencyError):
    """Raised when a provider's request cap (local rpd, or the upstream
    API's own quota) is reached and won't recover within this run.

    A leaf of :class:`ExternalDependencyError` — callers who only care that
    "something upstream failed" can catch the parent; callers who want to
    react specifically to rate limiting (back off, rotate to a different
    provider/key, or abort this run) should catch this class by name.
    """


@runtime_checkable
class RateLimitStrategy(Protocol):
    """Protocol for injectable rate-limiting strategies.

    Pass an instance to ``EndpointProvider(rate_limit=...)``.  Omit the argument
    (or pass ``None``) for internal services with no external rate limits.
    """

    async def acquire(self, estimated_tokens: int = 0, request_units: int = 1) -> None:
        """Await until a request slot is available.  May raise :exc:`RateLimitExhausted`.

        request_units: How many RPM/RPD units this call consumes — pass
                        ``len(texts)`` when a single call batches multiple
                        inputs into one HTTP request. Defaults to 1.
        """
        ...

    def record_usage(self, actual_tokens: int) -> None:
        """Correct the token estimate after a successful call.

        Optional — useful when the API response includes the exact token count.
        For embeddings the estimate is usually close enough; omitting the call is fine.
        """
        ...


class SlidingWindowStrategy:
    """Sliding-window rate limiter that tracks RPM, TPM, and RPD.

    Thread-safe (``threading.Lock`` guards mutable state) and async-safe
    (``asyncio.sleep`` yields the event loop during waits).

    A single instance can be shared across threads — each thread's ``asyncio.run()``
    call uses ``asyncio.sleep`` in its own event loop, but state is protected by the
    threading lock.

    Args:
        rpm: Max requests per minute.  ``0`` disables this limit.  Counted in
             ``request_units`` (see ``acquire()``), not in number of ``acquire()``
             calls — a single call batching 50 texts consumes 50 units.
        tpm: Max tokens per minute (estimate: 4 chars ≈ 1 token).  ``0`` disables.
        rpd: Max requests per day.  When reached, :exc:`RateLimitExhausted` is raised.
             ``0`` disables this hard cap.  Also counted in ``request_units``.

    Usage::

        strategy = SlidingWindowStrategy(rpm=10, tpm=40_000, rpd=1_500)
        provider = EndpointProvider(
            url="https://generativelanguage.googleapis.com/...",
            dimension=768,
            api_key="AIza...",
            rate_limit=strategy,
        )
    """

    _WINDOW = 60.0  # sliding window in seconds

    def __init__(self, rpm: int = 0, tpm: int = 0, rpd: int = 0) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self._rpm_window: deque[tuple[float, int]] = deque()
        self._tpm_window: deque[tuple[float, int]] = deque()
        self._daily_count: int = 0
        self._lock = threading.Lock()  # threading.Lock: safe across event loops

    async def acquire(self, estimated_tokens: int = 0, request_units: int = 1) -> None:
        """Async-friendly wait loop.  Uses asyncio.sleep to yield the event loop.

        request_units: How many RPM/RPD units this call consumes — pass
                        ``len(texts)`` when a single call batches multiple
                        inputs into one HTTP request. Defaults to 1.
        """
        while True:
            wait = self._compute_wait(estimated_tokens, request_units)
            if wait == 0:
                return
            await asyncio.sleep(wait)

    def record_usage(self, actual_tokens: int) -> None:
        """Replace the last TPM estimate with the actual token count."""
        with self._lock:
            now = time.monotonic()
            if self._tpm_window:
                self._tpm_window.pop()
            self._tpm_window.append((now, actual_tokens))

    def headroom(self) -> tuple[float, float]:
        """Non-blocking peek at how much RPM/TPM capacity is available right
        now, without claiming any of it (unlike ``acquire()``).

        Returns ``(remaining_request_units, remaining_tokens)``. A disabled
        dimension (``rpm``/``tpm`` == 0) reports ``float("inf")`` for that
        slot, matching ``acquire()``'s "0 disables this limit" contract.

        Deliberately ignores ``rpd`` — that's a whole-run hard cap enforced
        by ``acquire()`` raising :exc:`RateLimitExhausted`, not a per-batch
        sizing concern. Intended for callers like
        :class:`~chatbot_plugin_sdk.batching.EmbeddingBatchCoordinator` that
        want to size a batch to what's actually available in the current
        60s window *before* calling ``acquire()``, rather than forming a
        batch by item count alone and finding out only at ``acquire()``
        time that it must wait for the whole thing.
        """
        with self._lock:
            now = time.monotonic()
            self._evict_stale(now)
            remaining_units: float = float("inf")
            if self.rpm > 0:
                used = sum(u for _, u in self._rpm_window)
                remaining_units = max(0, self.rpm - used)
            remaining_tokens: float = float("inf")
            if self.tpm > 0:
                used = sum(t for _, t in self._tpm_window)
                remaining_tokens = max(0, self.tpm - used)
            return remaining_units, remaining_tokens

    # ── internals ──────────────────────────────────────────────────────────

    def _compute_wait(self, estimated_tokens: int, request_units: int = 1) -> float:
        """Return seconds to sleep, or 0 if a slot is available (and claim it)."""
        with self._lock:
            if self.rpd > 0 and self._daily_count + request_units > self.rpd:
                raise RateLimitExhausted(
                    f"Daily request cap of {self.rpd} reached. "
                    "Switch providers or wait until tomorrow."
                )
            now = time.monotonic()
            self._evict_stale(now)

            wait = 0.0
            if self.rpm > 0:
                wait = max(wait, self._rpm_wait(now, request_units))
            if self.tpm > 0 and estimated_tokens > 0:
                wait = max(wait, self._tpm_wait(now, estimated_tokens))

            if wait == 0:
                # Claim the slot
                if self.rpm > 0:
                    self._rpm_window.append((now, request_units))
                self._daily_count += request_units
                if self.tpm > 0:
                    self._tpm_window.append((now, estimated_tokens))
            return wait

    def _evict_stale(self, now: float) -> None:
        cutoff = now - self._WINDOW
        while self._rpm_window and self._rpm_window[0][0] < cutoff:
            self._rpm_window.popleft()
        while self._tpm_window and self._tpm_window[0][0] < cutoff:
            self._tpm_window.popleft()

    def _rpm_wait(self, now: float, request_units: int) -> float:
        used = sum(u for _, u in self._rpm_window)
        if used + request_units <= self.rpm:
            return 0.0
        if not self._rpm_window:
            return 0.0
        return max(0.0, self._WINDOW - (now - self._rpm_window[0][0]))

    def _tpm_wait(self, now: float, estimated_tokens: int) -> float:
        used = sum(t for _, t in self._tpm_window)
        if used + estimated_tokens <= self.tpm:
            return 0.0
        if not self._tpm_window:
            return 0.0
        return max(0.0, self._WINDOW - (now - self._tpm_window[0][0]))
