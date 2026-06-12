"""Async-safe sliding-window rate limiter for embedding API calls.

Design notes vs. scrape-analyzer's SlidingWindowStrategy:
  - Uses asyncio.sleep() instead of time.sleep() so it doesn't block the event loop.
  - Uses threading.Lock() (not asyncio.Lock) so state is safe when an EndpointProvider
    is shared across multiple asyncio.run() calls in a ThreadPoolExecutor.
  - Counts one request per embed() call regardless of batch size.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Protocol, runtime_checkable


class RateLimitExhausted(Exception):
    """Raised when the daily request cap (rpd) is reached.

    Callers may catch this to fall back to a different provider or abort.
    """


@runtime_checkable
class RateLimitStrategy(Protocol):
    """Protocol for injectable rate-limiting strategies.

    Pass an instance to ``EndpointProvider(rate_limit=...)``.  Omit the argument
    (or pass ``None``) for internal services with no external rate limits.
    """

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Await until a request slot is available.  May raise :exc:`RateLimitExhausted`."""
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
        rpm: Max requests per minute.  ``0`` disables this limit.
        tpm: Max tokens per minute (estimate: 4 chars ≈ 1 token).  ``0`` disables.
        rpd: Max requests per day.  When reached, :exc:`RateLimitExhausted` is raised.
             ``0`` disables this hard cap.

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
        self._rpm_window: deque[float] = deque()
        self._tpm_window: deque[tuple[float, int]] = deque()
        self._daily_count: int = 0
        self._lock = threading.Lock()  # threading.Lock: safe across event loops

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Async-friendly wait loop.  Uses asyncio.sleep to yield the event loop."""
        while True:
            wait = self._compute_wait(estimated_tokens)
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

    # ── internals ──────────────────────────────────────────────────────────

    def _compute_wait(self, estimated_tokens: int) -> float:
        """Return seconds to sleep, or 0 if a slot is available (and claim it)."""
        with self._lock:
            if self.rpd > 0 and self._daily_count >= self.rpd:
                raise RateLimitExhausted(
                    f"Daily request cap of {self.rpd} reached. "
                    "Switch providers or wait until tomorrow."
                )
            now = time.monotonic()
            self._evict_stale(now)

            wait = 0.0
            if self.rpm > 0:
                wait = max(wait, self._rpm_wait(now))
            if self.tpm > 0 and estimated_tokens > 0:
                wait = max(wait, self._tpm_wait(now, estimated_tokens))

            if wait == 0:
                # Claim the slot
                if self.rpm > 0:
                    self._rpm_window.append(now)
                self._daily_count += 1
                if self.tpm > 0:
                    self._tpm_window.append((now, estimated_tokens))
            return wait

    def _evict_stale(self, now: float) -> None:
        cutoff = now - self._WINDOW
        while self._rpm_window and self._rpm_window[0] < cutoff:
            self._rpm_window.popleft()
        while self._tpm_window and self._tpm_window[0][0] < cutoff:
            self._tpm_window.popleft()

    def _rpm_wait(self, now: float) -> float:
        if len(self._rpm_window) < self.rpm:
            return 0.0
        if not self._rpm_window:
            return 0.0
        return max(0.0, self._WINDOW - (now - self._rpm_window[0]))

    def _tpm_wait(self, now: float, estimated_tokens: int) -> float:
        used = sum(t for _, t in self._tpm_window)
        if used + estimated_tokens <= self.tpm:
            return 0.0
        if not self._tpm_window:
            return 0.0
        return max(0.0, self._WINDOW - (now - self._tpm_window[0][0]))
