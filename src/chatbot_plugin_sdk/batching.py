"""Coordinates embedding-API calls across concurrent callers sharing one
rate-limited provider.

Problem this solves: a caller that submits many chunks per call (e.g.
IngestProcessor.ingest() for one large article) and a rate-limited provider
(e.g. GeminiDenseProvider wrapping a shared SlidingWindowStrategy) can starve
or collide with other concurrent callers when each caller independently
sends its own embed_batch_size-sized batches straight at the provider — every
concurrent caller's batches compete for the same per-minute budget with no
coordination between them. See scrape-analyzer's specs/024-async-pipeline-refactor
research.md item 11 for the production trace that motivated this.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, List, Optional, TYPE_CHECKING

from chatbot_plugin_sdk.exceptions import EmbeddingError
from chatbot_plugin_sdk.rate_limit import estimate_tokens

if TYPE_CHECKING:
    from chatbot_plugin_sdk.protocols import DenseEmbeddingProvider


@dataclass
class EmbedWorkItem:
    text: str
    future: "asyncio.Future[list[float]]"


QueueFactory = Callable[[], "asyncio.Queue[EmbedWorkItem]"]


class EmbeddingBatchCoordinator:
    """Single background worker draining a shared queue of chunk-embed
    requests submitted by any number of concurrent callers, batching them
    into as few real provider calls as ``embed_batch_size`` allows.

    One worker by default: the underlying rate limit is a single shared
    resource regardless of worker count, so more workers would still
    serialize on it — one worker avoids wasted, uncoordinated collisions
    between concurrent callers by construction, rather than merely reducing
    their frequency.

    Batch formation is rate-limit-aware when ``dense`` exposes a ``rate_limit``
    attribute whose value has a ``headroom()`` method (``SlidingWindowStrategy``
    does; see its docstring) — the worker stops growing a batch as soon as the
    next item would exceed the *currently available* RPM/TPM window, instead
    of always growing to ``embed_batch_size`` and letting the provider's
    ``acquire()`` block on the whole thing. This matters specifically for TPM:
    without it, a batch sized purely by item count can sit right at the edge
    of (or past) the real per-minute token budget on every single call —
    small, unavoidable error in the token estimate (``estimate_tokens()`` is a
    4-chars-≈-1-token approximation, not a real tokenizer) then reliably tips
    real upstream usage over Google's actual quota even though the local
    estimate looked fine, producing recurring 429s despite locally-configured
    RPM/TPM matching the real quota. Right-sizing against live headroom
    shrinks the blast radius of that estimation error instead of eliminating
    it — ``acquire()`` remains the authoritative, blocking gate either way.
    When ``dense`` has no ``rate_limit`` (or that limiter has no
    ``headroom()``), batch formation falls back to the original count-only
    behavior.

    ``queue_factory`` is a pure dependency-inversion seam (DIP) — pass one
    to control queue behavior (bounded, priority-ordered, instrumented,
    etc.); the coordinator only ever calls the standard ``asyncio.Queue``
    interface (``put``/``get``/``get_nowait``) on whatever it returns.
    Not specifying one produces a plain FIFO ``asyncio.Queue()``, built
    lazily on first use — never at ``__init__`` time, so it binds to
    whichever event loop is actually running when work is first submitted.

    Usage::

        coordinator = EmbeddingBatchCoordinator(dense=my_provider, embed_batch_size=16)
        vectors = await coordinator.embed_many(["chunk 1", "chunk 2"])
        ...
        await coordinator.aclose()
    """

    def __init__(
        self,
        dense: "DenseEmbeddingProvider",
        embed_batch_size: int = 16,
        queue_factory: Optional[QueueFactory] = None,
    ) -> None:
        self._dense = dense
        self._embed_batch_size = embed_batch_size
        self._queue_factory: QueueFactory = queue_factory or (lambda: asyncio.Queue())
        self._queue: "asyncio.Queue[EmbedWorkItem] | None" = None
        self._worker_task: "asyncio.Task[None] | None" = None

    def _ensure_started(self) -> None:
        if self._queue is None:
            self._queue = self._queue_factory()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Submit texts as individual work items to the shared queue, await
        their vectors, return them in the same order as ``texts``.

        Safe to call concurrently — from multiple coroutines, against the
        same coordinator instance — each call's items interleave with any
        other concurrent call's items in the one shared queue and may end up
        batched together into the same provider call.
        """
        if not texts:
            return []
        self._ensure_started()
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        items = [EmbedWorkItem(text=t, future=loop.create_future()) for t in texts]
        for item in items:
            await self._queue.put(item)
        return list(await asyncio.gather(*(item.future for item in items)))

    def _headroom_fn(self):
        """Return the dense provider's rate limiter's headroom() bound method,
        or None if unavailable — either the provider exposes no rate_limit at
        all (e.g. FastEmbedDenseProvider, a local model with no upstream
        quota), or its rate_limit has no headroom() method (a custom
        RateLimitStrategy implementation that predates this). None means
        "fall back to count-only batching", not "unlimited"."""
        rate_limit = getattr(self._dense, "rate_limit", None)
        if rate_limit is None:
            return None
        return getattr(rate_limit, "headroom", None)

    async def _worker_loop(self) -> None:
        assert self._queue is not None
        current_batch: "Optional[List[EmbedWorkItem]]" = None
        # An item already popped off the queue while probing whether it fits
        # in the batch being formed, but that didn't fit and wasn't part of
        # any dispatched batch yet — becomes the next batch's first item.
        # Holding it here (rather than queue.put_nowait()'ing it back) keeps
        # strict FIFO order: put_nowait() would append it behind whatever
        # else is currently queued instead of leaving it at the front.
        pending: "Optional[EmbedWorkItem]" = None
        try:
            while True:
                if pending is not None:
                    item, pending = pending, None
                else:
                    item = await self._queue.get()
                batch = [item]
                headroom_fn = self._headroom_fn()
                if headroom_fn is not None:
                    remaining_units, remaining_tokens = headroom_fn()
                    running_tokens = estimate_tokens([item.text])
                    while len(batch) < self._embed_batch_size:
                        try:
                            next_item = self._queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        next_tokens = estimate_tokens([next_item.text])
                        if (
                            len(batch) + 1 > remaining_units
                            or running_tokens + next_tokens > remaining_tokens
                        ):
                            # Doesn't fit the window available right now — save it
                            # for the next batch rather than blocking this one on
                            # its acquire() wait too. The first item of a batch is
                            # never gated this way, so a batch is never empty.
                            pending = next_item
                            break
                        batch.append(next_item)
                        running_tokens += next_tokens
                else:
                    while len(batch) < self._embed_batch_size:
                        try:
                            batch.append(self._queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                current_batch = batch  # tracked so a cancellation mid-embed() can still resolve it
                texts = [b.text for b in batch]
                try:
                    vectors = await self._dense.embed(texts)
                    if len(vectors) != len(batch):
                        raise EmbeddingError(
                            f"Embedding provider returned {len(vectors)} vectors "
                            f"for a batch of {len(batch)} texts."
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — propagated to every item in this batch
                    # A mismatched/short vectors list must not leave any item's
                    # future unresolved — embed_many()'s gather() would hang on
                    # it forever otherwise (zip() would silently drop the rest).
                    for b in batch:
                        if not b.future.done():
                            b.future.set_exception(exc)
                    current_batch = None
                    continue
                for b, vec in zip(batch, vectors):
                    if not b.future.done():
                        b.future.set_result(vec)
                current_batch = None
        except asyncio.CancelledError:
            # Cancel whatever batch the worker was actively awaiting embed()
            # for when it was stopped (aclose()/set_queue()) — otherwise those
            # items' futures are never touched and their embed_many() callers
            # hang forever, not just "left unresolved". Then drain whatever's
            # still queued behind it for the same reason.
            if current_batch is not None:
                for b in current_batch:
                    if not b.future.done():
                        b.future.cancel()
            if pending is not None and not pending.future.done():
                pending.future.cancel()
            while True:
                try:
                    leftover = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not leftover.future.done():
                    leftover.future.cancel()
            raise

    async def aclose(self) -> None:
        """Cancel the background worker task. Idempotent — safe to call
        even if no work was ever submitted (worker never started)."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def get_queue(self) -> "asyncio.Queue[EmbedWorkItem] | None":
        """Return the coordinator's current queue, or None if no work has
        been submitted yet and set_queue() was never called (the queue is
        otherwise built lazily, from queue_factory, on first embed_many()
        call)."""
        return self._queue

    async def set_queue(self, queue: "asyncio.Queue[EmbedWorkItem]") -> None:
        """Replace the coordinator's queue.

        Safe at any point in the coordinator's lifecycle:
          - Before any work has been submitted (no worker running yet) —
            just swaps the queue reference.
          - While a worker is running — migrates every item still sitting in
            the current queue onto ``queue`` first (so nothing already
            submitted via ``embed_many()`` is silently lost), then stops the
            running worker and starts a fresh one against the new queue.

        Known limitation, shared with ``aclose()``: an item whose batch the
        worker has already claimed and is mid-``embed()`` call on is not
        migrated — the in-flight provider call is cancelled along with the
        worker, and that batch's ``embed_many()`` caller(s) receive
        ``asyncio.CancelledError`` rather than a result. Call this between
        runs, not concurrently with in-flight ``embed_many()`` calls, to
        avoid losing that in-flight work.
        """
        old_worker = self._worker_task
        old_queue = self._queue

        if old_queue is not None:
            while True:
                try:
                    item = old_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                await queue.put(item)

        if old_worker is not None:
            old_worker.cancel()
            try:
                await old_worker
            except asyncio.CancelledError:
                pass

        self._queue = queue

        if old_worker is not None:
            self._worker_task = asyncio.create_task(self._worker_loop())
