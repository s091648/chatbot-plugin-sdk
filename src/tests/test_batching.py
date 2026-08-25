"""Tests for EmbeddingBatchCoordinator."""
from __future__ import annotations

import asyncio

import pytest

from chatbot_plugin_sdk import EmbeddingBatchCoordinator
from chatbot_plugin_sdk.exceptions import EmbeddingError


def _dense(embed_side_effect=None):
    """A minimal DenseEmbeddingProvider stub with a recording embed()."""

    class _Dense:
        dimension = 3

        def __init__(self):
            self.calls: list[list[str]] = []

        async def embed(self, texts):
            self.calls.append(list(texts))
            if embed_side_effect is not None:
                return await embed_side_effect(texts)
            return [[0.1, 0.2, 0.3] for _ in texts]

    return _Dense()


# ── Single-caller behavior (parity with the pre-coordinator sequential loop) ───

class TestSingleCaller:
    @pytest.mark.asyncio
    async def test_returns_vectors_in_order(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        vectors = await coordinator.embed_many(["a", "b", "c"])
        assert vectors == [[0.1, 0.2, 0.3]] * 3
        await coordinator.aclose()

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_without_starting_worker(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        assert await coordinator.embed_many([]) == []
        assert dense.calls == []
        assert coordinator._worker_task is None

    @pytest.mark.asyncio
    async def test_one_call_over_batch_size_splits_into_multiple_provider_calls(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=2)
        vectors = await coordinator.embed_many(["a", "b", "c", "d", "e"])
        assert len(vectors) == 5
        assert len(dense.calls) >= 3  # 5 items, batch_size=2 -> at least ceil(5/2)=3 calls
        assert sum(len(c) for c in dense.calls) == 5
        await coordinator.aclose()


# ── Cross-call batching under concurrency ──────────────────────────────────────

class TestConcurrentCallers:
    @pytest.mark.asyncio
    async def test_two_concurrent_calls_share_one_worker_and_may_batch_together(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)

        results = await asyncio.gather(
            coordinator.embed_many(["a", "b"]),
            coordinator.embed_many(["c"]),
        )

        assert results[0] == [[0.1, 0.2, 0.3]] * 2
        assert results[1] == [[0.1, 0.2, 0.3]] * 1
        # Both callers' chunks were composed into one provider call (single worker,
        # no yield point between enqueue and the worker draining the queue).
        assert len(dense.calls) == 1
        assert sorted(dense.calls[0]) == ["a", "b", "c"]
        await coordinator.aclose()

    @pytest.mark.asyncio
    async def test_only_one_worker_task_regardless_of_caller_count(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        await asyncio.gather(*(coordinator.embed_many([f"chunk-{i}"]) for i in range(10)))
        assert coordinator._worker_task is not None
        await coordinator.aclose()


# ── Failure propagation ─────────────────────────────────────────────────────────

class TestBatchFailure:
    @pytest.mark.asyncio
    async def test_batch_failure_propagates_to_every_item_in_that_batch(self):
        async def _boom(texts):
            raise RuntimeError("provider exploded")

        dense = _dense(embed_side_effect=_boom)
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        with pytest.raises(RuntimeError, match="provider exploded"):
            await coordinator.embed_many(["a", "b"])
        await coordinator.aclose()

    @pytest.mark.asyncio
    async def test_vector_count_mismatch_fails_every_item_instead_of_hanging(self):
        """Regression: zip(batch, vectors) silently drops extra items when the
        provider returns fewer vectors than texts — every dropped item's future
        would never resolve, hanging embed_many()'s gather() forever. Must
        surface as an exception for every item in the batch instead."""
        async def _short_response(texts):
            return [[0.1, 0.2, 0.3]]  # always 1 vector, regardless of batch size

        dense = _dense(embed_side_effect=_short_response)
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        with pytest.raises(EmbeddingError, match="3 texts"):
            await asyncio.wait_for(coordinator.embed_many(["a", "b", "c"]), timeout=5)
        await coordinator.aclose()

    @pytest.mark.asyncio
    async def test_one_failed_batch_does_not_prevent_the_next_batch(self):
        call_count = {"n": 0}

        async def _fail_once(texts):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first batch fails")
            return [[0.1, 0.2, 0.3] for _ in texts]

        dense = _dense(embed_side_effect=_fail_once)
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        with pytest.raises(RuntimeError):
            await coordinator.embed_many(["a"])
        vectors = await coordinator.embed_many(["b"])
        assert vectors == [[0.1, 0.2, 0.3]]
        await coordinator.aclose()


# ── queue_factory injection (DIP) ───────────────────────────────────────────────

class TestQueueFactory:
    @pytest.mark.asyncio
    async def test_custom_queue_factory_is_used(self):
        created: list[asyncio.Queue] = []

        def factory():
            q = asyncio.Queue()
            created.append(q)
            return q

        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16, queue_factory=factory)
        await coordinator.embed_many(["a"])
        assert len(created) == 1
        assert coordinator._queue is created[0]
        await coordinator.aclose()

    @pytest.mark.asyncio
    async def test_default_factory_produces_plain_asyncio_queue(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        await coordinator.embed_many(["a"])
        assert type(coordinator._queue) is asyncio.Queue
        await coordinator.aclose()

    @pytest.mark.asyncio
    async def test_queue_not_built_until_first_use(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        assert coordinator._queue is None
        await coordinator.embed_many(["a"])
        assert coordinator._queue is not None
        await coordinator.aclose()


# ── get_queue() / set_queue() ───────────────────────────────────────────────────

class TestGetSetQueue:
    @pytest.mark.asyncio
    async def test_get_queue_returns_none_before_first_use(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        assert coordinator.get_queue() is None

    @pytest.mark.asyncio
    async def test_get_queue_returns_current_queue_after_first_use(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        await coordinator.embed_many(["a"])
        assert coordinator.get_queue() is coordinator._queue
        assert isinstance(coordinator.get_queue(), asyncio.Queue)
        await coordinator.aclose()

    @pytest.mark.asyncio
    async def test_set_queue_before_any_use_just_swaps_it(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        custom_queue = asyncio.Queue()
        await coordinator.set_queue(custom_queue)
        assert coordinator.get_queue() is custom_queue
        # A later embed_many() should pick up the manually-set queue, not
        # build a fresh one from the (unused, default) factory.
        await coordinator.embed_many(["a"])
        assert coordinator.get_queue() is custom_queue
        await coordinator.aclose()

    @pytest.mark.asyncio
    async def test_set_queue_migrates_queued_items_but_cancels_the_in_flight_batch(self):
        """Items still sitting in the old queue (not yet claimed) when
        set_queue() is called are migrated onto the new queue and still
        resolve. The item the worker had already claimed and was mid-embed()
        on is NOT migrated — its embed_many() caller gets CancelledError
        instead (documented limitation, see set_queue()'s docstring). Uses
        an Event (not sleep(0) counting) so "a" having actually been claimed
        by the worker is guaranteed, not assumed from scheduling order."""
        claimed_a = asyncio.Event()
        release = asyncio.Event()

        async def _blocking_first_call(texts):
            claimed_a.set()
            await release.wait()
            return [[0.1, 0.2, 0.3] for _ in texts]

        dense = _dense(embed_side_effect=_blocking_first_call)
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=1)

        first = asyncio.ensure_future(coordinator.embed_many(["a"]))
        await claimed_a.wait()  # worker has popped "a" off the queue and is now blocked inside embed()

        second = asyncio.ensure_future(coordinator.embed_many(["b"]))
        await asyncio.sleep(0)  # let "b" actually land in the queue (worker's blocked, won't touch it)

        new_queue = asyncio.Queue()
        await coordinator.set_queue(new_queue)
        release.set()  # unblocks the now-cancelled worker's embed() call, if it's even still checked

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=5)

        second_result = await asyncio.wait_for(second, timeout=5)
        assert second_result == [[0.1, 0.2, 0.3]]
        await coordinator.aclose()

    @pytest.mark.asyncio
    async def test_set_queue_restarts_worker_so_new_submissions_still_work(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        await coordinator.embed_many(["a"])  # starts the worker

        new_queue = asyncio.Queue()
        await coordinator.set_queue(new_queue)

        vectors = await asyncio.wait_for(coordinator.embed_many(["b"]), timeout=5)
        assert vectors == [[0.1, 0.2, 0.3]]
        await coordinator.aclose()


# ── Lifecycle ────────────────────────────────────────────────────────────────────

class TestAclose:
    @pytest.mark.asyncio
    async def test_aclose_before_any_use_is_a_noop(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        await coordinator.aclose()  # must not raise

    @pytest.mark.asyncio
    async def test_aclose_cancels_worker_task(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        await coordinator.embed_many(["a"])
        worker = coordinator._worker_task
        await coordinator.aclose()
        assert worker.cancelled() or worker.done()

    @pytest.mark.asyncio
    async def test_aclose_is_idempotent(self):
        dense = _dense()
        coordinator = EmbeddingBatchCoordinator(dense=dense, embed_batch_size=16)
        await coordinator.embed_many(["a"])
        await coordinator.aclose()
        await coordinator.aclose()  # must not raise
