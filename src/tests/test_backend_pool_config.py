"""Pool-tuning knobs on DatabaseConfig + backend/processor prewarm (v1.3.1)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin_sdk import (
    AsyncPgBackend,
    DatabaseConfig,
    EndpointProvider,
    IngestProcessor,
    SyncPgBackend,
)


def _cfg(**overrides) -> DatabaseConfig:
    base = dict(dbname="d", user="u", password="p", host="db.example.internal")
    base.update(overrides)
    return DatabaseConfig(**base)


class TestDatabaseConfigDefaults:
    def test_pool_defaults_preserve_legacy_sizing(self):
        c = _cfg()
        assert c.pool_size == 5
        assert c.max_overflow == 10

    def test_new_resilience_defaults(self):
        c = _cfg()
        assert c.pool_pre_ping is True
        assert c.connect_timeout == 30.0
        assert c.pool_recycle == 1800

    def test_overrides_apply(self):
        c = _cfg(pool_size=20, connect_timeout=45.0, pool_pre_ping=False)
        assert (c.pool_size, c.connect_timeout, c.pool_pre_ping) == (20, 45.0, False)


class TestAsyncPgBackendWiring:
    def test_engine_pool_sized_from_config(self):
        be = AsyncPgBackend(_cfg(pool_size=7, max_overflow=3))
        pool = be._engine.sync_engine.pool
        assert pool.size() == 7
        assert pool._max_overflow == 3
        assert be._pool_size == 7

    def test_engine_built_with_timeout_and_pre_ping_kwargs(self):
        with patch("chatbot_plugin_sdk.backends.async_pg.create_async_engine") as mk:
            AsyncPgBackend(_cfg(connect_timeout=12.5, pool_pre_ping=True, pool_recycle=900))
        kwargs = mk.call_args.kwargs
        assert kwargs["connect_args"] == {"timeout": 12.5}
        assert kwargs["pool_pre_ping"] is True
        assert kwargs["pool_recycle"] == 900

    async def test_prewarm_opens_and_releases_n_connections(self):
        be = AsyncPgBackend(_cfg(pool_size=4))
        conn = AsyncMock()
        conn.execute = AsyncMock()
        conn.close = AsyncMock()
        be._engine = MagicMock()
        be._engine.connect = AsyncMock(return_value=conn)

        await be.prewarm()

        assert be._engine.connect.await_count == 4
        assert conn.execute.await_count == 4
        assert conn.close.await_count == 4

    async def test_prewarm_explicit_count_overrides_pool_size(self):
        be = AsyncPgBackend(_cfg(pool_size=4))
        conn = AsyncMock()
        be._engine = MagicMock()
        be._engine.connect = AsyncMock(return_value=conn)

        await be.prewarm(2)

        assert be._engine.connect.await_count == 2

    async def test_prewarm_stops_on_connect_error_without_raising(self):
        be = AsyncPgBackend(_cfg(pool_size=5))
        be._engine = MagicMock()
        be._engine.connect = AsyncMock(side_effect=OSError("dns hiccup"))

        await be.prewarm()  # must not raise


class TestSyncPgBackendWiring:
    # psycopg2 is an optional extra ([sync]) and isn't in the dev group, so
    # patch the engine factory rather than pull a C driver into CI.
    def test_engine_built_with_pool_and_timeout_kwargs(self):
        with patch("chatbot_plugin_sdk.backends.sync_pg.create_engine") as mk, \
             patch("chatbot_plugin_sdk.backends.sync_pg.sessionmaker"):
            SyncPgBackend(_cfg(pool_size=9, max_overflow=1, connect_timeout=15.0))
        kwargs = mk.call_args.kwargs
        assert kwargs["pool_size"] == 9
        assert kwargs["max_overflow"] == 1
        assert kwargs["connect_args"] == {"connect_timeout": 15}

    async def test_prewarm_runs_sync_helper(self):
        with patch("chatbot_plugin_sdk.backends.sync_pg.create_engine"), \
             patch("chatbot_plugin_sdk.backends.sync_pg.sessionmaker"):
            be = SyncPgBackend(_cfg(pool_size=3))
        conn = MagicMock()
        be._engine = MagicMock()
        be._engine.connect = MagicMock(return_value=conn)

        await be.prewarm()

        assert be._engine.connect.call_count == 3
        assert conn.close.call_count == 3


class TestIngestProcessorPrewarm:
    def _processor(self) -> tuple[IngestProcessor, MagicMock]:
        backend = MagicMock()
        backend.schema = "vectors"
        backend.setup = AsyncMock()
        backend.prewarm = AsyncMock()
        processor = IngestProcessor()
        processor.configure(
            backend=backend,
            dense=EndpointProvider(url="http://embed:8080", dimension=768),
        )
        return processor, backend

    async def test_prewarm_runs_setup_then_backend_prewarm(self):
        processor, backend = self._processor()

        await processor.prewarm(6)

        backend.setup.assert_awaited_once()
        backend.prewarm.assert_awaited_once_with(6)

    async def test_prewarm_tolerates_backend_without_prewarm(self):
        processor, backend = self._processor()
        del backend.prewarm

        await processor.prewarm()  # must not raise

        backend.setup.assert_awaited_once()

    async def test_ensure_ready_runs_setup_once_under_concurrent_first_use(self):
        processor, backend = self._processor()

        await asyncio.gather(*(processor._ensure_ready() for _ in range(12)))

        assert backend.setup.await_count == 1
