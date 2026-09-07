from __future__ import annotations

import os
from dataclasses import dataclass

# Directory where fastembed downloads and caches ONNX models.
# Leave unset to use fastembed's default (~/.cache/fastembed).
FASTEMBED_CACHE_PATH: str | None = os.environ.get("FASTEMBED_CACHE_PATH") or None


@dataclass
class DatabaseConfig:
    """Connection parameters for PostgreSQL.  Passed to AsyncPgBackend or SyncPgBackend."""
    dbname: str
    user: str
    password: str
    host: str = "localhost"
    port: int = 5432
    schema: str = "vectors"
    articles_table: str = "articles"
    chunks_table: str = "article_chunks"

    # ── Connection-pool tuning (AsyncPgBackend / SyncPgBackend) ────────────
    # Defaults preserve the pre-1.3.1 behaviour (pool_size=5, max_overflow=10)
    # and only add a bounded connect timeout + pre-ping. A host that fans out
    # many concurrent ingest() calls should raise pool_size and keep
    # connect_timeout generous: every brand-new connection's DNS lookup runs
    # on asyncio's small default ThreadPoolExecutor, so a burst of cold
    # connects can queue past a tight timeout even when the DB is healthy.
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 1800
    pool_pre_ping: bool = True
    # Seconds to wait for a single new connection to be established (DNS + TCP
    # + TLS + auth). asyncpg's own default is 60s; psycopg2 has none.
    connect_timeout: float = 30.0
