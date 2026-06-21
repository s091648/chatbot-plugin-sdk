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
