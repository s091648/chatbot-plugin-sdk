from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Connection parameters for PostgreSQL.  Passed to AsyncPgBackend or SyncPgBackend."""
    dbname: str
    user: str
    password: str
    host: str = "localhost"
    port: int = 5432
    schema: str = "vectors"
