from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class DatabaseConfig:
    dbname: str
    user: str
    password: str
    host: str = "localhost"
    port: int = 5432
    schema: str = "vectors"


@dataclass
class _RuntimeDatabase:
    """Internal: holds live SQLAlchemy engine + session factory after configure()."""
    engine: Any       # AsyncEngine at runtime
    session_factory: Any  # async_sessionmaker[AsyncSession] at runtime
    schema: str
