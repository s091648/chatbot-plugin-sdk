from __future__ import annotations


class ToolboxError(Exception):
    """Base exception for all SDK errors."""


class NotConfiguredError(ToolboxError):
    """Raised when a required SDK setting is missing."""


class DatabaseError(ToolboxError):
    """Raised when a database operation fails."""


class EmbeddingError(ToolboxError):
    """Raised when embedding model call fails."""


class ChunkingError(ToolboxError):
    """Raised when text chunking fails."""
