"""SDK exception hierarchy."""

from __future__ import annotations


class ToolboxError(Exception):
    """Base exception for all toolbox SDK errors."""

    pass


class NotConfiguredError(ToolboxError):
    """Raised when a required SDK setting is missing."""

    pass


class DatabaseError(ToolboxError):
    """Raised when a database operation fails."""

    pass


class EmbeddingError(ToolboxError):
    """Raised when embedding model HTTP call fails."""

    pass


class ChunkingError(ToolboxError):
    """Raised when text chunking fails."""

    pass


class LLMError(ToolboxError):
    """Raised when LLM generation fails and no fallback is available."""

    pass
