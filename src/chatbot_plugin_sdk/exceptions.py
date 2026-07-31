from __future__ import annotations


class ToolboxError(Exception):
    """Base exception for all SDK errors.

    Any caller that only wants a single ``except`` clause for "something in
    this SDK went wrong" can catch this. More selective callers should
    prefer :class:`ExternalDependencyError` (network/infra failures worth
    retrying or falling back on) or a specific leaf class below.
    """


class NotConfiguredError(ToolboxError):
    """Raised when a processor is used before ``configure()`` is called.

    This signals caller/integration misuse (a programmer error), not a
    runtime condition to retry or fall back on — same category as the
    ``ValueError``/``TypeError`` raised by provider constructors for
    invalid arguments.
    """


class ChunkingError(ToolboxError):
    """Raised when text chunking fails."""


class ExternalDependencyError(ToolboxError):
    """Shared category for failures of infrastructure the SDK depends on
    but does not control (the configured Postgres backend, an embedding
    provider's API or model).

    This SDK has no HTTP layer of its own, so it does not map exceptions to
    status codes — that is the embedding application's job. What it does
    provide is this common ancestor so an application can, at its own API
    boundary, catch one type and translate it into its own
    external-dependency-failure category (e.g. a 502/503) without having to
    enumerate every specific leaf below.
    """


class DatabaseError(ExternalDependencyError):
    """Raised when a database operation fails."""


class EmbeddingError(ExternalDependencyError):
    """Raised when an embedding provider call fails for a reason other than
    a rate/quota limit (network error, malformed response, auth failure).

    See :class:`chatbot_plugin_sdk.rate_limit.RateLimitExhausted` for the
    rate/quota-limited case — kept as a distinct leaf (rather than folded
    into this one) because callers commonly want to react to it differently
    (back off or switch providers, instead of treating it as a hard failure).
    """
