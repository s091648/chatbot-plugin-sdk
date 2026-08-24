from __future__ import annotations
from contextlib import contextmanager
from typing import Any, ContextManager, Iterator, Protocol, runtime_checkable

# Exception contract for implementers of either protocol below: a failed
# embed() call must raise chatbot_plugin_sdk.exceptions.EmbeddingError (or a
# more specific subclass — e.g. RateLimitExhausted for a rate/quota limit),
# never let a provider-internal exception (an HTTP client error, a raw SDK
# exception from the underlying model API, etc.) propagate unwrapped. This
# lets IngestProcessor/RetrieveProcessor callers catch one SDK-owned
# exception family regardless of which provider is configured. See the
# built-in providers (gemini.py, huggingface.py, endpoint.py, local.py) for
# the pattern.


@runtime_checkable
class DenseEmbeddingProvider(Protocol):
    """HTTP endpoint 或 in-process callable，輸出 dense 向量。

    dimension 屬性供 ensure_ready() 在首次建表時決定 VECTOR(N) 的 N。
    """
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@runtime_checkable
class SparseEmbeddingProvider(Protocol):
    """HTTP endpoint 或 in-process callable，輸出 sparse 向量（token_id → weight）。

    dimension 屬性為詞彙表大小（SPLADE / BERT: 30522），供 setup() 建立 SPARSEVEC(N) 欄位使用。
    """
    dimension: int

    async def embed(self, texts: list[str]) -> list[dict[str, float]]:
        ...


# ---------------------------------------------------------------------------
# Tracing — kept as duck-typed Protocols (not a hard `opentelemetry` import)
# so the SDK stays usable with zero tracing dependency installed at all.
# `opentelemetry.trace`'s real `Tracer`/`Span` classes already structurally
# satisfy these two methods, so passing a real OTel tracer in needs no
# adapter — see `default_tracer()` below for how the built-in providers
# resolve one when the caller doesn't inject their own.
# ---------------------------------------------------------------------------


@runtime_checkable
class Span(Protocol):
    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        ...


@runtime_checkable
class Tracer(Protocol):
    def start_as_current_span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> ContextManager[Span]:
        ...


class NoOpSpan:
    """Zero-cost Span used wherever no real tracer is configured."""

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass


class NoOpTracer:
    """Zero-cost Tracer — the SDK's fallback when opentelemetry-api isn't
    installed (it's an optional dependency, see pyproject.toml's `otel`
    extra) and no tracer was explicitly injected. Every call is a no-op, so
    providers/processors can call tracing methods unconditionally without
    needing to check whether tracing is actually configured."""

    @contextmanager
    def start_as_current_span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> Iterator[Span]:
        yield NoOpSpan()


def default_tracer(name: str) -> Tracer:
    """Resolve the tracer a provider/processor should use when the caller
    doesn't inject its own: real OpenTelemetry if opentelemetry-api happens
    to be installed (unchanged behavior for any existing caller that already
    depends on it directly, e.g. chatbot-plugin), otherwise a NoOpTracer so
    tracing calls are free no-ops instead of an ImportError at import time."""
    try:
        from opentelemetry import trace as _otel_trace
    except ImportError:
        return NoOpTracer()
    return _otel_trace.get_tracer(name)
