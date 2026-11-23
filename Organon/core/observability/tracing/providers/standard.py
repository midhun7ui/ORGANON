"""
ORGANON - Standard Tracing Provider

Provides a lightweight, dependency-free tracing implementation.
"""

from time import perf_counter_ns, time_ns
from typing import Any
from uuid import uuid4
from contextvars import ContextVar
from typing import Any
from core.observability.tracing.base import Span, Tracer
from contextlib import contextmanager
from collections.abc import Iterator


class StandardSpan(Span):
    """Represents an individual tracing span."""

    def __init__(
        self,
        name: str,
        *,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a new span."""

        self.name = name
        self.span_id = uuid4().hex
        self.parent_span_id = parent_span_id

        self.attributes = dict(attributes) if attributes else {}

        self.status = "unset"
        self.exception_type: str | None = None

        self.start_time = time_ns()
        self.end_time: int | None = None

        self._start_counter = perf_counter_ns()
        self.duration_ns: int | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach metadata to the span."""

        self.attributes[key] = value

    def record_exception(self, exception: Exception) -> None:
        """Record the exception type without retaining sensitive messages."""

        self.exception_type = type(exception).__name__

    def set_status(self, status: str) -> None:
        """Set the span execution status."""

        self.status = status

    def end(self) -> None:
        """Finish the span and record its duration."""

        if self.end_time is not None:
            return

        self.end_time = time_ns()
        self.duration_ns = perf_counter_ns() - self._start_counter


class StandardTracer(Tracer):
    """Dependency-free tracing provider for ORGANON."""

    def __init__(self) -> None:
        """Initialize the tracer context."""

        self._current_span: ContextVar[StandardSpan | None] = ContextVar(
            "organon_current_span",
            default=None,
        )

    def get_current_span(self) -> StandardSpan | None:
        """Return the currently active span."""
        return self._current_span.get()

    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> StandardSpan:
        """Create a new span without changing the active context."""

        parent_span = self.get_current_span()

        return StandardSpan(
            name=name,
            parent_span_id=parent_span.span_id if parent_span else None,
            attributes=attributes,
        )

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[StandardSpan]:
        """Create a span and make it active within its context."""

        span = self.start_span(
            name=name,
            attributes=attributes,
        )

        token = self._current_span.set(span)

        try:
            with span:
                yield span

        finally:
            self._current_span.reset(token)
