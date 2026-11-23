"""
ORGANON - Tracing Abstraction

Defines the provider-independent tracing interface.
"""

from abc import ABC, abstractmethod
from typing import Any


class Span(ABC):
    """Represents a single operation within a trace."""

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None:
        """Attach metadata to the span."""
        pass

    @abstractmethod
    def record_exception(self, exception: Exception) -> None:
        """Record an exception occurring within the span."""
        pass

    @abstractmethod
    def set_status(self, status: str) -> None:
        """Set the span execution status."""
        pass

    @abstractmethod
    def end(self) -> None:
        """Finish the span and record its duration."""
        pass

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_value is not None:
            self.record_exception(exc_value)
            self.set_status("error")

        self.end()


class Tracer(ABC):
    """Creates and manages tracing spans."""

    @abstractmethod
    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Create a new span without making it the current span."""
        pass

    @abstractmethod
    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Create a span and make it current within its context."""
        pass