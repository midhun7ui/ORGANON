
"""
ORGANON - Logging Abstraction

Defines the provider-independent logging interface.
"""

from abc import ABC, abstractmethod
from typing import Any


class Logger(ABC):
    """Abstract interface for ORGANON logging providers."""

    @abstractmethod
    def debug(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record detailed diagnostic information."""
        ...

    @abstractmethod
    def info(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record normal application events."""
        ...

    @abstractmethod
    def warning(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record unexpected but recoverable situations."""
        ...

    @abstractmethod
    def error(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record application errors."""
        ...

    @abstractmethod
    def critical(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record severe application failures."""
        ...
