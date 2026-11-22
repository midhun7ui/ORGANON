"""
ORGANON - Standard Logging Provider

Implements the Logger interface using Python's built-in logging module.
"""

import logging
from typing import Any
    
from core.observability.logging.base import Logger


class StandardLogger(Logger):
    """Standard Python logging implementation for ORGANON."""

    def __init__(
        self,
        name: str = "organon",
        level: str | int = "INFO",
    ) -> None:
        """Initialize the standard logging provider."""

        self._logger = logging.getLogger(name)

        if isinstance(level, str):
            log_level = logging.getLevelName(level.upper())

            if not isinstance(log_level, int):
                raise ValueError(f"Invalid logging level: {level}")

        elif isinstance(level, int) and not isinstance(level, bool):
            if level < 0:
                raise ValueError(f"Invalid logging level: {level}")

            log_level = level

        else:
            raise ValueError(f"Invalid logging level: {level}")

        self._logger.setLevel(log_level)

    def _log(
        self,
        level: int,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a log message with optional structured context."""

        self._logger.log(
            level,
            message,
            extra={
                "organon_context": dict(context) if context else {},
            },
        )

    def debug(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        self._log(logging.DEBUG, message, context)

    def info(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        self._log(logging.INFO, message, context)

    def warning(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        self._log(logging.WARNING, message, context)

    def error(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        self._log(logging.ERROR, message, context)

    def critical(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        self._log(logging.CRITICAL, message, context)
