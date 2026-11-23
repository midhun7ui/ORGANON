"""
ORGANON - Observability Manager

Coordinates logging, tracing, and metrics providers.
"""

from typing import Any

from core.config.schema import ObservabilityConfig
from core.registry import Registry


class ObservabilityManager:
    """Manages ORGANON observability providers."""

    def __init__(
        self,
        config: ObservabilityConfig,
        registry: Registry,
    ) -> None:
        """Initialize the observability manager."""

        self.config = config
        self.registry = registry

        self.logger: Any | None = None
        self.tracer: Any | None = None
        self.metrics: Any | None = None

        self._initialized = False
        self._closed = False

    async def initialize(self) -> None:
        """Initialize enabled observability providers."""

        if self._initialized:
            return

        if self._closed:
            raise RuntimeError("Cannot initialize a closed ObservabilityManager.")

        # Respect the master observability switch.
        if not self.config.enabled:
            self._initialized = True
            return

        # Initialize the configured logging provider.
        if self.config.logging.enabled:
            provider_factory = self.registry.get(
                category="logging",
                name=self.config.logging.provider,
            )

            self.logger = provider_factory(**self.config.logging.config)

        # Initialize the configured tracing provider.
        if self.config.tracing.enabled:
            provider_factory = self.registry.get(
                category="tracing",
                name=self.config.tracing.provider,
            )

            self.tracer = provider_factory(**self.config.tracing.config)

        self._initialized = True
        # Validate observability configuration.
        self.config.validate()

    async def close(self) -> None:
        """Close initialized observability providers."""

        if self._closed:
            return

        providers = [
            self.logger,
            self.tracer,
            self.metrics,
        ]

        cleanup_errors = []

        # Close providers in reverse initialization order.
        for provider in reversed(providers):
            if provider is None:
                continue

            close = getattr(provider, "close", None)

            if close is None:
                continue

            try:
                result = close()

                if hasattr(result, "__await__"):
                    await result

            except Exception as error:
                cleanup_errors.append(error)

        # Clear provider references even if cleanup failed.
        self.logger = None
        self.tracer = None
        self.metrics = None

        self._initialized = False
        self._closed = True

        # Report errors after attempting every cleanup.
        if cleanup_errors:
            error = RuntimeError("One or more observability providers failed to close.")

            for cleanup_error in cleanup_errors:
                error.add_note(f"{type(cleanup_error).__name__}: {cleanup_error}")

            raise error from cleanup_errors[0]
