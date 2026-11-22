"""
ORGANON - Configuration Schema

Defines the structure of provider configurations.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderConfig:
    """Configuration for an individual provider."""

    provider: str
    config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate provider configuration."""

        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("Provider name must be a non-empty string.")

        if not isinstance(self.config, dict):
            raise ValueError("Provider config must be a dictionary.")


@dataclass
class ObservabilityProviderConfig:
    """Configuration for an observability provider."""

    enabled: bool = False
    provider: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate observability provider configuration."""

        if not isinstance(self.enabled, bool):
            raise ValueError("Provider enabled must be a boolean.")

        if self.enabled and (
            not isinstance(self.provider, str) or not self.provider.strip()
        ):
            raise ValueError("Enabled provider must have a valid provider name.")

        if self.provider is not None and not isinstance(self.provider, str):
            raise ValueError("Provider name must be a string or None.")

        if not isinstance(self.config, dict):
            raise ValueError("Provider config must be a dictionary.")


@dataclass
class ObservabilityConfig:
    """Configuration for ORGANON observability."""

    enabled: bool = False

    logging: ObservabilityProviderConfig = field(
        default_factory=lambda: ObservabilityProviderConfig(
            enabled=True,
            provider="standard",
        )
    )

    tracing: ObservabilityProviderConfig = field(
        default_factory=ObservabilityProviderConfig
    )

    metrics: ObservabilityProviderConfig = field(
        default_factory=ObservabilityProviderConfig
    )

    def validate(self) -> None:
        """Validate observability configuration."""

        if not isinstance(self.enabled, bool):
            raise ValueError("Observability enabled must be a boolean.")

        self.logging.validate()
        self.tracing.validate()
        self.metrics.validate()


@dataclass
class OrganonConfig:
    """Main configuration for the ORGANON application."""

    metadata_store: ProviderConfig
    vector_store: ProviderConfig
    embedding: ProviderConfig

    observability: ObservabilityConfig = field(
        default_factory=ObservabilityConfig
    )

    def validate(self) -> None:
        """Validate the complete ORGANON configuration."""

        if not isinstance(self.metadata_store, ProviderConfig):
            raise ValueError("Invalid metadata store configuration.")

        if not isinstance(self.vector_store, ProviderConfig):
            raise ValueError("Invalid vector store configuration.")

        if not isinstance(self.embedding, ProviderConfig):
            raise ValueError("Invalid embedding configuration.")

        if not isinstance(self.observability, ObservabilityConfig):
            raise ValueError("Invalid observability configuration.")

        self.metadata_store.validate()
        self.vector_store.validate()
        self.embedding.validate()
        self.observability.validate()