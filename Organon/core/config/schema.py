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
class OrganonConfig:
    """Main configuration for the ORGANON application."""

    vector_store: ProviderConfig
    embedding: ProviderConfig

    def validate(self) -> None:
        """Validate the complete ORGANON configuration."""

        if not isinstance(self.vector_store, ProviderConfig):
            raise ValueError("Invalid vector store configuration.")

        if not isinstance(self.embedding, ProviderConfig):
            raise ValueError("Invalid embedding configuration.")

        self.vector_store.validate()
        self.embedding.validate()
