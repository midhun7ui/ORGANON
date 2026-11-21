"""
ORGANON - Metadata Store Registration

Registers metadata store providers with the ORGANON registry.
"""

from typing import Any

from core.registry import Registry
from core.metadata.base import MetadataStore
from core.metadata.providers.postgresql.database import PostgreSQLDatabase
from core.metadata.providers.postgresql.store import PostgreSQLMetadataStore


def register_metadata_stores(registry: Registry) -> None:
    """Register available metadata store providers."""

    registry.register(
        category="metadata_store",
        name="postgresql",
        provider=PostgreSQLMetadataStore,
    )


def create_metadata_store(
    registry: Registry,
    provider: str,
    config: dict[str, Any],
) -> MetadataStore:
    """Create a metadata store instance from the registered provider."""

    # Retrieve the provider factory from the registry.
    provider_factory = registry.get("metadata_store", provider)

    # Create the database manager using the supplied configuration.
    database_url = config["database_url"]
    database = PostgreSQLDatabase(database_url)

    # Inject the database manager into the metadata provider.
    metadata_store = provider_factory(database=database)

    return metadata_store