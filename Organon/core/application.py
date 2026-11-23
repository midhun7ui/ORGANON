
"""
ORGANON - Application Manager

Handles application initialization and component lifecycle.
"""

import inspect
from typing import Any

from core.config.schema import OrganonConfig
from core.registry import Registry
from core.security.manager import SecurityManager

from core.engine.engine import OrganonEngine

from core.metadata.base import MetadataStore
from core.metadata.registration import (
    create_metadata_store,
    register_metadata_stores,
)

from core.vectorstores.registration import (
    create_vector_store,
    register_vector_stores,
)

from core.embeddings.registration import (
    create_embedding_provider,
    register_embedding_providers,
)

from core.documents.registration import register_document_loaders
from core.chunking.registration import register_chunkers

from core.observability.manager import ObservabilityManager
from core.observability.logging.providers.register import (
    register_logging_providers,
)
from core.observability.tracing.providers.register import (
    register_tracing_providers,
)


class Organon:
    """Main application manager for ORGANON."""

    def __init__(
        self,
        config: OrganonConfig,
        security_manager: SecurityManager,
    ) -> None:
        config.validate()

        self.config = config
        self.security_manager = security_manager

        # Engine will be created during initialization.
        self.engine: OrganonEngine | None = None

        # Metadata store will be created during initialization.
        self.metadata_store: MetadataStore | None = None

        # Observability manager will be initialized with the application.
        self.observability_manager: ObservabilityManager | None = None

        # Track application lifecycle.
        self._initialized = False
        self._closed = False

        # Store created provider instances.
        self._providers: dict[str, Any] = {}

        # Registry manages available provider factories.
        self.registry = Registry()

    async def _cleanup_providers(self) -> list[Exception]:
        """Release resources held by initialized providers."""

        cleanup_errors = []

        # Clean up providers in reverse creation order.
        for provider in reversed(list(self._providers.values())):
            close = getattr(provider, "close", None)

            if not callable(close):
                continue

            try:
                result = close()

                if inspect.isawaitable(result):
                    await result

            except Exception as exc:
                cleanup_errors.append(exc)

        # Clear provider references after all cleanup attempts.
        self._providers.clear()

        return cleanup_errors

    async def initialize(self) -> None:
        """Initialize the ORGANON application."""

        # Prevent duplicate initialization.
        if self._initialized:
            return

        # Prevent initialization after shutdown.
        if self._closed:
            raise RuntimeError(
                "Cannot initialize a closed ORGANON application."
            )

        try:
            # Register available provider implementations.
            register_metadata_stores(self.registry)
            register_vector_stores(self.registry)
            register_embedding_providers(self.registry)

            # Register document loaders and chunkers.
            register_document_loaders(self.registry)
            register_chunkers(self.registry)

            # Register logging and tracing providers.
            register_logging_providers(self.registry)
            register_tracing_providers(self.registry)

            # Read provider configurations.
            metadata_store_config = self.config.metadata_store
            vector_store_config = self.config.vector_store
            embedding_config = self.config.embedding

            # Create metadata store instance.
            metadata_store = create_metadata_store(
                registry=self.registry,
                provider=metadata_store_config.provider,
                config=metadata_store_config.config,
            )

            self._providers["metadata_store"] = metadata_store
            self.metadata_store = metadata_store

            # Create vector store instance.
            vector_store = create_vector_store(
                registry=self.registry,
                provider=vector_store_config.provider,
                config=vector_store_config.config,
            )

            self._providers["vector_store"] = vector_store

            # Create embedding provider instance.
            embedding_provider = create_embedding_provider(
                registry=self.registry,
                provider=embedding_config.provider,
                config=embedding_config.config,
            )

            self._providers["embedding"] = embedding_provider

            # Connect to the metadata database.
            await metadata_store.connect()

            # Connect to the vector database.
            await vector_store.connect()

            # Initialize the embedding provider.
            await embedding_provider.connect()

            # Create the observability manager.
            observability_manager = ObservabilityManager(
                config=self.config.observability,
                registry=self.registry,
            )

            # Track it before initialization so application cleanup
            # can manage it if initialization fails.
            self._providers["observability"] = observability_manager

            # Initialize configured observability providers.
            await observability_manager.initialize()

            self.observability_manager = observability_manager

            # Retrieve configured observability providers.
            logger = observability_manager.logger
            tracer = observability_manager.tracer

            # Create the main ORGANON engine.
            self.engine = OrganonEngine(
                security_manager=self.security_manager,
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                logger=logger,
                tracer=tracer,
            )

            # Mark application as initialized.
            self._initialized = True

        except Exception as exc:
            # Reset application component references.
            self.engine = None
            self.metadata_store = None
            self.observability_manager = None

            # Attempt cleanup of all created providers.
            cleanup_errors = await self._cleanup_providers()

            # Attach cleanup failures without hiding the original error.
            for error in cleanup_errors:
                exc.add_note(
                    f"Provider cleanup failed: {error!r}"
                )

            # Re-raise the original initialization error.
            raise

    async def close(self) -> None:
        """Close the ORGANON application and release resources."""

        # Avoid closing the application multiple times.
        if self._closed:
            return

        # Release application component references.
        self.engine = None
        self.metadata_store = None
        self.observability_manager = None

        # Attempt to clean up all initialized providers.
        cleanup_errors = await self._cleanup_providers()

        # Update lifecycle state even if cleanup fails.
        self._initialized = False
        self._closed = True

        # Report cleanup failures.
        if cleanup_errors:
            error = RuntimeError(
                "One or more providers failed to close."
            )

            for cleanup_error in cleanup_errors:
                error.add_note(
                    f"Provider cleanup failed: {cleanup_error!r}"
                )

            raise error
