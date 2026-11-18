
# The registry allows ORGANON to register, discover, and retrieve
# different provider implementations without hardcoding providers
# into the core engine.

"""
ORGANON - Provider Registry

Manages provider registration and discovery.
"""

# Any: Represents a value of any Python type.
# Callable: Represents an object that can be called like a function.
from typing import Any, Callable

# Import our custom exception for handling registry-related errors.
from core.exceptions import RegistryError


# A ProviderFactory is a callable that can accept any arguments
# and return any type of object.
#
# This allows us to register different provider classes or functions
# without restricting their implementation.
ProviderFactory = Callable[..., Any]


class Registry:
    """Central registry for ORGANON provider implementations."""

    def __init__(self) -> None:
        # Initialize an empty dictionary to store registered providers.
        #
        # Dictionary structure:
        # {
        #     ("embeddings", "openai"): OpenAIEmbeddingProvider,
        #     ("embeddings", "huggingface"): HuggingFaceEmbeddingProvider
        # }
        #
        # The key is a tuple containing:
        # (provider_category, provider_name)
        #
        # The value is the provider factory (class or callable).
        self._providers: dict[tuple[str, str], ProviderFactory] = {}

    def register(
        self,
        category: str,
        name: str,
        provider: ProviderFactory,
    ) -> None:
        """Register a provider under a category and name."""

        # Validate the category and provider name.
        # Empty values are not allowed.
        if not category or not name:
            raise RegistryError(
                "provider category and name cannot be empty."
            )

        # Normalize category and name to lowercase.
        # This makes registration case-insensitive.
        #
        # Example:
        # ("Embeddings", "OpenAI") becomes ("embeddings", "openai")
        key = (category.lower(), name.lower())

        # Check whether a provider with this category and name
        # is already registered.
        if key in self._providers:
            raise RegistryError(
                f"Provider '{name}' is already registered "
                f"under category '{category}'."
            )

        # Store the provider factory in the dictionary.
        # The provider can be retrieved later using its category and name.
        self._providers[key] = provider

    def get(self, category: str, name: str) -> ProviderFactory:
        """Retrieve a registered provider."""

        # Normalize category and name to lowercase
        # to maintain case-insensitive lookup.
        key = (category.lower(), name.lower())

        # Try to retrieve the provider using its dictionary key.
        try:
            return self._providers[key]

        # If the key doesn't exist, Python raises a KeyError.
        # We convert it into our custom RegistryError.
        except KeyError:
            raise RegistryError(
                f"Provider '{name}' was not found "
                f"under category '{category}'."
            ) from None

    def unregister(self, category: str, name: str) -> None:
        """Remove a registered provider."""

        # Normalize category and name before searching.
        key = (category.lower(), name.lower())

        # Check whether the provider exists before removing it.
        if key not in self._providers:
            raise RegistryError(
                f"Provider '{name}' is not registered "
                f"under category '{category}'."
            )

        # Remove the provider from the registry.
        del self._providers[key]

    def list_providers(self, category: str) -> list[str]:
        """List registered providers in a category."""

        # Normalize the category to lowercase.
        normalized_category = category.lower()

        # Iterate through all registered providers.
        # Return only the provider names that belong to the requested category.
        return [
            name
            for provider_category, name in self._providers
            if provider_category == normalized_category
        ]
