
from core.registry import Registry
from core.documents.base import DocumentLoader
from core.documents.loaders.text import TextLoader


def register_document_loaders(registry: Registry) -> None:
    """Register available document loader providers."""

    registry.register(
        category="document_loader",
        name="text",
        provider=TextLoader,
    )


def create_document_loader(
    registry: Registry,
    provider: str,
    config: dict | None = None,
) -> DocumentLoader:
    """Create a document loader from the registry."""

    provider_factory = registry.get("document_loader", provider)

    loader = provider_factory(**(config or {}))

    return loader
