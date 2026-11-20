
import pytest

from core.registry import Registry
from core.documents.loaders.text import TextLoader
from core.documents.registration import (
    register_document_loaders,
    create_document_loader,
)
from core.exceptions import RegistryError


def test_register_document_loaders():
    registry = Registry()

    register_document_loaders(registry)

    providers = registry.list_providers("document_loader")

    assert "text" in providers


def test_create_text_loader():
    registry = Registry()

    register_document_loaders(registry)

    loader = create_document_loader(
        registry=registry,
        provider="text",
    )

    assert isinstance(loader, TextLoader)


def test_create_unknown_loader():
    registry = Registry()

    with pytest.raises(RegistryError):
        create_document_loader(
            registry=registry,
            provider="unknown",
        )
