
import pytest

from core.config.schema import OrganonConfig, ProviderConfig


def test_provider_config_valid():
    config = ProviderConfig(
        provider="chroma",
        config={"host": "localhost", "port": 8000},
    )

    config.validate()


def test_provider_config_rejects_empty_provider():
    config = ProviderConfig(provider="", config={})

    with pytest.raises(ValueError):
        config.validate()


def test_provider_config_rejects_invalid_config():
    config = ProviderConfig(
        provider="chroma",
        config="localhost",
    )

    with pytest.raises(ValueError):
        config.validate()


def test_organon_config_valid():
    config = OrganonConfig(
        vector_store=ProviderConfig("chroma", {}),
        embedding=ProviderConfig("sentence_transformers", {}),
    )

    config.validate()


def test_organon_config_rejects_invalid_provider():
    config = OrganonConfig(
        vector_store="chroma",
        embedding=ProviderConfig("sentence_transformers", {}),
    )

    with pytest.raises(ValueError):
        config.validate()
