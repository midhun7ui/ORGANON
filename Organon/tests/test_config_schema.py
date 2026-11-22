
import pytest

from core.config.schema import (
    OrganonConfig,
    ProviderConfig,
    ObservabilityConfig,
    ObservabilityProviderConfig,
)


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
        metadata_store=ProviderConfig("postgresql", {}),
        vector_store=ProviderConfig("chroma", {}),
        embedding=ProviderConfig("sentence_transformers", {}),
    )

    config.validate()


def test_organon_config_rejects_invalid_provider():
    config = OrganonConfig(
        metadata_store=ProviderConfig("postgresql", {}),
        vector_store="chroma",
        embedding=ProviderConfig("sentence_transformers", {}),
    )

    with pytest.raises(ValueError):
        config.validate()

def test_observability_config_defaults():
    config = ObservabilityConfig()

    assert config.enabled is False

    assert config.logging.enabled is True
    assert config.logging.provider == "standard"

    assert config.tracing.enabled is False
    assert config.metrics.enabled is False

    config.validate()


def test_observability_provider_config_valid():
    config = ObservabilityProviderConfig(
        enabled=True,
        provider="opentelemetry",
        config={"endpoint": "localhost"},
    )

    config.validate()


def test_observability_provider_rejects_missing_provider():
    config = ObservabilityProviderConfig(
        enabled=True,
        provider=None,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_observability_provider_rejects_invalid_config():
    config = ObservabilityProviderConfig(
        enabled=True,
        provider="standard",
        config="invalid",
    )

    with pytest.raises(ValueError):
        config.validate()


def test_organon_config_has_default_observability():
    config = OrganonConfig(
        metadata_store=ProviderConfig("postgresql", {}),
        vector_store=ProviderConfig("chroma", {}),
        embedding=ProviderConfig("sentence_transformers", {}),
    )

    assert isinstance(config.observability, ObservabilityConfig)
    assert config.observability.enabled is False

    config.validate()


def test_organon_config_rejects_invalid_observability():
    config = OrganonConfig(
        metadata_store=ProviderConfig("postgresql", {}),
        vector_store=ProviderConfig("chroma", {}),
        embedding=ProviderConfig("sentence_transformers", {}),
        observability="invalid",
    )

    with pytest.raises(ValueError):
        config.validate()