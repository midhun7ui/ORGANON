
import pytest

from core.config.loader import load_yaml, parse_config, load_config
from core.exceptions import ConfigurationError
from core.config.schema import OrganonConfig, ProviderConfig


def test_load_valid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
metadata_store:
  provider: postgresql
  config:
    database_url: postgresql+asyncpg://localhost/organon

vector_store:
  provider: chroma
  config:
    host: localhost
    port: 8000
""",
        encoding="utf-8",
    )

    result = load_yaml(config_file)

    assert result["metadata_store"]["provider"] == "postgresql"
    assert result["vector_store"]["provider"] == "chroma"
    assert result["vector_store"]["config"]["port"] == 8000


def test_load_missing_yaml_file():
    with pytest.raises(ConfigurationError):
        load_yaml("missing_config.yaml")


def test_load_invalid_yaml(tmp_path):
    config_file = tmp_path / "invalid.yaml"

    config_file.write_text(
        "vector_store: [invalid yaml",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_yaml(config_file)


def test_load_yaml_with_non_dictionary_root(tmp_path):
    config_file = tmp_path / "invalid_root.yaml"

    config_file.write_text(
        "- chroma\n- sentence_transformers",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_yaml(config_file)


def test_parse_valid_config():
    data = {
        "metadata_store": {
            "provider": "postgresql",
            "config": {"database_url": "postgresql+asyncpg://localhost/organon"},
        },
        "vector_store": {
            "provider": "chroma",
            "config": {"host": "localhost", "port": 8000},
        },
        "embedding": {
            "provider": "sentence_transformers",
            "config": {"device": "cpu"},
        },
    }

    result = parse_config(data)

    assert isinstance(result, OrganonConfig)
    assert isinstance(result.metadata_store, ProviderConfig)
    assert isinstance(result.vector_store, ProviderConfig)
    assert isinstance(result.embedding, ProviderConfig)

    assert result.metadata_store.provider == "postgresql"
    assert result.vector_store.provider == "chroma"
    assert result.embedding.provider == "sentence_transformers"


def test_parse_config_rejects_missing_section():
    data = {
        "vector_store": {
            "provider": "chroma",
            "config": {},
        }
    }

    with pytest.raises(ConfigurationError):
        parse_config(data)


def test_parse_config_rejects_invalid_provider():
    data = {
        "metadata_store": {
            "provider": "postgresql",
            "config": {},
        },
        "vector_store": {
            "provider": "",
            "config": {},
        },
        "embedding": {
            "provider": "sentence_transformers",
            "config": {},
        },
    }

    with pytest.raises(ConfigurationError):
        parse_config(data)


def test_parse_config_rejects_invalid_input():
    with pytest.raises(ConfigurationError):
        parse_config("invalid")


def test_load_config_returns_organon_config(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
metadata_store:
  provider: postgresql
  config:
    database_url: postgresql+asyncpg://localhost/organon

vector_store:
  provider: chroma
  config:
    host: localhost
    port: 8000

embedding:
  provider: sentence_transformers
  config:
    model_name: all-MiniLM-L6-v2
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert isinstance(config, OrganonConfig)
    assert config.metadata_store.provider == "postgresql"
    assert config.vector_store.provider == "chroma"
    assert config.embedding.provider == "sentence_transformers"

def test_parse_config_without_observability():
    data = {
        "metadata_store": {"provider": "postgresql", "config": {}},
        "vector_store": {"provider": "chroma", "config": {}},
        "embedding": {"provider": "sentence_transformers", "config": {}},
    }

    result = parse_config(data)

    assert result.observability.enabled is False
    assert result.observability.logging.enabled is True
    assert result.observability.logging.provider == "standard"
    assert result.observability.tracing.enabled is False
    assert result.observability.metrics.enabled is False

def test_parse_config_with_observability_enabled():
    data = {
        "metadata_store": {"provider": "postgresql", "config": {}},
        "vector_store": {"provider": "chroma", "config": {}},
        "embedding": {"provider": "sentence_transformers", "config": {}},
        "observability": {
            "enabled": True,
        },
    }

    result = parse_config(data)

    assert result.observability.enabled is True
    assert result.observability.logging.enabled is True
    assert result.observability.logging.provider == "standard"

def test_parse_config_with_custom_observability_providers():
    data = {
        "metadata_store": {"provider": "postgresql", "config": {}},
        "vector_store": {"provider": "chroma", "config": {}},
        "embedding": {"provider": "sentence_transformers", "config": {}},
        "observability": {
            "enabled": True,
            "logging": {
                "enabled": True,
                "provider": "custom_logger",
                "config": {"level": "DEBUG"},
            },
            "tracing": {
                "enabled": True,
                "provider": "opentelemetry",
                "config": {"endpoint": "localhost:4317"},
            },
            "metrics": {
                "enabled": True,
                "provider": "prometheus",
                "config": {"port": 9090},
            },
        },
    }

    result = parse_config(data)

    assert result.observability.logging.provider == "custom_logger"
    assert result.observability.logging.config["level"] == "DEBUG"

    assert result.observability.tracing.enabled is True
    assert result.observability.tracing.provider == "opentelemetry"

    assert result.observability.metrics.enabled is True
    assert result.observability.metrics.provider == "prometheus"

def test_parse_config_rejects_invalid_observability():
    data = {
        "metadata_store": {"provider": "postgresql", "config": {}},
        "vector_store": {"provider": "chroma", "config": {}},
        "embedding": {"provider": "sentence_transformers", "config": {}},
        "observability": "invalid",
    }

    with pytest.raises(ConfigurationError):
        parse_config(data)