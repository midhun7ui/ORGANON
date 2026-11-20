from core.config.loader import resolve_environment_variables
import pytest
from core.exceptions import ConfigurationError

def test_resolves_environment_variable(monkeypatch):
    monkeypatch.setenv("CHROMA_HOST", "localhost")

    config = {
        "host": "${CHROMA_HOST}"
    }

    resolved = resolve_environment_variables(config)

    assert resolved["host"] == "localhost"

def test_resolves_nested_environment_variables(monkeypatch):
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PORT", "8000")

    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "host": "${CHROMA_HOST}",
                "port": "${CHROMA_PORT}",
            },
        },
        "allowed_hosts": ["${CHROMA_HOST}"],
    }

    resolved = resolve_environment_variables(config)

    assert resolved["vector_store"]["config"]["host"] == "localhost"
    assert resolved["vector_store"]["config"]["port"] == "8000"
    assert resolved["allowed_hosts"] == ["localhost"]

def test_raises_error_for_missing_environment_variable():
    config = {"host": "${MISSING_CHROMA_HOST}"}

    with pytest.raises(ConfigurationError, match="MISSING_CHROMA_HOST"):
        resolve_environment_variables(config)

def test_resolves_multiple_variables_in_one_string(monkeypatch):
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PORT", "8000")

    config = {
        "url": "http://${CHROMA_HOST}:${CHROMA_PORT}"
    }

    resolved = resolve_environment_variables(config)

    assert resolved["url"] == "http://localhost:8000"

def test_preserves_non_string_values():
    config = {
        "port": 8000,
        "enabled": True,
        "optional": None,
    }

    resolved = resolve_environment_variables(config)

    assert resolved["port"] == 8000
    assert resolved["enabled"] is True
    assert resolved["optional"] is None

from core.config.loader import load_config


def test_load_config_resolves_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_HOST", "localhost")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
vector_store:
  provider: chroma
  config:
    host: ${CHROMA_HOST}
    port: 8000

embedding:
  provider: sentence_transformers
  config:
    model_name: test-model
    device: cpu
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.vector_store.config["host"] == "localhost"
    assert config.vector_store.config["port"] == 8000