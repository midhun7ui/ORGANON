"""
ORGANON - Configuration Loader

Loads configuration data from external sources.
"""

import os
import re

from pathlib import Path
from typing import Any

import yaml

from core.exceptions import ConfigurationError
from core.config.schema import OrganonConfig, ProviderConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load configuration data from a YAML file."""

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML configuration file: {path}") from exc

    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration file: {path}") from exc

    # Ensure the YAML root is a dictionary.
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration file must contain a YAML mapping.")

    return data


def parse_config(data: dict[str, Any]) -> OrganonConfig:
    """Convert configuration data into an OrganonConfig object."""

    if not isinstance(data, dict):
        raise ConfigurationError("Configuration data must be a dictionary.")

    try:
        # Extract provider configurations.
        metadata_store_data = data["metadata_store"]
        vector_store_data = data["vector_store"]
        embedding_data = data["embedding"]

        # Create provider configuration objects.
        metadata_store = ProviderConfig(**metadata_store_data)
        vector_store = ProviderConfig(**vector_store_data)
        embedding = ProviderConfig(**embedding_data)

        # Create the main application configuration.
        config = OrganonConfig(
            metadata_store=metadata_store,
            vector_store=vector_store,
            embedding=embedding,
        )

        # Validate the complete configuration.
        config.validate()

        return config

    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid ORGANON configuration: {exc}") from exc


def resolve_environment_variables(data: Any) -> Any:
    """Resolve environment variable references in configuration data."""

    if isinstance(data, dict):
        return {
            key: resolve_environment_variables(value) for key, value in data.items()
        }

    if isinstance(data, list):
        return [resolve_environment_variables(item) for item in data]

    if isinstance(data, str):
        pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"

        def replace(match: re.Match) -> str:
            variable = match.group(1)
            value = os.environ.get(variable)

            if value is None:
                raise ConfigurationError(
                    f"Environment variable '{variable}' is not set."
                )

            return value

        return re.sub(pattern, replace, data)

    return data


def load_config(path: str | Path) -> OrganonConfig:
    """Load and validate ORGANON configuration from a YAML file."""

    data = load_yaml(path)
    data = resolve_environment_variables(data)

    return parse_config(data)
