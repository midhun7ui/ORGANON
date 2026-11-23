
import pytest

from core.observability.logging.providers.register import (
    register_logging_providers,
)
from core.observability.logging.providers.standard import StandardLogger
from core.registry import Registry
from core.exceptions import RegistryError


def test_standard_logger_is_registered():
    registry = Registry()

    register_logging_providers(registry)

    provider = registry.get("logging", "standard")

    assert provider is StandardLogger


def test_registered_logger_can_be_instantiated():
    registry = Registry()

    register_logging_providers(registry)

    provider_factory = registry.get("logging", "standard")
    logger = provider_factory(level="DEBUG")

    assert isinstance(logger, StandardLogger)


def test_logging_provider_registration_is_not_duplicated():
    registry = Registry()

    register_logging_providers(registry)

    with pytest.raises(RegistryError):
        register_logging_providers(registry)
