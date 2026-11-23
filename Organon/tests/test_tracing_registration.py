import pytest

from core.exceptions import RegistryError
from core.observability.tracing.providers.register import (
    register_tracing_providers,
)
from core.observability.tracing.providers.standard import StandardTracer
from core.registry import Registry


def test_standard_tracer_is_registered():

    registry = Registry()

    register_tracing_providers(registry)

    provider_factory = registry.get(
        category="tracing",
        name="standard",
    )

    assert provider_factory is StandardTracer


def test_registered_tracer_can_be_instantiated():

    registry = Registry()

    register_tracing_providers(registry)

    provider_factory = registry.get(
        category="tracing",
        name="standard",
    )

    tracer = provider_factory()

    assert isinstance(tracer, StandardTracer)


def test_duplicate_tracing_registration_raises_error():

    registry = Registry()

    register_tracing_providers(registry)

    with pytest.raises(RegistryError):
        register_tracing_providers(registry)