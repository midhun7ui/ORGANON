
import pytest

from core.config.schema import (
    ObservabilityConfig,
    ObservabilityProviderConfig,
)
from core.observability.manager import ObservabilityManager
from core.observability.logging.providers.register import (
    register_logging_providers,
)
from core.observability.logging.providers.standard import StandardLogger
from core.registry import Registry


@pytest.mark.asyncio
async def test_manager_does_not_initialize_when_disabled():
    config = ObservabilityConfig(enabled=False)
    registry = Registry()

    manager = ObservabilityManager(config, registry)

    await manager.initialize()

    assert manager.logger is None
    assert manager._initialized is True


@pytest.mark.asyncio
async def test_manager_initializes_standard_logger():
    config = ObservabilityConfig(
        enabled=True,
        logging=ObservabilityProviderConfig(
            enabled=True,
            provider="standard",
            config={"level": "DEBUG"},
        ),
    )

    registry = Registry()
    register_logging_providers(registry)

    manager = ObservabilityManager(config, registry)

    await manager.initialize()

    assert isinstance(manager.logger, StandardLogger)
    assert manager.logger._logger.level == 10
    assert manager._initialized is True


@pytest.mark.asyncio
async def test_manager_skips_disabled_logging():
    config = ObservabilityConfig(
        enabled=True,
        logging=ObservabilityProviderConfig(enabled=False),
    )

    registry = Registry()
    manager = ObservabilityManager(config, registry)

    await manager.initialize()

    assert manager.logger is None
    assert manager._initialized is True


@pytest.mark.asyncio
async def test_manager_initialization_is_idempotent():
    config = ObservabilityConfig(enabled=True)

    registry = Registry()
    register_logging_providers(registry)

    manager = ObservabilityManager(config, registry)

    await manager.initialize()
    first_logger = manager.logger

    await manager.initialize()

    assert manager.logger is first_logger


@pytest.mark.asyncio
async def test_manager_closes_synchronous_provider():
    class TestProvider:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    provider = TestProvider()

    config = ObservabilityConfig(
        enabled=True,
        logging=ObservabilityProviderConfig(
            enabled=True,
            provider="test",
        ),
    )

    registry = Registry()
    registry.register("logging", "test", lambda: provider)

    manager = ObservabilityManager(config, registry)

    await manager.initialize()
    await manager.close()

    assert provider.closed is True
    assert manager.logger is None
    assert manager._closed is True


@pytest.mark.asyncio
async def test_manager_closes_asynchronous_provider():
    class TestProvider:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    provider = TestProvider()

    config = ObservabilityConfig(
        enabled=True,
        logging=ObservabilityProviderConfig(
            enabled=True,
            provider="test",
        ),
    )

    registry = Registry()
    registry.register("logging", "test", lambda: provider)

    manager = ObservabilityManager(config, registry)

    await manager.initialize()
    await manager.close()

    assert provider.closed is True
    assert manager._closed is True


@pytest.mark.asyncio
async def test_manager_cannot_initialize_after_close():
    manager = ObservabilityManager(
        ObservabilityConfig(enabled=True),
        Registry(),
    )

    await manager.close()

    with pytest.raises(RuntimeError):
        await manager.initialize()



@pytest.mark.asyncio
async def test_manager_continues_cleanup_when_provider_fails():
    class FailingProvider:
        def close(self):
            raise RuntimeError("Cleanup failed")

    class WorkingProvider:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    failing_provider = FailingProvider()
    working_provider = WorkingProvider()

    config = ObservabilityConfig(
        enabled=True,
        logging=ObservabilityProviderConfig(
            enabled=True,
            provider="test",
        ),
    )

    registry = Registry()
    registry.register(
        "logging",
        "test",
        lambda: working_provider,
    )

    manager = ObservabilityManager(config, registry)

    await manager.initialize()

    # Simulate another initialized provider.
    manager.tracer = failing_provider

    with pytest.raises(RuntimeError):
        await manager.close()

    assert working_provider.closed is True
