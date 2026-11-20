from unittest.mock import AsyncMock, MagicMock

import pytest

from core.application import Organon
from core.security.manager import SecurityManager
from core.config.schema import OrganonConfig, ProviderConfig


@pytest.fixture
def security_manager():
    return MagicMock(spec=SecurityManager)


@pytest.fixture
def app_config():
    return OrganonConfig(
        vector_store=ProviderConfig(
            provider="chroma",
            config={
                "host": "localhost",
                "port": 8000,
                "collection_name": "organon",
            },
        ),
        embedding=ProviderConfig(
            provider="sentence_transformers",
            config={
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
            },
        ),
    )


@pytest.fixture
def mock_providers(monkeypatch):
    vector_store = MagicMock()
    vector_store.connect = AsyncMock()
    vector_store.close = AsyncMock()

    embedding_provider = MagicMock()
    embedding_provider.connect = AsyncMock()
    embedding_provider.close = AsyncMock()

    monkeypatch.setattr(
        "core.application.create_vector_store",
        lambda **kwargs: vector_store,
    )

    monkeypatch.setattr(
        "core.application.create_embedding_provider",
        lambda **kwargs: embedding_provider,
    )

    return vector_store, embedding_provider


@pytest.mark.asyncio
async def test_application_initialization(security_manager, app_config, mock_providers):
    app = Organon(
        config=app_config,
        security_manager=security_manager,
    )

    await app.initialize()

    assert app._initialized is True
    assert app._closed is False
    assert app.engine is not None


@pytest.mark.asyncio
async def test_application_initialization_is_idempotent(
    security_manager, app_config, mock_providers
):
    app = Organon(config=app_config, security_manager=security_manager)

    await app.initialize()
    await app.initialize()

    assert app._initialized is True
    mock_providers[0].connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_application_close(security_manager, app_config, mock_providers):
    app = Organon(config=app_config, security_manager=security_manager)

    await app.initialize()
    await app.close()

    assert app._initialized is False
    assert app._closed is True
    assert app.engine is None


@pytest.mark.asyncio
async def test_application_close_is_idempotent(security_manager, app_config):
    app = Organon(config=app_config, security_manager=security_manager)

    await app.close()
    await app.close()

    assert app._closed is True


@pytest.mark.asyncio
async def test_cannot_initialize_after_close(security_manager, app_config):
    app = Organon(config=app_config, security_manager=security_manager)

    await app.close()

    with pytest.raises(RuntimeError):
        await app.initialize()


@pytest.mark.asyncio
async def test_cleanup_continues_when_provider_close_fails(
    security_manager, app_config, monkeypatch
):
    app = Organon(config=app_config, security_manager=security_manager)

    first_provider = MagicMock()
    first_provider.close = AsyncMock()

    second_provider = MagicMock()
    second_provider.close = AsyncMock(side_effect=RuntimeError("Cleanup failed"))

    app._providers = {
        "first": first_provider,
        "second": second_provider,
    }

    with pytest.raises(RuntimeError, match="failed to close"):
        await app.close()

    # Both providers should receive a cleanup attempt.
    first_provider.close.assert_awaited_once()
    second_provider.close.assert_awaited_once()

    # Application should still be marked as closed.
    assert app._closed is True
    assert app._initialized is False
    assert app._providers == {}


@pytest.mark.asyncio
async def test_initialization_failure_cleans_up_providers(
    security_manager, app_config, monkeypatch
):
    vector_store = MagicMock()
    vector_store.connect = AsyncMock(side_effect=RuntimeError("Connection failed"))
    vector_store.close = AsyncMock()

    embedding_provider = MagicMock()
    embedding_provider.close = AsyncMock()

    monkeypatch.setattr(
        "core.application.create_vector_store",
        lambda **kwargs: vector_store,
    )

    monkeypatch.setattr(
        "core.application.create_embedding_provider",
        lambda **kwargs: embedding_provider,
    )

    app = Organon(config=app_config, security_manager=security_manager)

    with pytest.raises(RuntimeError, match="Connection failed"):
        await app.initialize()

    # Both providers should be cleaned up.
    vector_store.close.assert_awaited_once()
    embedding_provider.close.assert_awaited_once()

    # Application should not be initialized.
    assert app.engine is None
    assert app._initialized is False
    assert app._providers == {}


@pytest.mark.asyncio
async def test_application_registers_document_loaders(
    mock_providers,
    app_config,
):
    application = Organon(
        config=app_config,
        security_manager=MagicMock(),
    )

    await application.initialize()

    providers = application.registry.list_providers("document_loader")

    assert "text" in providers

    chunkers = application.registry.list_providers("chunker")

    assert "fixed_size" in chunkers

    await application.close()
