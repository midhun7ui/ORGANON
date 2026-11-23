from unittest.mock import AsyncMock
import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, call
from core.engine.engine import OrganonEngine
from core.engine.security_context import SecurityContext
from core.exceptions import AuthorizationError
from core.exceptions import ValidationError
from contextlib import contextmanager


class RecordingSpan:
    """A test span that records tracing events."""

    def __init__(self, name, attributes=None):
        self.name = name
        self.attributes = attributes or {}
        self.exceptions = []
        self.status = "unset"
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exception):
        self.exceptions.append(exception)

    def set_status(self, status):
        self.status = status

    def end(self):
        self.ended = True


class RecordingTracer:
    """A test tracer that records created spans."""

    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name, *, attributes=None):
        span = RecordingSpan(name, attributes)
        self.spans.append(span)

        try:
            yield span

        except BaseException as error:
            span.record_exception(error)
            span.set_status("error")
            raise

        finally:
            span.end()

class BrokenTracer:
    """A tracer that simulates provider initialization failures."""

    def start_as_current_span(self, name, *, attributes=None):
        raise RuntimeError("Tracing service unavailable.")


@pytest.mark.asyncio
async def test_store_documents_uses_authenticated_tenant():

    # Create a trusted security context.
    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:write"}),
    )

    # Mock the security manager.
    security_manager = AsyncMock()
    security_manager.authorize_resources.return_value = security_context

    # Mock the vector store.
    vector_store = AsyncMock()

    # Mock the embedding provider.
    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    # Create the engine.
    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # Store documents through the engine.
    await engine.store_documents(
        credentials={"token": "test-token"},
        ids=["doc-001"],
        documents=["ORGANON is a modular RAG framework."],
    )

    # Verify authorization.
    security_manager.authorize_resources.assert_awaited_once_with(
        credentials={"token": "test-token"},
        action="document:write",
        resource_ids=["doc-001"],
        resource_type="document",
    )

    # Verify document embedding generation.
    embedding_provider.embed_documents.assert_awaited_once_with(
        ["ORGANON is a modular RAG framework."]
    )

    # Verify that the trusted tenant ID reached the vector store.
    vector_store.upsert.assert_awaited_once_with(
        ids=["doc-001"],
        vectors=[[0.1, 0.2, 0.3]],
        documents=["ORGANON is a modular RAG framework."],
        metadatas=None,
        tenant_id="tenant-A",
    )


@pytest.mark.asyncio
async def test_store_documents_rejects_unauthorized_user():

    # Mock the security manager to reject the request.
    security_manager = AsyncMock()
    security_manager.authorize_resources.side_effect = AuthorizationError(
        "User is not allowed to write documents."
    )

    # Mock the vector store.
    vector_store = AsyncMock()

    # Mock the embedding provider.
    embedding_provider = AsyncMock()

    # Create the engine.
    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # Verify that the authorization error is raised.
    with pytest.raises(AuthorizationError):
        await engine.store_documents(
            credentials={"token": "invalid-token"},
            ids=["doc-001"],
            documents=["Test document"],
        )

    # Ensure the vector store is never accessed.
    vector_store.upsert.assert_not_awaited()

    # Ensure embeddings are never generated.
    embedding_provider.embed_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_documents_uses_authenticated_tenant():

    # Create a trusted security context.
    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:read"}),
    )

    # Mock the security manager.
    security_manager = AsyncMock()
    security_manager.authorize.return_value = security_context

    # Mock the vector store.
    vector_store = AsyncMock()

    # Mock the embedding provider.
    embedding_provider = AsyncMock()
    embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]

    # Create the engine.
    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # Retrieve documents.
    await engine.retrieve_documents(
        credentials={"token": "test-token"},
        query="What is ORGANON?",
        top_k=5,
    )

    # Verify read authorization.
    security_manager.authorize.assert_awaited_once_with(
        credentials={"token": "test-token"},
        action="document:read",
    )

    # Verify query embedding generation.
    embedding_provider.embed_query.assert_awaited_once_with("What is ORGANON?")

    # Verify tenant-scoped search.
    vector_store.search.assert_awaited_once_with(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        filters=None,
        tenant_id="tenant-A",
    )


@pytest.mark.asyncio
async def test_retrieve_documents_rejects_unauthorized_user():

    # Mock the security manager to reject the request.
    security_manager = AsyncMock()
    security_manager.authorize.side_effect = AuthorizationError(
        "User is not allowed to read documents."
    )

    # Mock the vector store.
    vector_store = AsyncMock()

    # Mock the embedding provider.
    embedding_provider = AsyncMock()

    # Create the engine.
    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # Verify that retrieval raises an authorization error.
    with pytest.raises(AuthorizationError):
        await engine.retrieve_documents(
            credentials={"token": "invalid-token"},
            query="What is ORGANON?",
        )

    # Ensure the vector store is never accessed.
    vector_store.search.assert_not_awaited()

    # Ensure embeddings are never generated.
    embedding_provider.embed_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_documents_rejects_empty_documents_before_embedding():

    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:write"}),
    )

    security_manager = AsyncMock()
    security_manager.authorize_resources.return_value = security_context

    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    with pytest.raises(ValidationError):
        await engine.store_documents(
            credentials={"token": "test-token"},
            ids=[],
            documents=[],
        )

    # Embedding generation must never happen.
    embedding_provider.embed_documents.assert_not_awaited()

    # Vector store must never be called.
    vector_store.upsert.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, top_k",
    [
        ("", 5),
        ("   ", 5),
        ("What is ORGANON?", 0),
        ("What is ORGANON?", -1),
        ("What is ORGANON?", True),
    ],
)
async def test_retrieve_documents_rejects_invalid_inputs_before_embedding(
    query,
    top_k,
):
    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:read"}),
    )

    security_manager = AsyncMock()
    security_manager.authorize.return_value = security_context

    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    with pytest.raises(ValidationError):
        await engine.retrieve_documents(
            credentials={"token": "test-token"},
            query=query,
            top_k=top_k,
        )

    embedding_provider.embed_query.assert_not_awaited()
    vector_store.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_documents_rejects_invalid_metadata_before_embedding():
    security_manager = AsyncMock()
    security_manager.authorize_resources.return_value = SecurityContext(
        user_id="user-1",
        tenant_id="tenant-1",
        permissions=frozenset({"document:write"}),
    )

    embedding_provider = AsyncMock()
    vector_store = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    with pytest.raises(ValidationError):
        await engine.store_documents(
            credentials={"token": "valid"},
            ids=["doc-1"],
            documents=["ORGANON is a modular RAG framework."],
            metadatas=[{"tags": ["Python", "AI"]}],
        )

    embedding_provider.embed_documents.assert_not_awaited()
    vector_store.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_connect_initializes_providers():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    await engine.connect()

    # Both providers should be initialized.
    vector_store.connect.assert_awaited_once()
    embedding_provider.connect.assert_awaited_once()

    # Engine should now be connected.
    assert engine._is_connected is True

@pytest.mark.asyncio
async def test_engine_connect_does_not_connect_providers_twice():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # First startup.
    await engine.connect()

    # Second startup.
    await engine.connect()

    # Providers should only connect once.
    vector_store.connect.assert_awaited_once()
    embedding_provider.connect.assert_awaited_once()

    assert engine._is_connected is True

@pytest.mark.asyncio
async def test_engine_close_shuts_down_providers_in_reverse_order():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    await engine.connect()

    # Track the exact order of provider shutdown.
    shutdown_order = Mock()
    shutdown_order.attach_mock(
        embedding_provider.close,
        "embedding_close",
    )
    shutdown_order.attach_mock(
        vector_store.close,
        "vector_store_close",
    )

    await engine.close()

    # Embeddings must close before the vector store.
    assert shutdown_order.mock_calls == [
        call.embedding_close(),
        call.vector_store_close(),
    ]

    assert engine._is_connected is False

@pytest.mark.asyncio
async def test_engine_connect_rolls_back_when_embedding_provider_fails():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    # Simulate an embedding provider startup failure.
    embedding_provider.connect.side_effect = RuntimeError(
        "Embedding provider failed to initialize."
    )

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # The original startup error should be preserved.
    with pytest.raises(
        RuntimeError,
        match="Embedding provider failed to initialize.",
    ):
        await engine.connect()

    # Both attempted providers should receive cleanup.
    embedding_provider.close.assert_awaited_once()
    vector_store.close.assert_awaited_once()

    # The engine must not report itself as connected.
    assert engine._is_connected is False

@pytest.mark.asyncio
async def test_engine_connect_preserves_startup_error_when_cleanup_fails():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    embedding_provider.connect.side_effect = RuntimeError(
        "Embedding startup failed."
    )

    embedding_provider.close.side_effect = RuntimeError(
        "Embedding cleanup failed."
    )

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding startup failed.",
    ) as exc_info:
        await engine.connect()

    # Vector store cleanup must still happen.
    vector_store.close.assert_awaited_once()

    # Engine must remain disconnected.
    assert engine._is_connected is False

    # Cleanup failure should be recorded as an exception note.
    assert any(
        "Embedding cleanup failed." in note
        for note in exc_info.value.__notes__
    )  

@pytest.mark.asyncio
async def test_engine_close_continues_when_provider_cleanup_fails():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    await engine.connect()

    # Simulate an embedding provider shutdown failure.
    embedding_provider.close.side_effect = RuntimeError(
        "Embedding shutdown failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding shutdown failed.",
    ):
        await engine.close()

    # Vector store must still be closed.
    vector_store.close.assert_awaited_once()

    # Engine must update its lifecycle state.
    assert engine._is_connected is False


@pytest.mark.asyncio
async def test_engine_connect_handles_concurrent_calls():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    # Control when the vector store finishes connecting.
    connection_started = asyncio.Event()
    allow_connection = asyncio.Event()

    async def delayed_connect():
        connection_started.set()
        await allow_connection.wait()

    vector_store.connect.side_effect = delayed_connect

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # Start the first connection and wait until it enters the provider.
    first_task = asyncio.create_task(engine.connect())
    await connection_started.wait()

    # Start another connection while the first is still running.
    second_task = asyncio.create_task(engine.connect())

    # Allow the first connection to finish.
    allow_connection.set()

    await asyncio.gather(first_task, second_task)

    # Each provider should have connected exactly once.
    vector_store.connect.assert_awaited_once()
    embedding_provider.connect.assert_awaited_once()

    assert engine._is_connected is True

@pytest.mark.asyncio
async def test_engine_close_handles_concurrent_calls():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    await engine.connect()

    # Pause the first provider shutdown.
    shutdown_started = asyncio.Event()
    allow_shutdown = asyncio.Event()

    async def delayed_close():
        shutdown_started.set()
        await allow_shutdown.wait()

    embedding_provider.close.side_effect = delayed_close

    # Start the first shutdown.
    first_task = asyncio.create_task(engine.close())
    await shutdown_started.wait()

    # Start another shutdown while the first is running.
    second_task = asyncio.create_task(engine.close())

    # Allow the first shutdown to finish.
    allow_shutdown.set()

    await asyncio.gather(first_task, second_task)

    # Each provider should close exactly once.
    embedding_provider.close.assert_awaited_once()
    vector_store.close.assert_awaited_once()

    assert engine._is_connected is False



@pytest.mark.asyncio
async def test_engine_connect_and_close_are_serialized():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    connection_started = asyncio.Event()
    allow_connection = asyncio.Event()

    async def delayed_connect():
        connection_started.set()
        await allow_connection.wait()

    vector_store.connect.side_effect = delayed_connect

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    # Start connection.
    connect_task = asyncio.create_task(engine.connect())
    await connection_started.wait()

    # Request shutdown while connection is still running.
    close_task = asyncio.create_task(engine.close())

    # Allow initialization to finish.
    allow_connection.set()

    await asyncio.gather(connect_task, close_task)

    vector_store.connect.assert_awaited_once()
    embedding_provider.connect.assert_awaited_once()

    embedding_provider.close.assert_awaited_once()
    vector_store.close.assert_awaited_once()

    assert engine._is_connected is False


@pytest.mark.asyncio
async def test_engine_close_retries_failed_provider_cleanup():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    await engine.connect()

    # First attempt fails; second attempt succeeds.
    vector_store.close.side_effect = [
        RuntimeError("Vector store cleanup failed"),
        None,
    ]

    # First shutdown should report the failure.
    with pytest.raises(
        RuntimeError,
        match="Vector store cleanup failed",
    ):
        await engine.close()

    # Embedding provider closed successfully.
    embedding_provider.close.assert_awaited_once()

    # Vector store cleanup failed on the first attempt.
    assert vector_store.close.await_count == 1

    # Retry shutdown.
    await engine.close()

    # Embedding provider should not be closed again.
    embedding_provider.close.assert_awaited_once()

    # Vector store should have been retried.
    assert vector_store.close.await_count == 2

    assert engine._is_connected is False
    assert engine._providers_to_close == []


@pytest.mark.asyncio
async def test_engine_connect_rolls_back_when_cancelled():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    connection_started = asyncio.Event()
    allow_connection = asyncio.Event()

    async def delayed_connect():
        connection_started.set()
        await allow_connection.wait()

    embedding_provider.connect.side_effect = delayed_connect

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    connect_task = asyncio.create_task(engine.connect())

    await connection_started.wait()

    # Cancel startup while the embedding provider is initializing.
    connect_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await connect_task

    # Both attempted providers should receive cleanup.
    embedding_provider.close.assert_awaited_once()
    vector_store.close.assert_awaited_once()

    assert engine._is_connected is False
    assert engine._providers_to_close == []


@pytest.mark.asyncio
async def test_engine_connect_preserves_cancellation_when_cleanup_fails():

    security_manager = AsyncMock()
    vector_store = AsyncMock()
    embedding_provider = AsyncMock()

    connection_started = asyncio.Event()
    allow_connection = asyncio.Event()

    async def delayed_connect():
        connection_started.set()
        await allow_connection.wait()

    embedding_provider.connect.side_effect = delayed_connect

    # First cleanup fails; retry succeeds.
    embedding_provider.close.side_effect = [
        RuntimeError("Embedding cleanup failed"),
        None,
    ]

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    connect_task = asyncio.create_task(engine.connect())

    await connection_started.wait()

    connect_task.cancel()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await connect_task

    # Original cancellation must remain the raised exception.
    assert isinstance(exc_info.value, asyncio.CancelledError)

    # Both providers were given a cleanup attempt.
    embedding_provider.close.assert_awaited_once()
    vector_store.close.assert_awaited_once()

    # Failed cleanup must be remembered.
    assert engine._providers_to_close == [embedding_provider]

    assert engine._is_connected is False

    # Retry the failed cleanup.
    await engine.close()

    assert embedding_provider.close.await_count == 2
    assert engine._providers_to_close == []
    assert engine._is_connected is False


@pytest.mark.asyncio
async def test_store_documents_emits_successful_logging_events():

    # Create a trusted security context.
    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:write"}),
    )

    # Mock the security manager.
    security_manager = AsyncMock()
    security_manager.authorize_resources.return_value = security_context

    # Mock the vector store.
    vector_store = AsyncMock()

    # Mock the embedding provider.
    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    # Mock the logger.
    logger = Mock()

    # Create the engine with the logger.
    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        logger=logger,
    )

    # Store documents.
    await engine.store_documents(
        credentials={"token": "secret-token"},
        ids=["doc-001"],
        documents=["ORGANON is a modular RAG framework."],
    )

    # Verify the ingestion started event.
    logger.info.assert_any_call(
        "Document ingestion started.",
        context={
            "event": "document_ingestion.started",
            "document_count": 1,
        },
    )

    # Verify the ingestion completed event.
    logger.info.assert_any_call(
        "Document ingestion completed.",
        context={
            "event": "document_ingestion.completed",
            "document_count": 1,
        },
    )

    # Verify no error event was emitted.
    logger.error.assert_not_called()

    # Verify sensitive information is not logged.
    for logged_call in logger.mock_calls:
        assert "secret-token" not in str(logged_call)
        assert "ORGANON is a modular RAG framework." not in str(logged_call)

@pytest.mark.asyncio
async def test_store_documents_emits_failure_logging_event():

    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:write"}),
    )

    security_manager = AsyncMock()
    security_manager.authorize_resources.side_effect = AuthorizationError(
        "Permission denied."
    )

    vector_store = AsyncMock()

    embedding_provider = AsyncMock()

    logger = Mock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        logger=logger,
    )

    with pytest.raises(AuthorizationError):
        await engine.store_documents(
            credentials={"token": "secret-token"},
            ids=["doc-001"],
            documents=["Sensitive document content."],
        )

    # Verify ingestion started.
    logger.info.assert_any_call(
        "Document ingestion started.",
        context={
            "event": "document_ingestion.started",
            "document_count": 1,
        },
    )

    # Verify ingestion failure.
    logger.error.assert_called_once_with(
        "Document ingestion failed.",
        context={
            "event": "document_ingestion.failed",
            "document_count": 1,
        },
    )

    # Verify embedding and storage never happened.
    embedding_provider.embed_documents.assert_not_called()
    vector_store.upsert.assert_not_called()

    # Verify sensitive information is not logged.
    for logged_call in logger.mock_calls:
        assert "secret-token" not in str(logged_call)
        assert "Sensitive document content." not in str(logged_call)

@pytest.mark.asyncio
async def test_retrieve_documents_emits_successful_logging_events():

    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:read"}),
    )

    security_manager = AsyncMock()
    security_manager.authorize.return_value = security_context

    vector_store = AsyncMock()
    vector_store.search.return_value = []

    embedding_provider = AsyncMock()
    embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]

    logger = Mock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        logger=logger,
    )

    await engine.retrieve_documents(
        credentials={"token": "secret-token"},
        query="How does ORGANON work?",
        top_k=5,
    )

    logger.info.assert_any_call(
        "Document retrieval started.",
        context={
            "event": "document_retrieval.started",
            "top_k": 5,
        },
    )

    logger.info.assert_any_call(
        "Document retrieval completed.",
        context={
            "event": "document_retrieval.completed",
            "top_k": 5,
            "result_count": 0,
        },
    )

    logger.error.assert_not_called()

    for logged_call in logger.mock_calls:
        assert "secret-token" not in str(logged_call)
        assert "How does ORGANON work?" not in str(logged_call)

@pytest.mark.asyncio
async def test_retrieve_documents_emits_failure_logging_event():

    security_manager = AsyncMock()
    security_manager.authorize.side_effect = AuthorizationError(
        "Permission denied."
    )

    vector_store = AsyncMock()
    embedding_provider = AsyncMock()
    logger = Mock()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        logger=logger,
    )

    with pytest.raises(AuthorizationError):
        await engine.retrieve_documents(
            credentials={"token": "secret-token"},
            query="Sensitive search query",
            top_k=5,
        )

    logger.info.assert_any_call(
        "Document retrieval started.",
        context={
            "event": "document_retrieval.started",
            "top_k": 5,
        },
    )

    logger.error.assert_called_once_with(
        "Document retrieval failed.",
        context={
            "event": "document_retrieval.failed",
            "top_k": 5,
        },
    )

    embedding_provider.embed_query.assert_not_called()
    vector_store.search.assert_not_called()

    for logged_call in logger.mock_calls:
        assert "secret-token" not in str(logged_call)
        assert "Sensitive search query" not in str(logged_call)


@pytest.mark.asyncio
async def test_store_documents_continues_when_logger_fails():

    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:write"}),
    )

    security_manager = AsyncMock()
    security_manager.authorize_resources.return_value = security_context

    vector_store = AsyncMock()

    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    logger = Mock()
    logger.info.side_effect = RuntimeError("Logging service unavailable.")
    logger.error.side_effect = RuntimeError("Logging service unavailable.")

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        logger=logger,
    )

    # Logging failures must not interrupt ingestion.
    await engine.store_documents(
        credentials={"token": "secret-token"},
        ids=["doc-001"],
        documents=["ORGANON is a modular RAG framework."],
    )

    # Verify that the actual operations completed.
    embedding_provider.embed_documents.assert_awaited_once()
    vector_store.upsert.assert_awaited_once()

@pytest.mark.asyncio
async def test_retrieve_documents_continues_when_logger_fails():

    security_context = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:read"}),
    )

    security_manager = AsyncMock()
    security_manager.authorize.return_value = security_context

    vector_store = AsyncMock()
    vector_store.search.return_value = []

    embedding_provider = AsyncMock()
    embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]

    logger = Mock()
    logger.info.side_effect = RuntimeError("Logging service unavailable.")
    logger.error.side_effect = RuntimeError("Logging service unavailable.")

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        logger=logger,
    )

    results = await engine.retrieve_documents(
        credentials={"token": "secret-token"},
        query="How does ORGANON work?",
        top_k=5,
    )

    # Verify retrieval completed despite logging failures.
    assert results == []

    embedding_provider.embed_query.assert_awaited_once()
    vector_store.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_documents_creates_tracing_span():

    security_manager = AsyncMock()
    security_manager.authorize_resources.return_value = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:write"}),
    )

    vector_store = AsyncMock()

    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    tracer = RecordingTracer()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        tracer=tracer,
    )

    await engine.store_documents(
        credentials={"token": "test-token"},
        ids=["doc-001"],
        documents=["ORGANON is a modular RAG framework."],
    )

    assert len(tracer.spans) == 1

    span = tracer.spans[0]

    assert span.name == "document_ingestion"
    assert span.attributes["document_count"] == 1
    assert span.exceptions == []
    assert span.ended is True


@pytest.mark.asyncio
async def test_retrieve_documents_creates_tracing_span():

    security_manager = AsyncMock()
    security_manager.authorize.return_value = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:read"}),
    )

    vector_store = AsyncMock()
    vector_store.search.return_value = []

    embedding_provider = AsyncMock()
    embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]

    tracer = RecordingTracer()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        tracer=tracer,
    )

    await engine.retrieve_documents(
        credentials={"token": "test-token"},
        query="What is ORGANON?",
        top_k=5,
    )

    assert len(tracer.spans) == 1

    span = tracer.spans[0]

    assert span.name == "document_retrieval"
    assert span.attributes["top_k"] == 5
    assert span.exceptions == []
    assert span.ended is True

@pytest.mark.asyncio
async def test_store_documents_records_tracing_exception():

    security_manager = AsyncMock()
    security_manager.authorize_resources.side_effect = AuthorizationError(
        "Permission denied."
    )

    vector_store = AsyncMock()
    embedding_provider = AsyncMock()
    tracer = RecordingTracer()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        tracer=tracer,
    )

    with pytest.raises(AuthorizationError, match="Permission denied."):
        await engine.store_documents(
            credentials={"token": "test-token"},
            ids=["doc-001"],
            documents=["Test document"],
        )

    assert len(tracer.spans) == 1

    span = tracer.spans[0]

    assert span.name == "document_ingestion"
    assert len(span.exceptions) == 1
    assert isinstance(span.exceptions[0], AuthorizationError)
    assert span.status == "error"
    assert span.ended is True

@pytest.mark.asyncio
async def test_retrieve_documents_records_tracing_exception():

    security_manager = AsyncMock()
    security_manager.authorize.side_effect = AuthorizationError(
        "Permission denied."
    )

    vector_store = AsyncMock()
    embedding_provider = AsyncMock()
    tracer = RecordingTracer()

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        tracer=tracer,
    )

    with pytest.raises(AuthorizationError, match="Permission denied."):
        await engine.retrieve_documents(
            credentials={"token": "test-token"},
            query="What is ORGANON?",
        )

    assert len(tracer.spans) == 1

    span = tracer.spans[0]

    assert span.name == "document_retrieval"
    assert len(span.exceptions) == 1
    assert isinstance(span.exceptions[0], AuthorizationError)
    assert span.status == "error"
    assert span.ended is True

@pytest.mark.asyncio
async def test_store_documents_continues_when_tracer_fails():

    security_manager = AsyncMock()
    security_manager.authorize_resources.return_value = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:write"}),
    )

    vector_store = AsyncMock()

    embedding_provider = AsyncMock()
    embedding_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        tracer=BrokenTracer(),
    )

    await engine.store_documents(
        credentials={"token": "test-token"},
        ids=["doc-001"],
        documents=["ORGANON is a modular RAG framework."],
    )

    embedding_provider.embed_documents.assert_awaited_once()
    vector_store.upsert.assert_awaited_once()

@pytest.mark.asyncio
async def test_retrieve_documents_continues_when_tracer_fails():

    security_manager = AsyncMock()
    security_manager.authorize.return_value = SecurityContext(
        user_id="user-001",
        tenant_id="tenant-A",
        permissions=frozenset({"document:read"}),
    )

    vector_store = AsyncMock()
    vector_store.search.return_value = []

    embedding_provider = AsyncMock()
    embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]

    engine = OrganonEngine(
        security_manager=security_manager,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        tracer=BrokenTracer(),
    )

    results = await engine.retrieve_documents(
        credentials={"token": "test-token"},
        query="What is ORGANON?",
        top_k=5,
    )

    assert results == []

    embedding_provider.embed_query.assert_awaited_once()
    vector_store.search.assert_awaited_once()