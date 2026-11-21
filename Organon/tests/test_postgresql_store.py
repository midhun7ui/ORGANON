import os
import uuid

import pytest
from sqlalchemy import delete
import asyncio
from core.documents.models import DocumentRecord, VersionStatus
from core.metadata.providers.postgresql.database import PostgreSQLDatabase
from core.metadata.providers.postgresql.models import DocumentORM
from core.metadata.providers.postgresql.store import PostgreSQLMetadataStore


@pytest.fixture
async def metadata_store():
    # Read the database URL from the environment.
    database_url = os.environ["ORGANON_DATABASE_URL"]

    # Initialize the database and metadata store.
    database = PostgreSQLDatabase(database_url)
    store = PostgreSQLMetadataStore(database)

    # Generate unique identifiers to avoid test data conflicts.
    tenant_id = f"test-company-{uuid.uuid4()}"
    document_id = f"test-document-{uuid.uuid4()}"

    try:
        yield store, tenant_id, document_id

    finally:
        # Clean up the document created by the test.
        async with database.session() as session:
            async with session.begin():
                await session.execute(
                    delete(DocumentORM).where(
                        DocumentORM.tenant_id == tenant_id,
                        DocumentORM.external_id == document_id,
                    )
                )

        # Release database resources.
        await database.close()


@pytest.mark.asyncio
async def test_create_and_get_document(metadata_store):
    store, tenant_id, document_id = metadata_store

    # Create a new document.
    document = DocumentRecord(
        id=document_id,
        tenant_id=tenant_id,
    )

    created = await store.create_document(document)

    # Verify the created document.
    assert created.id == document_id
    assert created.tenant_id == tenant_id
    assert created.current_version_id is None

    # Retrieve the document from PostgreSQL.
    retrieved = await store.get_document(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    # Verify the retrieved document.
    assert retrieved is not None
    assert retrieved.id == document_id
    assert retrieved.tenant_id == tenant_id


@pytest.mark.asyncio
async def test_create_document_versions(metadata_store):
    store, tenant_id, document_id = metadata_store

    # Create the parent document first.
    document = DocumentRecord(
        id=document_id,
        tenant_id=tenant_id,
    )
    await store.create_document(document)

    # Create the first version.
    first_version = await store.create_version(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    assert first_version.version_number == 1
    assert first_version.status.value == "processing"

    # Create the second version.
    second_version = await store.create_version(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    assert second_version.version_number == 2
    assert second_version.status.value == "processing"
    assert second_version.document_id == document_id


@pytest.mark.asyncio
async def test_get_version(metadata_store):
    store, tenant_id, document_id = metadata_store

    # Create a document.
    await store.create_document(DocumentRecord(id=document_id, tenant_id=tenant_id))

    # Create a version.
    version = await store.create_version(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    # Retrieve that specific version.
    retrieved = await store.get_version(
        document_id=document_id,
        version_id=version.id,
        tenant_id=tenant_id,
    )

    assert retrieved is not None
    assert retrieved.id == version.id
    assert retrieved.document_id == document_id
    assert retrieved.version_number == 1


@pytest.mark.asyncio
async def test_list_versions(metadata_store):
    store, tenant_id, document_id = metadata_store

    # Create a document.
    await store.create_document(DocumentRecord(id=document_id, tenant_id=tenant_id))

    # Create three versions.
    for _ in range(3):
        await store.create_version(
            document_id=document_id,
            tenant_id=tenant_id,
        )

    # Retrieve all versions.
    versions = await store.list_versions(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    assert len(versions) == 3

    # Versions should be ordered newest first.
    assert [version.version_number for version in versions] == [3, 2, 1]


@pytest.mark.asyncio
async def test_mark_version_ready(metadata_store):
    store, tenant_id, document_id = metadata_store

    # Create a document.
    await store.create_document(DocumentRecord(id=document_id, tenant_id=tenant_id))

    # Create a version in PROCESSING state.
    version = await store.create_version(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    assert version.status.value == "processing"

    # Mark the version as READY after processing.
    ready_version = await store.mark_version_ready(
        document_id=document_id,
        version_id=version.id,
        tenant_id=tenant_id,
    )

    assert ready_version.status.value == "ready"
    assert ready_version.id == version.id


@pytest.mark.asyncio
async def test_activate_version(metadata_store):
    store, tenant_id, document_id = metadata_store

    # Create a document.
    await store.create_document(DocumentRecord(id=document_id, tenant_id=tenant_id))

    # Create and prepare the first version.
    first_version = await store.create_version(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    await store.mark_version_ready(
        document_id=document_id,
        version_id=first_version.id,
        tenant_id=tenant_id,
    )

    # Activate the first version.
    active_version = await store.activate_version(
        document_id=document_id,
        version_id=first_version.id,
        tenant_id=tenant_id,
    )

    assert active_version.status.value == "active"

    # Verify that the document points to the active version.
    document = await store.get_document(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    assert document.current_version_id == first_version.id

    # Create and prepare a second version.
    second_version = await store.create_version(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    await store.mark_version_ready(
        document_id=document_id,
        version_id=second_version.id,
        tenant_id=tenant_id,
    )

    # Activate the second version.
    new_active_version = await store.activate_version(
        document_id=document_id,
        version_id=second_version.id,
        tenant_id=tenant_id,
    )

    assert new_active_version.status.value == "active"

    # Verify that the first version is now archived.
    archived_version = await store.get_version(
        document_id=document_id,
        version_id=first_version.id,
        tenant_id=tenant_id,
    )

    assert archived_version.status.value == "archived"

    # Verify that the document points to the second version.
    updated_document = await store.get_document(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    assert updated_document.current_version_id == second_version.id


@pytest.mark.asyncio
async def test_mark_version_failed(metadata_store):
    store, tenant_id, document_id = metadata_store

    # Create a document.
    await store.create_document(DocumentRecord(id=document_id, tenant_id=tenant_id))

    # Create a version.
    version = await store.create_version(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    # Mark the version as failed.
    failed_version = await store.mark_version_failed(
        document_id=document_id,
        version_id=version.id,
        tenant_id=tenant_id,
    )

    # Verify the failed state.
    assert failed_version.status.value == "failed"

    # Confirm that the document has no active version.
    document = await store.get_document(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    assert document.current_version_id is None


@pytest.mark.asyncio
async def test_concurrent_version_creation(metadata_store):
    store, tenant_id, document_id = metadata_store

    # Create the parent document.
    await store.create_document(DocumentRecord(id=document_id, tenant_id=tenant_id))

    # Create 10 versions concurrently.
    versions = await asyncio.gather(
        *[
            store.create_version(
                document_id=document_id,
                tenant_id=tenant_id,
            )
            for _ in range(10)
        ]
    )

    # Extract and sort the generated version numbers.
    version_numbers = sorted(version.version_number for version in versions)

    # Every version number must be unique and sequential.
    assert version_numbers == list(range(1, 11))

    # Every version must have a unique ID.
    version_ids = [version.id for version in versions]
    assert len(set(version_ids)) == 10


@pytest.mark.asyncio
async def test_concurrent_version_activation(metadata_store):
    store, tenant_id, document_id = metadata_store

    # Create a document.
    await store.create_document(DocumentRecord(id=document_id, tenant_id=tenant_id))

    # Create two versions.
    version_1 = await store.create_version(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    version_2 = await store.create_version(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    # Both versions must be READY before activation.
    await store.mark_version_ready(
        document_id=document_id,
        version_id=version_1.id,
        tenant_id=tenant_id,
    )

    await store.mark_version_ready(
        document_id=document_id,
        version_id=version_2.id,
        tenant_id=tenant_id,
    )

    # Attempt to activate both versions concurrently.
    results = await asyncio.gather(
        store.activate_version(
            document_id=document_id,
            version_id=version_1.id,
            tenant_id=tenant_id,
        ),
        store.activate_version(
            document_id=document_id,
            version_id=version_2.id,
            tenant_id=tenant_id,
        ),
    )

    # Both activation operations should complete.
    assert len(results) == 2

    # Retrieve the final document and its versions.
    document = await store.get_document(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    versions = await store.list_versions(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    active_versions = [
        version for version in versions if version.status == VersionStatus.ACTIVE
    ]

    # Exactly one version must remain ACTIVE.
    assert len(active_versions) == 1

    # The document pointer must reference that active version.
    assert document.current_version_id == active_versions[0].id
