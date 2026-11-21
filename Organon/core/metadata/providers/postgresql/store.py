"""
ORGANON - PostgreSQL Metadata Store

Implements the MetadataStore interface using PostgreSQL
and SQLAlchemy's asynchronous database operations.
"""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.exc import IntegrityError

from core.documents.models import (
    DocumentRecord,
    DocumentVersion,
    VersionStatus,
)
from core.exceptions import (
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
    InvalidVersionStateError,
    ValidationError,
)
from core.metadata.base import MetadataStore
from core.metadata.providers.postgresql.database import PostgreSQLDatabase
from core.metadata.providers.postgresql.models import (
    DocumentORM,
    DocumentVersionORM,
)


class PostgreSQLMetadataStore(MetadataStore):
    """PostgreSQL implementation of ORGANON's MetadataStore."""

    def __init__(self, database: PostgreSQLDatabase) -> None:
        async def connect(self) -> None:
            """Initialize the PostgreSQL metadata provider."""

            await self.database.connect()

        async def close(self) -> None:
            """Release PostgreSQL database resources."""

            await self.database.close()

        # Database manager handles connections and async sessions.
        self.database = database

    # ---------------------------------------------------------
    # Internal mapping and validation helpers
    # ---------------------------------------------------------

    @staticmethod
    def _to_document_record(
        document: DocumentORM,
    ) -> DocumentRecord:
        """Convert a PostgreSQL document into an ORGANON document."""

        return DocumentRecord(
            id=document.external_id,
            tenant_id=document.tenant_id,
            current_version_id=(
                str(document.current_version_id)
                if document.current_version_id
                else None
            ),
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    @staticmethod
    def _to_document_version(
        version: DocumentVersionORM,
        document_external_id: str,
    ) -> DocumentVersion:
        """Convert a PostgreSQL version into an ORGANON version."""

        return DocumentVersion(
            id=str(version.id),
            document_id=document_external_id,
            version_number=version.version_number,
            status=VersionStatus(version.status),
            created_at=version.created_at,
            activated_at=version.activated_at,
        )

    @staticmethod
    def _parse_version_id(version_id: str) -> UUID:
        """Validate and convert a version ID into a UUID."""

        try:
            return UUID(version_id)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValidationError("Version ID must be a valid UUID.") from exc

    async def _get_locked_document(
        self,
        session,
        document_id: str,
        tenant_id: str,
    ) -> DocumentORM:
        """
        Retrieve and lock a document belonging to a tenant.

        The row lock prevents concurrent operations from modifying
        the same document's version lifecycle simultaneously.
        """

        statement = (
            select(DocumentORM)
            .where(
                DocumentORM.external_id == document_id,
                DocumentORM.tenant_id == tenant_id,
            )
            .with_for_update()
        )

        result = await session.execute(statement)
        document = result.scalar_one_or_none()

        if document is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

        return document

    # ---------------------------------------------------------
    # Document operations
    # ---------------------------------------------------------

    async def create_document(
        self,
        document: DocumentRecord,
    ) -> DocumentRecord:
        """Create a new document record in PostgreSQL."""

        # Map the ORGANON model to the PostgreSQL ORM model.
        database_document = DocumentORM(
            tenant_id=document.tenant_id,
            external_id=document.id,
        )

        try:
            async with self.database.session() as session:
                async with session.begin():

                    session.add(database_document)

                    # Send the INSERT to PostgreSQL.
                    await session.flush()

                    # Retrieve generated UUID and database timestamps.
                    await session.refresh(database_document)

                    result = self._to_document_record(database_document)

        except IntegrityError as exc:
            # Translate only the document identity constraint violation.
            constraint_name = getattr(
                exc.orig,
                "constraint_name",
                None,
            )

            if constraint_name == "uq_documents_tenant_external_id":
                raise DocumentAlreadyExistsError(
                    f"Document '{document.id}' already exists "
                    f"for tenant '{document.tenant_id}'."
                ) from exc

            # Do not hide unrelated database integrity errors.
            raise

        return result

    async def get_document(
        self,
        document_id: str,
        *,
        tenant_id: str,
    ) -> DocumentRecord | None:
        """Retrieve a document belonging to a specific tenant."""

        # Always filter using both the external ID and tenant ID.
        statement = select(DocumentORM).where(
            DocumentORM.external_id == document_id,
            DocumentORM.tenant_id == tenant_id,
        )

        async with self.database.session() as session:
            result = await session.execute(statement)
            document = result.scalar_one_or_none()

        if document is None:
            return None

        return self._to_document_record(document)

    # ---------------------------------------------------------
    # Version creation and retrieval
    # ---------------------------------------------------------

    async def create_version(
        self,
        document_id: str,
        *,
        tenant_id: str,
    ) -> DocumentVersion:
        """Create the next version of a document atomically."""

        async with self.database.session() as session:
            async with session.begin():

                # Lock the document before calculating its next version.
                document = await self._get_locked_document(
                    session,
                    document_id,
                    tenant_id,
                )

                # Find the highest existing version number.
                version_statement = select(
                    func.max(DocumentVersionORM.version_number)
                ).where(DocumentVersionORM.document_id == document.id)

                version_result = await session.execute(version_statement)
                latest_version = version_result.scalar_one()

                next_version = (latest_version or 0) + 1

                # New versions always begin in PROCESSING state.
                database_version = DocumentVersionORM(
                    document_id=document.id,
                    version_number=next_version,
                    status=VersionStatus.PROCESSING.value,
                )

                session.add(database_version)

                await session.flush()
                await session.refresh(database_version)

                created_version = self._to_document_version(
                    database_version,
                    document.external_id,
                )

        return created_version

    async def get_version(
        self,
        document_id: str,
        version_id: str,
        *,
        tenant_id: str,
    ) -> DocumentVersion | None:
        """Retrieve a specific version belonging to a tenant's document."""

        parsed_version_id = self._parse_version_id(version_id)

        statement = (
            select(
                DocumentVersionORM,
                DocumentORM.external_id,
            )
            .join(
                DocumentORM,
                DocumentVersionORM.document_id == DocumentORM.id,
            )
            .where(
                DocumentORM.external_id == document_id,
                DocumentORM.tenant_id == tenant_id,
                DocumentVersionORM.id == parsed_version_id,
            )
        )

        async with self.database.session() as session:
            result = await session.execute(statement)
            row = result.one_or_none()

        if row is None:
            return None

        version, external_id = row

        return self._to_document_version(
            version,
            external_id,
        )

    async def list_versions(
        self,
        document_id: str,
        *,
        tenant_id: str,
        limit: int = 50,
        before_version_number: int | None = None,
    ) -> list[DocumentVersion]:
        """List document versions using cursor-based pagination."""

        # Validate pagination inputs.
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValidationError("Limit must be a positive integer.")

        if before_version_number is not None:
            if (
                isinstance(before_version_number, bool)
                or not isinstance(before_version_number, int)
                or before_version_number <= 0
            ):
                raise ValidationError(
                    "Before version number must be a positive integer."
                )

        statement = (
            select(
                DocumentVersionORM,
                DocumentORM.external_id,
            )
            .join(
                DocumentORM,
                DocumentVersionORM.document_id == DocumentORM.id,
            )
            .where(
                DocumentORM.external_id == document_id,
                DocumentORM.tenant_id == tenant_id,
            )
        )

        # Fetch only versions older than the supplied cursor.
        if before_version_number is not None:
            statement = statement.where(
                DocumentVersionORM.version_number < before_version_number
            )

        # Newest versions appear first.
        statement = statement.order_by(desc(DocumentVersionORM.version_number)).limit(
            limit
        )

        async with self.database.session() as session:
            result = await session.execute(statement)
            rows = result.all()

        return [
            self._to_document_version(version, external_id)
            for version, external_id in rows
        ]

    # ---------------------------------------------------------
    # Version activation
    # ---------------------------------------------------------

    async def activate_version(
        self,
        document_id: str,
        version_id: str,
        *,
        tenant_id: str,
    ) -> DocumentVersion:
        """
        Atomically activate a ready document version.

        The previous active version is archived before the new
        version becomes active. Both changes happen in one
        PostgreSQL transaction.
        """

        parsed_version_id = self._parse_version_id(version_id)

        async with self.database.session() as session:
            async with session.begin():

                # 1. Lock the parent document first.
                document = await self._get_locked_document(
                    session,
                    document_id,
                    tenant_id,
                )

                # 2. Retrieve and lock the requested version.
                version_statement = (
                    select(DocumentVersionORM)
                    .where(
                        DocumentVersionORM.id == parsed_version_id,
                        DocumentVersionORM.document_id == document.id,
                    )
                    .with_for_update()
                )

                result = await session.execute(version_statement)
                version = result.scalar_one_or_none()

                if version is None:
                    raise DocumentNotFoundError(
                        f"Version '{version_id}' was not found."
                    )

                # 3. Only READY versions can become active.
                if version.status != VersionStatus.READY.value:
                    raise InvalidVersionStateError(
                        f"Cannot activate version '{version_id}' "
                        f"from state '{version.status}'."
                    )

                # 4. Find and lock the currently active version.
                previous_version_statement = (
                    select(DocumentVersionORM)
                    .where(
                        DocumentVersionORM.document_id == document.id,
                        DocumentVersionORM.status == VersionStatus.ACTIVE.value,
                    )
                    .with_for_update()
                )

                result = await session.execute(previous_version_statement)
                previous_version = result.scalar_one_or_none()

                # 5. Archive the previous active version first.
                if previous_version is not None:
                    previous_version.status = VersionStatus.ARCHIVED.value

                    # Flush the archive update before activating
                    # the new version to satisfy the unique index.
                    await session.flush()

                activated_at = datetime.now(timezone.utc)

                # 6. Activate the requested version.
                version.status = VersionStatus.ACTIVE.value
                version.activated_at = activated_at

                # 7. Update the document's current version pointer.
                document.current_version_id = version.id
                document.updated_at = activated_at

                # 8. Persist both changes inside this transaction.
                await session.flush()

                # Refresh the activated version from PostgreSQL.
                await session.refresh(version)

                activated_version = self._to_document_version(
                    version,
                    document.external_id,
                )

        # The transaction commits when session.begin() exits.
        return activated_version

    # ---------------------------------------------------------
    # Version lifecycle transitions
    # ---------------------------------------------------------

    async def mark_version_ready(
        self,
        document_id: str,
        version_id: str,
        *,
        tenant_id: str,
    ) -> DocumentVersion:
        """Mark a successfully ingested version as ready for activation."""

        parsed_version_id = self._parse_version_id(version_id)

        async with self.database.session() as session:
            async with session.begin():

                # Lock the parent document first.
                document = await self._get_locked_document(
                    session,
                    document_id,
                    tenant_id,
                )

                # Retrieve the version belonging to this document.
                statement = (
                    select(DocumentVersionORM)
                    .where(
                        DocumentVersionORM.id == parsed_version_id,
                        DocumentVersionORM.document_id == document.id,
                    )
                    .with_for_update()
                )

                result = await session.execute(statement)
                version = result.scalar_one_or_none()

                if version is None:
                    raise DocumentNotFoundError(
                        f"Version '{version_id}' was not found."
                    )

                # Only PROCESSING versions can become READY.
                if version.status != VersionStatus.PROCESSING.value:
                    raise InvalidVersionStateError(
                        f"Cannot mark version '{version_id}' as ready "
                        f"from state '{version.status}'."
                    )

                version.status = VersionStatus.READY.value

                await session.flush()
                await session.refresh(version)

                ready_version = self._to_document_version(
                    version,
                    document.external_id,
                )

        return ready_version

    async def mark_version_failed(
        self,
        document_id: str,
        version_id: str,
        *,
        tenant_id: str,
    ) -> DocumentVersion:
        """Mark a failed ingestion version as failed."""

        parsed_version_id = self._parse_version_id(version_id)

        async with self.database.session() as session:
            async with session.begin():

                # Lock the parent document first.
                document = await self._get_locked_document(
                    session,
                    document_id,
                    tenant_id,
                )

                # Retrieve the version belonging to this document.
                statement = (
                    select(DocumentVersionORM)
                    .where(
                        DocumentVersionORM.id == parsed_version_id,
                        DocumentVersionORM.document_id == document.id,
                    )
                    .with_for_update()
                )

                result = await session.execute(statement)
                version = result.scalar_one_or_none()

                if version is None:
                    raise DocumentNotFoundError(
                        f"Version '{version_id}' was not found."
                    )

                # Only PROCESSING versions can become FAILED.
                if version.status != VersionStatus.PROCESSING.value:
                    raise InvalidVersionStateError(
                        f"Cannot mark version '{version_id}' as failed "
                        f"from state '{version.status}'."
                    )

                version.status = VersionStatus.FAILED.value

                await session.flush()
                await session.refresh(version)

                failed_version = self._to_document_version(
                    version,
                    document.external_id,
                )

        return failed_version
