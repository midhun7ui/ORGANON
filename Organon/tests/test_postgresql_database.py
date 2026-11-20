
import os

import pytest

from core.metadata.providers.postgresql.database import PostgreSQLDatabase
from sqlalchemy import text
import uuid

from sqlalchemy import select
from core.metadata.providers.postgresql.models import DocumentORM

@pytest.mark.asyncio
async def test_postgresql_database_connection():
    database_url = os.environ["ORGANON_DATABASE_URL"]

    database = PostgreSQLDatabase(database_url)

    try:
        await database.connect()
    finally:
        await database.close()

@pytest.mark.asyncio
async def test_postgresql_database_session():
    database_url = os.environ["ORGANON_DATABASE_URL"]

    database = PostgreSQLDatabase(database_url)

    try:
        async with database.session() as session:
            result = await session.execute(text("SELECT 1"))

            assert result.scalar_one() == 1

    finally:
        await database.close()

@pytest.mark.asyncio
async def test_postgresql_transaction_rollback():
    database_url = os.environ["ORGANON_DATABASE_URL"]
    database = PostgreSQLDatabase(database_url)

    document_id = uuid.uuid4()
    external_id = f"rollback-test-{document_id}"

    try:
        with pytest.raises(RuntimeError):
            async with database.session() as session:
                async with session.begin():
                    document = DocumentORM(
                        id=document_id,
                        tenant_id="test-tenant",
                        external_id=external_id,
                    )

                    session.add(document)

                    raise RuntimeError("Simulated failure")

        async with database.session() as session:
            result = await session.execute(
                select(DocumentORM).where(
                    DocumentORM.id == document_id
                )
            )

            assert result.scalar_one_or_none() is None

    finally:
        await database.close()