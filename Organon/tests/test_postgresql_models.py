from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint

from core.metadata.providers.postgresql.models import (
    Base,
    DocumentORM,
    DocumentVersionORM,
)

import pytest

from core.documents.models import DocumentRecord
from core.metadata.providers.postgresql.database import PostgreSQLDatabase
from core.metadata.providers.postgresql.store import PostgreSQLMetadataStore


from sqlalchemy import Index


def test_metadata_contains_expected_tables():
    assert set(Base.metadata.tables.keys()) == {
        "documents",
        "document_versions",
    }


def test_document_version_has_document_foreign_key():
    table = DocumentVersionORM.__table__

    foreign_keys = [
        fk for fk in table.foreign_keys if fk.target_fullname == "documents.id"
    ]

    assert len(foreign_keys) == 1
    assert foreign_keys[0].ondelete == "CASCADE"


def test_document_current_version_has_composite_foreign_key():
    table = DocumentORM.__table__

    constraints = [
        constraint
        for constraint in table.constraints
        if isinstance(constraint, ForeignKeyConstraint)
        and constraint.name == "fk_documents_current_version"
    ]

    assert len(constraints) == 1

    foreign_key = constraints[0]

    assert [element.target_fullname for element in foreign_key.elements] == [
        "document_versions.id",
        "document_versions.document_id",
    ]


def test_document_version_has_unique_version_number():
    constraints = DocumentVersionORM.__table__.constraints

    assert any(
        isinstance(constraint, UniqueConstraint)
        and constraint.name == "uq_document_versions_number"
        for constraint in constraints
    )


def test_document_version_has_valid_status_constraint():
    constraints = DocumentVersionORM.__table__.constraints

    assert any(
        isinstance(constraint, CheckConstraint)
        and constraint.name == "ck_document_versions_valid_status"
        for constraint in constraints
    )


def test_document_version_has_positive_number_constraint():
    constraints = DocumentVersionORM.__table__.constraints

    assert any(
        isinstance(constraint, CheckConstraint)
        and constraint.name == "ck_document_versions_positive_number"
        for constraint in constraints
    )

def test_document_has_only_one_active_version_index():
    indexes = DocumentVersionORM.__table__.indexes

    active_index = next(
        (
            index
            for index in indexes
            if index.name == "uq_document_versions_one_active"
        ),
        None,
    )

    assert active_index is not None
    assert active_index.unique is True

    assert [column.name for column in active_index.columns] == [
        "document_id"
    ]

    condition = active_index.dialect_options["postgresql"]["where"]

    assert str(condition) == "status = 'active'"

