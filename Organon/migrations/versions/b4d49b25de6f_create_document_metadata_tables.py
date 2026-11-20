"""create document metadata tables

Revision ID: b4d49b25de6f
Revises:
Create Date: 2026-11-21 02:21:48.511679
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b4d49b25de6f"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create document metadata tables."""

    # 1. Create documents without its circular foreign key.
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("external_id", sa.String(), nullable=False),
        sa.Column("current_version_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "external_id",
            name="uq_documents_tenant_external_id",
        ),
    )

    # 2. Create document_versions.
    op.create_table(
        "document_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default="processing",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "activated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            "status IN ('processing', 'ready', 'active', 'archived', 'failed')",
            name="ck_document_versions_valid_status",
        ),
        sa.CheckConstraint(
            "version_number > 0",
            name="ck_document_versions_positive_number",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "version_number",
            name="uq_document_versions_number",
        ),
        sa.UniqueConstraint(
            "id",
            "document_id",
            name="uq_document_versions_id_document",
        ),
    )

    # 3. Enforce only one active version per document.
    op.create_index(
        "uq_document_versions_one_active",
        "document_versions",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    # 4. Add the circular foreign key after both tables exist.
    op.create_foreign_key(
        "fk_documents_current_version",
        "documents",
        "document_versions",
        ["current_version_id", "id"],
        ["id", "document_id"],
    )


def downgrade() -> None:
    """Remove document metadata tables."""

    # Remove circular foreign key first.
    op.drop_constraint(
        "fk_documents_current_version",
        "documents",
        type_="foreignkey",
    )

    op.drop_index(
        "uq_document_versions_one_active",
        table_name="document_versions",
    )

    op.drop_table("document_versions")
    op.drop_table("documents")
