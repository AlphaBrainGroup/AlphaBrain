"""Add normalized fingerprints for template duplicate detection.

Revision ID: 0008_template_fingerprints
Revises: 0007_user_gpu_refresh_interval
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from alphabrain_ui.template_fingerprints import (
    TEMPLATE_FINGERPRINT_VERSION,
    template_spec_fingerprint,
)

revision = "0008_template_fingerprints"
down_revision = "0007_user_gpu_refresh_interval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("experiment_templates") as batch:
        batch.add_column(
            sa.Column("spec_fingerprint", sa.String(length=64), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column(
                "fingerprint_version",
                sa.Integer(),
                nullable=False,
                server_default=str(TEMPLATE_FINGERPRINT_VERSION),
            )
        )

    templates = sa.table(
        "experiment_templates",
        sa.column("id", sa.String(length=36)),
        sa.column("spec", sa.JSON()),
        sa.column("spec_fingerprint", sa.String(length=64)),
        sa.column("fingerprint_version", sa.Integer()),
    )
    connection = op.get_bind()
    for row in connection.execute(sa.select(templates.c.id, templates.c.spec)).mappings():
        connection.execute(
            templates.update()
            .where(templates.c.id == row["id"])
            .values(
                spec_fingerprint=template_spec_fingerprint(row["spec"] or {}),
                fingerprint_version=TEMPLATE_FINGERPRINT_VERSION,
            )
        )

    op.create_index(
        "ix_experiment_templates_spec_fingerprint",
        "experiment_templates",
        ["spec_fingerprint"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_experiment_templates_spec_fingerprint",
        table_name="experiment_templates",
    )
    with op.batch_alter_table("experiment_templates") as batch:
        batch.drop_column("fingerprint_version")
        batch.drop_column("spec_fingerprint")
