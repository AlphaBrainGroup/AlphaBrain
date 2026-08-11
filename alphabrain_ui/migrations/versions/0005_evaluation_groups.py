"""Add grouped and specialized evaluation metadata.

Revision ID: 0005_evaluation_groups
Revises: 0004_resources_datasets
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_evaluation_groups"
down_revision = "0004_resources_datasets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_groups",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("result_summary", sa.JSON(), nullable=False),
        sa.Column("result_path", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in ("owner_id", "name", "kind", "status", "created_at"):
        op.create_index(
            f"ix_evaluation_groups_{column}",
            "evaluation_groups",
            [column],
            unique=False,
        )

    # SQLite cannot add foreign keys in place.  Alembic's batch mode performs
    # a table copy there and emits normal ALTER statements on other databases.
    with op.batch_alter_table("evaluation_runs") as batch:
        batch.add_column(sa.Column("group_id", sa.String(length=36), nullable=True))
        batch.add_column(
            sa.Column("position", sa.Integer(), nullable=False, server_default=sa.text("0"))
        )
        batch.add_column(
            sa.Column(
                "evaluation_kind",
                sa.String(length=32),
                nullable=False,
                server_default="standard",
            )
        )
        batch.add_column(
            sa.Column(
                "source_kind",
                sa.String(length=32),
                nullable=False,
                server_default="temporary_checkpoint",
            )
        )
        batch.add_column(sa.Column("deployment_id", sa.String(length=36), nullable=True))
        batch.add_column(
            sa.Column(
                "result_schema_version",
                sa.String(length=32),
                nullable=False,
                server_default="evaluation-result-v1",
            )
        )
        batch.create_foreign_key(
            "fk_evaluation_runs_group_id_evaluation_groups",
            "evaluation_groups",
            ["group_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_foreign_key(
            "fk_evaluation_runs_deployment_id_model_deployments",
            "model_deployments",
            ["deployment_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index("ix_evaluation_runs_group_id", ["group_id"], unique=False)
        batch.create_index("ix_evaluation_runs_evaluation_kind", ["evaluation_kind"], unique=False)
        batch.create_index("ix_evaluation_runs_source_kind", ["source_kind"], unique=False)
        batch.create_index("ix_evaluation_runs_deployment_id", ["deployment_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("evaluation_runs") as batch:
        batch.drop_index("ix_evaluation_runs_deployment_id")
        batch.drop_index("ix_evaluation_runs_source_kind")
        batch.drop_index("ix_evaluation_runs_evaluation_kind")
        batch.drop_index("ix_evaluation_runs_group_id")
        batch.drop_constraint(
            "fk_evaluation_runs_deployment_id_model_deployments", type_="foreignkey"
        )
        batch.drop_constraint(
            "fk_evaluation_runs_group_id_evaluation_groups", type_="foreignkey"
        )
        batch.drop_column("result_schema_version")
        batch.drop_column("deployment_id")
        batch.drop_column("source_kind")
        batch.drop_column("evaluation_kind")
        batch.drop_column("position")
        batch.drop_column("group_id")

    for column in reversed(("owner_id", "name", "kind", "status", "created_at")):
        op.drop_index(f"ix_evaluation_groups_{column}", table_name="evaluation_groups")
    op.drop_table("evaluation_groups")
