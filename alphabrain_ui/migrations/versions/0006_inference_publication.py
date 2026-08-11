"""Add managed inference history and model publications.

Revision ID: 0006_inference_publication
Revises: 0005_evaluation_groups
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_inference_publication"
down_revision = "0005_evaluation_groups"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inference_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("deployment_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("image_count", sa.Integer(), nullable=False),
        sa.Column("save_inputs", sa.Boolean(), nullable=False),
        sa.Column("request_summary", sa.JSON(), nullable=False),
        sa.Column("deployment_metadata", sa.JSON(), nullable=False),
        sa.Column("output", sa.JSON(), nullable=False),
        sa.Column("input_paths", sa.JSON(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["deployment_id"], ["model_deployments.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id"),
    )
    for column in ("request_id", "owner_id", "deployment_id", "status", "created_at"):
        op.create_index(f"ix_inference_runs_{column}", "inference_runs", [column], unique=False)

    op.create_table(
        "utility_gpu_reservations",
        sa.Column("gpu_index", sa.Integer(), nullable=False),
        sa.Column("utility_run_id", sa.String(length=36), nullable=False),
        sa.Column("reserved_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["utility_run_id"], ["utility_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("gpu_index"),
    )
    op.create_index(
        "ix_utility_gpu_reservations_utility_run_id",
        "utility_gpu_reservations",
        ["utility_run_id"],
        unique=False,
    )

    op.create_table(
        "model_publications",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=36), nullable=True),
        sa.Column("utility_run_id", sa.String(length=36), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("repo_id", sa.String(length=256), nullable=False),
        sa.Column("revision", sa.String(length=128), nullable=False),
        sa.Column("private", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("result_url", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["checkpoint_id"], ["checkpoints.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["utility_run_id"], ["utility_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in (
        "owner_id",
        "checkpoint_id",
        "utility_run_id",
        "provider",
        "repo_id",
        "status",
        "created_at",
    ):
        op.create_index(f"ix_model_publications_{column}", "model_publications", [column], unique=False)


def downgrade() -> None:
    for column in reversed(
        (
            "owner_id",
            "checkpoint_id",
            "utility_run_id",
            "provider",
            "repo_id",
            "status",
            "created_at",
        )
    ):
        op.drop_index(f"ix_model_publications_{column}", table_name="model_publications")
    op.drop_table("model_publications")
    op.drop_index(
        "ix_utility_gpu_reservations_utility_run_id",
        table_name="utility_gpu_reservations",
    )
    op.drop_table("utility_gpu_reservations")
    for column in reversed(("request_id", "owner_id", "deployment_id", "status", "created_at")):
        op.drop_index(f"ix_inference_runs_{column}", table_name="inference_runs")
    op.drop_table("inference_runs")
