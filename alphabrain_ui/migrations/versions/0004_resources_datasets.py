"""Add managed resources, utility runs, and dataset registry.

Revision ID: 0004_resources_datasets
Revises: 0003_evaluation_runs
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_resources_datasets"
down_revision = "0003_evaluation_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "utility_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("resource_id", sa.String(length=160), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("queue_class", sa.String(length=16), nullable=False),
        sa.Column("requested_gpu_count", sa.Integer(), nullable=False),
        sa.Column("requested_gpu_ids", sa.JSON(), nullable=False),
        sa.Column("assigned_gpu_ids", sa.JSON(), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("command", sa.JSON(), nullable=False),
        sa.Column("environment", sa.JSON(), nullable=False),
        sa.Column("cwd", sa.Text(), nullable=False),
        sa.Column("output_path", sa.Text(), nullable=False),
        sa.Column("log_path", sa.Text(), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("pgid", sa.Integer(), nullable=True),
        sa.Column("process_created_at", sa.Float(), nullable=True),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in ("owner_id", "kind", "resource_id", "status", "queue_class", "queued_at"):
        op.create_index(f"ix_utility_runs_{column}", "utility_runs", [column], unique=False)

    op.create_table(
        "dataset_registrations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("storage_mode", sa.String(length=16), nullable=False),
        sa.Column("visibility", sa.String(length=16), nullable=False),
        sa.Column("format", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("dataset_id", sa.String(length=128), nullable=False),
        sa.Column("dataset_mix", sa.String(length=128), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("episode_count", sa.Integer(), nullable=False),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("validation", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("copy_run_id", sa.String(length=36), nullable=True),
        sa.Column("stats_run_id", sa.String(length=36), nullable=True),
        sa.Column("stats_status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["copy_run_id"], ["utility_runs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["stats_run_id"], ["utility_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("path"),
    )
    for column in ("owner_id", "name", "storage_mode", "visibility", "format", "status", "fingerprint", "copy_run_id", "stats_run_id", "created_at"):
        op.create_index(f"ix_dataset_registrations_{column}", "dataset_registrations", [column], unique=False)

    op.create_table(
        "dataset_mixtures",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("visibility", sa.String(length=16), nullable=False),
        sa.Column("members", sa.JSON(), nullable=False),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in ("owner_id", "name", "visibility", "created_at"):
        op.create_index(f"ix_dataset_mixtures_{column}", "dataset_mixtures", [column], unique=False)


def downgrade() -> None:
    for column in reversed(("owner_id", "name", "visibility", "created_at")):
        op.drop_index(f"ix_dataset_mixtures_{column}", table_name="dataset_mixtures")
    op.drop_table("dataset_mixtures")
    for column in reversed(("owner_id", "name", "storage_mode", "visibility", "format", "status", "fingerprint", "copy_run_id", "stats_run_id", "created_at")):
        op.drop_index(f"ix_dataset_registrations_{column}", table_name="dataset_registrations")
    op.drop_table("dataset_registrations")
    for column in reversed(("owner_id", "kind", "resource_id", "status", "queue_class", "queued_at")):
        op.drop_index(f"ix_utility_runs_{column}", table_name="utility_runs")
    op.drop_table("utility_runs")
