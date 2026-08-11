"""Add managed benchmark evaluation runs.

Revision ID: 0003_evaluation_runs
Revises: 0002_model_deployments
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_evaluation_runs"
down_revision = "0002_model_deployments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=36), nullable=True),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("checkpoint_path", sa.Text(), nullable=False),
        sa.Column("combination_id", sa.String(length=128), nullable=False),
        sa.Column("adapter_id", sa.String(length=128), nullable=False),
        sa.Column("backbone_id", sa.String(length=128), nullable=False),
        sa.Column("action_head_id", sa.String(length=128), nullable=False),
        sa.Column("benchmark_id", sa.String(length=64), nullable=False),
        sa.Column("benchmark_status", sa.String(length=32), nullable=False),
        sa.Column("preset", sa.String(length=32), nullable=False),
        sa.Column("suite", sa.String(length=128), nullable=False),
        sa.Column("task_set", sa.String(length=512), nullable=False),
        sa.Column("split", sa.String(length=128), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("model_parameters", sa.JSON(), nullable=False),
        sa.Column("wandb", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_gpu_count", sa.Integer(), nullable=False),
        sa.Column("requested_gpu_ids", sa.JSON(), nullable=False),
        sa.Column("assigned_gpu_ids", sa.JSON(), nullable=False),
        sa.Column("server_port", sa.Integer(), nullable=True),
        sa.Column("command", sa.JSON(), nullable=False),
        sa.Column("environment", sa.JSON(), nullable=False),
        sa.Column("cwd", sa.Text(), nullable=False),
        sa.Column("output_dir", sa.Text(), nullable=False),
        sa.Column("config_path", sa.Text(), nullable=False),
        sa.Column("log_path", sa.Text(), nullable=False),
        sa.Column("progress_path", sa.Text(), nullable=False),
        sa.Column("result_path", sa.Text(), nullable=False),
        sa.Column("result_summary", sa.JSON(), nullable=False),
        sa.Column("wandb_status", sa.String(length=32), nullable=False),
        sa.Column("wandb_error", sa.Text(), nullable=False),
        sa.Column("wandb_run_url", sa.Text(), nullable=False),
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
        sa.ForeignKeyConstraint(["checkpoint_id"], ["checkpoints.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("output_dir", name="uq_evaluation_runs_output_dir"),
    )
    for column in (
        "owner_id",
        "checkpoint_id",
        "name",
        "combination_id",
        "adapter_id",
        "backbone_id",
        "action_head_id",
        "benchmark_id",
        "status",
        "server_port",
        "wandb_status",
        "queued_at",
    ):
        op.create_index(f"ix_evaluation_runs_{column}", "evaluation_runs", [column], unique=False)

    op.create_table(
        "evaluation_gpu_reservations",
        sa.Column("gpu_index", sa.Integer(), nullable=False),
        sa.Column("evaluation_id", sa.String(length=36), nullable=False),
        sa.Column("reserved_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["evaluation_id"], ["evaluation_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("gpu_index"),
    )
    op.create_index(
        "ix_evaluation_gpu_reservations_evaluation_id",
        "evaluation_gpu_reservations",
        ["evaluation_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_evaluation_gpu_reservations_evaluation_id", table_name="evaluation_gpu_reservations")
    op.drop_table("evaluation_gpu_reservations")
    for column in reversed(
        (
            "owner_id",
            "checkpoint_id",
            "name",
            "combination_id",
            "adapter_id",
            "backbone_id",
            "action_head_id",
            "benchmark_id",
            "status",
            "server_port",
            "wandb_status",
            "queued_at",
        )
    ):
        op.drop_index(f"ix_evaluation_runs_{column}", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
