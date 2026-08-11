"""Add managed model deployments.

Revision ID: 0002_model_deployments
Revises: 0001_initial
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_model_deployments"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_deployments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=36), nullable=True),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("checkpoint_path", sa.Text(), nullable=False),
        sa.Column("combination_id", sa.String(length=128), nullable=False),
        sa.Column("adapter_id", sa.String(length=128), nullable=False),
        sa.Column("backbone_id", sa.String(length=128), nullable=False),
        sa.Column("action_head_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_gpu_count", sa.Integer(), nullable=False),
        sa.Column("requested_gpu_ids", sa.JSON(), nullable=False),
        sa.Column("assigned_gpu_ids", sa.JSON(), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("endpoint_scope", sa.String(length=16), nullable=False),
        sa.Column("bind_host", sa.String(length=128), nullable=False),
        sa.Column("advertised_host", sa.String(length=255), nullable=False),
        sa.Column("port", sa.Integer(), nullable=True),
        sa.Column("idle_timeout_seconds", sa.Integer(), nullable=False),
        sa.Column("api_key_hash", sa.String(length=64), nullable=False),
        sa.Column("api_key_prefix", sa.String(length=16), nullable=False),
        sa.Column("command", sa.JSON(), nullable=False),
        sa.Column("environment", sa.JSON(), nullable=False),
        sa.Column("log_path", sa.Text(), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("pgid", sa.Integer(), nullable=True),
        sa.Column("process_created_at", sa.Float(), nullable=True),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["checkpoint_id"], ["checkpoints.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in (
        "owner_id",
        "checkpoint_id",
        "name",
        "combination_id",
        "adapter_id",
        "backbone_id",
        "action_head_id",
        "status",
        "port",
        "queued_at",
    ):
        op.create_index(f"ix_model_deployments_{column}", "model_deployments", [column], unique=False)

    op.create_table(
        "deployment_gpu_reservations",
        sa.Column("gpu_index", sa.Integer(), nullable=False),
        sa.Column("deployment_id", sa.String(length=36), nullable=False),
        sa.Column("reserved_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["deployment_id"], ["model_deployments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("gpu_index"),
    )
    op.create_index(
        "ix_deployment_gpu_reservations_deployment_id",
        "deployment_gpu_reservations",
        ["deployment_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_deployment_gpu_reservations_deployment_id", table_name="deployment_gpu_reservations")
    op.drop_table("deployment_gpu_reservations")
    for column in reversed(
        (
            "owner_id",
            "checkpoint_id",
            "name",
            "combination_id",
            "adapter_id",
            "backbone_id",
            "action_head_id",
            "status",
            "port",
            "queued_at",
        )
    ):
        op.drop_index(f"ix_model_deployments_{column}", table_name="model_deployments")
    op.drop_table("model_deployments")
