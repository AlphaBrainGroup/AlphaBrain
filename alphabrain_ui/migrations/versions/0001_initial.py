"""Initial AlphaBrain UI schema.

Revision ID: 0001_initial
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("theme", sa.String(length=16), nullable=False),
        sa.Column("experimental_enabled", sa.Boolean(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_local", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_role", "users", ["role"], unique=False)
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("actor_id", sa.String(length=36), nullable=True),
        sa.Column("action", sa.String(length=96), nullable=False),
        sa.Column("target_type", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=128), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_events_action", "audit_events", ["action"], unique=False)
    op.create_index("ix_audit_events_created_at", "audit_events", ["created_at"], unique=False)

    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("csrf_token", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_auth_sessions_expires_at", "auth_sessions", ["expires_at"], unique=False)
    op.create_index("ix_auth_sessions_token_hash", "auth_sessions", ["token_hash"], unique=True)
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"], unique=False)

    op.create_table(
        "experiment_templates",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("visibility", sa.String(length=16), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_experiment_templates_name", "experiment_templates", ["name"], unique=False)
    op.create_index("ix_experiment_templates_owner_id", "experiment_templates", ["owner_id"], unique=False)
    op.create_index("ix_experiment_templates_visibility", "experiment_templates", ["visibility"], unique=False)

    op.create_table(
        "experiments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("family", sa.String(length=64), nullable=False),
        sa.Column("compatibility", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("resolved", sa.JSON(), nullable=False),
        sa.Column("config_snapshot_path", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_experiments_created_at", "experiments", ["created_at"], unique=False)
    op.create_index("ix_experiments_family", "experiments", ["family"], unique=False)
    op.create_index("ix_experiments_name", "experiments", ["name"], unique=False)
    op.create_index("ix_experiments_owner_id", "experiments", ["owner_id"], unique=False)
    op.create_index("ix_experiments_status", "experiments", ["status"], unique=False)

    op.create_table(
        "system_settings",
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("updated_by", sa.String(length=36), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["updated_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("key"),
    )

    op.create_table(
        "experiment_stages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=96), nullable=False),
        sa.Column("phase", sa.String(length=64), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("dependency_stage_id", sa.String(length=36), nullable=True),
        sa.Column("resolved", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["dependency_stage_id"], ["experiment_stages.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_experiment_stages_experiment_id", "experiment_stages", ["experiment_id"], unique=False)

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("stage_id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_gpu_count", sa.Integer(), nullable=False),
        sa.Column("requested_gpu_ids", sa.JSON(), nullable=False),
        sa.Column("assigned_gpu_ids", sa.JSON(), nullable=False),
        sa.Column("command", sa.JSON(), nullable=False),
        sa.Column("environment", sa.JSON(), nullable=False),
        sa.Column("cwd", sa.Text(), nullable=False),
        sa.Column("output_dir", sa.Text(), nullable=False),
        sa.Column("input_checkpoint_path", sa.Text(), nullable=False),
        sa.Column("log_path", sa.Text(), nullable=False),
        sa.Column("metrics_path", sa.Text(), nullable=False),
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
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["stage_id"], ["experiment_stages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("output_dir", name="uq_jobs_output_dir"),
    )
    op.create_index("ix_jobs_experiment_id", "jobs", ["experiment_id"], unique=False)
    op.create_index("ix_jobs_input_checkpoint_path", "jobs", ["input_checkpoint_path"], unique=False)
    op.create_index("ix_jobs_owner_id", "jobs", ["owner_id"], unique=False)
    op.create_index("ix_jobs_queued_at", "jobs", ["queued_at"], unique=False)
    op.create_index("ix_jobs_stage_id", "jobs", ["stage_id"], unique=False)
    op.create_index("ix_jobs_status", "jobs", ["status"], unique=False)

    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("kind", sa.String(length=48), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("path"),
    )
    op.create_index("ix_artifacts_experiment_id", "artifacts", ["experiment_id"], unique=False)
    op.create_index("ix_artifacts_kind", "artifacts", ["kind"], unique=False)

    op.create_table(
        "checkpoints",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("is_complete", sa.Boolean(), nullable=False),
        sa.Column("is_resumable", sa.Boolean(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("path"),
    )
    op.create_index("ix_checkpoints_experiment_id", "checkpoints", ["experiment_id"], unique=False)
    op.create_index("ix_checkpoints_step", "checkpoints", ["step"], unique=False)

    op.create_table(
        "gpu_reservations",
        sa.Column("gpu_index", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("reserved_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("gpu_index"),
    )
    op.create_index("ix_gpu_reservations_job_id", "gpu_reservations", ["job_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_gpu_reservations_job_id", table_name="gpu_reservations")
    op.drop_table("gpu_reservations")

    op.drop_index("ix_checkpoints_step", table_name="checkpoints")
    op.drop_index("ix_checkpoints_experiment_id", table_name="checkpoints")
    op.drop_table("checkpoints")

    op.drop_index("ix_artifacts_kind", table_name="artifacts")
    op.drop_index("ix_artifacts_experiment_id", table_name="artifacts")
    op.drop_table("artifacts")

    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_stage_id", table_name="jobs")
    op.drop_index("ix_jobs_queued_at", table_name="jobs")
    op.drop_index("ix_jobs_owner_id", table_name="jobs")
    op.drop_index("ix_jobs_input_checkpoint_path", table_name="jobs")
    op.drop_index("ix_jobs_experiment_id", table_name="jobs")
    op.drop_table("jobs")

    op.drop_index("ix_experiment_stages_experiment_id", table_name="experiment_stages")
    op.drop_table("experiment_stages")

    op.drop_table("system_settings")

    op.drop_index("ix_experiments_status", table_name="experiments")
    op.drop_index("ix_experiments_owner_id", table_name="experiments")
    op.drop_index("ix_experiments_name", table_name="experiments")
    op.drop_index("ix_experiments_family", table_name="experiments")
    op.drop_index("ix_experiments_created_at", table_name="experiments")
    op.drop_table("experiments")

    op.drop_index("ix_experiment_templates_visibility", table_name="experiment_templates")
    op.drop_index("ix_experiment_templates_owner_id", table_name="experiment_templates")
    op.drop_index("ix_experiment_templates_name", table_name="experiment_templates")
    op.drop_table("experiment_templates")

    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_token_hash", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_expires_at", table_name="auth_sessions")
    op.drop_table("auth_sessions")

    op.drop_index("ix_audit_events_created_at", table_name="audit_events")
    op.drop_index("ix_audit_events_action", table_name="audit_events")
    op.drop_table("audit_events")

    op.drop_index("ix_users_username", table_name="users")
    op.drop_index("ix_users_role", table_name="users")
    op.drop_table("users")
