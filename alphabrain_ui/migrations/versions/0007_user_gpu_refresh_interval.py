"""Add the per-user GPU refresh interval preference.

Revision ID: 0007_user_gpu_refresh_interval
Revises: 0006_inference_publication
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_user_gpu_refresh_interval"
down_revision = "0006_inference_publication"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(
            sa.Column(
                "gpu_refresh_interval_seconds",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("5"),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_column("gpu_refresh_interval_seconds")
