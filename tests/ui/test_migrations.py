from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import MetaData, create_engine, inspect, select

from alphabrain_ui.database import Base, Database, EvaluationRun, Job, ModelDeployment
from alphabrain_ui.template_fingerprints import TEMPLATE_FINGERPRINT_VERSION, template_spec_fingerprint


def test_frozen_initial_migration_matches_models_and_downgrades(tmp_path: Path) -> None:
    database = Database(tmp_path / "migration.sqlite3")
    database.migrate()

    inspector = inspect(database.engine)
    business_tables = set(inspector.get_table_names()) - {"alembic_version"}
    assert business_tables == set(Base.metadata.tables)
    assert {item["name"] for item in inspector.get_columns("jobs")} == {column.name for column in Job.__table__.columns}
    job_uniques = {tuple(item["column_names"]) for item in inspector.get_unique_constraints("jobs")}
    assert ("output_dir",) in job_uniques
    assert "ix_jobs_input_checkpoint_path" in {item["name"] for item in inspector.get_indexes("jobs")}
    assert {item["name"] for item in inspector.get_columns("model_deployments")} == {
        column.name for column in ModelDeployment.__table__.columns
    }
    assert {item["name"] for item in inspector.get_columns("evaluation_runs")} == {
        column.name for column in EvaluationRun.__table__.columns
    }

    config = Config()
    config.set_main_option("script_location", str(Path(__file__).resolve().parents[2] / "alphabrain_ui/migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.path}")
    command.downgrade(config, "base")
    assert not (set(inspect(database.engine).get_table_names()) - {"alembic_version"})


def test_template_fingerprint_migration_backfills_and_downgrades(tmp_path: Path) -> None:
    path = tmp_path / "template-fingerprint.sqlite3"
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).resolve().parents[2] / "alphabrain_ui/migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(config, "0007_user_gpu_refresh_interval")

    engine = create_engine(f"sqlite:///{path}")
    metadata = MetaData()
    metadata.reflect(engine)
    now = datetime.now(timezone.utc)
    spec = {
        "dataset": {"id": "libero"},
        "parameters": {"run_id": "migration-run", "learning_rate": 0.0001},
        "resources": {"allocation": "fixed", "num_gpus": 1, "gpu_ids": [0]},
    }
    with engine.begin() as connection:
        connection.execute(
            metadata.tables["users"].insert().values(
                id="user-1",
                username="admin",
                display_name="Admin",
                password_hash="",
                role="administrator",
                language="zh-CN",
                theme="light",
                gpu_refresh_interval_seconds=5,
                experimental_enabled=False,
                is_active=True,
                is_local=True,
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            metadata.tables["experiment_templates"].insert().values(
                id="template-1",
                owner_id="user-1",
                name="Existing template",
                description="",
                visibility="private",
                spec=spec,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )

    command.upgrade(config, "head")
    upgraded = MetaData()
    upgraded.reflect(engine)
    templates = upgraded.tables["experiment_templates"]
    with engine.connect() as connection:
        row = connection.execute(select(templates).where(templates.c.id == "template-1")).mappings().one()
    assert row["spec_fingerprint"] == template_spec_fingerprint(spec)
    assert row["fingerprint_version"] == TEMPLATE_FINGERPRINT_VERSION
    assert "ix_experiment_templates_spec_fingerprint" in {
        item["name"] for item in inspect(engine).get_indexes("experiment_templates")
    }

    command.downgrade(config, "0007_user_gpu_refresh_interval")
    column_names = {item["name"] for item in inspect(engine).get_columns("experiment_templates")}
    assert "spec_fingerprint" not in column_names
    assert "fingerprint_version" not in column_names
