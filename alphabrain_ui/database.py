from __future__ import annotations

import contextlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class UTCDateTime(TypeDecorator[datetime]):
    """Persist UTC in SQLite and always restore timezone-aware values."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:  # type: ignore[no-untyped-def]
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:  # type: ignore[no-untyped-def]
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(128), default="")
    password_hash: Mapped[str] = mapped_column(Text, default="")
    role: Mapped[str] = mapped_column(String(32), default="researcher", index=True)
    language: Mapped[str] = mapped_column(String(16), default="zh-CN")
    theme: Mapped[str] = mapped_column(String(16), default="light")
    gpu_refresh_interval_seconds: Mapped[int] = mapped_column(Integer, default=5)
    experimental_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_local: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    csrf_token: Mapped[str] = mapped_column(String(128))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    user: Mapped[User] = relationship()


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[object] = mapped_column(JSON)
    updated_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class ExperimentTemplate(Base):
    __tablename__ = "experiment_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    visibility: Mapped[str] = mapped_column(String(16), default="private", index=True)
    spec: Mapped[dict] = mapped_column(JSON)
    spec_fingerprint: Mapped[str] = mapped_column(String(64), default="", index=True)
    fingerprint_version: Mapped[int] = mapped_column(Integer, default=1)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    owner: Mapped[User] = relationship()


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    family: Mapped[str] = mapped_column(String(64), index=True)
    compatibility: Mapped[str] = mapped_column(String(32), default="verified")
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    spec: Mapped[dict] = mapped_column(JSON)
    resolved: Mapped[dict] = mapped_column(JSON)
    config_snapshot_path: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    owner: Mapped[User] = relationship()
    stages: Mapped[list[ExperimentStage]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", order_by="ExperimentStage.position"
    )


class ExperimentStage(Base):
    __tablename__ = "experiment_stages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(96))
    phase: Mapped[str] = mapped_column(String(64), default="train")
    position: Mapped[int] = mapped_column(Integer, default=0)
    dependency_stage_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiment_stages.id", ondelete="SET NULL"), nullable=True
    )
    resolved: Mapped[dict] = mapped_column(JSON)
    experiment: Mapped[Experiment] = relationship(back_populates="stages", foreign_keys=[experiment_id])
    jobs: Mapped[list[Job]] = relationship(back_populates="stage", cascade="all, delete-orphan")


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("output_dir", name="uq_jobs_output_dir"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    stage_id: Mapped[str] = mapped_column(ForeignKey("experiment_stages.id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    requested_gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    requested_gpu_ids: Mapped[list] = mapped_column(JSON, default=list)
    assigned_gpu_ids: Mapped[list] = mapped_column(JSON, default=list)
    command: Mapped[list] = mapped_column(JSON, default=list)
    environment: Mapped[dict] = mapped_column(JSON, default=dict)
    cwd: Mapped[str] = mapped_column(Text)
    output_dir: Mapped[str] = mapped_column(Text, default="")
    input_checkpoint_path: Mapped[str] = mapped_column(Text, default="", index=True)
    log_path: Mapped[str] = mapped_column(Text, default="")
    metrics_path: Mapped[str] = mapped_column(Text, default="")
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pgid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    process_created_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    queued_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    stop_requested_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    stage: Mapped[ExperimentStage] = relationship(back_populates="jobs")
    experiment: Mapped[Experiment] = relationship()
    owner: Mapped[User] = relationship()


class GPUReservation(Base):
    __tablename__ = "gpu_reservations"

    gpu_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    reserved_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    checkpoint_id: Mapped[str | None] = mapped_column(
        ForeignKey("checkpoints.id", ondelete="SET NULL"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(160), index=True)
    checkpoint_path: Mapped[str] = mapped_column(Text)
    combination_id: Mapped[str] = mapped_column(String(128), index=True)
    adapter_id: Mapped[str] = mapped_column(String(128), index=True)
    backbone_id: Mapped[str] = mapped_column(String(128), index=True)
    action_head_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    requested_gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    requested_gpu_ids: Mapped[list] = mapped_column(JSON, default=list)
    assigned_gpu_ids: Mapped[list] = mapped_column(JSON, default=list)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    endpoint_scope: Mapped[str] = mapped_column(String(16), default="local")
    bind_host: Mapped[str] = mapped_column(String(128), default="127.0.0.1")
    advertised_host: Mapped[str] = mapped_column(String(255), default="127.0.0.1")
    port: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    idle_timeout_seconds: Mapped[int] = mapped_column(Integer, default=1800)
    api_key_hash: Mapped[str] = mapped_column(String(64))
    api_key_prefix: Mapped[str] = mapped_column(String(16))
    command: Mapped[list] = mapped_column(JSON, default=list)
    environment: Mapped[dict] = mapped_column(JSON, default=dict)
    log_path: Mapped[str] = mapped_column(Text, default="")
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pgid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    process_created_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    queued_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    ready_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    stop_requested_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    owner: Mapped[User] = relationship()
    checkpoint: Mapped[Checkpoint | None] = relationship(foreign_keys=[checkpoint_id])


class DeploymentGPUReservation(Base):
    __tablename__ = "deployment_gpu_reservations"

    gpu_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    deployment_id: Mapped[str] = mapped_column(
        ForeignKey("model_deployments.id", ondelete="CASCADE"), index=True
    )
    reserved_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (UniqueConstraint("output_dir", name="uq_evaluation_runs_output_dir"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    group_id: Mapped[str | None] = mapped_column(
        ForeignKey("evaluation_groups.id", ondelete="SET NULL"), nullable=True, index=True
    )
    position: Mapped[int] = mapped_column(Integer, default=0)
    evaluation_kind: Mapped[str] = mapped_column(String(32), default="standard", index=True)
    source_kind: Mapped[str] = mapped_column(String(32), default="temporary_checkpoint", index=True)
    deployment_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_deployments.id", ondelete="SET NULL"), nullable=True, index=True
    )
    result_schema_version: Mapped[str] = mapped_column(
        String(32), default="evaluation-result-v1"
    )
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    checkpoint_id: Mapped[str | None] = mapped_column(
        ForeignKey("checkpoints.id", ondelete="SET NULL"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(160), index=True)
    checkpoint_path: Mapped[str] = mapped_column(Text)
    combination_id: Mapped[str] = mapped_column(String(128), index=True)
    adapter_id: Mapped[str] = mapped_column(String(128), index=True)
    backbone_id: Mapped[str] = mapped_column(String(128), index=True)
    action_head_id: Mapped[str] = mapped_column(String(128), index=True)
    benchmark_id: Mapped[str] = mapped_column(String(64), index=True)
    benchmark_status: Mapped[str] = mapped_column(String(32), default="verified")
    preset: Mapped[str] = mapped_column(String(32), default="quick")
    suite: Mapped[str] = mapped_column(String(128), default="")
    task_set: Mapped[str] = mapped_column(String(512), default="")
    split: Mapped[str] = mapped_column(String(128), default="")
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    model_parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    wandb: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    requested_gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    requested_gpu_ids: Mapped[list] = mapped_column(JSON, default=list)
    assigned_gpu_ids: Mapped[list] = mapped_column(JSON, default=list)
    server_port: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    command: Mapped[list] = mapped_column(JSON, default=list)
    environment: Mapped[dict] = mapped_column(JSON, default=dict)
    cwd: Mapped[str] = mapped_column(Text)
    output_dir: Mapped[str] = mapped_column(Text)
    config_path: Mapped[str] = mapped_column(Text, default="")
    log_path: Mapped[str] = mapped_column(Text, default="")
    progress_path: Mapped[str] = mapped_column(Text, default="")
    result_path: Mapped[str] = mapped_column(Text, default="")
    result_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    wandb_status: Mapped[str] = mapped_column(String(32), default="disabled", index=True)
    wandb_error: Mapped[str] = mapped_column(Text, default="")
    wandb_run_url: Mapped[str] = mapped_column(Text, default="")
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pgid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    process_created_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    queued_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    stop_requested_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    owner: Mapped[User] = relationship()
    checkpoint: Mapped[Checkpoint | None] = relationship(foreign_keys=[checkpoint_id])
    group: Mapped[EvaluationGroup | None] = relationship(back_populates="runs")
    deployment: Mapped[ModelDeployment | None] = relationship(foreign_keys=[deployment_id])


class EvaluationGroup(Base):
    """One user request expanded into one or more managed evaluation runs."""

    __tablename__ = "evaluation_groups"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    kind: Mapped[str] = mapped_column(String(32), default="batch", index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    spec: Mapped[dict] = mapped_column(JSON, default=dict)
    result_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    result_path: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    owner: Mapped[User] = relationship()
    runs: Mapped[list[EvaluationRun]] = relationship(
        back_populates="group", order_by="EvaluationRun.position"
    )


class EvaluationGPUReservation(Base):
    __tablename__ = "evaluation_gpu_reservations"

    gpu_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), index=True
    )
    reserved_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class InferenceRun(Base):
    """One audited Playground request to a UI-managed deployment."""

    __tablename__ = "inference_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    deployment_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_deployments.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    batch_size: Mapped[int] = mapped_column(Integer, default=1)
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    save_inputs: Mapped[bool] = mapped_column(Boolean, default=False)
    request_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    deployment_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    output_json: Mapped[dict] = mapped_column("output", JSON, default=dict)
    input_paths: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    owner: Mapped[User] = relationship()
    deployment: Mapped[ModelDeployment | None] = relationship(foreign_keys=[deployment_id])


class UtilityRun(Base):
    """A managed non-training task such as a download, copy, or preprocessing run."""

    __tablename__ = "utility_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    resource_id: Mapped[str] = mapped_column(String(160), default="", index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    queue_class: Mapped[str] = mapped_column(String(16), default="cpu", index=True)
    requested_gpu_count: Mapped[int] = mapped_column(Integer, default=0)
    requested_gpu_ids: Mapped[list] = mapped_column(JSON, default=list)
    assigned_gpu_ids: Mapped[list] = mapped_column(JSON, default=list)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    command: Mapped[list] = mapped_column(JSON, default=list)
    environment: Mapped[dict] = mapped_column(JSON, default=dict)
    cwd: Mapped[str] = mapped_column(Text)
    output_path: Mapped[str] = mapped_column(Text, default="")
    log_path: Mapped[str] = mapped_column(Text, default="")
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pgid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    process_created_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    queued_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    stop_requested_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    owner: Mapped[User] = relationship()


class UtilityGPUReservation(Base):
    __tablename__ = "utility_gpu_reservations"

    gpu_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    utility_run_id: Mapped[str] = mapped_column(
        ForeignKey("utility_runs.id", ondelete="CASCADE"), index=True
    )
    reserved_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class DatasetRegistration(Base):
    """A validated server-local dataset, either referenced in-place or managed by the UI."""

    __tablename__ = "dataset_registrations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    path: Mapped[str] = mapped_column(Text, unique=True)
    source_path: Mapped[str] = mapped_column(Text, default="")
    storage_mode: Mapped[str] = mapped_column(String(16), default="reference", index=True)
    visibility: Mapped[str] = mapped_column(String(16), default="shared", index=True)
    format: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    status: Mapped[str] = mapped_column(String(32), default="validating", index=True)
    dataset_id: Mapped[str] = mapped_column(String(128), default="lerobot")
    dataset_mix: Mapped[str] = mapped_column(String(128), default="")
    fingerprint: Mapped[str] = mapped_column(String(64), default="", index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    episode_count: Mapped[int] = mapped_column(Integer, default=0)
    step_count: Mapped[int] = mapped_column(Integer, default=0)
    validation: Mapped[dict] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    copy_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("utility_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    stats_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("utility_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    stats_status: Mapped[str] = mapped_column(String(32), default="unknown")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    owner: Mapped[User] = relationship()


class DatasetMixture(Base):
    __tablename__ = "dataset_mixtures"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    visibility: Mapped[str] = mapped_column(String(16), default="shared", index=True)
    members: Mapped[list] = mapped_column(JSON, default=list)
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    owner: Mapped[User] = relationship()


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    name: Mapped[str] = mapped_column(String(256))
    path: Mapped[str] = mapped_column(Text, unique=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True)
    path: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(String(256))
    step: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    is_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    is_resumable: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class ModelPublication(Base):
    """Status and provenance for publishing a checkpoint to a model registry."""

    __tablename__ = "model_publications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    checkpoint_id: Mapped[str | None] = mapped_column(
        ForeignKey("checkpoints.id", ondelete="SET NULL"), nullable=True, index=True
    )
    utility_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("utility_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    provider: Mapped[str] = mapped_column(String(32), default="huggingface", index=True)
    repo_id: Mapped[str] = mapped_column(String(256), index=True)
    revision: Mapped[str] = mapped_column(String(128), default="main")
    private: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    source_path: Mapped[str] = mapped_column(Text)
    result_url: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    owner: Mapped[User] = relationship()
    checkpoint: Mapped[Checkpoint | None] = relationship(foreign_keys=[checkpoint_id])
    utility_run: Mapped[UtilityRun | None] = relationship(foreign_keys=[utility_run_id])


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    actor_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(96), index=True)
    target_type: Mapped[str] = mapped_column(String(64), default="")
    target_id: Mapped[str] = mapped_column(String(128), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.engine = create_engine(
            f"sqlite:///{path}",
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=NullPool,
            future=True,
        )

        @event.listens_for(self.engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def migrate(self) -> None:
        """Apply bundled Alembic migrations to the UI state database."""
        from alembic import command
        from alembic.config import Config

        migrations = Path(__file__).with_name("migrations")
        config = Config()
        config.set_main_option("script_location", str(migrations))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{self.path}")
        command.upgrade(config, "head")

    @contextlib.contextmanager
    def session(self) -> Iterator[Session]:
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
