from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


ExecutionBackend = Literal[
    "alphabrain_cl",
    "alphabrain_finetune",
    "alphabrain_eval",
    "alphabrain_vlm_cotrain",
    "alphabrain_world_model",
]

JobStatus = Literal["queued", "running", "completed", "failed"]
Severity = Literal["low", "medium", "high", "critical"]
TrainingStrategy = Literal["continual_learning", "fine_tune", "vlm_cotrain", "world_model_verification"]


class ArtifactBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    videos: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    metrics_json: str | None = None


class TaskBreakdownEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_name: str
    success_rate: float = Field(ge=0.0, le=1.0)
    attempts: int | None = Field(default=None, ge=0)


class FailureCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    label: str
    failure_pattern: str
    affected_tasks: list[str] = Field(default_factory=list)
    share_of_failures: float = Field(ge=0.0, le=1.0)
    failure_count: int = Field(ge=0)
    severity: Severity = "medium"


class EvalRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    project_id: str
    benchmark_suite: str
    checkpoint: str | None = None
    overall_success: float = Field(ge=0.0, le=1.0)
    task_breakdown: list[TaskBreakdownEntry] = Field(default_factory=list)
    failure_clusters: list[FailureCluster] = Field(default_factory=list)
    artifacts: ArtifactBundle = Field(default_factory=ArtifactBundle)
    created_at: datetime = Field(default_factory=utc_now)


class TargetDataSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch_episodes: int = Field(ge=0)
    teleop_corrections: int = Field(ge=0)
    lighting_variants: int = Field(default=0, ge=0)
    language_augmentations: int = Field(default=0, ge=0)


class SourceRatio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    real: float = Field(ge=0.0, le=1.0)
    synthetic: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def ensure_total_is_one(self) -> "SourceRatio":
        total = round(self.real + self.synthetic, 6)
        if total != 1.0:
            raise ValueError("source_ratio.real + source_ratio.synthetic must equal 1.0")
        return self


class PatchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    project_id: str
    based_on_eval_run: str
    root_causes: list[str] = Field(default_factory=list)
    target_data_spec: TargetDataSpec
    annotation_schema: str
    source_ratio: SourceRatio
    training_strategy: TrainingStrategy
    execution_backend: ExecutionBackend
    verification_spec: str
    expected_uplift: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)


class IterationResultSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success_before: float = Field(ge=0.0, le=1.0)
    success_after: float = Field(ge=0.0, le=1.0)


class IterationArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logs: list[str] = Field(default_factory=list)
    videos: list[str] = Field(default_factory=list)
    eval_runs: list[str] = Field(default_factory=list)


class IterationJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration_id: str
    project_id: str
    plan_id: str
    based_on_checkpoint: str
    status: JobStatus
    execution_backend: ExecutionBackend
    config: dict[str, Any] = Field(default_factory=dict)
    output_checkpoint: str | None = None
    result_summary: IterationResultSummary | None = None
    artifacts: IterationArtifacts = Field(default_factory=IterationArtifacts)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ReusableAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    type: Literal["recipe", "template", "failure_pattern", "verification_setup"]
    name: str
    source_project: str
    reuse_count: int = Field(default=0, ge=0)
    linked_iteration: str
    description: str


class ImprovementReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration_id: str
    plan_id: str
    project_id: str
    success_before: float = Field(ge=0.0, le=1.0)
    success_after: float = Field(ge=0.0, le=1.0)
    uplift: float = Field(ge=0.0, le=1.0)
    summary: str
    assets_created: list[ReusableAsset] = Field(default_factory=list)


class PlanGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_run_id: str | None = None
    eval_run: EvalRun | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "PlanGenerationRequest":
        if not self.eval_run_id and self.eval_run is None:
            raise ValueError("either eval_run_id or eval_run must be provided")
        return self


class IterationStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    checkpoint: str
    execution_backend: ExecutionBackend | None = None
    config: dict[str, Any] = Field(default_factory=dict)