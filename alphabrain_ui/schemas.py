from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


def normalize_username(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 64 or not all(c.isalnum() or c in "-_." for c in value):
        raise ValueError("username must contain only letters, numbers, -, _, or .")
    return value


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


def _inline_secret_paths(node: Any, prefix: str = "") -> list[str]:
    """Find credential-shaped values before a generic spec reaches SQLite."""

    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            normalized = str(key).lower()
            is_secret = normalized in {
                "token",
                "api_key",
                "apikey",
                "secret",
                "password",
                "passwd",
            } or normalized.endswith(("_api_key", "_access_token", "_auth_token", "_secret", "_password"))
            if is_secret and value not in (None, ""):
                found.append(path)
            found.extend(_inline_secret_paths(value, path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_inline_secret_paths(value, f"{prefix}[{index}]"))
    return found


def reject_inline_secrets(value: dict[str, Any]) -> dict[str, Any]:
    paths = _inline_secret_paths(value)
    if paths:
        raise ValueError(f"credentials cannot be stored in experiment specifications: {', '.join(paths)}")
    return value


WandbUploadCategory = Literal["metrics", "config", "system", "checkpoints", "videos"]


class WandbRunConfig(APIModel):
    """Persistable per-experiment W&B preferences; never contains credentials."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    enabled: bool = False
    mode: Literal["online", "offline", "disabled"] = "online"
    project: str = Field(default="AlphaBrain", min_length=1, max_length=128)
    entity: str = Field(default="", max_length=128)
    run_name: str = Field(default="", max_length=256)
    group: str = Field(default="", max_length=256)
    job_type: str = Field(default="", max_length=128)
    tags: list[str] = Field(default_factory=list, max_length=64)
    notes: str = Field(default="", max_length=10_000)
    categories: list[WandbUploadCategory] = Field(
        default_factory=lambda: ["metrics", "config", "system"],
        max_length=5,
    )

    @field_validator("project", "entity", "run_name", "group", "job_type")
    @classmethod
    def trim_short_text(cls, value: str, info: ValidationInfo) -> str:
        value = value.strip()
        if info.field_name == "project" and not value:
            raise ValueError("W&B project cannot be blank")
        if any(ord(character) < 32 for character in value):
            raise ValueError("W&B fields cannot contain control characters")
        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for raw in values:
            value = raw.strip()
            if not value or len(value) > 128 or "," in value or any(ord(character) < 32 for character in value):
                raise ValueError("W&B tags must be 1-128 characters and cannot contain commas or controls")
            if value not in result:
                result.append(value)
        return result

    @field_validator("categories")
    @classmethod
    def unique_categories(cls, values: list[WandbUploadCategory]) -> list[WandbUploadCategory]:
        return list(dict.fromkeys(values))


class WandbAPIKeyUpdate(APIModel):
    api_key: str = Field(min_length=8, max_length=1024)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 8 or any(character.isspace() or ord(character) < 33 for character in value):
            raise ValueError("W&B API key must contain at least 8 non-whitespace characters")
        return value


class WandbSecretStatus(APIModel):
    configured: bool


class ReferenceResultsMatchRequest(APIModel):
    """A bounded, scalar-only signature used to select local reference data."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    benchmark_id: str = Field(
        min_length=2,
        max_length=160,
        pattern=r"^[a-z][a-z0-9_.-]{1,159}$",
    )
    signature: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("signature", mode="before")
    @classmethod
    def validate_signature(cls, value: Any) -> dict[str, str | int | float | bool]:
        if not isinstance(value, dict) or len(value) > 64:
            raise ValueError("reference signature must be an object with at most 64 fields")
        normalized: dict[str, str | int | float | bool] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if (
                len(key) < 2
                or len(key) > 160
                or not key[0].islower()
                or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in key)
            ):
                raise ValueError("reference signature contains an invalid key")
            if isinstance(item, str):
                if not item or len(item) > 256 or any(ord(character) < 32 for character in item):
                    raise ValueError("reference signature contains invalid text")
            elif isinstance(item, bool):
                pass
            elif isinstance(item, int):
                pass
            elif isinstance(item, float):
                if not isfinite(item):
                    raise ValueError("reference signature numbers must be finite")
            else:
                raise ValueError("reference signature values must be scalar")
            normalized[key] = item
        return normalized


class SetupRequest(APIModel):
    mode: Literal["personal", "lab"] = "personal"
    username: str = "admin"
    display_name: str = "Administrator"
    password: str = ""

    @field_validator("username")
    @classmethod
    def valid_username(cls, value: str) -> str:
        return normalize_username(value)


class LoginRequest(APIModel):
    username: str
    password: str


class UserCreate(APIModel):
    username: str
    display_name: str = ""
    password: str = Field(min_length=8)
    role: Literal["administrator", "researcher"] = "researcher"

    @field_validator("username")
    @classmethod
    def valid_username(cls, value: str) -> str:
        return normalize_username(value)


class UserUpdate(APIModel):
    display_name: str | None = None
    password: str | None = Field(default=None, min_length=8)
    role: Literal["administrator", "researcher"] | None = None
    is_active: bool | None = None


class PreferenceUpdate(APIModel):
    language: Literal["zh-CN", "en-US"] | None = None
    theme: Literal["light", "dark"] | None = None
    gpu_refresh_interval_seconds: int | None = Field(default=None, ge=0, le=3600)
    experimental_enabled: bool | None = None

    @field_validator("gpu_refresh_interval_seconds")
    @classmethod
    def valid_gpu_refresh_interval(cls, value: int | None) -> int | None:
        if value is not None and value != 0 and value < 2:
            raise ValueError("gpu_refresh_interval_must_be_zero_or_at_least_two_seconds")
        return value


class UserOut(APIModel):
    id: str
    username: str
    display_name: str
    role: str
    language: str
    theme: str
    gpu_refresh_interval_seconds: int
    experimental_enabled: bool
    is_active: bool
    created_at: datetime


class SettingsUpdate(APIModel):
    deployment_mode: Literal["personal", "lab"] | None = None
    experimental_globally_enabled: bool | None = None
    environment: dict[str, str] | None = None
    results_roots: list[str] | None = None
    storage_monitor_path: str | None = Field(default=None, max_length=4096)
    dataset_roots: list[str] | None = None
    managed_dataset_root: str | None = Field(default=None, max_length=4096)
    pretrained_root: str | None = Field(default=None, max_length=4096)
    cpu_utility_concurrency: int | None = Field(default=None, ge=1, le=16)
    disk_min_free_gib: float | None = Field(default=None, ge=0)
    disk_min_free_percent: float | None = Field(default=None, ge=0, le=100)
    secure_cookies: bool | None = None
    model_server_python: str | None = Field(default=None, max_length=4096)
    remote_training_enabled: bool | None = None
    remote_training_host: str | None = Field(default=None, max_length=255)
    remote_training_user: str | None = Field(default=None, max_length=255)
    remote_training_port: int | None = Field(default=None, ge=1, le=65535)
    remote_training_repo_root: str | None = Field(default=None, max_length=4096)
    remote_training_identity_file: str | None = Field(default=None, max_length=4096)
    remote_training_gpu_ids: list[int] | None = None
    remote_training_setup_command: str | None = Field(default=None, max_length=4096)
    admin_password: str | None = Field(default=None, min_length=8)


class TemplateCreate(APIModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = ""
    visibility: Literal["private", "shared"] = "private"
    spec: dict[str, Any]
    allow_duplicate: bool = False

    _reject_credentials = field_validator("spec")(reject_inline_secrets)


class TemplateUpdate(APIModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = None
    visibility: Literal["private", "shared"] | None = None
    spec: dict[str, Any] | None = None
    allow_duplicate: bool = False

    _reject_credentials = field_validator("spec")(lambda value: reject_inline_secrets(value) if value else value)


class TemplateOut(APIModel):
    id: str
    owner_id: str
    name: str
    description: str
    visibility: str
    spec: dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime


class ExperimentRequest(APIModel):
    name: str = Field(min_length=1, max_length=160)
    spec: dict[str, Any]
    acknowledge_experimental: bool = False

    _reject_credentials = field_validator("spec")(reject_inline_secrets)


class DatasetValidationRequest(APIModel):
    path: str = Field(min_length=1, max_length=4096)
    dataset_id: str = Field(min_length=1, max_length=128)
    dataset_mix: str | None = Field(default=None, max_length=128)


class DatasetInspectionRequest(APIModel):
    path: str = Field(min_length=1, max_length=4096)


class SecretTokenUpdate(APIModel):
    token: str = Field(min_length=8, max_length=4096)

    @field_validator("token")
    @classmethod
    def validate_token(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 8 or any(character.isspace() or ord(character) < 33 for character in value):
            raise ValueError("token must contain at least 8 non-whitespace characters")
        return value


class ResourceInstallRequest(APIModel):
    target_root: str | None = Field(default=None, max_length=4096)


class ResourceRegisterRequest(APIModel):
    path: str = Field(min_length=1, max_length=4096)


class ResourcePreprocessRequest(APIModel):
    kind: Literal["t5", "reason1", "umt5", "reason1_projection"]
    inputs: dict[str, str] = Field(default_factory=dict)
    gpu_count: int = Field(default=1, ge=1, le=64)
    gpu_ids: list[int] = Field(default_factory=list)


class DatasetRegistrationCreate(APIModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=10_000)
    path: str = Field(min_length=1, max_length=4096)
    storage_mode: Literal["reference", "managed_copy"] = "reference"
    visibility: Literal["private", "shared"] = "shared"
    dataset_id: str = Field(default="lerobot", min_length=1, max_length=128)
    dataset_mix: str = Field(default="", max_length=128)


class DatasetMixtureMember(APIModel):
    registration_id: str = Field(min_length=1, max_length=36)
    pattern: str = Field(default="", max_length=1024)
    weight: float = Field(default=1.0, gt=0)
    robot_type: str = Field(default="", max_length=128)
    trajectory_limit: int | None = Field(default=None, ge=1)


class DatasetMixtureCreate(APIModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=10_000)
    visibility: Literal["private", "shared"] = "shared"
    members: list[DatasetMixtureMember] = Field(min_length=1, max_length=128)
    options: dict[str, Any] = Field(default_factory=dict)


class DatasetMixtureUpdate(APIModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=10_000)
    visibility: Literal["private", "shared"] | None = None
    members: list[DatasetMixtureMember] | None = Field(default=None, min_length=1, max_length=128)
    options: dict[str, Any] | None = None


class DeploymentCheckpointSource(APIModel):
    kind: Literal["indexed", "local"]
    checkpoint_id: str | None = None
    path: str | None = Field(default=None, max_length=4096)


class DeploymentResources(APIModel):
    strategy: Literal["auto", "fixed"] = "auto"
    gpu_count: int = Field(default=1, ge=1, le=64)
    gpu_ids: list[int] = Field(default_factory=list)


class DeploymentEndpoint(APIModel):
    scope: Literal["local", "lan"] = "local"
    advertised_host: str | None = Field(default=None, max_length=255)
    port: int | None = Field(default=None, ge=1024, le=65535)
    idle_timeout_seconds: int = Field(default=1800, ge=-1, le=31_536_000)


class DeploymentRequest(APIModel):
    name: str = Field(min_length=1, max_length=160)
    checkpoint_source: DeploymentCheckpointSource
    combination_id: str | None = Field(default=None, max_length=128)
    resources: DeploymentResources = Field(default_factory=DeploymentResources)
    endpoint: DeploymentEndpoint = Field(default_factory=DeploymentEndpoint)
    parameters: dict[str, Any] = Field(default_factory=dict)
    acknowledge_experimental: bool = False

    @field_validator("name")
    @classmethod
    def deployment_name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("deployment name cannot be blank")
        return value


class ModelPublicationRequest(APIModel):
    checkpoint_id: str = Field(min_length=1, max_length=36)
    repo_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(default="main", min_length=1, max_length=128)
    private: bool = True

    @field_validator("repo_id")
    @classmethod
    def valid_huggingface_repo_id(cls, value: str) -> str:
        value = value.strip()
        parts = value.split("/")
        if (
            len(parts) != 2
            or any(not part or len(part) > 96 for part in parts)
            or any(not all(char.isalnum() or char in "._-" for char in part) for part in parts)
            or any(part.startswith((".", "-")) or part.endswith((".", "-")) for part in parts)
        ):
            raise ValueError("repo_id must use the namespace/model-name format")
        return value

    @field_validator("revision")
    @classmethod
    def valid_huggingface_revision(cls, value: str) -> str:
        value = value.strip()
        if not value or value.startswith("/") or value.endswith("/") or ".." in value:
            raise ValueError("invalid Hugging Face revision")
        if not all(char.isalnum() or char in "._/-" for char in value):
            raise ValueError("invalid Hugging Face revision")
        return value


class LoraMergeRequest(APIModel):
    model: Literal["qwengr00t", "neurovla", "llamaoft", "paligemma"]
    output_name: str | None = Field(default=None, max_length=200)
    resources: DeploymentResources = Field(default_factory=DeploymentResources)

    @field_validator("output_name")
    @classmethod
    def validate_output_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value.endswith(".pt") or not all(char.isalnum() or char in "._-" for char in value):
            raise ValueError("output_name must be a safe .pt filename")
        return value

EvaluationWandbCategory = Literal["summary", "tasks", "config", "videos"]
EvaluationKind = Literal[
    "standard",
    "batch",
    "cl_matrix",
    "rl_iterations",
    "online_stdp",
    "world_model_video",
]
EvaluationSourceKind = Literal["temporary_checkpoint", "managed_deployment"]


class EvaluationCheckpointSource(APIModel):
    kind: Literal["indexed", "local"]
    checkpoint_id: str | None = None
    path: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def source_matches_kind(self) -> EvaluationCheckpointSource:
        if self.kind == "indexed":
            if not (self.checkpoint_id or "").strip():
                raise ValueError("checkpoint_id is required for an indexed checkpoint")
            self.path = None
        else:
            if not (self.path or "").strip():
                raise ValueError("path is required for a local checkpoint")
            self.path = str(self.path).strip()
            self.checkpoint_id = None
        return self


class EvaluationCheckpointInspectionRequest(EvaluationCheckpointSource):
    """Checkpoint-only inspection with an optional ambiguity resolution."""

    combination_id: str | None = Field(default=None, max_length=128)

    @field_validator("combination_id")
    @classmethod
    def trim_inspection_combination(cls, value: str | None) -> str | None:
        value = value.strip() if value is not None else None
        return value or None


class EvaluationResources(APIModel):
    strategy: Literal["auto", "fixed"] = "auto"
    gpu_count: int = Field(default=1, ge=1, le=64)
    gpu_ids: list[int] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def fixed_gpu_count_matches_ids(self) -> EvaluationResources:
        if self.strategy == "fixed":
            if len(self.gpu_ids) != self.gpu_count or len(set(self.gpu_ids)) != len(self.gpu_ids):
                raise ValueError("fixed evaluation resources require one unique GPU ID per requested GPU")
        else:
            self.gpu_ids = []
        return self


class EvaluationWandbConfig(APIModel):
    """Per-evaluation upload preferences; credentials are never accepted here."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    enabled: bool = False
    mode: Literal["online", "offline", "disabled"] = "online"
    project: str = Field(default="AlphaBrain", min_length=1, max_length=128)
    entity: str = Field(default="", max_length=128)
    run_name: str = Field(default="", max_length=256)
    group: str = Field(default="", max_length=256)
    tags: list[str] = Field(default_factory=list, max_length=64)
    notes: str = Field(default="", max_length=10_000)
    categories: list[EvaluationWandbCategory] = Field(
        default_factory=lambda: ["summary", "tasks", "config"],
        max_length=4,
    )

    @field_validator("project", "entity", "run_name", "group")
    @classmethod
    def trim_evaluation_wandb_text(cls, value: str, info: ValidationInfo) -> str:
        value = value.strip()
        if info.field_name == "project" and not value:
            raise ValueError("W&B project cannot be blank")
        if any(ord(character) < 32 for character in value):
            raise ValueError("W&B fields cannot contain control characters")
        return value

    @field_validator("tags")
    @classmethod
    def normalize_evaluation_tags(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for raw in values:
            value = raw.strip()
            if not value or len(value) > 128 or "," in value or any(ord(character) < 32 for character in value):
                raise ValueError("W&B tags must be 1-128 characters and cannot contain commas or controls")
            if value not in result:
                result.append(value)
        return result

    @field_validator("categories")
    @classmethod
    def unique_evaluation_categories(
        cls, values: list[EvaluationWandbCategory]
    ) -> list[EvaluationWandbCategory]:
        return list(dict.fromkeys(values))


class EvaluationRequest(APIModel):
    name: str = Field(min_length=1, max_length=160)
    kind: EvaluationKind = "standard"
    source_kind: EvaluationSourceKind = "temporary_checkpoint"
    checkpoint_source: EvaluationCheckpointSource | None = None
    checkpoint_sources: list[EvaluationCheckpointSource] = Field(default_factory=list, max_length=64)
    suites: list[str] = Field(default_factory=list, max_length=32)
    deployment_id: str | None = Field(default=None, max_length=36)
    combination_id: str | None = Field(default=None, max_length=128)
    benchmark_id: Literal["libero", "robocasa365", "robocasa_tabletop", "libero_plus"]
    preset: Literal["quick", "standard", "full", "custom"] = "quick"
    suite: str | None = Field(default=None, max_length=128)
    task_set: str | None = Field(default=None, max_length=512)
    split: str | None = Field(default=None, max_length=128)
    parameters: dict[str, Any] = Field(default_factory=dict)
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    resources: EvaluationResources = Field(default_factory=EvaluationResources)
    wandb: EvaluationWandbConfig = Field(default_factory=EvaluationWandbConfig)
    acknowledge_experimental: bool = False

    _reject_evaluation_credentials = field_validator("parameters", "model_parameters")(reject_inline_secrets)

    @field_validator("name")
    @classmethod
    def evaluation_name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("evaluation name cannot be blank")
        return value

    @field_validator("suite", "task_set", "split")
    @classmethod
    def trim_optional_evaluation_text(cls, value: str | None) -> str | None:
        value = value.strip() if value is not None else None
        return value or None

    @field_validator("suites")
    @classmethod
    def normalize_evaluation_suites(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def evaluation_kind_contract(self) -> EvaluationRequest:
        if self.source_kind == "managed_deployment":
            if not self.deployment_id:
                raise ValueError("deployment_id is required for a managed deployment source")
            if self.kind not in {"standard", "batch", "world_model_video"}:
                raise ValueError(
                    "managed deployment sources support standard, batch, and world-model video evaluations"
                )
            if self.checkpoint_sources:
                raise ValueError("managed deployment batch evaluations cannot select checkpoint_sources")
            if self.kind == "batch" and not self.suites:
                raise ValueError("managed deployment batch evaluations require suites")
            # A managed deployment is the immutable source of its checkpoint
            # and model combination.  Discard a legacy client placeholder so it
            # cannot be mistaken for an independently resolved checkpoint.
            self.checkpoint_source = None
        else:
            self.deployment_id = None
            if self.checkpoint_source is None:
                raise ValueError("checkpoint_source is required for a temporary checkpoint source")
        if self.kind not in {"cl_matrix", "rl_iterations"} and self.resources.gpu_count != 1:
            raise ValueError("only CL matrix and RL iteration evaluations can atomically request multiple GPUs")
        if (
            self.kind == "batch"
            and self.source_kind == "temporary_checkpoint"
            and not (self.checkpoint_sources or self.suites)
        ):
            raise ValueError("batch evaluation requires checkpoint_sources or suites")
        return self


class Issue(APIModel):
    level: Literal["error", "warning", "info"]
    code: str
    message: str
    message_i18n: dict[str, str] = Field(default_factory=dict)
    field: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ResolveResponse(APIModel):
    resolved: dict[str, Any]
    compatibility: str
    issues: list[Issue] = Field(default_factory=list)
    command_preview: list[dict[str, Any]] = Field(default_factory=list)
    diff: dict[str, Any] = Field(default_factory=dict)


class PreflightResponse(ResolveResponse):
    can_submit: bool


class JobOut(APIModel):
    id: str
    experiment_id: str
    stage_id: str
    owner_id: str
    status: str
    requested_gpu_count: int
    requested_gpu_ids: list[int]
    assigned_gpu_ids: list[int]
    output_dir: str
    log_path: str
    metrics_path: str
    pid: int | None
    exit_code: int | None
    error: str
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class EvaluationRunOut(APIModel):
    id: str
    group_id: str | None
    position: int
    evaluation_kind: str
    source_kind: str
    deployment_id: str | None
    result_schema_version: str
    owner_id: str
    checkpoint_id: str | None
    name: str
    checkpoint_path: str
    combination_id: str
    adapter_id: str
    backbone_id: str
    action_head_id: str
    benchmark_id: str
    benchmark_status: str
    preset: str
    suite: str
    task_set: str
    split: str
    parameters: dict[str, Any]
    model_parameters: dict[str, Any]
    wandb: dict[str, Any]
    status: str
    requested_gpu_count: int
    requested_gpu_ids: list[int]
    assigned_gpu_ids: list[int]
    server_port: int | None
    output_dir: str
    config_path: str
    log_path: str
    progress_path: str
    result_path: str
    result_summary: dict[str, Any]
    wandb_status: str
    wandb_error: str
    wandb_run_url: str
    pid: int | None
    exit_code: int | None
    error: str
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    stop_requested_at: datetime | None
    created_at: datetime
    updated_at: datetime


class EvaluationGroupOut(APIModel):
    id: str
    owner_id: str
    name: str
    kind: str
    status: str
    spec: dict[str, Any]
    result_summary: dict[str, Any]
    result_path: str
    error: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    runs: list[EvaluationRunOut] = Field(default_factory=list)


class EvaluationSummary(APIModel):
    num_tasks: int = Field(default=0, ge=0)
    num_episodes: int = Field(default=0, ge=0)
    num_successes: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluationTaskMetric(APIModel):
    id: str
    name: str = ""
    num_episodes: int = Field(default=0, ge=0)
    num_successes: int = Field(default=0, ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationEpisodeResult(APIModel):
    task_id: str
    task_name: str = ""
    episode_index: int = Field(ge=0)
    success: bool
    steps: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    video: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationVideoArtifact(APIModel):
    path: str
    task_id: str = ""
    task_name: str = ""
    episode_index: int | None = Field(default=None, ge=0)
    success: bool | None = None


class EvaluationArtifactV2(APIModel):
    kind: str = "file"
    path: str
    name: str = ""
    media_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationMatrixV2(APIModel):
    id: str
    name: str = ""
    row_labels: list[str] = Field(default_factory=list)
    column_labels: list[str] = Field(default_factory=list)
    values: list[list[float | None]] = Field(default_factory=list)
    value_format: str = "rate"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def matrix_dimensions_match(self) -> EvaluationMatrixV2:
        if self.row_labels and len(self.values) != len(self.row_labels):
            raise ValueError("matrix row count must match row_labels")
        if self.column_labels and any(len(row) != len(self.column_labels) for row in self.values):
            raise ValueError("matrix column count must match column_labels")
        return self


class EvaluationSeriesPointV2(APIModel):
    x: float | int | str
    y: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationSeriesV2(APIModel):
    id: str
    name: str = ""
    x_label: str = ""
    y_label: str = ""
    points: list[EvaluationSeriesPointV2] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResultV1(APIModel):
    schema_version: Literal["evaluation-result-v1"] = "evaluation-result-v1"
    status: Literal["running", "completed", "failed", "partial"]
    benchmark: str
    checkpoint: str = ""
    suite: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    summary: EvaluationSummary
    tasks: list[EvaluationTaskMetric] = Field(default_factory=list)
    episodes: list[EvaluationEpisodeResult] = Field(default_factory=list)
    videos: list[EvaluationVideoArtifact] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvaluationResultV2(APIModel):
    """Extensible result envelope used by grouped and specialized evaluations.

    The v1 task/episode/video fields remain present so existing clients can
    render v2 results while progressively adopting matrices, series, and
    generic artifacts.
    """

    schema_version: Literal["evaluation-result-v2"] = "evaluation-result-v2"
    kind: EvaluationKind
    status: Literal["running", "completed", "failed", "partial"]
    benchmark: str = ""
    checkpoint: str = ""
    suite: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    summary: dict[str, Any] = Field(default_factory=dict)
    tasks: list[EvaluationTaskMetric] = Field(default_factory=list)
    episodes: list[EvaluationEpisodeResult] = Field(default_factory=list)
    videos: list[EvaluationVideoArtifact] = Field(default_factory=list)
    artifacts: list[EvaluationArtifactV2] = Field(default_factory=list)
    matrices: list[EvaluationMatrixV2] = Field(default_factory=list)
    series: list[EvaluationSeriesV2] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    children: list[dict[str, Any]] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvaluationCompareRequest(APIModel):
    evaluation_ids: list[str] = Field(min_length=2, max_length=8)

    @field_validator("evaluation_ids")
    @classmethod
    def unique_evaluation_ids(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("evaluation IDs must be unique")
        return values


class StageOut(APIModel):
    id: str
    name: str
    phase: str
    position: int
    dependency_stage_id: str | None
    resolved: dict[str, Any]
    jobs: list[JobOut] = Field(default_factory=list)


class ExperimentOut(APIModel):
    id: str
    owner_id: str
    name: str
    family: str
    compatibility: str
    status: str
    spec: dict[str, Any]
    resolved: dict[str, Any]
    config_snapshot_path: str
    created_at: datetime
    updated_at: datetime
    stages: list[StageOut] = Field(default_factory=list)


class StopRequest(APIModel):
    force: bool = False


class DeleteRequest(APIModel):
    confirmation: str
