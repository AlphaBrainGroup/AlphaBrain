from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import secrets
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload
from starlette.exceptions import HTTPException as StarletteHTTPException

from .checkpoint_tools import build_lora_merge_command, discover_lora_bundle, safe_output_name
from .builtin_presets import (
    builtin_checkpoint_candidates,
    builtin_templates,
    is_builtin_checkpoint_id,
    is_builtin_template_id,
)
from .database import (
    AuditEvent,
    Checkpoint,
    Database,
    DatasetMixture,
    DatasetRegistration,
    DeploymentGPUReservation,
    EvaluationGroup,
    EvaluationGPUReservation,
    EvaluationRun,
    Experiment,
    ExperimentStage,
    ExperimentTemplate,
    GPUReservation,
    InferenceRun,
    Job,
    ModelPublication,
    ModelDeployment,
    UtilityRun,
    UtilityGPUReservation,
    User,
    new_id,
    utcnow,
)
from .datasets import list_directories, validate_dataset_directory
from .dataset_registry import inspect_dataset, is_within_roots, preview_dataset, resolve_local_directory
from .dependencies import admin_user, assert_owner_or_admin, current_user, get_db
from .deployment_preflight import validate_deployment_request
from .deployment_registry import get_deployment_catalog, inspect_checkpoint as inspect_deployment_checkpoint
from .deployments import (
    DEPLOYMENT_ACTIVE_STATUSES,
    DEPLOYMENT_TERMINAL_STATUSES,
    DeploymentManager,
    generate_api_key,
)
from .gpu import DemoGPUMonitor, GPUMonitor, storage_snapshot
from .inference import (
    InferenceCoordinator,
    InferenceInputError,
    InferenceTransportError,
    decode_inference_request,
    infer_via_websocket,
    save_inference_inputs,
)
from .evaluation_preflight import validate_evaluation_request
from .evaluation_groups import EvaluationChildSpec, expand_evaluation_request
from .evaluation_registry import (
    evaluate_catalog_readiness,
    get_evaluation_catalog,
    inspect_evaluation_checkpoint,
)
from .evaluations import (
    EVALUATION_ACTIVE_STATUSES,
    EVALUATION_TERMINAL_STATUSES,
    EvaluationManager,
    refresh_evaluation_group,
)
from .jobs import ACTIVE_STATUSES, TERMINAL_STATUSES, JobManager
from .launchers import build_launch_plan, normalize_family
from .preflight import effective_environment, run_preflight
from .remote_metrics import RemoteMetricsCollector
from .remote_training import RemoteTrainingConfig, validate_remote_training_config
from .runtime import RuntimeConfig
from .schemas import (
    DatasetInspectionRequest,
    DatasetValidationRequest,
    DatasetMixtureCreate,
    DatasetMixtureUpdate,
    DatasetRegistrationCreate,
    DeleteRequest,
    DeploymentRequest,
    EvaluationCompareRequest,
    EvaluationCheckpointInspectionRequest,
    EvaluationGroupOut,
    EvaluationRequest,
    EvaluationRunOut,
    ExperimentOut,
    ExperimentRequest,
    JobOut,
    LoginRequest,
    LoraMergeRequest,
    ModelPublicationRequest,
    PreferenceUpdate,
    PreflightResponse,
    ReferenceResultsMatchRequest,
    ResolveResponse,
    SettingsUpdate,
    SecretTokenUpdate,
    ResourceInstallRequest,
    ResourcePreprocessRequest,
    ResourceRegisterRequest,
    SetupRequest,
    TemplateCreate,
    TemplateOut,
    TemplateUpdate,
    UserCreate,
    UserOut,
    UserUpdate,
    WandbAPIKeyUpdate,
    WandbSecretStatus,
)
from .services import AuthService, SettingsService, add_audit
from .template_fingerprints import (
    TEMPLATE_FINGERPRINT_VERSION,
    canonical_template_json,
    normalize_template_name,
    template_spec_fingerprint,
)
from .resource_catalog import WORLD_MODEL_RESOURCES, install_command, preprocess_command, resource_catalog
from .reference_results import (
    ReferenceResultsError,
    load_reference_snapshot,
    match_reference_results,
)
from .registry import catalog_for_runtime
from .registry_overlay import RegistryOverlayError, RegistryOverlayStore, build_registry_view
from .secrets import HuggingFaceSecretStore
from .system_metrics import collect_system_metrics
from .version import __version__
from .wandb import WandbSecretStore, wandb_category_catalog, wandb_requires_api_key
from .utilities import UTILITY_ACTIVE_STATUSES, UtilityManager, serialize_utility

SAFE_ENV_KEYS = {
    "PRETRAINED_MODELS_DIR",
    "LIBERO_DATA_ROOT",
    "LEROBOT_LIBERO_DATA_DIR",
    "LIBERO_HOME",
    "LIBERO_PLUS_HOME",
    "LIBERO_PYTHON",
    "LIBERO_PLUS_PYTHON",
    "ROBOCASA365_PYTHON",
    "ROBOCASA_TABLETOP_PYTHON",
    "ROBOCASA_TABLETOP_DATA_ROOT",
    "ROBOCASA365_DATA_ROOT",
    "WANDB_BASE_URL",
    "WANDB_MODE",
    "HF_HOME",
}


REGISTRY_OVERLAY_ERROR_MESSAGES: dict[str, dict[str, str]] = {
    "overlay_mapping_required": {
        "zh-CN": "注册表叠加中的该字段必须是对象。",
        "en-US": "This registry overlay field must be an object.",
    },
    "overlay_field_forbidden": {
        "zh-CN": "注册表叠加包含不允许修改的执行字段。",
        "en-US": "The registry overlay contains an operational field that cannot be changed.",
    },
    "overlay_credentials_forbidden": {
        "zh-CN": "注册表叠加中禁止保存凭据或密钥。",
        "en-US": "Credentials and secrets cannot be stored in a registry overlay.",
    },
    "overlay_schema_mismatch": {
        "zh-CN": "注册表叠加的 schema 不正确。",
        "en-US": "The registry overlay schema is invalid.",
    },
    "overlay_schema_version_unsupported": {
        "zh-CN": "不支持该注册表叠加版本。",
        "en-US": "This registry overlay schema version is not supported.",
    },
    "overlay_invalid_version": {
        "zh-CN": "注册表叠加版本号格式不正确。",
        "en-US": "The registry overlay version is invalid.",
    },
    "overlay_invalid_id": {
        "zh-CN": "注册表叠加包含格式不正确的 ID。",
        "en-US": "The registry overlay contains an invalid ID.",
    },
    "overlay_id_conflict": {
        "zh-CN": "注册表叠加 ID 与现有或新增条目冲突。",
        "en-US": "A registry overlay ID conflicts with an existing or added entry.",
    },
    "overlay_untrusted_base": {
        "zh-CN": "注册表叠加只能继承内置可信条目。",
        "en-US": "Registry overlay entries may only derive from trusted built-in entries.",
    },
    "overlay_unknown_component_reference": {
        "zh-CN": "注册表叠加引用了未知组件。",
        "en-US": "The registry overlay references an unknown component.",
    },
    "overlay_unknown_operational_reference": {
        "zh-CN": "注册表叠加引用了未接通的执行适配器。",
        "en-US": "The registry overlay references an unwired operational adapter.",
    },
    "overlay_unknown_disable_target": {
        "zh-CN": "注册表叠加尝试禁用不存在的条目。",
        "en-US": "The registry overlay tries to disable an entry that does not exist.",
    },
    "overlay_invalid_yaml": {
        "zh-CN": "已保存的注册表叠加不是有效的 YAML。",
        "en-US": "The saved registry overlay is not valid YAML.",
    },
    "overlay_too_large": {
        "zh-CN": "注册表叠加超过大小限制。",
        "en-US": "The registry overlay exceeds the size limit.",
    },
}

REFERENCE_RESULTS_ERROR_MESSAGES: dict[str, dict[str, str]] = {
    "reference_snapshot_unavailable": {
        "zh-CN": "本地参考结果快照不可用。",
        "en-US": "The local reference-results snapshot is unavailable.",
    },
    "reference_invalid_id": {
        "zh-CN": "Benchmark ID 或参考结果 ID 格式不正确。",
        "en-US": "The benchmark or reference-result ID is invalid.",
    },
    "reference_invalid_signature": {
        "zh-CN": "评测签名格式不正确，只允许标量字段。",
        "en-US": "The evaluation signature is invalid; only scalar fields are allowed.",
    },
}


def _localized_error_detail(
    code: str,
    *,
    messages: dict[str, dict[str, str]],
    path: str = "",
    value: Any = None,
) -> dict[str, Any]:
    fallback = {
        "zh-CN": f"请求校验失败（{code}）。",
        "en-US": f"Request validation failed ({code}).",
    }
    localized = messages.get(code, fallback)
    result: dict[str, Any] = {
        "code": code,
        "message": localized["en-US"],
        "message_i18n": localized,
    }
    if path:
        result["path"] = path
    if value is not None:
        result["context"] = value
    return result


def _overlay_audit_detail(status: dict[str, Any]) -> dict[str, Any]:
    """Return only non-sensitive overlay metadata for the audit trail."""

    keys = (
        "configured",
        "sha256",
        "overlay_version",
        "component_additions",
        "combination_additions",
        "deployment_combination_additions",
        "disabled_entries",
    )
    return {key: status.get(key) for key in keys if key in status}


def _public_overlay_status(status: dict[str, Any]) -> dict[str, Any]:
    """Hide the host path and keep the API shape stable when no overlay exists."""

    result = {
        "configured": bool(status.get("configured", False)),
        "sha256": status.get("sha256"),
        "overlay_version": status.get("overlay_version"),
        "component_additions": int(status.get("component_additions", 0)),
        "combination_additions": int(status.get("combination_additions", 0)),
        "deployment_combination_additions": int(
            status.get("deployment_combination_additions", 0)
        ),
        "disabled_entries": int(status.get("disabled_entries", 0)),
    }
    return result


def _set_session_cookies(response: Response, raw: str, csrf: str, secure: bool, max_age: int) -> None:
    response.set_cookie(
        AuthService.cookie_name,
        raw,
        max_age=max_age,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        AuthService.csrf_cookie_name,
        csrf,
        max_age=max_age,
        httponly=False,
        secure=secure,
        samesite="lax",
        path="/",
    )


def _clear_session_cookies(response: Response) -> None:
    response.delete_cookie(AuthService.cookie_name, path="/")
    response.delete_cookie(AuthService.csrf_cookie_name, path="/")


def _serialize_user(user: User) -> dict[str, Any]:
    return UserOut.model_validate(user).model_dump(mode="json")


def _serialize_template(template: ExperimentTemplate) -> dict[str, Any]:
    result = TemplateOut.model_validate(template).model_dump(mode="json")
    result["owner_name"] = template.owner.display_name or template.owner.username
    return result


def _serialize_dataset(registration: DatasetRegistration) -> dict[str, Any]:
    return {
        "id": registration.id,
        "owner_id": registration.owner_id,
        "owner_name": (
            registration.owner.display_name or registration.owner.username
            if getattr(registration, "owner", None)
            else ""
        ),
        "name": registration.name,
        "description": registration.description,
        "path": registration.path,
        "source_path": registration.source_path,
        "storage_mode": registration.storage_mode,
        "visibility": registration.visibility,
        "format": registration.format,
        "status": registration.status,
        "dataset_id": registration.dataset_id,
        "dataset_mix": registration.dataset_mix,
        "fingerprint": registration.fingerprint,
        "size_bytes": registration.size_bytes,
        "episode_count": registration.episode_count,
        "step_count": registration.step_count,
        "validation": registration.validation,
        "metadata": registration.metadata_json,
        "copy_run_id": registration.copy_run_id,
        "stats_run_id": registration.stats_run_id,
        "stats_status": registration.stats_status,
        "created_at": registration.created_at,
        "updated_at": registration.updated_at,
    }


def _serialize_mixture(mixture: DatasetMixture, registrations: dict[str, DatasetRegistration]) -> dict[str, Any]:
    resolved_members = []
    for member in mixture.members or []:
        registration = registrations.get(str(member.get("registration_id", "")))
        resolved_members.append(
            {
                **member,
                "dataset_path": registration.path if registration else None,
                "dataset_name": registration.name if registration else None,
                "dataset_status": registration.status if registration else "missing",
                "dataset_format": registration.format if registration else "unknown",
                "format_family": (
                    str((registration.validation or {}).get("format_family", "unknown"))
                    if registration else "unknown"
                ),
                "builder_support": (
                    str((registration.validation or {}).get("builder_support", "unsupported"))
                    if registration else "unsupported"
                ),
            }
        )
    return {
        "id": mixture.id,
        "owner_id": mixture.owner_id,
        "owner_name": (
            mixture.owner.display_name or mixture.owner.username
            if getattr(mixture, "owner", None)
            else ""
        ),
        "name": mixture.name,
        "description": mixture.description,
        "visibility": mixture.visibility,
        "members": mixture.members,
        "resolved_members": resolved_members,
        "mixture_spec": [
            {
                "path": item.get("dataset_path"),
                "pattern": item.get("pattern", ""),
                "weight": item.get("weight", 1.0),
                "robot_type": item.get("robot_type", ""),
                "trajectory_limit": item.get("trajectory_limit"),
            }
            for item in resolved_members
            if item.get("dataset_path")
        ],
        "options": mixture.options,
        "version": mixture.version,
        "created_at": mixture.created_at,
        "updated_at": mixture.updated_at,
    }


def _latest_metric(path_value: str) -> dict[str, Any] | None:
    path = Path(path_value)
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 128 * 1024))
            lines = handle.read().splitlines()
    except OSError:
        return None
    for raw in reversed(lines):
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _queue_positions(db: Session) -> dict[str, int]:
    job_rows = db.execute(select(Job.id, Job.queued_at, Job.created_at).where(Job.status == "queued")).all()
    deployment_rows = db.execute(
        select(ModelDeployment.id, ModelDeployment.queued_at, ModelDeployment.created_at).where(
            ModelDeployment.status == "queued"
        )
    ).all()
    evaluation_rows = db.execute(
        select(EvaluationRun.id, EvaluationRun.queued_at, EvaluationRun.created_at).where(
            EvaluationRun.status == "queued"
        )
    ).all()
    utility_rows = db.execute(
        select(UtilityRun.id, UtilityRun.queued_at, UtilityRun.created_at).where(
            UtilityRun.status == "queued",
            UtilityRun.queue_class == "gpu",
        )
    ).all()
    workloads = [
        (queued_at, created_at, "job", row_id) for row_id, queued_at, created_at in job_rows
    ] + [
        (queued_at, created_at, "deployment", row_id)
        for row_id, queued_at, created_at in deployment_rows
    ] + [
        (queued_at, created_at, "evaluation", row_id)
        for row_id, queued_at, created_at in evaluation_rows
    ] + [
        (queued_at, created_at, "utility", row_id)
        for row_id, queued_at, created_at in utility_rows
    ]
    workloads.sort(key=lambda item: (item[0], item[1]))
    return {
        f"{kind}:{row_id}": position
        for position, (_queued_at, _created_at, kind, row_id) in enumerate(workloads, start=1)
    }


def _job_queue_positions(db: Session) -> dict[str, int]:
    positions = _queue_positions(db)
    return {key.removeprefix("job:"): value for key, value in positions.items() if key.startswith("job:")}


def _deployment_queue_positions(db: Session) -> dict[str, int]:
    positions = _queue_positions(db)
    return {
        key.removeprefix("deployment:"): value
        for key, value in positions.items()
        if key.startswith("deployment:")
    }


def _evaluation_queue_positions(db: Session) -> dict[str, int]:
    positions = _queue_positions(db)
    return {
        key.removeprefix("evaluation:"): value
        for key, value in positions.items()
        if key.startswith("evaluation:")
    }


def _gpu_utility_queue_positions(db: Session) -> dict[str, int]:
    positions = _queue_positions(db)
    return {
        key.removeprefix("utility:"): value
        for key, value in positions.items()
        if key.startswith("utility:")
    }


def _serialize_job(job: Job, queue_position: int | None = None) -> dict[str, Any]:
    result = JobOut.model_validate(job).model_dump(mode="json")
    latest = _latest_metric(job.metrics_path)
    latest_metrics = latest.get("metrics") if latest and isinstance(latest.get("metrics"), dict) else None
    progress = None
    if latest:
        current = latest.get("step", latest.get("iteration"))
        parameters = job.experiment.spec.get("parameters", {}) if isinstance(job.experiment.spec, dict) else {}
        maximum = (
            parameters.get("max_train_steps", parameters.get("max_steps")) if isinstance(parameters, dict) else None
        )
        if isinstance(current, (int, float)) and isinstance(maximum, (int, float)) and maximum > 0:
            progress = min(1.0, max(0.0, float(current) / float(maximum)))
    result.update(
        {
            "name": job.experiment.name if getattr(job, "experiment", None) else job.stage.name,
            "owner_name": job.owner.display_name or job.owner.username if getattr(job, "owner", None) else "",
            "stage_name": job.stage.name if getattr(job, "stage", None) else "",
            "gpu_ids": result["assigned_gpu_ids"] or result["requested_gpu_ids"],
            "command_preview": " ".join(str(part) for part in job.command),
            "error_summary": job.error,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "queue_position": queue_position,
            "latest_metrics": latest_metrics,
            "progress": progress,
        }
    )
    return result


def _serialize_deployment(
    deployment: ModelDeployment,
    queue_position: int | None = None,
    controller_available: bool | None = None,
) -> dict[str, Any]:
    endpoint_url = None
    if deployment.port:
        endpoint_url = f"ws://{deployment.advertised_host}:{deployment.port}"
    public_parameters = {
        str(key): value
        for key, value in (deployment.parameters or {}).items()
        if not str(key).startswith("_")
    }
    result = {
        "id": deployment.id,
        "name": deployment.name,
        "owner_id": deployment.owner_id,
        "owner_name": (
            deployment.owner.display_name or deployment.owner.username
            if getattr(deployment, "owner", None)
            else ""
        ),
        "checkpoint_id": deployment.checkpoint_id,
        "checkpoint_path": deployment.checkpoint_path,
        "combination_id": deployment.combination_id,
        "adapter_id": deployment.adapter_id,
        "backbone_id": deployment.backbone_id,
        "action_head_id": deployment.action_head_id,
        "status": deployment.status,
        "queue_position": queue_position,
        "requested_gpu_count": deployment.requested_gpu_count,
        "requested_gpu_ids": deployment.requested_gpu_ids,
        "assigned_gpu_ids": deployment.assigned_gpu_ids,
        "gpu_ids": deployment.assigned_gpu_ids or deployment.requested_gpu_ids,
        "parameters": public_parameters,
        "endpoint_scope": deployment.endpoint_scope,
        "bind_host": deployment.bind_host,
        "advertised_host": deployment.advertised_host,
        "port": deployment.port,
        "endpoint": {
            "scope": deployment.endpoint_scope,
            "bind_host": deployment.bind_host,
            "advertised_host": deployment.advertised_host,
            "port": deployment.port,
            "url": endpoint_url,
        },
        "idle_timeout_seconds": deployment.idle_timeout_seconds,
        "api_key_prefix": deployment.api_key_prefix,
        "command_preview": " ".join(str(value) for value in deployment.command),
        "log_path": deployment.log_path,
        "pid": deployment.pid,
        "exit_code": deployment.exit_code,
        "error": deployment.error,
        "error_summary": deployment.error,
        "queued_at": deployment.queued_at,
        "started_at": deployment.started_at,
        "ready_at": deployment.ready_at,
        "finished_at": deployment.finished_at,
        "created_at": deployment.created_at,
        "updated_at": deployment.updated_at,
    }
    if controller_available is not None:
        result["managed_inference_available"] = controller_available
    return result


def _serialize_inference_run(run: InferenceRun, *, include_output: bool = True) -> dict[str, Any]:
    return {
        "id": run.id,
        "request_id": run.request_id,
        "deployment_id": run.deployment_id,
        "owner_id": run.owner_id,
        "status": run.status,
        "batch_size": run.batch_size,
        "image_count": run.image_count,
        "save_inputs": run.save_inputs,
        "inputs_saved": bool(run.input_paths),
        "request_summary": run.request_summary,
        "deployment_metadata": run.deployment_metadata,
        "output": run.output_json if include_output else None,
        "latency_ms": run.latency_ms,
        "error": run.error,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
    }


def _serialize_model_publication(publication: ModelPublication) -> dict[str, Any]:
    return {
        "id": publication.id,
        "owner_id": publication.owner_id,
        "checkpoint_id": publication.checkpoint_id,
        "utility_run_id": publication.utility_run_id,
        "provider": publication.provider,
        "repo_id": publication.repo_id,
        "revision": publication.revision,
        "private": publication.private,
        "status": publication.status,
        "source_path": publication.source_path,
        "result_url": publication.result_url,
        "metadata": publication.metadata_json,
        "error": publication.error,
        "created_at": publication.created_at,
        "started_at": publication.started_at,
        "finished_at": publication.finished_at,
        "updated_at": publication.updated_at,
    }


def _load_evaluation_result(evaluation: EvaluationRun) -> dict[str, Any] | None:
    """Load only the versioned result owned by this evaluation directory."""

    result_path = Path(evaluation.result_path)
    output_root = Path(evaluation.output_dir)
    try:
        resolved_root = output_root.resolve(strict=True)
        resolved_result = result_path.resolve(strict=True)
        if not resolved_result.is_file() or not resolved_result.is_relative_to(resolved_root):
            return None
        with resolved_result.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") not in {
        "evaluation-result-v1",
        "evaluation-result-v2",
    }:
        return None
    return value


def _artifact_token(relative_path: str) -> str:
    return base64.urlsafe_b64encode(relative_path.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_artifact_token(token: str) -> str:
    if not token or len(token) > 8192:
        raise ValueError("invalid_artifact_id")
    try:
        padding = "=" * (-len(token) % 4)
        value = base64.urlsafe_b64decode((token + padding).encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid_artifact_id") from exc
    path = Path(value)
    if path.is_absolute() or not value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("invalid_artifact_id")
    return path.as_posix()


def _evaluation_artifact_path(evaluation: EvaluationRun, artifact_id: str) -> Path:
    relative = _decode_artifact_token(artifact_id)
    try:
        root = Path(evaluation.output_dir).resolve(strict=True)
        raw_candidate = root / relative
        if raw_candidate.is_symlink():
            raise ValueError("evaluation_artifact_not_found")
        candidate = raw_candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("evaluation_artifact_not_found") from exc
    if not candidate.is_file() or candidate.is_symlink() or not candidate.is_relative_to(root):
        raise ValueError("evaluation_artifact_not_found")
    return candidate


def _evaluation_artifacts(evaluation: EvaluationRun) -> list[dict[str, Any]]:
    """Build a bounded artifact list without following links outside a run."""

    root = Path(evaluation.output_dir)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError:
        return []
    result = _load_evaluation_result(evaluation) or {}
    declared_videos = {
        str(item.get("path")): item
        for item in result.get("videos", [])
        if isinstance(item, dict) and item.get("path")
    }
    declared_artifacts = {
        str(item.get("path")): item
        for item in result.get("artifacts", [])
        if isinstance(item, dict) and item.get("path")
    }
    candidates: set[Path] = set()
    for relative in declared_videos:
        path = Path(relative)
        if not path.is_absolute():
            candidates.add(resolved_root / path)
    for relative in declared_artifacts:
        path = Path(relative)
        if not path.is_absolute():
            candidates.add(resolved_root / path)
    for fixed in (
        "evaluation-result-v1.json",
        "evaluation-result-v2.json",
        "evaluation-config.yaml",
        "runner.log",
        "eval.log",
        "server.log",
        "progress.jsonl",
    ):
        candidates.add(resolved_root / fixed)
    video_root = resolved_root / "videos"
    if video_root.is_dir():
        for index, path in enumerate(video_root.rglob("*")):
            if index >= 5000:
                break
            if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"}:
                candidates.add(path)
    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates):
        try:
            if candidate.is_symlink():
                continue
            resolved = candidate.resolve(strict=True)
            if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
                continue
            relative = resolved.relative_to(resolved_root).as_posix()
            metadata = declared_videos.get(relative, declared_artifacts.get(relative, {}))
            stat = resolved.stat()
        except OSError:
            continue
        artifact_id = _artifact_token(relative)
        kind = str(metadata.get("kind") or (
            "video" if resolved.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"} else "file"
        ))
        rows.append(
            {
                "id": artifact_id,
                "name": resolved.name,
                "path": relative,
                "kind": kind,
                "size_bytes": stat.st_size,
                "task_id": metadata.get("task_id"),
                "task_name": metadata.get("task_name"),
                "episode_index": metadata.get("episode_index"),
                "success": metadata.get("success"),
                "url": f"/api/v1/evaluations/{evaluation.id}/artifacts/{artifact_id}",
            }
        )
    return rows


def _latest_evaluation_progress(path_value: str) -> dict[str, Any] | None:
    latest = _latest_metric(path_value)
    return latest if isinstance(latest, dict) else None


def _evaluation_progress_value(evaluation: EvaluationRun, event: dict[str, Any] | None) -> float:
    if evaluation.status == "queued":
        return 0.0
    if evaluation.status in EVALUATION_TERMINAL_STATUSES:
        return 1.0 if evaluation.status == "completed" else 0.0
    stage = str((event or {}).get("stage", ""))
    return {
        "server_start": 0.05,
        "server_ready": 0.15,
        "simulation": 0.2,
        "aggregation": 0.95,
        "evaluation": 1.0,
    }.get(stage, 0.05 if evaluation.status == "starting" else 0.2)


def _serialize_evaluation(
    evaluation: EvaluationRun,
    queue_position: int | None = None,
    *,
    include_result: bool = False,
    include_artifacts: bool = False,
) -> dict[str, Any]:
    result = EvaluationRunOut.model_validate(evaluation).model_dump(mode="json")
    progress_event = _latest_evaluation_progress(evaluation.progress_path)
    result.update(
        {
            "owner_name": (
                evaluation.owner.display_name or evaluation.owner.username
                if getattr(evaluation, "owner", None)
                else ""
            ),
            "queue_position": queue_position,
            "gpu_ids": evaluation.assigned_gpu_ids or evaluation.requested_gpu_ids,
            "output_path": evaluation.output_dir,
            "error_summary": evaluation.error,
            "command_preview": " ".join(str(value) for value in evaluation.command),
            "phase": str((progress_event or {}).get("stage", "")),
            "progress": _evaluation_progress_value(evaluation, progress_event),
            "latest_progress": progress_event,
        }
    )
    if include_result:
        result["result"] = _load_evaluation_result(evaluation)
    if include_artifacts:
        artifacts = _evaluation_artifacts(evaluation)
        result["artifacts"] = artifacts
        result["videos"] = [item for item in artifacts if item.get("kind") == "video"]
    return result


def _load_evaluation_group_result(group: EvaluationGroup) -> dict[str, Any] | None:
    path = Path(group.result_path)
    try:
        with path.resolve(strict=True).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != "evaluation-result-v2":
        return None
    return value


def _serialize_evaluation_group(
    db: Session,
    group: EvaluationGroup,
    *,
    include_result: bool = False,
) -> dict[str, Any]:
    refresh_evaluation_group(db, group.id)
    db.flush()
    runs = list(
        db.execute(
            select(EvaluationRun)
            .where(EvaluationRun.group_id == group.id)
            .options(selectinload(EvaluationRun.owner))
            .order_by(EvaluationRun.position)
        ).scalars()
    )
    value = EvaluationGroupOut.model_validate(group).model_dump(mode="json", exclude={"runs"})
    value["owner_name"] = (
        group.owner.display_name or group.owner.username
        if getattr(group, "owner", None)
        else ""
    )
    value["runs"] = [_serialize_evaluation(run) for run in runs]
    if include_result:
        value["result"] = _load_evaluation_group_result(group)
    return value


def _serialize_experiment(experiment: Experiment) -> dict[str, Any]:
    result = ExperimentOut.model_validate(experiment).model_dump(mode="json")
    architecture = experiment.spec.get("architecture", {}) if isinstance(experiment.spec, dict) else {}
    training = experiment.spec.get("training", {}) if isinstance(experiment.spec, dict) else {}
    dataset = experiment.spec.get("dataset", {}) if isinstance(experiment.spec, dict) else {}
    result.update(
        {
            "owner_name": experiment.owner.display_name or experiment.owner.username,
            "architecture": (
                architecture.get("framework") or architecture.get("backbone") or experiment.spec.get("backbone_id")
            ),
            "method": training.get("method") or training.get("family") or experiment.spec.get("method_id"),
            "dataset": dataset.get("id") or dataset.get("mix") or experiment.spec.get("dataset_id"),
            "config": experiment.resolved,
        }
    )
    return result


def _can_experimental(db: Session, user: User) -> bool:
    return bool(SettingsService(db).get("experimental_globally_enabled", False) and user.experimental_enabled)


def _get_experiment(db: Session, experiment_id: str) -> Experiment:
    experiment = db.execute(
        select(Experiment)
        .where(Experiment.id == experiment_id)
        .options(selectinload(Experiment.stages).selectinload(ExperimentStage.jobs))
    ).scalar_one_or_none()
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment_not_found")
    return experiment


def _get_job(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    return job


def _get_deployment(db: Session, deployment_id: str) -> ModelDeployment:
    deployment = db.execute(
        select(ModelDeployment)
        .where(ModelDeployment.id == deployment_id)
        .options(selectinload(ModelDeployment.owner))
    ).scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="deployment_not_found")
    return deployment


def _get_evaluation(db: Session, evaluation_id: str) -> EvaluationRun:
    evaluation = db.execute(
        select(EvaluationRun)
        .where(EvaluationRun.id == evaluation_id)
        .options(selectinload(EvaluationRun.owner))
    ).scalar_one_or_none()
    if evaluation is None:
        raise HTTPException(status_code=404, detail="evaluation_not_found")
    return evaluation


def _get_evaluation_group(db: Session, group_id: str) -> EvaluationGroup:
    group = db.execute(
        select(EvaluationGroup)
        .where(EvaluationGroup.id == group_id)
        .options(selectinload(EvaluationGroup.owner))
    ).scalar_one_or_none()
    if group is None:
        raise HTTPException(status_code=404, detail="evaluation_group_not_found")
    return group


def _gpu_reservations(db: Session) -> dict[int, str]:
    reservations = {
        int(row.gpu_index): str(row.job_id)
        for row in db.execute(select(GPUReservation)).scalars().all()
    }
    reservations.update(
        {
            int(row.gpu_index): f"deployment:{row.deployment_id}"
            for row in db.execute(select(DeploymentGPUReservation)).scalars().all()
        }
    )
    reservations.update(
        {
            int(row.gpu_index): f"evaluation:{row.evaluation_id}"
            for row in db.execute(select(EvaluationGPUReservation)).scalars().all()
        }
    )
    reservations.update(
        {
            int(row.gpu_index): f"utility:{row.utility_run_id}"
            for row in db.execute(select(UtilityGPUReservation)).scalars().all()
        }
    )
    return reservations


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):  # type: ignore[no-untyped-def]
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and "." not in Path(path).name:
                return await super().get_response("index.html", scope)
            raise
        if response.status_code == 404 and "." not in Path(path).name:
            return await super().get_response("index.html", scope)
        return response


def create_app(config: RuntimeConfig | None = None) -> FastAPI:
    runtime = config or RuntimeConfig.from_env()
    runtime.ensure_directories()
    database = Database(runtime.database_path)
    database.migrate()
    if runtime.demo_mode:
        from .demo import prepare_demo_environment

        prepare_demo_environment(runtime, database)
    gpu_monitor = DemoGPUMonitor() if runtime.demo_mode else GPUMonitor()
    wandb_secret_store = WandbSecretStore(runtime.state_dir)
    hf_secret_store = HuggingFaceSecretStore(runtime.state_dir)
    registry_overlay_store = RegistryOverlayStore(runtime.state_dir)

    def runtime_registry_catalog() -> dict[str, Any]:
        return catalog_for_runtime(
            registry_overlay_store.effective_catalog(),
            demo_mode=runtime.demo_mode,
        )

    remote_metrics_collector = RemoteMetricsCollector()
    with database.session() as startup_db:
        utility_concurrency = int(SettingsService(startup_db).get("cpu_utility_concurrency", 2))
    utility_manager = UtilityManager(
        runtime,
        database,
        hf_secret_store,
        cpu_concurrency=utility_concurrency,
        interval=min(1.0, max(0.1, runtime.scheduler_interval)),
    )
    scheduler_lock = asyncio.Lock()
    deployment_manager = DeploymentManager(
        runtime,
        database,
        gpu_monitor,
        lock=scheduler_lock,
        source_catalog_provider=runtime_registry_catalog,
    )
    inference_coordinator = InferenceCoordinator()
    evaluation_manager = EvaluationManager(
        runtime,
        database,
        gpu_monitor,
        lock=scheduler_lock,
        controller_secret_store=deployment_manager.controller_secret_store,
        source_catalog_provider=runtime_registry_catalog,
    )
    manager = JobManager(
        runtime,
        database,
        gpu_monitor,
        lock=scheduler_lock,
        deployment_manager=deployment_manager,
        evaluation_manager=evaluation_manager,
        utility_manager=utility_manager,
        wandb_secret_store=wandb_secret_store,
    )

    def secure_cookies(settings: SettingsService) -> bool:
        # An environment-level HTTPS policy is a security floor; a stale or
        # default database value must never turn Secure cookies back off.
        return bool(runtime.secure_cookies or settings.get("secure_cookies", False))

    def public_settings(settings: SettingsService) -> dict[str, Any]:
        values = settings.all()
        values["secure_cookies"] = secure_cookies(settings)
        values["secure_cookies_locked"] = bool(runtime.secure_cookies)
        return values

    def configured_storage_monitor_path(settings: SettingsService) -> Path:
        """Resolve the dashboard storage target, preserving the old default."""

        configured = str(settings.get("storage_monitor_path", "") or "").strip()
        if not configured:
            results_roots = settings.get("results_roots", ["results"])
            configured = str(results_roots[0]) if isinstance(results_roots, list) and results_roots else "results"
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = runtime.repo_root / path
        return path.resolve(strict=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await manager.start()
        await utility_manager.start()
        try:
            yield
        finally:
            try:
                await utility_manager.shutdown()
            finally:
                try:
                    await manager.shutdown()
                finally:
                    close_gpu_monitor = getattr(gpu_monitor, "close", None)
                    if callable(close_gpu_monitor):
                        close_gpu_monitor()

    app = FastAPI(
        title="AlphaBrain UI API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.config = runtime
    app.state.database = database
    app.state.gpu_monitor = gpu_monitor
    app.state.job_manager = manager
    app.state.deployment_manager = deployment_manager
    app.state.inference_coordinator = inference_coordinator
    app.state.evaluation_manager = evaluation_manager
    app.state.wandb_secret_store = wandb_secret_store
    app.state.hf_secret_store = hf_secret_store
    app.state.registry_overlay_store = registry_overlay_store
    app.state.utility_manager = utility_manager

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default response includes the rejected input value. That is
        # unsafe for credential-shaped fields such as a W&B API key, so retain
        # the useful location/message while never reflecting request values.
        errors = [
            {key: value for key, value in item.items() if key not in {"input", "ctx"}}
            for item in exc.errors()
        ]
        if request.url.path == "/api/v1/reference-results/match":
            first = errors[0] if errors else {}
            location = [str(value) for value in first.get("loc", [])]
            if first.get("type") == "extra_forbidden":
                code = "reference_unknown_field"
            elif "signature" in location:
                code = "reference_invalid_signature"
            else:
                code = "reference_invalid_id"
            return JSONResponse(
                status_code=422,
                content={
                    "detail": _localized_error_detail(
                        code,
                        messages=REFERENCE_RESULTS_ERROR_MESSAGES,
                        path=".".join(location[1:]),
                    )
                },
            )
        if request.url.path.startswith("/api/v1/registry/overlay"):
            return JSONResponse(
                status_code=422,
                content={
                    "detail": _localized_error_detail(
                        "overlay_mapping_required",
                        messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                    )
                },
            )
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.middleware("http")
    async def csrf_protection(request: Request, call_next):  # type: ignore[no-untyped-def]
        unsafe = request.method in {"POST", "PUT", "PATCH", "DELETE"}
        exempt = request.url.path in {"/api/v1/setup", "/api/v1/auth/login"}
        if unsafe and request.url.path.startswith("/api/v1") and not exempt:
            with database.session() as db:
                settings = SettingsService(db)
                if settings.get("initialized", False) and settings.get("deployment_mode") == "lab":
                    row = AuthService(db, runtime.session_days).get_session(request.cookies.get(AuthService.cookie_name))
                    header = request.headers.get("X-CSRF-Token")
                    cookie = request.cookies.get(AuthService.csrf_cookie_name)
                    if row is None or not header or header != cookie or header != row.csrf_token:
                        return JSONResponse(status_code=403, content={"detail": "csrf_validation_failed"})
        return await call_next(request)

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "repo_root": str(runtime.repo_root),
            "demo_mode": runtime.demo_mode,
            "gpu_monitor": {"available": gpu_monitor.available, "error": gpu_monitor.error},
        }

    @app.get("/api/v1/setup/status")
    def setup_status(db: Session = Depends(get_db)) -> dict[str, Any]:
        settings = SettingsService(db)
        initialized = bool(settings.get("initialized", False))
        return {
            "initialized": initialized,
            "deployment_mode": (settings.get("deployment_mode", "personal") if initialized else runtime.initial_mode),
            "default_language": "zh-CN",
        }

    @app.post("/api/v1/setup")
    def setup(payload: SetupRequest, response: Response, db: Session = Depends(get_db)) -> dict[str, Any]:
        settings = SettingsService(db)
        if settings.get("initialized", False):
            raise HTTPException(status_code=409, detail="already_initialized")
        if payload.mode == "lab" and len(payload.password) < 8:
            raise HTTPException(status_code=422, detail="password_must_have_8_characters")
        auth = AuthService(db, runtime.session_days)
        password = payload.password if payload.mode == "lab" else secrets.token_urlsafe(32)
        user = User(
            username=payload.username,
            display_name=payload.display_name or payload.username,
            password_hash=auth.hash_password(password),
            role="administrator",
            language="zh-CN",
            is_local=payload.mode == "personal",
        )
        db.add(user)
        db.flush()
        settings.set("deployment_mode", payload.mode, user.id)
        settings.set("initialized", True, user.id)
        add_audit(db, "setup.complete", actor_id=user.id, target_type="system", detail={"mode": payload.mode})
        if payload.mode == "lab":
            raw, session = auth.create_session(user)
            _set_session_cookies(
                response,
                raw,
                session.csrf_token,
                secure_cookies(settings),
                runtime.session_days * 86400,
            )
        return {"user": _serialize_user(user), "deployment_mode": payload.mode}

    @app.post("/api/v1/auth/login")
    def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> dict[str, Any]:
        settings = SettingsService(db)
        if not settings.get("initialized", False):
            raise HTTPException(status_code=428, detail="setup_required")
        if settings.get("deployment_mode") != "lab":
            raise HTTPException(status_code=409, detail="login_not_required_in_personal_mode")
        auth = AuthService(db, runtime.session_days)
        user = auth.authenticate(payload.username, payload.password)
        if user is None:
            add_audit(db, "auth.login_failed", detail={"username": payload.username})
            db.commit()
            raise HTTPException(status_code=401, detail="invalid_credentials")
        raw, session = auth.create_session(user)
        _set_session_cookies(
            response,
            raw,
            session.csrf_token,
            secure_cookies(settings),
            runtime.session_days * 86400,
        )
        add_audit(db, "auth.login", actor_id=user.id, target_type="user", target_id=user.id)
        return {"user": _serialize_user(user), "csrf_token": session.csrf_token}

    @app.post("/api/v1/auth/logout")
    def logout(
        request: Request,
        response: Response,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        AuthService(db, runtime.session_days).revoke(request.cookies.get(AuthService.cookie_name))
        _clear_session_cookies(response)
        add_audit(db, "auth.logout", actor_id=user.id, target_type="user", target_id=user.id)
        return {"ok": True}

    @app.get("/api/v1/auth/me")
    def me(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
        return {
            "user": _serialize_user(user),
            "deployment_mode": SettingsService(db).get("deployment_mode"),
            "experimental_available": bool(SettingsService(db).get("experimental_globally_enabled", False)),
        }

    @app.patch("/api/v1/users/me/preferences")
    def update_preferences(
        payload: PreferenceUpdate,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        values = payload.model_dump(exclude_none=True)
        if values.get("experimental_enabled") and not SettingsService(db).get("experimental_globally_enabled", False):
            raise HTTPException(status_code=409, detail="experimental_globally_disabled")
        for key, value in values.items():
            setattr(user, key, value)
        add_audit(db, "user.preferences_update", actor_id=user.id, target_type="user", target_id=user.id)
        return _serialize_user(user)

    @app.get("/api/v1/users")
    def list_users(_admin: User = Depends(admin_user), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
        return [_serialize_user(row) for row in db.execute(select(User).order_by(User.created_at)).scalars().all()]

    @app.post("/api/v1/users")
    def create_user(
        payload: UserCreate,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        if db.execute(select(User).where(User.username == payload.username)).scalar_one_or_none():
            raise HTTPException(status_code=409, detail="username_exists")
        user = User(
            username=payload.username.strip(),
            display_name=payload.display_name or payload.username,
            password_hash=AuthService(db).hash_password(payload.password),
            role=payload.role,
        )
        db.add(user)
        db.flush()
        add_audit(db, "user.create", actor_id=admin.id, target_type="user", target_id=user.id)
        return _serialize_user(user)

    @app.patch("/api/v1/users/{user_id}")
    def update_user(
        user_id: str,
        payload: UserUpdate,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        target = db.get(User, user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="user_not_found")
        values = payload.model_dump(exclude_none=True)
        password = values.pop("password", None)
        new_role = values.get("role", target.role)
        new_active = values.get("is_active", target.is_active)
        removes_administrator = (
            target.role == "administrator" and target.is_active and (new_role != "administrator" or not new_active)
        )
        if target.id == admin.id and removes_administrator:
            raise HTTPException(status_code=409, detail="cannot_remove_current_administrator")
        if removes_administrator:
            other_administrators = db.execute(
                select(func.count(User.id)).where(
                    User.id != target.id,
                    User.role == "administrator",
                    User.is_active.is_(True),
                )
            ).scalar_one()
            if not other_administrators:
                raise HTTPException(status_code=409, detail="cannot_remove_last_administrator")
        if target.is_active and not new_active:
            active_jobs = db.execute(
                select(func.count(Job.id)).where(
                    Job.owner_id == target.id,
                    Job.status.in_(ACTIVE_STATUSES | {"queued", "blocked"}),
                )
            ).scalar_one()
            active_deployments = db.execute(
                select(func.count(ModelDeployment.id)).where(
                    ModelDeployment.owner_id == target.id,
                    ModelDeployment.status.in_(DEPLOYMENT_ACTIVE_STATUSES | {"queued"}),
                )
            ).scalar_one()
            active_evaluations = db.execute(
                select(func.count(EvaluationRun.id)).where(
                    EvaluationRun.owner_id == target.id,
                    EvaluationRun.status.in_({"queued", "starting", "running", "stopping"}),
                )
            ).scalar_one()
            if active_jobs or active_deployments or active_evaluations:
                raise HTTPException(status_code=409, detail="user_has_active_jobs")
        if password:
            target.password_hash = AuthService(db).hash_password(password)
        for key, value in values.items():
            setattr(target, key, value)
        add_audit(db, "user.update", actor_id=admin.id, target_type="user", target_id=target.id)
        return _serialize_user(target)

    @app.delete("/api/v1/users/{user_id}")
    def deactivate_user(
        user_id: str,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        target = db.get(User, user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="user_not_found")
        if target.id == admin.id:
            raise HTTPException(status_code=409, detail="cannot_deactivate_current_administrator")
        active_jobs = db.execute(
            select(func.count(Job.id)).where(
                Job.owner_id == target.id,
                Job.status.in_(ACTIVE_STATUSES | {"queued", "blocked"}),
            )
        ).scalar_one()
        active_deployments = db.execute(
            select(func.count(ModelDeployment.id)).where(
                ModelDeployment.owner_id == target.id,
                ModelDeployment.status.in_(DEPLOYMENT_ACTIVE_STATUSES | {"queued"}),
            )
        ).scalar_one()
        active_evaluations = db.execute(
            select(func.count(EvaluationRun.id)).where(
                EvaluationRun.owner_id == target.id,
                EvaluationRun.status.in_({"queued", "starting", "running", "stopping"}),
            )
        ).scalar_one()
        if active_jobs or active_deployments or active_evaluations:
            raise HTTPException(status_code=409, detail="user_has_active_jobs")
        target.is_active = False
        add_audit(db, "user.deactivate", actor_id=admin.id, target_type="user", target_id=target.id)
        return {"ok": True}

    @app.get("/api/v1/settings")
    def get_settings(_admin: User = Depends(admin_user), db: Session = Depends(get_db)) -> dict[str, Any]:
        return public_settings(SettingsService(db))

    @app.get("/api/v1/settings/wandb", response_model=WandbSecretStatus)
    def get_wandb_secret_status(_admin: User = Depends(admin_user)) -> dict[str, bool]:
        return wandb_secret_store.status()

    @app.put("/api/v1/settings/wandb/api-key", response_model=WandbSecretStatus)
    def set_wandb_api_key(
        payload: WandbAPIKeyUpdate,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        try:
            result = wandb_secret_store.set_api_key(payload.api_key)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_wandb_api_key") from exc
        add_audit(
            db,
            "settings.wandb_api_key_set",
            actor_id=admin.id,
            target_type="system",
            detail={"configured": True},
        )
        return result

    @app.delete("/api/v1/settings/wandb/api-key", response_model=WandbSecretStatus)
    def delete_wandb_api_key(
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        result = wandb_secret_store.delete_api_key()
        add_audit(
            db,
            "settings.wandb_api_key_delete",
            actor_id=admin.id,
            target_type="system",
            detail={"configured": False},
        )
        return result

    @app.get("/api/v1/settings/huggingface/download-token")
    def get_hf_download_token_status(_admin: User = Depends(admin_user)) -> dict[str, bool]:
        return hf_secret_store.global_status()

    @app.put("/api/v1/settings/huggingface/download-token")
    def set_hf_download_token(
        payload: SecretTokenUpdate,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        result = hf_secret_store.set_global_download_token(payload.token)
        add_audit(db, "settings.hf_download_token_set", actor_id=admin.id, target_type="system")
        return result

    @app.delete("/api/v1/settings/huggingface/download-token")
    def delete_hf_download_token(
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        result = hf_secret_store.delete_global_download_token()
        add_audit(db, "settings.hf_download_token_delete", actor_id=admin.id, target_type="system")
        return result

    @app.get("/api/v1/users/me/huggingface/publish-token")
    def get_hf_publish_token_status(user: User = Depends(current_user)) -> dict[str, bool]:
        return hf_secret_store.user_status(user.id)

    @app.put("/api/v1/users/me/huggingface/publish-token")
    def set_hf_publish_token(
        payload: SecretTokenUpdate,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        result = hf_secret_store.set_user_publish_token(user.id, payload.token)
        add_audit(db, "user.hf_publish_token_set", actor_id=user.id, target_type="user", target_id=user.id)
        return result

    @app.delete("/api/v1/users/me/huggingface/publish-token")
    def delete_hf_publish_token(
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        result = hf_secret_store.delete_user_publish_token(user.id)
        add_audit(db, "user.hf_publish_token_delete", actor_id=user.id, target_type="user", target_id=user.id)
        return result

    @app.patch("/api/v1/settings")
    def update_settings(
        payload: SettingsUpdate,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        values = payload.model_dump(exclude_none=True)
        password = values.pop("admin_password", None)
        configured_python = str(values.get("model_server_python", "") or "").strip()
        if configured_python:
            expanded = Path(configured_python).expanduser()
            resolved_python = (
                str(expanded.resolve(strict=False))
                if expanded.is_absolute() or "/" in configured_python
                else shutil.which(configured_python)
            )
            if (
                not resolved_python
                or not Path(resolved_python).is_file()
                or not os.access(resolved_python, os.X_OK)
            ):
                raise HTTPException(status_code=422, detail="model_server_python_not_found")
            values["model_server_python"] = resolved_python
        environment = values.get("environment")
        if environment is not None:
            unsafe = [key for key in environment if key not in SAFE_ENV_KEYS]
            if unsafe:
                raise HTTPException(status_code=422, detail={"code": "unsafe_environment_keys", "keys": unsafe})
        settings = SettingsService(db)
        current_settings = settings.all()
        remote_keys = {
            "remote_training_enabled",
            "remote_training_host",
            "remote_training_user",
            "remote_training_port",
            "remote_training_repo_root",
            "remote_training_identity_file",
            "remote_training_gpu_ids",
            "remote_training_setup_command",
        }
        if remote_keys.intersection(values):
            candidate_settings = dict(current_settings)
            candidate_settings.update({key: values[key] for key in remote_keys if key in values})
            try:
                remote_config = RemoteTrainingConfig.from_settings(candidate_settings)
                validate_remote_training_config(remote_config)
            except (TypeError, ValueError) as exc:
                code = str(exc) if str(exc).startswith("remote_training_") else "remote_training_config_invalid"
                raise HTTPException(status_code=422, detail=code) from exc
            if remote_config.identity_file:
                identity_file = Path(remote_config.identity_file).expanduser().resolve(strict=False)
                if not identity_file.is_file():
                    raise HTTPException(status_code=422, detail="remote_training_identity_file_not_found")
                values["remote_training_identity_file"] = str(identity_file)
            if remote_config.enabled and shutil.which("ssh") is None:
                raise HTTPException(status_code=422, detail="remote_training_ssh_not_found")
            changed_remote_keys = {
                key for key in remote_keys if key in values and values[key] != current_settings.get(key)
            }
            if changed_remote_keys:
                active_training = db.execute(
                    select(func.count(Job.id)).where(Job.status.in_(ACTIVE_STATUSES | {"queued", "blocked"}))
                ).scalar_one()
                if active_training:
                    raise HTTPException(status_code=409, detail="cannot_change_remote_training_with_active_jobs")
        results_roots = values.get("results_roots")
        if results_roots is not None:
            if not results_roots:
                raise HTTPException(status_code=422, detail="results_roots_cannot_be_empty")
            invalid_roots = []
            for value in results_roots:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = runtime.repo_root / path
                snapshot = storage_snapshot(path, 0, 0)
                if snapshot["error"]:
                    invalid_roots.append({"path": str(path), "error": snapshot["error"]})
            if invalid_roots:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "results_roots_unavailable", "items": invalid_roots},
                )
        if "storage_monitor_path" in values:
            configured = str(values["storage_monitor_path"] or "").strip()
            if not configured:
                values["storage_monitor_path"] = ""
            else:
                path = Path(configured).expanduser()
                if not path.is_absolute():
                    path = runtime.repo_root / path
                path = path.resolve(strict=False)
                if not path.exists():
                    raise HTTPException(
                        status_code=422,
                        detail={"code": "storage_monitor_path_not_found", "path": str(path)},
                    )
                if not path.is_dir():
                    raise HTTPException(
                        status_code=422,
                        detail={"code": "storage_monitor_path_not_directory", "path": str(path)},
                    )
                try:
                    shutil.disk_usage(path)
                except OSError as exc:
                    raise HTTPException(
                        status_code=422,
                        detail={"code": "storage_monitor_path_unavailable", "path": str(path)},
                    ) from exc
                values["storage_monitor_path"] = str(path)
        dataset_roots = values.get("dataset_roots")
        if dataset_roots is not None:
            current_dataset_roots = current_settings.get("dataset_roots", ["data"])
            if [str(value) for value in dataset_roots] == [str(value) for value in current_dataset_roots]:
                # The environment form submits all of its fields. Do not let
                # an unchanged legacy/default root block an unrelated SSH
                # settings update merely because that local path is currently
                # absent. A genuinely changed root is still validated below.
                values.pop("dataset_roots")
                dataset_roots = None
        if dataset_roots is not None:
            if not dataset_roots:
                raise HTTPException(status_code=422, detail="dataset_roots_cannot_be_empty")
            normalized_roots = []
            for value in dataset_roots:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = runtime.repo_root / path
                path = path.resolve(strict=False)
                if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
                    raise HTTPException(status_code=422, detail={"code": "dataset_root_unavailable", "path": str(path)})
                normalized_roots.append(str(path))
            values["dataset_roots"] = normalized_roots
        for path_key in ("managed_dataset_root", "pretrained_root"):
            if path_key not in values:
                continue
            if not str(values[path_key]).strip():
                values[path_key] = ""
                continue
            path = Path(str(values[path_key])).expanduser()
            if not path.is_absolute():
                path = runtime.repo_root / path
            path = path.resolve(strict=False)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise HTTPException(status_code=422, detail={"code": f"{path_key}_unavailable", "path": str(path)}) from exc
            values[path_key] = str(path)
        old_mode = settings.get("deployment_mode")
        new_mode = values.get("deployment_mode", old_mode)
        if old_mode != new_mode:
            active_jobs = db.execute(
                select(func.count(Job.id)).where(Job.status.in_(ACTIVE_STATUSES | {"queued", "blocked"}))
            ).scalar_one()
            active_deployments = db.execute(
                select(func.count(ModelDeployment.id)).where(
                    ModelDeployment.status.in_(DEPLOYMENT_ACTIVE_STATUSES | {"queued"})
                )
            ).scalar_one()
            active_evaluations = db.execute(
                select(func.count(EvaluationRun.id)).where(
                    EvaluationRun.status.in_({"queued", "starting", "running", "stopping"})
                )
            ).scalar_one()
            active_utilities = db.execute(
                select(func.count(UtilityRun.id)).where(UtilityRun.status.in_(UTILITY_ACTIVE_STATUSES | {"queued"}))
            ).scalar_one()
            if active_jobs or active_deployments or active_evaluations or active_utilities:
                raise HTTPException(status_code=409, detail="cannot_switch_mode_with_active_jobs")
            if new_mode == "lab":
                if not password:
                    raise HTTPException(status_code=422, detail="admin_password_required_for_lab_mode")
                admin.password_hash = AuthService(db).hash_password(password)
                admin.is_local = False
            else:
                admin.is_local = True
        settings.update(values, admin.id)
        if "cpu_utility_concurrency" in values:
            utility_manager.cpu_concurrency = int(values["cpu_utility_concurrency"])
        add_audit(db, "settings.update", actor_id=admin.id, target_type="system", detail={"keys": list(values)})
        return public_settings(settings)

    def load_registry_overlay() -> dict[str, Any] | None:
        try:
            return registry_overlay_store.load()
        except RegistryOverlayError as exc:
            raise HTTPException(
                status_code=503,
                detail=_localized_error_detail(
                    exc.code,
                    messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                ),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=_localized_error_detail(
                    str(exc),
                    messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                ),
            ) from exc

    def registry_overlay_response() -> dict[str, Any]:
        overlay = load_registry_overlay()
        try:
            status_value = registry_overlay_store.status()
        except (RegistryOverlayError, RuntimeError) as exc:
            code = exc.code if isinstance(exc, RegistryOverlayError) else str(exc)
            raise HTTPException(
                status_code=503,
                detail=_localized_error_detail(
                    code,
                    messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                ),
            ) from exc
        return {"status": _public_overlay_status(status_value), "overlay": overlay}

    def effective_registry_catalog() -> dict[str, Any]:
        try:
            return runtime_registry_catalog()
        except (RegistryOverlayError, RuntimeError) as exc:
            code = exc.code if isinstance(exc, RegistryOverlayError) else str(exc)
            raise HTTPException(
                status_code=503,
                detail=_localized_error_detail(
                    code,
                    messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                ),
            ) from exc

    @app.get("/api/v1/registry")
    def registry(_user: User = Depends(current_user)) -> dict[str, Any]:
        """Return the per-process effective catalog without mutating global loaders."""

        overlay = load_registry_overlay()
        try:
            catalog = effective_registry_catalog()
            status_value = registry_overlay_store.status()
        except (RegistryOverlayError, RuntimeError) as exc:
            code = exc.code if isinstance(exc, RegistryOverlayError) else str(exc)
            raise HTTPException(
                status_code=503,
                detail=_localized_error_detail(
                    code,
                    messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                ),
            ) from exc
        return {
            "view": build_registry_view(catalog, overlay),
            "overlay_status": _public_overlay_status(status_value),
        }

    @app.get("/api/v1/registry/overlay")
    def get_registry_overlay(_admin: User = Depends(admin_user)) -> dict[str, Any]:
        return registry_overlay_response()

    @app.post("/api/v1/registry/overlay/preview")
    def preview_registry_overlay(
        payload: dict[str, Any],
        _admin: User = Depends(admin_user),
    ) -> dict[str, Any]:
        try:
            result = registry_overlay_store.preview(payload)
        except RegistryOverlayError as exc:
            status_code = 413 if exc.code == "overlay_too_large" else (
                409 if exc.code == "overlay_id_conflict" else 422
            )
            raise HTTPException(
                status_code=status_code,
                detail=_localized_error_detail(
                    exc.code,
                    messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                    path=exc.path,
                    value=exc.detail,
                ),
            ) from exc
        catalog = catalog_for_runtime(
            result["effective_catalog"],
            demo_mode=runtime.demo_mode,
        )
        return {
            "overlay": result["overlay"],
            "view": build_registry_view(catalog, result["overlay"]),
        }

    @app.put("/api/v1/registry/overlay")
    def put_registry_overlay(
        payload: dict[str, Any],
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        try:
            previous = registry_overlay_store.status()
        except (RegistryOverlayError, RuntimeError) as exc:
            code = exc.code if isinstance(exc, RegistryOverlayError) else str(exc)
            raise HTTPException(
                status_code=503,
                detail=_localized_error_detail(
                    code,
                    messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                ),
            ) from exc
        try:
            status_value = registry_overlay_store.save(payload)
        except RegistryOverlayError as exc:
            status_code = 413 if exc.code == "overlay_too_large" else (
                409 if exc.code == "overlay_id_conflict" else 422
            )
            raise HTTPException(
                status_code=status_code,
                detail=_localized_error_detail(
                    exc.code,
                    messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                    path=exc.path,
                    value=exc.detail,
                ),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=_localized_error_detail(
                    str(exc),
                    messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                ),
            ) from exc
        if previous.get("sha256") != status_value.get("sha256"):
            add_audit(
                db,
                "registry_overlay.update",
                actor_id=admin.id,
                target_type="registry_overlay",
                target_id=str(status_value.get("overlay_version") or "active"),
                detail={
                    "before": _overlay_audit_detail(previous),
                    "after": _overlay_audit_detail(status_value),
                },
            )
        return registry_overlay_response()

    @app.delete("/api/v1/registry/overlay")
    def delete_registry_overlay(
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        try:
            previous = registry_overlay_store.status()
        except (RegistryOverlayError, RuntimeError):
            # A regular overlay file with invalid YAML/schema must remain
            # recoverable from the UI.  Do not reflect parse details in the
            # audit trail; the store still refuses to unlink non-regular paths.
            previous = {"configured": True, "invalid": True}
        try:
            status_value = registry_overlay_store.delete()
        except (RegistryOverlayError, RuntimeError) as exc:
            code = exc.code if isinstance(exc, RegistryOverlayError) else str(exc)
            raise HTTPException(
                status_code=503,
                detail=_localized_error_detail(
                    code,
                    messages=REGISTRY_OVERLAY_ERROR_MESSAGES,
                ),
            ) from exc
        if previous.get("configured"):
            add_audit(
                db,
                "registry_overlay.delete",
                actor_id=admin.id,
                target_type="registry_overlay",
                target_id=str(previous.get("overlay_version") or "active"),
                detail={"previous": _overlay_audit_detail(previous), "configured": False},
            )
        return {"status": _public_overlay_status(status_value), "overlay": None}

    @app.get("/api/v1/reference-results")
    def reference_results(_user: User = Depends(current_user)) -> dict[str, Any]:
        try:
            return load_reference_snapshot()
        except ReferenceResultsError as exc:
            raise HTTPException(
                status_code=503,
                detail=_localized_error_detail(
                    exc.code,
                    messages=REFERENCE_RESULTS_ERROR_MESSAGES,
                ),
            ) from exc

    @app.post("/api/v1/reference-results/match")
    def match_reference_result_sets(
        payload: ReferenceResultsMatchRequest,
        _user: User = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            result = match_reference_results(
                payload.benchmark_id,
                payload.signature,
            )
        except ReferenceResultsError as exc:
            status_code = (
                422
                if exc.code in {"reference_invalid_id", "reference_invalid_signature"}
                else 503
            )
            raise HTTPException(
                status_code=status_code,
                detail=_localized_error_detail(
                    exc.code,
                    messages=REFERENCE_RESULTS_ERROR_MESSAGES,
                    path=exc.path if status_code == 422 else "",
                    value=exc.detail if status_code == 422 else None,
                ),
            ) from exc
        return result

    @app.get("/api/v1/capabilities")
    def capabilities(
        include_experimental: bool = Query(False),
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        allowed = _can_experimental(db, user)
        include = bool(include_experimental and allowed)
        try:
            from .capabilities import get_catalog

            catalog = get_catalog(
                include_experimental=include,
                source_catalog=effective_registry_catalog(),
                repo_root=runtime.repo_root,
                environment=effective_environment(
                    runtime.repo_root,
                    SettingsService(db).get("environment", {}),
                ),
            )
            if hasattr(catalog, "model_dump"):
                catalog = catalog.model_dump()
            for combination in catalog.get("combinations", []):
                combination["wandb_categories"] = wandb_category_catalog(combination)
            return {"catalog": catalog, "include_experimental": include}
        except ImportError:
            return {
                "catalog": {"components": [], "combinations": []},
                "include_experimental": include,
            }

    @app.get("/api/v1/deployment/capabilities")
    def deployment_capabilities(
        include_experimental: bool | None = Query(default=None),
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        experimental_allowed = _can_experimental(db, user)
        include = experimental_allowed and include_experimental is not False
        catalog = get_deployment_catalog(
            include_experimental=include,
            source_catalog=effective_registry_catalog(),
        )
        return {
            "catalog": catalog,
            "include_experimental": include,
            "experimental_allowed": experimental_allowed,
        }

    @app.get("/api/v1/datasets/directories")
    def browse_dataset_directories(
        path: str | None = Query(default=None, max_length=4096),
        _user: User = Depends(current_user),
    ) -> dict[str, Any]:
        try:
            return list_directories(path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/datasets/validate")
    def validate_dataset(
        payload: DatasetValidationRequest,
        _user: User = Depends(current_user),
    ) -> dict[str, Any]:
        return validate_dataset_directory(
            payload.path,
            payload.dataset_id,
            payload.dataset_mix,
            base_dir=runtime.repo_root,
        )

    def configured_dataset_roots(settings: SettingsService) -> list[Path]:
        values = settings.get("dataset_roots", [str(runtime.repo_root / "data")])
        roots: list[Path] = []
        for value in values if isinstance(values, list) else []:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = runtime.repo_root / path
            roots.append(path.resolve(strict=False))
        return roots or [(runtime.repo_root / "data").resolve(strict=False)]

    def inspect_allowed_dataset_path(value: str, settings: SettingsService) -> tuple[Path, dict[str, Any]]:
        try:
            source = resolve_local_directory(value, base_dir=runtime.repo_root)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        roots = configured_dataset_roots(settings)
        if not is_within_roots(source, roots):
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "dataset_path_outside_configured_roots",
                    "roots": [str(root) for root in roots],
                },
            )
        return source, inspect_dataset(source)

    packages_root = (runtime.state_dir / "artifacts" / "packages").resolve(strict=False)

    def create_package_run(
        *,
        owner_id: str,
        kind: str,
        source: Path,
        label: str,
        resource_id: str,
        reference_key: str,
        reference_id: str,
        db: Session,
    ) -> UtilityRun:
        active_runs = db.execute(
            select(UtilityRun).where(
                UtilityRun.kind == kind,
                UtilityRun.status.in_(UTILITY_ACTIVE_STATUSES | {"queued"}),
            )
        ).scalars().all()
        if any(str((run.parameters or {}).get(reference_key) or "") == reference_id for run in active_runs):
            raise HTTPException(status_code=409, detail="package_already_active")
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-.")[:96] or "model"
        output = packages_root / owner_id / f"{safe_label}-{{run_id}}.zip"
        temporary = output.parent / f".{output.name}.partial-{{run_id}}"
        return utility_manager.create_run(
            owner_id=owner_id,
            kind=kind,
            resource_id=resource_id,
            command=[
                sys.executable,
                "-m",
                "alphabrain_ui.utility_worker",
                "archive",
                "--source",
                str(source),
                "--output",
                str(output),
                "--run-id",
                "{run_id}",
                "--progress",
                "{progress_path}",
            ],
            cwd=runtime.repo_root,
            output_path=str(output),
            parameters={
                reference_key: reference_id,
                "source_name": source.name,
                "_cleanup_paths": [str(temporary)],
            },
            db_session=db,
        )

    def find_dataset(db: Session, user: User, dataset_id: str) -> DatasetRegistration:
        row = db.execute(
            select(DatasetRegistration)
            .options(selectinload(DatasetRegistration.owner))
            .where(DatasetRegistration.id == dataset_id)
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="dataset_registration_not_found")
        if user.role != "administrator" and row.owner_id != user.id and row.visibility != "shared":
            raise HTTPException(status_code=403, detail="dataset_registration_forbidden")
        return row

    @app.get("/api/v1/resources")
    def list_resources(
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        settings = SettingsService(db)
        resources = resource_catalog(runtime.repo_root, settings.all())
        active = db.execute(
            select(UtilityRun).where(UtilityRun.status.in_(UTILITY_ACTIVE_STATUSES | {"queued"}))
        ).scalars().all()
        active_by_resource = {row.resource_id: serialize_utility(row) for row in active if row.resource_id}
        for item in resources:
            item["active_run"] = active_by_resource.get(item["id"])
        return {"items": resources, "hf_download_token": hf_secret_store.global_status()}

    @app.post("/api/v1/resources/{resource_id}/install")
    def install_resource(
        resource_id: str,
        payload: ResourceInstallRequest,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        duplicate = db.execute(
            select(UtilityRun).where(
                UtilityRun.resource_id == resource_id,
                UtilityRun.status.in_(UTILITY_ACTIVE_STATUSES | {"queued"}),
            )
        ).scalar_one_or_none()
        if duplicate:
            raise HTTPException(status_code=409, detail="resource_install_already_active")
        settings = SettingsService(db)
        catalog = {item["id"]: item for item in resource_catalog(runtime.repo_root, settings.all())}
        definition = catalog.get(resource_id)
        if definition is None:
            raise HTTPException(status_code=404, detail="resource_not_found")
        if not definition.get("installable"):
            raise HTTPException(status_code=409, detail="resource_not_installable")
        if definition.get("requires_hf_token") and not hf_secret_store.global_status()["configured"]:
            raise HTTPException(status_code=409, detail="hf_download_token_required")
        target_value = payload.target_root or definition.get("target_path")
        if not target_value:
            raise HTTPException(status_code=422, detail="resource_target_root_required")
        target = Path(str(target_value)).expanduser()
        if not target.is_absolute():
            target = runtime.repo_root / target
        target = target.resolve(strict=False)
        target.mkdir(parents=True, exist_ok=True)
        try:
            command, environment, output_path = install_command(
                resource_id, repo_root=runtime.repo_root, target_root=target
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        configured_environment = dict(settings.get("environment", {}))
        if resource_id.startswith("pretrained."):
            configured_environment["PRETRAINED_MODELS_DIR"] = str(target)
            settings.set("pretrained_root", str(target), admin.id)
        elif resource_id == "dataset.libero":
            configured_environment["LEROBOT_LIBERO_DATA_DIR"] = str(target)
        settings.set("environment", configured_environment, admin.id)
        run = utility_manager.create_run(
            owner_id=admin.id,
            kind="resource_download",
            resource_id=resource_id,
            command=command,
            environment=environment,
            cwd=runtime.repo_root,
            output_path=output_path,
            parameters={
                "hf_auth": "global",
                "_cleanup_paths": [
                    str(
                        (
                            Path(output_path).parent / f".{Path(output_path).name}.partial-{{run_id}}"
                        )
                        if resource_id.startswith("pretrained.")
                        else Path(output_path) / ".libero.partial-{run_id}"
                    )
                ],
            },
            db_session=db,
        )
        add_audit(db, "resource.install", actor_id=admin.id, target_type="resource", target_id=resource_id, detail={"run_id": run.id})
        return serialize_utility(run)

    @app.put("/api/v1/resources/{resource_id}/path")
    def register_resource_path(
        resource_id: str,
        payload: ResourceRegisterRequest,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        if resource_id not in WORLD_MODEL_RESOURCES:
            raise HTTPException(status_code=409, detail="resource_path_registration_not_supported")
        try:
            path = resolve_local_directory(payload.path, base_dir=runtime.repo_root)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not any(path.iterdir()):
            raise HTTPException(status_code=422, detail="resource_directory_empty")
        settings = SettingsService(db)
        paths = dict(settings.get("resource_paths", {}))
        paths[resource_id] = str(path)
        settings.set("resource_paths", paths, admin.id)
        add_audit(db, "resource.path_register", actor_id=admin.id, target_type="resource", target_id=resource_id, detail={"path": str(path)})
        return next(item for item in resource_catalog(runtime.repo_root, settings.all()) if item["id"] == resource_id)

    @app.post("/api/v1/resources/{resource_id}/preprocess")
    def preprocess_resource(
        resource_id: str,
        payload: ResourcePreprocessRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        definition = WORLD_MODEL_RESOURCES.get(resource_id)
        if definition is None or payload.kind not in definition.get("preprocess", []):
            raise HTTPException(status_code=422, detail="preprocess_not_supported_for_resource")
        settings = SettingsService(db)
        resource_paths = settings.get("resource_paths", {})
        resource_path = str(resource_paths.get(resource_id) or "") if isinstance(resource_paths, dict) else ""
        inputs = dict(payload.inputs)
        if resource_path:
            if payload.kind == "t5":
                inputs.setdefault("text_encoder_dir", str(Path(resource_path) / "text_encoder"))
                inputs.setdefault("tokenizer_dir", str(Path(resource_path) / "tokenizer"))
            elif payload.kind == "reason1":
                inputs.setdefault("reason1_path", resource_path)
            elif payload.kind == "umt5":
                inputs.setdefault("wan_dir", resource_path)
        output_key = "dst" if payload.kind == "reason1_projection" else "output_path"
        output_value = str(inputs.get(output_key) or "")
        if not output_value:
            raise HTTPException(status_code=422, detail="preprocess_output_path_required")
        output = Path(output_value).expanduser()
        if not output.is_absolute():
            output = runtime.repo_root / output
        output = output.resolve(strict=False)
        allowed_output_roots = configured_dataset_roots(settings)
        pretrained_value = str(settings.get("pretrained_root", "") or "")
        if pretrained_value:
            pretrained_path = Path(pretrained_value).expanduser()
            if not pretrained_path.is_absolute():
                pretrained_path = runtime.repo_root / pretrained_path
            allowed_output_roots.append(pretrained_path.resolve(strict=False))
        allowed_output_roots.append((runtime.state_dir / "artifacts").resolve(strict=False))
        if not is_within_roots(output, allowed_output_roots):
            raise HTTPException(status_code=403, detail={"code": "preprocess_output_outside_configured_roots", "roots": [str(root) for root in allowed_output_roots]})
        if output.exists():
            raise HTTPException(status_code=409, detail="preprocess_output_already_exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        inputs[output_key] = str(output)
        if payload.kind == "reason1_projection":
            source = Path(str(inputs.get("src") or "")).expanduser()
            if not source.is_absolute():
                source = runtime.repo_root / source
            if not source.is_file():
                raise HTTPException(status_code=422, detail="projection_source_not_found")
            inputs["src"] = str(source.resolve())
        try:
            command, environment, output_path = preprocess_command(
                payload.kind, repo_root=runtime.repo_root, inputs=inputs
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        run = utility_manager.create_run(
            owner_id=user.id,
            kind=f"world_model_preprocess_{payload.kind}",
            resource_id=resource_id,
            command=command,
            environment=environment,
            cwd=runtime.repo_root,
            output_path=output_path,
            queue_class="gpu",
            requested_gpu_count=payload.gpu_count,
            requested_gpu_ids=payload.gpu_ids,
        )
        add_audit(db, "resource.preprocess", actor_id=user.id, target_type="resource", target_id=resource_id, detail={"run_id": run.id, "kind": payload.kind})
        return serialize_utility(run)

    @app.get("/api/v1/utilities")
    def list_utilities(
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        statement = select(UtilityRun).order_by(UtilityRun.created_at.desc())
        if user.role != "administrator":
            statement = statement.where(UtilityRun.owner_id == user.id)
        rows = db.execute(statement.limit(500)).scalars().all()
        queued = [row for row in reversed(rows) if row.status == "queued"]
        positions = {row.id: index for index, row in enumerate(queued, start=1)}
        return [serialize_utility(row, queue_position=positions.get(row.id)) for row in rows]

    @app.post("/api/v1/utilities/{run_id}/cancel")
    async def cancel_utility(
        run_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        run = db.get(UtilityRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="utility_not_found")
        assert_owner_or_admin(user, run.owner_id)
        await utility_manager.stop(run_id)
        db.expire_all()
        updated = db.get(UtilityRun, run_id)
        add_audit(db, "utility.cancel", actor_id=user.id, target_type="utility", target_id=run_id)
        return serialize_utility(updated)

    @app.get("/api/v1/utilities/{run_id}/log")
    def get_utility_log(
        run_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> Response:
        run = db.get(UtilityRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="utility_not_found")
        assert_owner_or_admin(user, run.owner_id)
        path = Path(run.log_path).resolve(strict=False)
        logs_root = (runtime.state_dir / "logs" / "utilities").resolve(strict=False)
        if logs_root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="utility_log_not_found")
        return FileResponse(path, media_type="text/plain", filename=f"utility-{run.id}.log")

    @app.get("/api/v1/utilities/{run_id}/output")
    def get_utility_output(
        run_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> Response:
        run = db.get(UtilityRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="utility_not_found")
        assert_owner_or_admin(user, run.owner_id)
        if run.kind not in {"checkpoint_package", "training_package"}:
            raise HTTPException(status_code=409, detail="utility_output_not_downloadable")
        if run.status != "completed":
            raise HTTPException(status_code=409, detail="utility_output_not_ready")
        try:
            path = Path(run.output_path).resolve(strict=True)
        except OSError as exc:
            raise HTTPException(status_code=404, detail="utility_output_not_found") from exc
        if path.is_symlink() or not path.is_file() or not path.is_relative_to(packages_root):
            raise HTTPException(status_code=403, detail="utility_output_outside_package_root")
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @app.get("/api/v1/datasets")
    def list_registered_datasets(
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        statement = select(DatasetRegistration).options(selectinload(DatasetRegistration.owner))
        if user.role != "administrator":
            statement = statement.where(
                (DatasetRegistration.visibility == "shared") | (DatasetRegistration.owner_id == user.id)
            )
        rows = db.execute(statement.order_by(DatasetRegistration.updated_at.desc())).scalars().all()
        return [_serialize_dataset(row) for row in rows]

    @app.post("/api/v1/datasets/inspect")
    def inspect_dataset_for_registration(
        payload: DatasetInspectionRequest,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        _source, report = inspect_allowed_dataset_path(payload.path, SettingsService(db))
        return report

    @app.post("/api/v1/datasets")
    def register_dataset(
        payload: DatasetRegistrationCreate,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        settings = SettingsService(db)
        source, report = inspect_allowed_dataset_path(payload.path, settings)
        if not report["valid"]:
            raise HTTPException(status_code=422, detail={"code": "dataset_validation_failed", "report": report})
        registration = DatasetRegistration(
            owner_id=user.id,
            name=payload.name.strip(),
            description=payload.description,
            source_path=str(source),
            storage_mode=payload.storage_mode,
            visibility=payload.visibility,
            format=str(report["format"]),
            status="ready" if payload.storage_mode == "reference" else "copying",
            dataset_id=payload.dataset_id,
            dataset_mix=payload.dataset_mix,
            fingerprint=str(report.get("fingerprint", "")),
            size_bytes=int(report.get("size_bytes", 0)),
            episode_count=int(report.get("episode_count", 0)),
            step_count=int(report.get("step_count", 0)),
            validation=report,
        )
        if payload.storage_mode == "reference":
            registration.path = str(source)
        else:
            managed_value = settings.get("managed_dataset_root", str(runtime.state_dir / "datasets"))
            managed_root = Path(str(managed_value)).expanduser().resolve(strict=False)
            managed_root.mkdir(parents=True, exist_ok=True)
            registration.path = str(managed_root / f"{payload.name.strip().replace(' ', '-')}-{new_id()[:8]}")
        db.add(registration)
        try:
            db.flush()
        except IntegrityError as exc:
            raise HTTPException(status_code=409, detail="dataset_path_already_registered") from exc
        if payload.storage_mode == "managed_copy":
            command = [
                os.sys.executable, "-m", "alphabrain_ui.utility_worker", "copy-tree",
                "--source", str(source), "--target", registration.path, "--run-id", registration.id,
            ]
            run = utility_manager.create_run(
                owner_id=user.id, kind="dataset_copy", command=command, cwd=runtime.repo_root,
                output_path=registration.path, parameters={"registration_id": registration.id}, db_session=db,
            )
            registration.copy_run_id = run.id
        add_audit(db, "dataset.register", actor_id=user.id, target_type="dataset", target_id=registration.id, detail={"storage_mode": payload.storage_mode})
        db.flush()
        return _serialize_dataset(registration)

    @app.get("/api/v1/datasets/mixtures")
    def list_dataset_mixtures(
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        statement = select(DatasetMixture).options(selectinload(DatasetMixture.owner))
        if user.role != "administrator":
            statement = statement.where((DatasetMixture.visibility == "shared") | (DatasetMixture.owner_id == user.id))
        rows = db.execute(statement.order_by(DatasetMixture.updated_at.desc())).scalars().all()
        ids = {str(member.get("registration_id")) for row in rows for member in (row.members or [])}
        registrations = {row.id: row for row in db.execute(select(DatasetRegistration).where(DatasetRegistration.id.in_(ids))).scalars().all()} if ids else {}
        return [_serialize_mixture(row, registrations) for row in rows]

    def validate_mixture_members(db: Session, user: User, members: list[dict[str, Any]]) -> dict[str, DatasetRegistration]:
        registrations: dict[str, DatasetRegistration] = {}
        for member in members:
            registration = find_dataset(db, user, str(member["registration_id"]))
            if registration.status != "ready":
                raise HTTPException(status_code=409, detail={"code": "dataset_not_ready", "dataset_id": registration.id})
            registrations[registration.id] = registration
        return registrations

    @app.post("/api/v1/datasets/mixtures")
    def create_dataset_mixture(
        payload: DatasetMixtureCreate,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        members = [member.model_dump(mode="json") for member in payload.members]
        registrations = validate_mixture_members(db, user, members)
        row = DatasetMixture(owner_id=user.id, name=payload.name.strip(), description=payload.description, visibility=payload.visibility, members=members, options=payload.options)
        db.add(row)
        db.flush()
        add_audit(db, "dataset_mixture.create", actor_id=user.id, target_type="dataset_mixture", target_id=row.id)
        return _serialize_mixture(row, registrations)

    @app.patch("/api/v1/datasets/mixtures/{mixture_id}")
    def update_dataset_mixture(
        mixture_id: str,
        payload: DatasetMixtureUpdate,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        row = db.get(DatasetMixture, mixture_id)
        if row is None:
            raise HTTPException(status_code=404, detail="dataset_mixture_not_found")
        assert_owner_or_admin(user, row.owner_id)
        values = payload.model_dump(exclude_none=True, mode="json")
        members = values.get("members", row.members)
        registrations = validate_mixture_members(db, user, members)
        for key, value in values.items():
            setattr(row, key, value)
        row.version += 1
        add_audit(db, "dataset_mixture.update", actor_id=user.id, target_type="dataset_mixture", target_id=row.id)
        return _serialize_mixture(row, registrations)

    @app.delete("/api/v1/datasets/mixtures/{mixture_id}")
    def delete_dataset_mixture(
        mixture_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        row = db.get(DatasetMixture, mixture_id)
        if row is None:
            raise HTTPException(status_code=404, detail="dataset_mixture_not_found")
        assert_owner_or_admin(user, row.owner_id)
        db.delete(row)
        add_audit(db, "dataset_mixture.delete", actor_id=user.id, target_type="dataset_mixture", target_id=mixture_id)
        return {"ok": True}

    @app.get("/api/v1/datasets/{dataset_id}")
    def get_registered_dataset(
        dataset_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        return _serialize_dataset(find_dataset(db, user, dataset_id))

    @app.post("/api/v1/datasets/{dataset_id}/validate")
    def revalidate_registered_dataset(
        dataset_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        row = find_dataset(db, user, dataset_id)
        assert_owner_or_admin(user, row.owner_id)
        report = inspect_dataset(Path(row.path))
        row.validation = report
        row.status = "ready" if report["valid"] else "invalid"
        row.format = str(report["format"])
        row.episode_count = int(report.get("episode_count", 0))
        row.step_count = int(report.get("step_count", 0))
        row.size_bytes = int(report.get("size_bytes", 0))
        previous_fingerprint = row.fingerprint
        row.fingerprint = str(report.get("fingerprint", ""))
        if previous_fingerprint and previous_fingerprint != row.fingerprint:
            row.stats_status = "stale"
        add_audit(db, "dataset.validate", actor_id=user.id, target_type="dataset", target_id=row.id)
        return _serialize_dataset(row)

    @app.get("/api/v1/datasets/{dataset_id}/preview")
    def get_dataset_preview(
        dataset_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        row = find_dataset(db, user, dataset_id)
        return preview_dataset(Path(row.path), limit=limit)

    @app.post("/api/v1/datasets/{dataset_id}/stats")
    def compute_dataset_stats(
        dataset_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        row = find_dataset(db, user, dataset_id)
        assert_owner_or_admin(user, row.owner_id)
        if row.status != "ready":
            raise HTTPException(status_code=409, detail="dataset_not_ready")
        output = runtime.state_dir / "artifacts" / "dataset-stats" / f"{row.id}.json"
        command = [os.sys.executable, "-m", "alphabrain_ui.utility_worker", "dataset-stats", "--source", row.path, "--output", str(output)]
        run = utility_manager.create_run(
            owner_id=user.id, kind="dataset_stats", command=command, cwd=runtime.repo_root,
            output_path=str(output), parameters={"registration_id": row.id}, db_session=db,
        )
        row.stats_run_id = run.id
        row.stats_status = "queued"
        add_audit(db, "dataset.stats", actor_id=user.id, target_type="dataset", target_id=row.id, detail={"run_id": run.id})
        return serialize_utility(run)

    @app.delete("/api/v1/datasets/{dataset_id}")
    def remove_dataset_registration(
        dataset_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        row = find_dataset(db, user, dataset_id)
        assert_owner_or_admin(user, row.owner_id)
        active_run = db.execute(
            select(UtilityRun).where(
                UtilityRun.id.in_([value for value in (row.copy_run_id, row.stats_run_id) if value]),
                UtilityRun.status.in_(UTILITY_ACTIVE_STATUSES | {"queued"}),
            )
        ).scalar_one_or_none() if row.copy_run_id or row.stats_run_id else None
        if active_run:
            raise HTTPException(status_code=409, detail="dataset_has_active_utility")
        path = row.path
        storage_mode = row.storage_mode
        db.delete(row)
        add_audit(db, "dataset.unregister", actor_id=user.id, target_type="dataset", target_id=dataset_id, detail={"data_deleted": False})
        return {"ok": True, "data_deleted": False, "path": path, "storage_mode": storage_mode}

    def configured_builtin_templates(settings: SettingsService) -> list[dict[str, Any]]:
        pretrained_root = Path(str(
            os.environ.get("PRETRAINED_MODELS_DIR")
            or settings.get("pretrained_root", "")
            or "data/pretrained_models"
        )).expanduser()
        if not pretrained_root.is_absolute():
            pretrained_root = runtime.repo_root / pretrained_root
        return builtin_templates(
            pretrained_root=pretrained_root,
            resource_paths=settings.get("resource_paths", {}),
            environment={
                **os.environ,
                **{str(key): str(value) for key, value in settings.get("environment", {}).items()},
            },
        )

    def template_duplicate_matches(
        *,
        name: str,
        spec: dict[str, Any],
        user: User,
        db: Session,
        exclude_id: str | None = None,
        candidate_owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        owner_id = candidate_owner_id or user.id
        candidate_name = normalize_template_name(name)
        candidate_json = canonical_template_json(spec)
        candidate_fingerprint = template_spec_fingerprint(spec)
        matches: list[dict[str, Any]] = []

        rows = (
            db.execute(
                select(ExperimentTemplate).where(
                    (ExperimentTemplate.visibility == "shared")
                    | (ExperimentTemplate.owner_id == user.id)
                    | (ExperimentTemplate.owner_id == owner_id)
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            if row.id == exclude_id:
                continue
            if (
                row.fingerprint_version != TEMPLATE_FINGERPRINT_VERSION
                or not row.spec_fingerprint
            ):
                row.spec_fingerprint = template_spec_fingerprint(row.spec)
                row.fingerprint_version = TEMPLATE_FINGERPRINT_VERSION
            same_spec = bool(
                row.spec_fingerprint == candidate_fingerprint
                and canonical_template_json(row.spec) == candidate_json
            )
            same_name = bool(
                row.owner_id == owner_id
                and normalize_template_name(row.name) == candidate_name
            )
            if not same_spec and not same_name:
                continue
            matches.append(
                {
                    "id": row.id,
                    "name": row.name,
                    "owner_id": row.owner_id,
                    "owner_name": row.owner.display_name or row.owner.username,
                    "visibility": "shared" if row.visibility == "shared" else "personal",
                    "builtin": False,
                    "same_name": same_name,
                    "same_spec": same_spec,
                }
            )

        for preset in configured_builtin_templates(SettingsService(db)):
            same_spec = bool(
                template_spec_fingerprint(preset["spec"]) == candidate_fingerprint
                and canonical_template_json(preset["spec"]) == candidate_json
            )
            if not same_spec:
                continue
            matches.append(
                {
                    "id": preset["id"],
                    "name": preset["name"],
                    "owner_id": None,
                    "owner_name": preset.get("owner_name", "AlphaBrain"),
                    "visibility": "shared",
                    "builtin": True,
                    "same_name": False,
                    "same_spec": True,
                }
            )
        matches.sort(
            key=lambda item: (
                not (item["same_name"] and item["same_spec"]),
                not item["same_spec"],
                not item["same_name"],
                item["name"].casefold(),
            )
        )
        return matches

    def reject_unconfirmed_template_duplicate(
        *,
        name: str,
        spec: dict[str, Any],
        allow_duplicate: bool,
        user: User,
        db: Session,
        exclude_id: str | None = None,
        candidate_owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        matches = template_duplicate_matches(
            name=name,
            spec=spec,
            user=user,
            db=db,
            exclude_id=exclude_id,
            candidate_owner_id=candidate_owner_id,
        )
        if matches and not allow_duplicate:
            raise HTTPException(
                status_code=409,
                detail={"code": "duplicate_template", "matches": matches},
            )
        return matches

    @app.get("/api/v1/templates")
    def list_templates(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
        settings = SettingsService(db)
        presets = configured_builtin_templates(settings)
        rows = (
            db.execute(
                select(ExperimentTemplate)
                .where((ExperimentTemplate.visibility == "shared") | (ExperimentTemplate.owner_id == user.id))
                .order_by(ExperimentTemplate.updated_at.desc())
            )
            .scalars()
            .all()
        )
        return [*presets, *[_serialize_template(row) for row in rows]]

    @app.post("/api/v1/templates")
    def create_template(
        payload: TemplateCreate,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        values = payload.model_dump(exclude={"allow_duplicate"})
        matches = reject_unconfirmed_template_duplicate(
            name=payload.name,
            spec=payload.spec,
            allow_duplicate=payload.allow_duplicate,
            user=user,
            db=db,
        )
        row = ExperimentTemplate(
            owner_id=user.id,
            **values,
            spec_fingerprint=template_spec_fingerprint(payload.spec),
            fingerprint_version=TEMPLATE_FINGERPRINT_VERSION,
        )
        db.add(row)
        db.flush()
        add_audit(
            db,
            "template.create",
            actor_id=user.id,
            target_type="template",
            target_id=row.id,
            detail={
                "duplicate_override": bool(payload.allow_duplicate and matches),
                "duplicate_match_ids": [item["id"] for item in matches],
            },
        )
        return _serialize_template(row)

    @app.get("/api/v1/templates/{template_id}")
    def get_template(
        template_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        if is_builtin_template_id(template_id):
            settings = SettingsService(db)
            presets = configured_builtin_templates(settings)
            preset = next((item for item in presets if item["id"] == template_id), None)
            if preset is None:
                raise HTTPException(status_code=404, detail="template_not_found")
            return preset
        row = db.get(ExperimentTemplate, template_id)
        if row is None or (row.visibility != "shared" and row.owner_id != user.id and user.role != "administrator"):
            raise HTTPException(status_code=404, detail="template_not_found")
        return _serialize_template(row)

    @app.patch("/api/v1/templates/{template_id}")
    def update_template(
        template_id: str,
        payload: TemplateUpdate,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        if is_builtin_template_id(template_id):
            raise HTTPException(status_code=409, detail="builtin_template_read_only")
        row = db.get(ExperimentTemplate, template_id)
        if row is None:
            raise HTTPException(status_code=404, detail="template_not_found")
        assert_owner_or_admin(user, row.owner_id)
        values = payload.model_dump(exclude_none=True, exclude={"allow_duplicate"})
        candidate_name = str(values.get("name", row.name))
        candidate_spec = values.get("spec", row.spec)
        identity_changed = bool(
            normalize_template_name(candidate_name) != normalize_template_name(row.name)
            or canonical_template_json(candidate_spec) != canonical_template_json(row.spec)
        )
        matches = (
            reject_unconfirmed_template_duplicate(
                name=candidate_name,
                spec=candidate_spec,
                allow_duplicate=payload.allow_duplicate,
                user=user,
                db=db,
                exclude_id=row.id,
                candidate_owner_id=row.owner_id,
            )
            if identity_changed
            else []
        )
        for key, value in values.items():
            setattr(row, key, value)
        if "spec" in values:
            row.spec_fingerprint = template_spec_fingerprint(candidate_spec)
            row.fingerprint_version = TEMPLATE_FINGERPRINT_VERSION
        row.version += 1
        add_audit(
            db,
            "template.update",
            actor_id=user.id,
            target_type="template",
            target_id=row.id,
            detail={
                "duplicate_override": bool(payload.allow_duplicate and matches),
                "duplicate_match_ids": [item["id"] for item in matches],
            },
        )
        return _serialize_template(row)

    @app.delete("/api/v1/templates/{template_id}")
    def delete_template(
        template_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        if is_builtin_template_id(template_id):
            raise HTTPException(status_code=409, detail="builtin_template_read_only")
        row = db.get(ExperimentTemplate, template_id)
        if row is None:
            raise HTTPException(status_code=404, detail="template_not_found")
        assert_owner_or_admin(user, row.owner_id)
        db.delete(row)
        add_audit(db, "template.delete", actor_id=user.id, target_type="template", target_id=template_id)
        return {"ok": True}

    def resolve_or_preflight(
        payload: ExperimentRequest,
        user: User,
        db: Session,
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[dict[str, Any]]]:
        if runtime.demo_mode:
            from .demo import demo_experiment_preflight

            return demo_experiment_preflight(runtime, payload)
        settings = SettingsService(db)
        effective_spec = json.loads(json.dumps(payload.spec))
        dataset_spec = effective_spec.get("dataset") if isinstance(effective_spec.get("dataset"), dict) else {}
        dataset_reference_issues: list[dict[str, Any]] = []

        def dataset_issue(code: str, message_zh: str, message_en: str, field: str) -> None:
            dataset_reference_issues.append(
                {
                    "level": "error",
                    "code": code,
                    "message": message_en,
                    "message_i18n": {"zh-CN": message_zh, "en-US": message_en},
                    "field": field,
                    "detail": {},
                }
            )

        registration_id = str(dataset_spec.get("registration_id") or "")
        mixture_id = str(dataset_spec.get("mixture_id") or "")
        if registration_id and mixture_id:
            dataset_issue(
                "dataset_source_ambiguous",
                "一次实验只能选择一个已登记数据集或一个数据组合。",
                "Choose either one registered dataset or one dataset mixture.",
                "dataset",
            )
        elif registration_id:
            registration = db.get(DatasetRegistration, registration_id)
            if registration is None or (
                user.role != "administrator"
                and registration.visibility == "private"
                and registration.owner_id != user.id
            ):
                dataset_issue(
                    "dataset_registration_not_found",
                    "所选已登记数据集不存在或当前用户无权使用。",
                    "The selected dataset registration does not exist or is not accessible.",
                    "dataset.registration_id",
                )
            elif registration.status != "ready":
                dataset_issue(
                    "dataset_registration_not_ready",
                    "所选数据集尚未通过校验或仍在复制。",
                    "The selected dataset is not validated and ready.",
                    "dataset.registration_id",
                )
            else:
                dataset_spec["root"] = registration.path
                dataset_spec["mixture_spec"] = [
                    {
                        "path": registration.path,
                        "pattern": registration.dataset_mix or "",
                        "weight": 1.0,
                        "robot_type": "",
                    }
                ]
                dataset_spec["source"] = {
                    "kind": "registration",
                    "id": registration.id,
                    "fingerprint": registration.fingerprint,
                }
        elif mixture_id:
            mixture = db.get(DatasetMixture, mixture_id)
            if mixture is None or (
                user.role != "administrator"
                and mixture.visibility == "private"
                and mixture.owner_id != user.id
            ):
                dataset_issue(
                    "dataset_mixture_not_found",
                    "所选数据组合不存在或当前用户无权使用。",
                    "The selected dataset mixture does not exist or is not accessible.",
                    "dataset.mixture_id",
                )
            else:
                registration_ids = {
                    str(member.get("registration_id") or "") for member in mixture.members or []
                }
                registrations = {
                    row.id: row
                    for row in db.execute(
                        select(DatasetRegistration).where(DatasetRegistration.id.in_(registration_ids))
                    ).scalars().all()
                }
                inaccessible = [
                    registration_id
                    for registration_id in registration_ids
                    if registration_id not in registrations
                    or registrations[registration_id].status != "ready"
                    or (
                        user.role != "administrator"
                        and registrations[registration_id].visibility == "private"
                        and registrations[registration_id].owner_id != user.id
                    )
                ]
                if inaccessible or not registration_ids:
                    dataset_issue(
                        "dataset_mixture_not_ready",
                        "数据组合包含缺失、未校验或无权访问的成员。",
                        "The dataset mixture contains missing, unvalidated, or inaccessible members.",
                        "dataset.mixture_id",
                    )
                else:
                    serialized = _serialize_mixture(mixture, registrations)
                    dataset_spec["mixture_spec"] = serialized["mixture_spec"]
                    dataset_spec["source"] = {
                        "kind": "mixture",
                        "id": mixture.id,
                        "version": mixture.version,
                    }
        elif dataset_spec.get("mixture_spec"):
            # Arbitrary paths in a generic experiment JSON would bypass the
            # Dataset Center root and visibility checks. Only server-resolved
            # registration/mixture references may populate this field.
            dataset_spec.pop("mixture_spec", None)
            dataset_issue(
                "untrusted_dataset_mixture_spec",
                "自定义 mixture_spec 必须先在数据中心登记为数据组合。",
                "A custom mixture_spec must first be registered in the Dataset Center.",
                "dataset.mixture_spec",
            )
        effective_spec["dataset"] = dataset_spec
        remote_training = RemoteTrainingConfig.from_settings(settings.all())
        result = run_preflight(
            repo_root=runtime.repo_root,
            state_dir=runtime.state_dir,
            name=payload.name,
            spec=effective_spec,
            configured_environment=settings.get("environment", {}),
            gpu_monitor=gpu_monitor,
            results_roots=settings.get("results_roots", ["results"]),
            disk_min_free_gib=float(settings.get("disk_min_free_gib", 100)),
            disk_min_free_percent=float(settings.get("disk_min_free_percent", 10)),
            include_experimental=_can_experimental(db, user),
            source_catalog=effective_registry_catalog(),
            remote_gpu_ids=list(remote_training.gpu_ids) if remote_training.enabled else None,
        )
        resolved, compatibility, issues, previews = result
        issues.extend(dataset_reference_issues)
        if wandb_requires_api_key(resolved) and not wandb_secret_store.status()["configured"]:
            issues.append(
                {
                    "level": "error",
                    "code": "wandb_api_key_missing",
                    "message": "W&B online mode requires an API key configured in Settings.",
                    "message_i18n": {
                        "zh-CN": "W&B 在线模式需要先在“设置 → 环境”中保存 API Key。",
                        "en-US": "W&B online mode requires an API key configured under Settings → Environment.",
                    },
                    "field": "wandb.mode",
                    "detail": {"settings_tab": "environment"},
                }
            )
        preview_outputs = {str(item.get("output_dir", "")) for item in previews if item.get("output_dir")}
        if preview_outputs:
            reserved = set(db.execute(select(Job.output_dir).where(Job.output_dir.in_(preview_outputs))).scalars().all())
            for output_dir in sorted(reserved):
                issues.append(
                    {
                        "level": "error",
                        "code": "output_directory_reserved",
                        "message": f"Another experiment already owns this output directory: {output_dir}",
                        "message_i18n": {
                            "zh-CN": f"该输出目录已被其他实验占用：{output_dir}",
                            "en-US": f"Another experiment already owns this output directory: {output_dir}",
                        },
                        "field": "parameters.run_id",
                        "detail": {"output_dir": output_dir},
                    }
                )
        return resolved, compatibility, issues, previews

    @app.post("/api/v1/experiments/resolve", response_model=ResolveResponse)
    def resolve_experiment_endpoint(
        payload: ExperimentRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        resolved, compatibility, issues, previews = resolve_or_preflight(payload, user, db)
        # Resolution returns all static information; callers may distinguish
        # warnings from launch-blocking errors through /preflight.
        return {
            "resolved": resolved,
            "compatibility": compatibility,
            "issues": issues,
            "command_preview": previews,
            "diff": resolved.get("diff", {}) if isinstance(resolved, dict) else {},
        }

    @app.post("/api/v1/experiments/preflight", response_model=PreflightResponse)
    def preflight_endpoint(
        payload: ExperimentRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        resolved, compatibility, issues, previews = resolve_or_preflight(payload, user, db)
        return {
            "resolved": resolved,
            "compatibility": compatibility,
            "issues": issues,
            "command_preview": previews,
            "diff": resolved.get("diff", {}) if isinstance(resolved, dict) else {},
            "can_submit": not any(item["level"] == "error" for item in issues),
        }

    @app.post("/api/v1/experiments/submit")
    def submit_experiment(
        payload: ExperimentRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        resolved, compatibility, issues, _previews = resolve_or_preflight(payload, user, db)
        errors = [item for item in issues if item["level"] == "error"]
        if errors:
            raise HTTPException(status_code=422, detail={"code": "preflight_failed", "issues": errors})
        if compatibility == "experimental" and not payload.acknowledge_experimental:
            raise HTTPException(status_code=409, detail="experimental_acknowledgement_required")

        experiment_id = new_id()
        settings = SettingsService(db)
        if runtime.demo_mode:
            from .demo import demo_launch_plan

            stages = demo_launch_plan(runtime, payload, resolved, experiment_id)
        else:
            stages = build_launch_plan(
                runtime.repo_root,
                runtime.state_dir,
                payload.name,
                resolved,
                snapshot_key=experiment_id,
                environment=settings.get("environment", {}),
                write_snapshots=True,
            )
        experiment = Experiment(
            id=experiment_id,
            owner_id=user.id,
            name=payload.name,
            family=normalize_family(resolved),
            compatibility=compatibility,
            status="queued",
            spec=payload.spec,
            resolved=resolved,
            config_snapshot_path=next((s.config_snapshot_path for s in stages if s.config_snapshot_path), ""),
        )
        db.add(experiment)
        db.flush()
        stage_ids = [new_id() for _ in stages]
        checkpoint_value = (
            resolved.get("checkpoint")
            or resolved.get("training", {}).get("checkpoint")
            or resolved.get("spec", {}).get("training", {}).get("checkpoint")
        )
        input_checkpoint_path = str(Path(str(checkpoint_value)).expanduser().resolve()) if checkpoint_value else ""
        created_jobs: list[Job] = []
        for position, launch in enumerate(stages):
            dependency = stage_ids[launch.dependency_position] if launch.dependency_position is not None else None
            stage = ExperimentStage(
                id=stage_ids[position],
                experiment_id=experiment.id,
                name=launch.name,
                phase=launch.phase,
                position=position,
                dependency_stage_id=dependency,
                resolved=launch.redacted_preview(),
            )
            db.add(stage)
            job_id = new_id()
            job = Job(
                id=job_id,
                experiment_id=experiment.id,
                stage_id=stage.id,
                owner_id=user.id,
                status="blocked" if dependency else "queued",
                requested_gpu_count=launch.requested_gpu_count,
                requested_gpu_ids=launch.requested_gpu_ids,
                command=launch.command,
                environment=launch.environment,
                cwd=launch.cwd,
                output_dir=launch.output_dir,
                input_checkpoint_path=input_checkpoint_path,
                log_path=str(runtime.state_dir / "logs" / f"{job_id}.log"),
                metrics_path=launch.metrics_path,
            )
            db.add(job)
            created_jobs.append(job)
        add_audit(
            db,
            "experiment.submit",
            actor_id=user.id,
            target_type="experiment",
            target_id=experiment.id,
            detail={"family": experiment.family, "compatibility": compatibility, "stages": len(stages)},
        )
        try:
            db.flush()
        except IntegrityError as exc:
            if "output_dir" in str(exc).lower():
                shutil.rmtree(runtime.state_dir / "configs" / experiment_id, ignore_errors=True)
                raise HTTPException(status_code=409, detail="output_directory_reserved") from exc
            raise
        return {
            "experiment_id": experiment.id,
            "jobs": [_serialize_job(job) for job in created_jobs],
            "issues": [item for item in issues if item["level"] != "error"],
        }

    @app.get("/api/v1/experiments")
    def list_experiments(
        limit: int = Query(100, ge=1, le=500),
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        rows = (
            db.execute(
                select(Experiment)
                .options(selectinload(Experiment.stages).selectinload(ExperimentStage.jobs))
                .order_by(Experiment.created_at.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return [_serialize_experiment(row) for row in rows]

    @app.get("/api/v1/experiments/{experiment_id}")
    def experiment_detail(
        experiment_id: str,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        return _serialize_experiment(_get_experiment(db, experiment_id))

    @app.get("/api/v1/jobs")
    def list_jobs(
        job_status: str | None = Query(None, alias="status"),
        limit: int = Query(200, ge=1, le=1000),
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        stmt = select(Job).order_by(Job.queued_at.desc()).limit(limit)
        if job_status:
            stmt = stmt.where(Job.status == job_status)
        positions = _job_queue_positions(db)
        return [_serialize_job(row, positions.get(row.id)) for row in db.execute(stmt).scalars().all()]

    @app.get("/api/v1/workloads")
    def list_workloads(
        workload_status: str | None = Query(None, alias="status"),
        kind: str | None = Query(None),
        limit: int = Query(300, ge=1, le=1000),
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        global_positions = _queue_positions(db)
        cpu_queued = db.execute(
            select(UtilityRun.id)
            .where(
                UtilityRun.status == "queued",
                UtilityRun.queue_class == "cpu",
            )
            .order_by(UtilityRun.queued_at, UtilityRun.created_at)
        ).scalars().all()
        cpu_positions = {run_id: position for position, run_id in enumerate(cpu_queued, start=1)}
        records: list[dict[str, Any]] = []

        def allowed(record_kind: str, status_value: str) -> bool:
            return (not kind or kind == record_kind) and (
                not workload_status or workload_status == status_value
            )

        if kind in {None, "training"}:
            non_terminal_experiment_ids = set(
                db.execute(
                    select(Job.experiment_id).where(Job.status.notin_(TERMINAL_STATUSES))
                ).scalars().all()
            )
            latest_packages: dict[str, UtilityRun] = {}
            for package_run in db.execute(
                select(UtilityRun)
                .where(UtilityRun.kind == "training_package")
                .order_by(UtilityRun.created_at.desc())
            ).scalars().all():
                packaged_job_id = str((package_run.parameters or {}).get("job_id") or "")
                if packaged_job_id and packaged_job_id not in latest_packages:
                    latest_packages[packaged_job_id] = package_run
            rows = db.execute(
                select(Job)
                .options(selectinload(Job.owner), selectinload(Job.experiment), selectinload(Job.stage))
                .order_by(Job.created_at.desc())
                .limit(limit)
            ).scalars().all()
            for row in rows:
                if not allowed("training", row.status):
                    continue
                serialized = _serialize_job(row, global_positions.get(f"job:{row.id}"))
                records.append(
                    {
                        "id": row.id,
                        "kind": "training",
                        "name": serialized["name"],
                        "owner_name": serialized["owner_name"],
                        "status": row.status,
                        "queue_scope": "gpu_global",
                        "queue_position": serialized["queue_position"],
                        "gpu_ids": serialized["gpu_ids"],
                        "created_at": row.created_at,
                        "started_at": row.started_at,
                        "finished_at": row.finished_at,
                        "error": row.error,
                        "detail_url": f"/jobs/{row.id}",
                        "can_delete": (
                            row.experiment_id not in non_terminal_experiment_ids
                            and (user.role == "administrator" or user.id == row.owner_id)
                        ),
                        "can_package": (
                            row.status in TERMINAL_STATUSES
                            and (user.role == "administrator" or user.id == row.owner_id)
                        ),
                        "package_run": (
                            serialize_utility(latest_packages[row.id])
                            if row.id in latest_packages
                            else None
                        ),
                    }
                )
        if kind in {None, "deployment"}:
            rows = db.execute(
                select(ModelDeployment)
                .options(selectinload(ModelDeployment.owner))
                .order_by(ModelDeployment.created_at.desc())
                .limit(limit)
            ).scalars().all()
            for row in rows:
                if not allowed("deployment", row.status):
                    continue
                records.append(
                    {
                        "id": row.id,
                        "kind": "deployment",
                        "name": row.name,
                        "owner_name": row.owner.display_name or row.owner.username,
                        "status": row.status,
                        "queue_scope": "gpu_global",
                        "queue_position": global_positions.get(f"deployment:{row.id}"),
                        "gpu_ids": row.assigned_gpu_ids or row.requested_gpu_ids,
                        "created_at": row.created_at,
                        "started_at": row.started_at,
                        "finished_at": row.finished_at,
                        "error": row.error,
                        "detail_url": f"/deployments/{row.id}",
                    }
                )
        if kind in {None, "evaluation"}:
            rows = db.execute(
                select(EvaluationRun)
                .options(selectinload(EvaluationRun.owner))
                .order_by(EvaluationRun.created_at.desc())
                .limit(limit)
            ).scalars().all()
            for row in rows:
                if not allowed("evaluation", row.status):
                    continue
                records.append(
                    {
                        "id": row.id,
                        "kind": "evaluation",
                        "name": row.name,
                        "owner_name": row.owner.display_name or row.owner.username,
                        "status": row.status,
                        "queue_scope": "gpu_global",
                        "queue_position": global_positions.get(f"evaluation:{row.id}"),
                        "gpu_ids": row.assigned_gpu_ids or row.requested_gpu_ids,
                        "created_at": row.created_at,
                        "started_at": row.started_at,
                        "finished_at": row.finished_at,
                        "error": row.error,
                        "detail_url": f"/evaluations/{row.id}",
                    }
                )
        if kind in {None, "utility"}:
            rows = db.execute(
                select(UtilityRun)
                .options(selectinload(UtilityRun.owner))
                .order_by(UtilityRun.created_at.desc())
                .limit(limit)
            ).scalars().all()
            for row in rows:
                if not allowed("utility", row.status):
                    continue
                records.append(
                    {
                        "id": row.id,
                        "kind": "utility",
                        "name": row.resource_id or row.kind,
                        "subtype": row.kind,
                        "owner_name": row.owner.display_name or row.owner.username,
                        "status": row.status,
                        "queue_scope": "gpu_global" if row.queue_class == "gpu" else "cpu_utility",
                        "queue_position": (
                            global_positions.get(f"utility:{row.id}")
                            if row.queue_class == "gpu"
                            else cpu_positions.get(row.id)
                        ),
                        "gpu_ids": row.assigned_gpu_ids or row.requested_gpu_ids,
                        "created_at": row.created_at,
                        "started_at": row.started_at,
                        "finished_at": row.finished_at,
                        "error": row.error,
                        "detail_url": "/resources",
                    }
                )
        records.sort(key=lambda item: item["created_at"], reverse=True)
        return records[:limit]

    @app.get("/api/v1/jobs/{job_id}")
    def job_detail(job_id: str, _user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
        job = _get_job(db, job_id)
        return _serialize_job(job, _job_queue_positions(db).get(job.id))

    @app.delete("/api/v1/jobs/{job_id}")
    def delete_job(
        job_id: str,
        payload: DeleteRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        job = _get_job(db, job_id)
        experiment = _get_experiment(db, job.experiment_id)
        assert_owner_or_admin(user, experiment.owner_id)
        if payload.confirmation != experiment.name:
            raise HTTPException(status_code=422, detail="confirmation_does_not_match_training_name")
        if any(
            stage_job.status not in TERMINAL_STATUSES
            for stage in experiment.stages
            for stage_job in stage.jobs
        ):
            raise HTTPException(status_code=409, detail="training_must_be_terminal_before_delete")
        add_audit(
            db,
            "training.delete",
            actor_id=user.id,
            target_type="experiment",
            target_id=experiment.id,
            detail={"job_id": job_id},
        )
        db.delete(experiment)
        return {"ok": True}

    @app.post("/api/v1/jobs/{job_id}/package")
    def package_training(
        job_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        job = _get_job(db, job_id)
        experiment = _get_experiment(db, job.experiment_id)
        assert_owner_or_admin(user, experiment.owner_id)
        if job.status not in TERMINAL_STATUSES:
            raise HTTPException(status_code=409, detail="training_must_be_terminal_before_package")
        raw_source = Path(job.output_dir)
        if raw_source.is_symlink():
            raise HTTPException(status_code=409, detail="training_output_invalid")
        try:
            source = raw_source.resolve(strict=True)
        except OSError as exc:
            raise HTTPException(status_code=404, detail="training_output_not_found") from exc
        if not source.is_dir():
            raise HTTPException(status_code=409, detail="training_output_invalid")
        configured_roots = []
        for value in SettingsService(db).get("results_roots", ["results"]):
            root = Path(str(value)).expanduser()
            if not root.is_absolute():
                root = runtime.repo_root / root
            configured_roots.append(root.resolve(strict=False))
        if not any(source != root and source.is_relative_to(root) for root in configured_roots):
            raise HTTPException(status_code=403, detail="training_output_outside_results_roots")
        run = create_package_run(
            owner_id=user.id,
            kind="training_package",
            source=source,
            label=experiment.name,
            resource_id=job.id,
            reference_key="job_id",
            reference_id=job.id,
            db=db,
        )
        add_audit(
            db,
            "training.package",
            actor_id=user.id,
            target_type="job",
            target_id=job.id,
            detail={"utility_run_id": run.id},
        )
        return serialize_utility(run)

    @app.post("/api/v1/jobs/{job_id}/cancel")
    async def cancel_job(
        job_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        job = _get_job(db, job_id)
        assert_owner_or_admin(user, job.owner_id)
        db.commit()
        try:
            await manager.cancel(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "job.cancel", actor_id=user.id, target_type="job", target_id=job_id)
        db.expire_all()
        return _serialize_job(_get_job(db, job_id))

    @app.post("/api/v1/jobs/{job_id}/stop")
    async def stop_job(
        job_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        job = _get_job(db, job_id)
        assert_owner_or_admin(user, job.owner_id)
        db.commit()
        try:
            await manager.request_stop(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "job.stop", actor_id=user.id, target_type="job", target_id=job_id)
        db.expire_all()
        return _serialize_job(_get_job(db, job_id))

    @app.post("/api/v1/jobs/{job_id}/terminate")
    async def terminate_job(
        job_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        job = _get_job(db, job_id)
        assert_owner_or_admin(user, job.owner_id)
        db.commit()
        try:
            await manager.terminate(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "job.terminate", actor_id=user.id, target_type="job", target_id=job_id)
        db.expire_all()
        return _serialize_job(_get_job(db, job_id))

    @app.post("/api/v1/jobs/{job_id}/force-kill")
    async def force_kill_job(
        job_id: str,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        _get_job(db, job_id)
        db.commit()
        try:
            await manager.force_kill(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "job.force_kill", actor_id=admin.id, target_type="job", target_id=job_id)
        db.expire_all()
        return _serialize_job(_get_job(db, job_id))

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        request: Request,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> StreamingResponse:
        job = _get_job(db, job_id)
        last_id = request.headers.get("Last-Event-ID", "0:0")
        try:
            log_offset, metrics_offset = (int(x) for x in last_id.split(":", 1))
        except (ValueError, TypeError):
            log_offset = metrics_offset = 0
        log_path, metrics_path = Path(job.log_path), Path(job.metrics_path)

        async def stream():
            nonlocal log_offset, metrics_offset
            previous_status = ""
            idle_ticks = 0
            while True:
                if await request.is_disconnected():
                    return
                emitted = False
                if log_path.exists():
                    with log_path.open("rb") as handle:
                        handle.seek(log_offset)
                        chunk = handle.read(64 * 1024)
                    if chunk:
                        log_offset += len(chunk)
                        data = json.dumps({"text": chunk.decode("utf-8", errors="replace")}, ensure_ascii=False)
                        yield f"id: {log_offset}:{metrics_offset}\nevent: log\ndata: {data}\n\n"
                        emitted = True
                if metrics_path.exists():
                    with metrics_path.open("rb") as handle:
                        handle.seek(metrics_offset)
                        raw = handle.read(128 * 1024)
                    if raw:
                        # Keep an incomplete trailing JSON line for the next poll.
                        complete = raw.rfind(b"\n")
                        if complete >= 0:
                            payload = raw[: complete + 1]
                            metrics_offset += complete + 1
                            for line in payload.splitlines():
                                try:
                                    metric = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                yield f"id: {log_offset}:{metrics_offset}\nevent: metric\ndata: {json.dumps(metric)}\n\n"
                                emitted = True
                with database.session() as event_db:
                    current = event_db.get(Job, job_id)
                    current_status = current.status if current else "deleted"
                if current_status != previous_status:
                    status_data = json.dumps({"status": current_status})
                    yield f"id: {log_offset}:{metrics_offset}\nevent: status\ndata: {status_data}\n\n"
                    previous_status = current_status
                    emitted = True
                if current_status in TERMINAL_STATUSES and not emitted:
                    yield f"event: end\ndata: {json.dumps({'status': current_status})}\n\n"
                    return
                if not emitted:
                    idle_ticks += 1
                    if idle_ticks >= 10:
                        yield ": heartbeat\n\n"
                        idle_ticks = 0
                else:
                    idle_ticks = 0
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/v1/deployments/preflight")
    def deployment_preflight(
        payload: DeploymentRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        if runtime.demo_mode:
            from .demo import demo_deployment_preflight

            return demo_deployment_preflight(payload, db)
        return validate_deployment_request(
            payload,
            db=db,
            runtime=runtime,
            gpu_monitor=gpu_monitor,
            include_experimental=_can_experimental(db, user),
            source_catalog=effective_registry_catalog(),
        )

    @app.post("/api/v1/deployments")
    def create_deployment(
        payload: DeploymentRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        if runtime.demo_mode:
            from .demo import demo_deployment_preflight

            preflight = demo_deployment_preflight(payload, db)
        else:
            preflight = validate_deployment_request(
                payload,
                db=db,
                runtime=runtime,
                gpu_monitor=gpu_monitor,
                include_experimental=_can_experimental(db, user),
                source_catalog=effective_registry_catalog(),
            )
        if not preflight["ok"]:
            raise HTTPException(
                status_code=422,
                detail={"code": "deployment_preflight_failed", "issues": preflight["issues"]},
            )
        resolved = preflight["resolved"]
        raw_key, key_hash, key_prefix = generate_api_key()
        deployment_id = new_id()
        endpoint = resolved["endpoint"]
        resources = resolved["resources"]
        parameters = dict(resolved["parameters"])
        parameters["_startup_timeout_seconds"] = int(resolved.get("startup_timeout_seconds", 900))
        deployment = ModelDeployment(
            id=deployment_id,
            owner_id=user.id,
            checkpoint_id=resolved.get("checkpoint_id"),
            name=payload.name.strip(),
            checkpoint_path=str(resolved["checkpoint_path"]),
            combination_id=str(resolved["combination_id"]),
            adapter_id=str(resolved["adapter_id"]),
            backbone_id=str(resolved["backbone_id"]),
            action_head_id=str(resolved["action_head_id"]),
            status="queued",
            requested_gpu_count=int(resources["gpu_count"]),
            requested_gpu_ids=list(resources["gpu_ids"]),
            assigned_gpu_ids=[],
            parameters=parameters,
            endpoint_scope=str(endpoint["scope"]),
            bind_host=str(endpoint["bind_host"]),
            advertised_host=str(endpoint["advertised_host"]),
            port=endpoint.get("port"),
            idle_timeout_seconds=int(endpoint["idle_timeout_seconds"]),
            api_key_hash=key_hash,
            api_key_prefix=key_prefix,
            command=[],
            environment={str(key): str(value) for key, value in resolved.get("environment", {}).items()},
            log_path=str(runtime.state_dir / "logs" / f"deployment-{deployment_id}.log"),
        )
        db.add(deployment)
        db.flush()
        try:
            deployment_manager.create_controller_key(deployment.id)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="deployment_controller_key_creation_failed") from exc
        add_audit(
            db,
            "deployment.create",
            actor_id=user.id,
            target_type="deployment",
            target_id=deployment.id,
            detail={
                "checkpoint_id": deployment.checkpoint_id,
                "combination_id": deployment.combination_id,
                "adapter_id": deployment.adapter_id,
                "endpoint_scope": deployment.endpoint_scope,
                "requested_gpu_count": deployment.requested_gpu_count,
            },
        )
        return {
            "deployment": _serialize_deployment(
                deployment,
                _deployment_queue_positions(db).get(deployment.id),
                deployment_manager.controller_key_configured(deployment.id),
            ),
            "api_key": raw_key,
        }

    @app.get("/api/v1/deployments")
    def list_deployments(
        deployment_status: str | None = Query(None, alias="status"),
        limit: int = Query(200, ge=1, le=1000),
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        statement = (
            select(ModelDeployment)
            .options(selectinload(ModelDeployment.owner))
            .order_by(ModelDeployment.queued_at.desc())
            .limit(limit)
        )
        if deployment_status:
            statement = statement.where(ModelDeployment.status == deployment_status)
        positions = _deployment_queue_positions(db)
        return [
            _serialize_deployment(
                row,
                positions.get(row.id),
                deployment_manager.controller_key_configured(row.id),
            )
            for row in db.execute(statement).scalars().all()
        ]

    @app.get("/api/v1/deployments/{deployment_id}")
    def deployment_detail(
        deployment_id: str,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        deployment = _get_deployment(db, deployment_id)
        return _serialize_deployment(
            deployment,
            _deployment_queue_positions(db).get(deployment.id),
            deployment_manager.controller_key_configured(deployment.id),
        )

    @app.post("/api/v1/deployments/{deployment_id}/infer")
    async def infer_with_deployment(
        deployment_id: str,
        instructions: str = Form(...),
        states: str | None = Form(default=None),
        image_counts: str | None = Form(default=None),
        save_inputs: bool = Form(default=False),
        images: list[UploadFile] | None = File(default=None),
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        deployment = _get_deployment(db, deployment_id)
        if deployment.status != "running" or not deployment.port:
            raise HTTPException(status_code=409, detail="deployment_not_ready_for_inference")
        controller_key = deployment_manager.controller_key(deployment.id)
        if not controller_key:
            raise HTTPException(status_code=409, detail="deployment_controller_unavailable")
        try:
            decoded = await decode_inference_request(
                instructions=instructions,
                states=states,
                image_counts=image_counts,
                images=images or [],
            )
        except InferenceInputError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

        run = InferenceRun(
            id=new_id(),
            request_id=decoded.request_id,
            owner_id=user.id,
            deployment_id=deployment.id,
            status="running",
            batch_size=decoded.batch_size,
            image_count=len(decoded.images),
            save_inputs=save_inputs,
            request_summary=decoded.summary,
            deployment_metadata={},
            output_json={},
            input_paths=[],
        )
        db.add(run)
        db.flush()
        if save_inputs:
            try:
                run.input_paths = save_inference_inputs(runtime.state_dir, run.id, decoded.images)
            except OSError as exc:
                run.status = "failed"
                run.error = "inference_input_save_failed"
                run.finished_at = utcnow()
                db.commit()
                raise HTTPException(status_code=507, detail="inference_input_save_failed") from exc
        db.commit()

        async def operation():  # type: ignore[no-untyped-def]
            return await infer_via_websocket(
                port=int(deployment.port),
                controller_key=controller_key,
                request=decoded,
            )

        try:
            output, metadata, latency_ms = await inference_coordinator.run(deployment.id, operation)
        except InferenceInputError as exc:
            current = db.get(InferenceRun, run.id)
            if current:
                current.status = "failed"
                current.error = exc.code
                current.finished_at = utcnow()
                db.commit()
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
        except InferenceTransportError as exc:
            current = db.get(InferenceRun, run.id)
            if current:
                current.status = "timed_out" if exc.code == "deployment_inference_timeout" else "failed"
                current.error = exc.code
                current.finished_at = utcnow()
                db.commit()
            raise HTTPException(
                status_code=504 if exc.code == "deployment_inference_timeout" else 502,
                detail=exc.code,
            ) from exc

        current = db.get(InferenceRun, run.id)
        assert current is not None
        current.status = "completed"
        current.output_json = output
        current.deployment_metadata = metadata
        current.latency_ms = latency_ms
        current.finished_at = utcnow()
        add_audit(
            db,
            "deployment.infer",
            actor_id=user.id,
            target_type="deployment",
            target_id=deployment.id,
            detail={
                "inference_run_id": current.id,
                "batch_size": current.batch_size,
                "image_count": current.image_count,
                "save_inputs": current.save_inputs,
            },
        )
        db.commit()
        return _serialize_inference_run(current)

    @app.get("/api/v1/inference-runs")
    def list_inference_runs(
        deployment_id: str | None = None,
        limit: int = Query(100, ge=1, le=500),
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        statement = select(InferenceRun).order_by(InferenceRun.created_at.desc()).limit(limit)
        if deployment_id:
            statement = statement.where(InferenceRun.deployment_id == deployment_id)
        return [
            _serialize_inference_run(row, include_output=False)
            for row in db.execute(statement).scalars().all()
        ]

    @app.get("/api/v1/inference-runs/{run_id}")
    def inference_run_detail(
        run_id: str,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        run = db.get(InferenceRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="inference_run_not_found")
        return _serialize_inference_run(run)

    @app.delete("/api/v1/deployments/{deployment_id}")
    def delete_deployment(
        deployment_id: str,
        payload: DeleteRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        deployment = _get_deployment(db, deployment_id)
        assert_owner_or_admin(user, deployment.owner_id)
        if payload.confirmation != deployment.name:
            raise HTTPException(status_code=422, detail="confirmation_does_not_match_deployment_name")
        if deployment.status not in DEPLOYMENT_TERMINAL_STATUSES:
            raise HTTPException(status_code=409, detail="deployment_must_be_terminal_before_delete")
        add_audit(
            db,
            "deployment.delete",
            actor_id=user.id,
            target_type="deployment",
            target_id=deployment.id,
        )
        deployment_manager.delete_controller_key(deployment.id)
        db.delete(deployment)
        return {"ok": True}

    @app.post("/api/v1/deployments/{deployment_id}/cancel")
    async def cancel_deployment(
        deployment_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        deployment = _get_deployment(db, deployment_id)
        assert_owner_or_admin(user, deployment.owner_id)
        # Lab-mode authentication updates the session's last_seen_at value.
        # Commit it before the manager opens its own SQLite write session.
        db.commit()
        try:
            await deployment_manager.cancel(deployment_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "deployment.cancel", actor_id=user.id, target_type="deployment", target_id=deployment_id)
        db.commit()
        db.expire_all()
        return _serialize_deployment(_get_deployment(db, deployment_id))

    @app.post("/api/v1/deployments/{deployment_id}/stop")
    async def stop_deployment(
        deployment_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        deployment = _get_deployment(db, deployment_id)
        assert_owner_or_admin(user, deployment.owner_id)
        db.commit()
        try:
            await deployment_manager.stop(deployment_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "deployment.stop", actor_id=user.id, target_type="deployment", target_id=deployment_id)
        db.commit()
        db.expire_all()
        return _serialize_deployment(_get_deployment(db, deployment_id))

    @app.post("/api/v1/deployments/{deployment_id}/restart")
    async def restart_deployment(
        deployment_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        deployment = _get_deployment(db, deployment_id)
        assert_owner_or_admin(user, deployment.owner_id)
        db.commit()
        try:
            await deployment_manager.restart(deployment_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "deployment.restart", actor_id=user.id, target_type="deployment", target_id=deployment_id)
        db.commit()
        db.expire_all()
        return {"deployment": _serialize_deployment(_get_deployment(db, deployment_id))}

    @app.post("/api/v1/deployments/{deployment_id}/rotate-key")
    async def rotate_deployment_key(
        deployment_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        deployment = _get_deployment(db, deployment_id)
        assert_owner_or_admin(user, deployment.owner_id)
        db.commit()
        try:
            _deployment, raw_key = await deployment_manager.rotate_key(deployment_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "deployment.rotate_key", actor_id=user.id, target_type="deployment", target_id=deployment_id)
        db.commit()
        db.expire_all()
        return {"deployment": _serialize_deployment(_get_deployment(db, deployment_id)), "api_key": raw_key}

    @app.post("/api/v1/deployments/{deployment_id}/force-kill")
    async def force_kill_deployment(
        deployment_id: str,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        _get_deployment(db, deployment_id)
        db.commit()
        try:
            await deployment_manager.stop(deployment_id, force=True)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "deployment.force_kill", actor_id=admin.id, target_type="deployment", target_id=deployment_id)
        db.expire_all()
        return _serialize_deployment(_get_deployment(db, deployment_id))

    @app.get("/api/v1/deployments/{deployment_id}/events")
    async def deployment_events(
        deployment_id: str,
        request: Request,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> StreamingResponse:
        deployment = _get_deployment(db, deployment_id)
        cursor = request.headers.get("Last-Event-ID") or request.query_params.get("cursor") or "0"
        try:
            log_offset = max(0, int(cursor))
        except (TypeError, ValueError):
            log_offset = 0
        log_path = Path(deployment.log_path)

        async def stream():
            nonlocal log_offset
            previous_status = ""
            idle_ticks = 0
            while True:
                if await request.is_disconnected():
                    return
                emitted = False
                if log_path.exists():
                    try:
                        size = log_path.stat().st_size
                        if log_offset > size:
                            log_offset = 0
                        with log_path.open("rb") as handle:
                            handle.seek(log_offset)
                            chunk = handle.read(64 * 1024)
                    except OSError:
                        chunk = b""
                    if chunk:
                        log_offset += len(chunk)
                        data = json.dumps(
                            {"type": "log", "line": chunk.decode("utf-8", errors="replace")},
                            ensure_ascii=False,
                        )
                        yield f"id: {log_offset}\nevent: log\ndata: {data}\n\n"
                        emitted = True
                with database.session() as event_db:
                    current = event_db.get(ModelDeployment, deployment_id)
                    current_status = current.status if current else "deleted"
                if current_status != previous_status:
                    data = json.dumps({"type": "status", "status": current_status})
                    yield f"id: {log_offset}\nevent: status\ndata: {data}\n\n"
                    previous_status = current_status
                    emitted = True
                if current_status in DEPLOYMENT_TERMINAL_STATUSES | {"deleted"} and not emitted:
                    data = json.dumps({"type": "end", "status": current_status})
                    yield f"event: end\ndata: {data}\n\n"
                    return
                if emitted:
                    idle_ticks = 0
                else:
                    idle_ticks += 1
                    if idle_ticks >= 10:
                        yield ": heartbeat\n\n"
                        idle_ticks = 0
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/evaluation-benchmarks")
    def evaluation_benchmarks(
        include_experimental: bool | None = Query(default=None),
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        experimental_allowed = _can_experimental(db, user)
        include = experimental_allowed and include_experimental is not False
        catalog = evaluate_catalog_readiness(
            get_evaluation_catalog(
                include_experimental=include,
                source_catalog=effective_registry_catalog(),
            ),
            SettingsService(db),
            runtime.repo_root,
        )
        return {
            "catalog": catalog,
            "benchmarks": catalog.get("benchmarks", []),
            "presets": catalog.get("presets", []),
            "combinations": catalog.get("combinations", []),
            "include_experimental": include,
            "experimental_allowed": experimental_allowed,
        }

    @app.post("/api/v1/evaluations/inspect-checkpoint")
    def inspect_evaluation_checkpoint_endpoint(
        payload: EvaluationCheckpointInspectionRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        """Statically detect the wired evaluation combinations for a checkpoint.

        This endpoint reads only small configuration/statistics files.  Model
        weights are never imported or deserialized in the UI process.
        """

        if runtime.demo_mode:
            from .demo import demo_checkpoint_inspection

            return demo_checkpoint_inspection(payload.checkpoint_id)
        settings = SettingsService(db)
        checkpoint_index = [
            {
                "id": row.id,
                "path": row.path,
                "name": row.name,
                "is_complete": row.is_complete,
            }
            for row in db.execute(select(Checkpoint)).scalars().all()
        ]
        inspection = inspect_evaluation_checkpoint(
            payload.model_dump(exclude={"combination_id"}, exclude_none=True),
            checkpoint_index=checkpoint_index,
            repo_root=runtime.repo_root,
            environment=effective_environment(
                runtime.repo_root,
                settings.get("environment", {}),
            ),
            combination_id=payload.combination_id,
            include_experimental=_can_experimental(db, user),
            source_catalog=effective_registry_catalog(),
        )
        issues = list(inspection.get("issues", []))
        if payload.kind == "indexed":
            row = db.get(Checkpoint, str(payload.checkpoint_id))
            if row is None:
                issues.append(
                    {
                        "level": "error",
                        "severity": "error",
                        "code": "checkpoint_index_not_found",
                        "field": "checkpoint_source.checkpoint_id",
                        "path": "checkpoint_source.checkpoint_id",
                        "message": "索引中的 Checkpoint 不存在。",
                        "message_i18n": {
                            "zh-CN": "索引中的 Checkpoint 不存在。",
                            "en-US": "The indexed checkpoint does not exist.",
                        },
                        "detail": {},
                    }
                )
            elif not row.is_complete:
                issues.append(
                    {
                        "level": "error",
                        "severity": "error",
                        "code": "checkpoint_incomplete",
                        "field": "checkpoint_source.checkpoint_id",
                        "path": "checkpoint_source.checkpoint_id",
                        "message": "Checkpoint 尚未标记为完整，不能用于评测。",
                        "message_i18n": {
                            "zh-CN": "Checkpoint 尚未标记为完整，不能用于评测。",
                            "en-US": "The checkpoint is not marked complete and cannot be evaluated.",
                        },
                        "detail": {},
                    }
                )
        has_errors = any(
            str(item.get("level") or item.get("severity")) == "error"
            for item in issues
        )
        return {
            "ok": bool(inspection.get("evaluation_candidates")) and not has_errors,
            "detected": inspection.get("detected", {}),
            "candidates": inspection.get("evaluation_candidates", []),
            "compatible_benchmark_ids": inspection.get("compatible_benchmark_ids", []),
            "issues": issues,
        }

    def resolve_evaluation(payload: EvaluationRequest, user: User, db: Session) -> dict[str, Any]:
        if runtime.demo_mode and payload.kind == "standard":
            from .demo import demo_evaluation_preflight

            return demo_evaluation_preflight(runtime, payload, db)
        children = expand_evaluation_request(payload)
        resolved_children: list[dict[str, Any]] = []
        child_preflights: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        for child in children:
            child_preflight = validate_evaluation_request(
                child.request,
                db=db,
                runtime=runtime,
                gpu_monitor=gpu_monitor,
                include_experimental=_can_experimental(db, user),
                wandb_configured=wandb_secret_store.status()["configured"],
                controller_secret_store=deployment_manager.controller_secret_store,
                source_catalog=effective_registry_catalog(),
            )
            child_preflights.append(child_preflight)
            for item in child_preflight.get("issues", []):
                issue = dict(item)
                issue.setdefault("detail", {})
                issue["detail"] = {
                    **dict(issue.get("detail") or {}),
                    "child_position": child.position,
                    "child_kind": child.evaluation_kind,
                    "child_name": child.name_suffix,
                }
                issues.append(issue)
            resolved_children.append(
                {
                    "position": child.position,
                    "evaluation_kind": child.evaluation_kind,
                    "name_suffix": child.name_suffix,
                    "request": child.request.model_dump(mode="json"),
                    "resolved": child_preflight.get("resolved", {}),
                    "ok": bool(child_preflight.get("ok")),
                }
            )
        if payload.kind == "standard" and len(resolved_children) == 1:
            # Preserve the original endpoint contract for existing clients.
            return child_preflights[0]
        can_submit = bool(resolved_children) and all(bool(child["ok"]) for child in resolved_children)
        output_root = next(
            (
                child["resolved"].get("output_root")
                for child in resolved_children
                if child["resolved"].get("output_root")
            ),
            str(runtime.repo_root / "results/evaluation"),
        )
        return {
            "ok": can_submit,
            "can_submit": can_submit,
            "compatibility": "experimental"
            if any(item.get("code", "").startswith("experimental_") for item in issues)
            else "verified",
            "items": issues,
            "issues": issues,
            "resolved": {
                "group_kind": payload.kind,
                "children": resolved_children,
                "output_root": output_root,
            },
            "command_preview": [
                child["resolved"].get("command", []) for child in resolved_children
            ],
            "diff": {},
        }

    @app.post("/api/v1/evaluations/preflight")
    def evaluation_preflight(
        payload: EvaluationRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        return resolve_evaluation(payload, user, db)

    @app.post("/api/v1/evaluations")
    def create_evaluation(
        payload: EvaluationRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        preflight = resolve_evaluation(payload, user, db)
        if not preflight.get("ok"):
            raise HTTPException(
                status_code=422,
                detail={"code": "evaluation_preflight_failed", "issues": preflight.get("issues", [])},
            )
        resolved = dict(preflight["resolved"])
        if payload.kind != "standard":
            group_id = new_id()
            output_root = Path(str(resolved["output_root"])).expanduser().resolve(strict=False)
            if output_root.name != "ui":
                output_root /= "ui"
            group_slug = re.sub(r"[^a-z0-9]+", "-", payload.name.lower()).strip("-")[:48] or "evaluation"
            group_root = output_root / "groups" / utcnow().strftime("%Y%m%d") / f"{group_id}-{group_slug}"
            group = EvaluationGroup(
                id=group_id,
                owner_id=user.id,
                name=payload.name.strip(),
                kind=payload.kind,
                status="queued",
                spec=payload.model_dump(mode="json"),
                result_summary={},
                result_path=str(group_root / "evaluation-result-v2.json"),
                error="",
            )
            db.add(group)
            db.flush()
            evaluations: list[EvaluationRun] = []
            for child in resolved.get("children", []):
                child_resolved = dict(child.get("resolved") or {})
                position = int(child.get("position", len(evaluations)))
                evaluation_kind = str(child.get("evaluation_kind") or "standard")
                suffix = str(child.get("name_suffix") or "").strip()
                evaluation_id = new_id()
                child_slug = re.sub(r"[^a-z0-9]+", "-", suffix.lower()).strip("-")[:40]
                output_dir = group_root / f"{position:03d}-{child_slug or evaluation_kind}"
                wandb_config = dict(child_resolved.get("wandb", {}))
                result_schema_version = (
                    "evaluation-result-v2"
                    if evaluation_kind in {
                        "cl_matrix", "rl_iterations", "online_stdp_baseline",
                        "online_stdp_adapted", "world_model_video",
                    }
                    else "evaluation-result-v1"
                )
                resources = dict(child_resolved.get("resources") or {})
                evaluation = EvaluationRun(
                    id=evaluation_id,
                    group_id=group.id,
                    position=position,
                    evaluation_kind=evaluation_kind,
                    source_kind=payload.source_kind,
                    deployment_id=payload.deployment_id,
                    result_schema_version=result_schema_version,
                    owner_id=user.id,
                    checkpoint_id=child_resolved.get("checkpoint_id"),
                    name=f"{payload.name.strip()}{f' · {suffix}' if suffix else ''}",
                    checkpoint_path=str(child_resolved.get("checkpoint_path") or ""),
                    combination_id=str(child_resolved.get("combination_id") or payload.combination_id or ""),
                    adapter_id=str(child_resolved.get("adapter_id") or ""),
                    backbone_id=str(child_resolved.get("backbone_id") or ""),
                    action_head_id=str(child_resolved.get("action_head_id") or ""),
                    benchmark_id=str(child_resolved.get("benchmark_id") or payload.benchmark_id),
                    benchmark_status=str(child_resolved.get("benchmark_status", "verified")),
                    preset=str(child_resolved.get("preset", payload.preset)),
                    suite=str(child_resolved.get("suite", "")),
                    task_set=str(child_resolved.get("task_set", "")),
                    split=str(child_resolved.get("split", "")),
                    parameters=dict(child_resolved.get("parameters", {})),
                    model_parameters=dict(child_resolved.get("server_parameters", {})),
                    wandb=wandb_config,
                    wandb_status="pending" if wandb_config.get("enabled") else "disabled",
                    status="queued",
                    requested_gpu_count=int(resources.get("gpu_count", 1)),
                    requested_gpu_ids=list(resources.get("gpu_ids", [])),
                    assigned_gpu_ids=[],
                    command=[],
                    environment={str(key): str(value) for key, value in child_resolved.get("environment", {}).items()},
                    cwd=str(runtime.repo_root),
                    output_dir=str(output_dir),
                    config_path=str(output_dir / "evaluation-config.yaml"),
                    log_path=str(output_dir / "runner.log"),
                    progress_path=str(output_dir / "progress.jsonl"),
                    result_path=str(output_dir / f"{result_schema_version}.json"),
                )
                db.add(evaluation)
                evaluations.append(evaluation)
            try:
                db.flush()
            except IntegrityError as exc:
                raise HTTPException(status_code=409, detail="evaluation_output_directory_reserved") from exc
            add_audit(
                db,
                "evaluation_group.create",
                actor_id=user.id,
                target_type="evaluation_group",
                target_id=group.id,
                detail={"kind": group.kind, "run_count": len(evaluations)},
            )
            return {
                "group": _serialize_evaluation_group(db, group),
                "evaluations": [_serialize_evaluation(row) for row in evaluations],
                # Backwards-compatible navigation target for older frontends.
                "evaluation": _serialize_evaluation(evaluations[0]),
                "issues": [item for item in preflight.get("issues", []) if item.get("level") != "error"],
            }
        evaluation_id = new_id()
        output_root = Path(str(resolved["output_root"])).expanduser().resolve(strict=False)
        if output_root.name != "ui":
            output_root /= "ui"
        slug = re.sub(r"[^a-z0-9]+", "-", payload.name.lower()).strip("-")[:48] or "evaluation"
        output_dir = output_root / utcnow().strftime("%Y%m%d") / f"{evaluation_id}-{slug}"
        wandb_config = dict(resolved.get("wandb", {}))
        evaluation = EvaluationRun(
            id=evaluation_id,
            group_id=None,
            position=0,
            evaluation_kind="standard",
            source_kind=payload.source_kind,
            deployment_id=payload.deployment_id,
            result_schema_version="evaluation-result-v1",
            owner_id=user.id,
            checkpoint_id=resolved.get("checkpoint_id"),
            name=payload.name.strip(),
            checkpoint_path=str(resolved["checkpoint_path"]),
            combination_id=str(resolved["combination_id"]),
            adapter_id=str(resolved["adapter_id"]),
            backbone_id=str(resolved.get("backbone_id") or ""),
            action_head_id=str(resolved.get("action_head_id") or ""),
            benchmark_id=str(resolved["benchmark_id"]),
            benchmark_status=str(resolved.get("benchmark_status", "verified")),
            preset=str(resolved.get("preset", payload.preset)),
            suite=str(resolved.get("suite", "")),
            task_set=str(resolved.get("task_set", "")),
            split=str(resolved.get("split", "")),
            parameters=dict(resolved.get("parameters", {})),
            model_parameters=dict(resolved.get("server_parameters", {})),
            wandb=wandb_config,
            wandb_status="pending" if wandb_config.get("enabled") else "disabled",
            status="queued",
            requested_gpu_count=int(resolved.get("resources", {}).get("gpu_count", 1)),
            requested_gpu_ids=list(resolved.get("resources", {}).get("gpu_ids", [])),
            assigned_gpu_ids=[],
            command=[],
            environment={str(key): str(value) for key, value in resolved.get("environment", {}).items()},
            cwd=str(runtime.repo_root),
            output_dir=str(output_dir),
            config_path=str(output_dir / "evaluation-config.yaml"),
            log_path=str(output_dir / "runner.log"),
            progress_path=str(output_dir / "progress.jsonl"),
            result_path=str(output_dir / "evaluation-result-v1.json"),
        )
        db.add(evaluation)
        try:
            db.flush()
        except IntegrityError as exc:
            raise HTTPException(status_code=409, detail="evaluation_output_directory_reserved") from exc
        add_audit(
            db,
            "evaluation.create",
            actor_id=user.id,
            target_type="evaluation",
            target_id=evaluation.id,
            detail={
                "checkpoint_id": evaluation.checkpoint_id,
                "combination_id": evaluation.combination_id,
                "benchmark_id": evaluation.benchmark_id,
                "preset": evaluation.preset,
            },
        )
        return {
            "evaluation": _serialize_evaluation(
                evaluation,
                _evaluation_queue_positions(db).get(evaluation.id),
            ),
            "issues": [item for item in preflight.get("issues", []) if item.get("level") != "error"],
        }

    @app.get("/api/v1/evaluations")
    def list_evaluations(
        evaluation_status: str | None = Query(None, alias="status"),
        benchmark_id: str | None = Query(None),
        limit: int = Query(200, ge=1, le=1000),
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        statement = (
            select(EvaluationRun)
            .options(selectinload(EvaluationRun.owner))
            .order_by(EvaluationRun.queued_at.desc())
            .limit(limit)
        )
        if evaluation_status:
            statement = statement.where(EvaluationRun.status == evaluation_status)
        if benchmark_id:
            statement = statement.where(EvaluationRun.benchmark_id == benchmark_id)
        positions = _evaluation_queue_positions(db)
        return [
            _serialize_evaluation(row, positions.get(row.id))
            for row in db.execute(statement).scalars().all()
        ]

    @app.get("/api/v1/evaluation-groups")
    def list_evaluation_groups(
        group_status: str | None = Query(None, alias="status"),
        kind: str | None = Query(None),
        limit: int = Query(200, ge=1, le=1000),
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        statement = (
            select(EvaluationGroup)
            .options(selectinload(EvaluationGroup.owner))
            .order_by(EvaluationGroup.created_at.desc())
            .limit(limit)
        )
        if group_status:
            statement = statement.where(EvaluationGroup.status == group_status)
        if kind:
            statement = statement.where(EvaluationGroup.kind == kind)
        return [
            _serialize_evaluation_group(db, group)
            for group in db.execute(statement).scalars().all()
        ]

    @app.get("/api/v1/evaluation-groups/{group_id}")
    def evaluation_group_detail(
        group_id: str,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        return _serialize_evaluation_group(
            db, _get_evaluation_group(db, group_id), include_result=True
        )

    @app.get("/api/v1/evaluation-groups/{group_id}/result")
    def evaluation_group_result(
        group_id: str,
        download: bool = Query(False),
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ):
        group = _get_evaluation_group(db, group_id)
        refresh_evaluation_group(db, group.id)
        result = _load_evaluation_group_result(group)
        if result is None:
            raise HTTPException(status_code=404, detail="evaluation_group_result_not_available")
        if download:
            return FileResponse(
                Path(group.result_path).resolve(strict=True),
                media_type="application/json",
                filename="evaluation-result-v2.json",
            )
        return result

    @app.post("/api/v1/evaluation-groups/{group_id}/cancel")
    async def cancel_evaluation_group(
        group_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        group = _get_evaluation_group(db, group_id)
        assert_owner_or_admin(user, group.owner_id)
        queued_ids = list(
            db.execute(
                select(EvaluationRun.id).where(
                    EvaluationRun.group_id == group.id,
                    EvaluationRun.status == "queued",
                )
            ).scalars()
        )
        db.commit()
        for evaluation_id in queued_ids:
            with contextlib.suppress(ValueError, KeyError):
                await evaluation_manager.cancel(evaluation_id)
        db.expire_all()
        current = _get_evaluation_group(db, group_id)
        add_audit(
            db,
            "evaluation_group.cancel",
            actor_id=user.id,
            target_type="evaluation_group",
            target_id=group_id,
        )
        return _serialize_evaluation_group(db, current, include_result=True)

    @app.post("/api/v1/evaluations/compare")
    def compare_evaluations(
        payload: EvaluationCompareRequest,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        rows_by_id = {
            row.id: row
            for row in db.execute(
                select(EvaluationRun)
                .where(EvaluationRun.id.in_(payload.evaluation_ids))
                .options(selectinload(EvaluationRun.owner))
            ).scalars()
        }
        if len(rows_by_id) != len(payload.evaluation_ids):
            raise HTTPException(status_code=404, detail="evaluation_not_found")
        rows = [rows_by_id[value] for value in payload.evaluation_ids]
        if any(row.status != "completed" for row in rows):
            raise HTTPException(status_code=409, detail="evaluation_not_completed")
        results = [_load_evaluation_result(row) for row in rows]
        if any(value is None for value in results):
            raise HTTPException(status_code=409, detail="evaluation_result_not_available")
        if any(value.get("status") != "completed" for value in results if value is not None):
            raise HTTPException(status_code=409, detail="evaluation_not_completed")

        def task_signature(row: EvaluationRun) -> tuple[str, str, str, str, str, str, int]:
            parameters = dict(row.parameters or {})
            try:
                task_limit = int(parameters.get("task_limit") or 0)
            except (TypeError, ValueError):
                task_limit = 0
            return (
                row.benchmark_id,
                row.suite,
                row.task_set,
                row.split,
                str(parameters.get("task_ids") or "").strip(),
                str(parameters.get("task_list") or "").strip(),
                task_limit,
            )

        signatures = {
            task_signature(row)
            for row in rows
        }
        compatible = len(signatures) == 1
        signature = ""
        if compatible:
            benchmark_id, suite, task_set, split, task_ids, task_list, task_limit = next(
                iter(signatures)
            )
            signature = " | ".join(
                (
                    benchmark_id,
                    suite,
                    task_set,
                    split,
                    f"task_ids={task_ids}",
                    f"task_list={task_list}",
                    f"task_limit={task_limit}",
                )
            )
        warnings: list[str] = []
        if not compatible:
            warnings.append("signature_mismatch")
        if len({json.dumps(row.parameters or {}, sort_keys=True) for row in rows}) > 1:
            warnings.append("parameters_differ")
        episode_counts = {
            int((result or {}).get("summary", {}).get("num_episodes", 0))
            for result in results
        }
        if len(episode_counts) > 1:
            warnings.append("episode_counts_differ")
        return {
            "benchmark_id": rows[0].benchmark_id if compatible else "",
            "signature": signature,
            "compatible": compatible,
            "warnings": warnings,
            "items": [
                {
                    "evaluation": _serialize_evaluation(row),
                    "result": result,
                }
                for row, result in zip(rows, results, strict=True)
            ],
        }

    @app.get("/api/v1/evaluations/{evaluation_id}")
    def evaluation_detail(
        evaluation_id: str,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        evaluation = _get_evaluation(db, evaluation_id)
        return _serialize_evaluation(
            evaluation,
            _evaluation_queue_positions(db).get(evaluation.id),
            include_result=True,
            include_artifacts=True,
        )

    @app.post("/api/v1/evaluations/{evaluation_id}/cancel")
    async def cancel_evaluation(
        evaluation_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        evaluation = _get_evaluation(db, evaluation_id)
        assert_owner_or_admin(user, evaluation.owner_id)
        db.commit()
        try:
            await evaluation_manager.cancel(evaluation_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "evaluation.cancel", actor_id=user.id, target_type="evaluation", target_id=evaluation_id)
        db.expire_all()
        return _serialize_evaluation(_get_evaluation(db, evaluation_id))

    @app.post("/api/v1/evaluations/{evaluation_id}/stop")
    async def stop_evaluation(
        evaluation_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        evaluation = _get_evaluation(db, evaluation_id)
        assert_owner_or_admin(user, evaluation.owner_id)
        db.commit()
        try:
            await evaluation_manager.stop(evaluation_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(db, "evaluation.stop", actor_id=user.id, target_type="evaluation", target_id=evaluation_id)
        db.expire_all()
        return _serialize_evaluation(_get_evaluation(db, evaluation_id))

    @app.post("/api/v1/evaluations/{evaluation_id}/force-kill")
    async def force_kill_evaluation(
        evaluation_id: str,
        admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        _get_evaluation(db, evaluation_id)
        db.commit()
        try:
            await evaluation_manager.stop(evaluation_id, force=True)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(
            db,
            "evaluation.force_kill",
            actor_id=admin.id,
            target_type="evaluation",
            target_id=evaluation_id,
        )
        db.expire_all()
        return _serialize_evaluation(_get_evaluation(db, evaluation_id))

    @app.post("/api/v1/evaluations/{evaluation_id}/wandb/retry")
    async def retry_evaluation_wandb(
        evaluation_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        evaluation = _get_evaluation(db, evaluation_id)
        assert_owner_or_admin(user, evaluation.owner_id)
        db.commit()
        try:
            evaluation_manager.request_upload(evaluation_id, wandb_secret_store)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        add_audit(
            db,
            "evaluation.wandb_retry",
            actor_id=user.id,
            target_type="evaluation",
            target_id=evaluation_id,
        )
        db.expire_all()
        return _serialize_evaluation(_get_evaluation(db, evaluation_id), include_result=True)

    @app.get("/api/v1/evaluations/{evaluation_id}/result")
    def evaluation_result(
        evaluation_id: str,
        download: bool = Query(False),
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ):
        evaluation = _get_evaluation(db, evaluation_id)
        result = _load_evaluation_result(evaluation)
        if result is None:
            raise HTTPException(status_code=404, detail="evaluation_result_not_available")
        if download:
            path = Path(evaluation.result_path).resolve(strict=True)
            return FileResponse(path, media_type="application/json", filename=path.name)
        return result

    @app.get("/api/v1/evaluations/{evaluation_id}/artifacts")
    def evaluation_artifacts(
        evaluation_id: str,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        return _evaluation_artifacts(_get_evaluation(db, evaluation_id))

    @app.get("/api/v1/evaluations/{evaluation_id}/artifacts/{artifact_id}")
    def evaluation_artifact(
        evaluation_id: str,
        artifact_id: str,
        download: bool = Query(False),
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> FileResponse:
        evaluation = _get_evaluation(db, evaluation_id)
        try:
            path = _evaluation_artifact_path(evaluation, artifact_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path,
            filename=path.name if download else None,
            content_disposition_type="attachment" if download else "inline",
        )

    @app.get("/api/v1/evaluations/{evaluation_id}/events")
    async def evaluation_events(
        evaluation_id: str,
        request: Request,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> StreamingResponse:
        evaluation = _get_evaluation(db, evaluation_id)
        cursor = request.headers.get("Last-Event-ID") or request.query_params.get("cursor") or "0:0"
        try:
            log_offset, progress_offset = (max(0, int(value)) for value in cursor.split(":", 1))
        except (TypeError, ValueError):
            log_offset = progress_offset = 0
        log_path = Path(evaluation.log_path)
        progress_path = Path(evaluation.progress_path)

        async def stream():
            nonlocal log_offset, progress_offset
            previous_status = ""
            idle_ticks = 0
            while True:
                if await request.is_disconnected():
                    return
                emitted = False
                if log_path.exists():
                    try:
                        size = log_path.stat().st_size
                        if log_offset > size:
                            log_offset = 0
                        with log_path.open("rb") as handle:
                            handle.seek(log_offset)
                            chunk = handle.read(64 * 1024)
                    except OSError:
                        chunk = b""
                    if chunk:
                        log_offset += len(chunk)
                        data = json.dumps(
                            {"type": "log", "line": chunk.decode("utf-8", errors="replace")},
                            ensure_ascii=False,
                        )
                        yield f"id: {log_offset}:{progress_offset}\nevent: log\ndata: {data}\n\n"
                        emitted = True
                if progress_path.exists():
                    try:
                        size = progress_path.stat().st_size
                        if progress_offset > size:
                            progress_offset = 0
                        with progress_path.open("rb") as handle:
                            handle.seek(progress_offset)
                            raw = handle.read(64 * 1024)
                    except OSError:
                        raw = b""
                    complete = raw.rfind(b"\n")
                    if complete >= 0:
                        payload = raw[: complete + 1]
                        progress_offset += complete + 1
                        for line in payload.splitlines():
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            phase = str(event.get("stage", "")) if isinstance(event, dict) else ""
                            progress_value = {
                                "server_start": 0.05,
                                "server_ready": 0.15,
                                "simulation": 0.2,
                                "aggregation": 0.95,
                                "evaluation": 1.0,
                            }.get(phase, 0.2)
                            data = json.dumps(
                                {"type": "progress", "phase": phase, "progress": progress_value, "data": event},
                                ensure_ascii=False,
                            )
                            yield f"id: {log_offset}:{progress_offset}\nevent: progress\ndata: {data}\n\n"
                            emitted = True
                with database.session() as event_db:
                    current = event_db.get(EvaluationRun, evaluation_id)
                    current_status = current.status if current else "deleted"
                if current_status != previous_status:
                    data = json.dumps({"type": "status", "status": current_status})
                    yield f"id: {log_offset}:{progress_offset}\nevent: status\ndata: {data}\n\n"
                    previous_status = current_status
                    emitted = True
                if current_status in EVALUATION_TERMINAL_STATUSES | {"deleted"} and not emitted:
                    data = json.dumps({"type": "end", "status": current_status})
                    yield f"event: end\ndata: {data}\n\n"
                    return
                if emitted:
                    idle_ticks = 0
                else:
                    idle_ticks += 1
                    if idle_ticks >= 10:
                        yield ": heartbeat\n\n"
                        idle_ticks = 0
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/training/target")
    def training_target(
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        remote_training = RemoteTrainingConfig.from_settings(SettingsService(db).all())
        return {
            "mode": "remote" if remote_training.enabled else "local",
            "gpu_ids": list(remote_training.gpu_ids) if remote_training.enabled else [],
        }

    @app.get("/api/v1/gpus")
    def gpus(_user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
        reservations = _gpu_reservations(db)
        items = gpu_monitor.snapshot(reservations)
        return {
            "available": gpu_monitor.available,
            "error": gpu_monitor.error,
            "items": [item.to_dict() for item in items],
        }

    @app.get("/api/v1/remote-training/metrics")
    def remote_training_metrics(
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        config = RemoteTrainingConfig.from_settings(SettingsService(db).all())
        if not config.enabled:
            return {"enabled": False}
        reservations = {
            int(row.gpu_index): str(row.job_id)
            for row in db.execute(select(GPUReservation)).scalars().all()
        }
        return remote_metrics_collector.snapshot(config, reservations)

    @app.get("/api/v1/storage")
    def storage(_user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
        settings = SettingsService(db)
        return [
            storage_snapshot(
                configured_storage_monitor_path(settings),
                float(settings.get("disk_min_free_gib", 100)),
                float(settings.get("disk_min_free_percent", 10)),
            )
        ]

    def _builtin_checkpoint_rows(db: Session, user: User) -> list[dict[str, Any]]:
        settings = SettingsService(db)
        pretrained_root = Path(str(
            os.environ.get("PRETRAINED_MODELS_DIR")
            or settings.get("pretrained_root", "")
            or "data/pretrained_models"
        )).expanduser()
        if not pretrained_root.is_absolute():
            pretrained_root = runtime.repo_root / pretrained_root
        environment = {
            **os.environ,
            **{str(key): str(value) for key, value in settings.get("environment", {}).items()},
        }
        rows = builtin_checkpoint_candidates(pretrained_root)
        for row in rows:
            inspection = inspect_deployment_checkpoint(
                row["path"],
                repo_root=runtime.repo_root,
                environment=environment,
                include_experimental=_can_experimental(db, user),
                source_catalog=effective_registry_catalog(),
            )
            deployable = bool(row["complete"] and inspection.get("deployable"))
            row["deployable"] = deployable
            row["can_deploy"] = deployable
            row["can_package"] = bool(row["complete"] and user.role == "administrator")
            row["inspection_summary"] = {
                "format": inspection.get("checkpoint", {}).get("format", "unknown"),
                "checkpoint_family": inspection.get("checkpoint", {}).get("checkpoint_family"),
                "checkpoint_format": inspection.get("checkpoint", {}).get("checkpoint_format", "unknown"),
                "checkpoint_format_label": inspection.get("checkpoint", {}).get("checkpoint_format_label", {}),
                "framework": inspection.get("detected", {}).get("framework"),
                "combination_id": inspection.get("detected", {}).get("combination_id"),
                "issue_codes": [str(item.get("code", "unknown")) for item in inspection.get("issues", [])],
            }
            row["checkpoint_family"] = inspection.get("checkpoint", {}).get("checkpoint_family")
            row["checkpoint_format"] = inspection.get("checkpoint", {}).get("checkpoint_format", "unknown")
            row["checkpoint_format_label_i18n"] = inspection.get("checkpoint", {}).get(
                "checkpoint_format_label", {}
            )
            if not row.get("combination_id"):
                row["combination_id"] = inspection.get("detected", {}).get("combination_id")
        return rows

    @app.get("/api/v1/checkpoints")
    def checkpoints(
        experiment_id: str | None = None,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        manager.refresh_checkpoints(experiment_id)
        db.expire_all()
        stmt = select(Checkpoint).order_by(Checkpoint.created_at.desc())
        if experiment_id:
            stmt = stmt.where(Checkpoint.experiment_id == experiment_id)
        rows = db.execute(stmt).scalars().all()
        experiment_ids = {row.experiment_id for row in rows}
        experiments = {
            row.id: row
            for row in db.execute(
                select(Experiment).where(Experiment.id.in_(experiment_ids)).options(selectinload(Experiment.owner))
            )
            .scalars()
            .all()
        }
        busy_experiments = set(
            db.execute(
                select(Job.experiment_id).where(
                    Job.experiment_id.in_(experiment_ids),
                    Job.status.in_(ACTIVE_STATUSES | {"queued", "blocked"}),
                )
            ).scalars()
        )
        referenced_checkpoints = set(
            db.execute(
                select(Job.input_checkpoint_path).where(
                    Job.input_checkpoint_path != "",
                    Job.status.in_(ACTIVE_STATUSES | {"queued", "blocked"}),
                )
            ).scalars()
        )
        referenced_checkpoints.update(
            db.execute(
                select(ModelDeployment.checkpoint_path).where(
                    ModelDeployment.status.in_(DEPLOYMENT_ACTIVE_STATUSES | {"queued"})
                )
            ).scalars()
        )
        referenced_checkpoints.update(
            db.execute(
                select(EvaluationRun.checkpoint_path).where(
                    EvaluationRun.status.in_({"queued", "starting", "running", "stopping"})
                )
            ).scalars()
        )
        indexed = [
            {
                "id": row.id,
                "experiment_id": row.experiment_id,
                "experiment_name": experiments[row.experiment_id].name,
                "owner_name": (
                    experiments[row.experiment_id].owner.display_name or experiments[row.experiment_id].owner.username
                ),
                "job_id": row.job_id,
                "path": row.path,
                "name": row.name,
                "step": row.step,
                "size_bytes": row.size_bytes,
                "is_complete": row.is_complete,
                "is_resumable": row.is_resumable,
                "complete": row.is_complete,
                "resumable": row.is_resumable,
                "deployable": row.is_complete,
                "metadata": row.metadata_json,
                "created_at": row.created_at,
                "can_delete": (
                    row.experiment_id not in busy_experiments
                    and row.path not in referenced_checkpoints
                    and (user.role == "administrator" or experiments[row.experiment_id].owner_id == user.id)
                ),
                "can_package": (
                    row.is_complete
                    and (user.role == "administrator" or experiments[row.experiment_id].owner_id == user.id)
                ),
            }
            for row in rows
            if row.experiment_id in experiments
        ]
        presets = [] if experiment_id else _builtin_checkpoint_rows(db, user)
        return [*presets, *indexed]

    @app.get("/api/v1/checkpoints/{checkpoint_id}")
    def checkpoint_detail(
        checkpoint_id: str,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        if is_builtin_checkpoint_id(checkpoint_id):
            preset = next(
                (item for item in _builtin_checkpoint_rows(db, _user) if item["id"] == checkpoint_id),
                None,
            )
            if preset is None:
                raise HTTPException(status_code=404, detail="checkpoint_not_found")
            settings = SettingsService(db)
            inspection = inspect_deployment_checkpoint(
                preset["path"],
                repo_root=runtime.repo_root,
                environment={
                    **os.environ,
                    **{str(key): str(value) for key, value in settings.get("environment", {}).items()},
                },
                include_experimental=_can_experimental(db, _user),
                source_catalog=effective_registry_catalog(),
            )
            return {
                **preset,
                "inspection": inspection,
                "tools": {
                    "publish_huggingface": {"available": False},
                    "merge_lora": {"available": False, "reason": "builtin_checkpoint_read_only", "experimental": False},
                    "add_qwen_special_tokens": {"available": False, "reason": "builtin_checkpoint_read_only"},
                },
            }
        manager.refresh_checkpoints()
        db.expire_all()
        checkpoint = db.get(Checkpoint, checkpoint_id)
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="checkpoint_not_found")
        experiment = db.execute(
            select(Experiment)
            .where(Experiment.id == checkpoint.experiment_id)
            .options(selectinload(Experiment.owner))
        ).scalar_one_or_none()
        if experiment is None:
            raise HTTPException(status_code=404, detail="experiment_not_found")
        inspection = inspect_deployment_checkpoint(
            checkpoint.path,
            repo_root=runtime.repo_root,
            include_experimental=_can_experimental(db, _user),
            source_catalog=effective_registry_catalog(),
        )
        lora_bundle = discover_lora_bundle(checkpoint.path)
        return {
            "id": checkpoint.id,
            "experiment_id": checkpoint.experiment_id,
            "experiment_name": experiment.name,
            "owner_name": experiment.owner.display_name or experiment.owner.username,
            "job_id": checkpoint.job_id,
            "path": checkpoint.path,
            "name": checkpoint.name,
            "step": checkpoint.step,
            "size_bytes": checkpoint.size_bytes,
            "complete": checkpoint.is_complete,
            "resumable": checkpoint.is_resumable,
            "can_package": (
                checkpoint.is_complete
                and (_user.role == "administrator" or experiment.owner_id == _user.id)
            ),
            "metadata": checkpoint.metadata_json,
            "created_at": checkpoint.created_at,
            "inspection": inspection,
            "tools": {
                "publish_huggingface": {"available": checkpoint.is_complete},
                "merge_lora": {
                    **lora_bundle,
                    "experimental": False,
                },
                "add_qwen_special_tokens": {
                    "available": False,
                    "reason": "Use the resource center model preparation flow; checkpoint mutation is disabled.",
                },
            },
        }

    @app.post("/api/v1/checkpoints/{checkpoint_id}/package")
    def package_checkpoint(
        checkpoint_id: str,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        if is_builtin_checkpoint_id(checkpoint_id):
            if user.role != "administrator":
                raise HTTPException(status_code=403, detail="administrator_required")
            preset = next(
                (item for item in _builtin_checkpoint_rows(db, user) if item["id"] == checkpoint_id),
                None,
            )
            if preset is None:
                raise HTTPException(status_code=404, detail="checkpoint_not_found")
            raw_source = Path(str(preset["path"]))
            label = str(preset.get("name") or raw_source.name)
        else:
            checkpoint = db.get(Checkpoint, checkpoint_id)
            if checkpoint is None:
                raise HTTPException(status_code=404, detail="checkpoint_not_found")
            experiment = db.get(Experiment, checkpoint.experiment_id)
            if experiment is None:
                raise HTTPException(status_code=404, detail="experiment_not_found")
            assert_owner_or_admin(user, experiment.owner_id)
            if not checkpoint.is_complete:
                raise HTTPException(status_code=409, detail="checkpoint_incomplete")
            job = db.get(Job, checkpoint.job_id) if checkpoint.job_id else None
            if job is None:
                raise HTTPException(status_code=409, detail="checkpoint_job_not_found")
            try:
                job_root = Path(job.output_dir).resolve(strict=True)
            except OSError as exc:
                raise HTTPException(status_code=404, detail="training_output_not_found") from exc
            raw_source = Path(checkpoint.path)
            label = f"{experiment.name}-{checkpoint.name}"
            try:
                candidate = raw_source.resolve(strict=True)
            except OSError as exc:
                raise HTTPException(status_code=404, detail="checkpoint_not_found_on_disk") from exc
            if not candidate.is_relative_to(job_root):
                raise HTTPException(status_code=403, detail="checkpoint_outside_job_output")
        if raw_source.is_symlink():
            raise HTTPException(status_code=409, detail="checkpoint_source_invalid")
        try:
            source = raw_source.resolve(strict=True)
        except OSError as exc:
            raise HTTPException(status_code=404, detail="checkpoint_not_found_on_disk") from exc
        if not (source.is_file() or source.is_dir()):
            raise HTTPException(status_code=409, detail="checkpoint_source_invalid")
        run = create_package_run(
            owner_id=user.id,
            kind="checkpoint_package",
            source=source,
            label=label,
            resource_id=checkpoint_id,
            reference_key="checkpoint_id",
            reference_id=checkpoint_id,
            db=db,
        )
        add_audit(
            db,
            "checkpoint.package",
            actor_id=user.id,
            target_type="checkpoint",
            target_id=checkpoint_id,
            detail={"utility_run_id": run.id},
        )
        return serialize_utility(run)

    @app.post("/api/v1/checkpoints/{checkpoint_id}/merge-lora")
    def merge_lora_checkpoint(
        checkpoint_id: str,
        payload: LoraMergeRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        checkpoint = db.get(Checkpoint, checkpoint_id)
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="checkpoint_not_found")
        experiment = db.get(Experiment, checkpoint.experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail="experiment_not_found")
        assert_owner_or_admin(user, experiment.owner_id)
        bundle = discover_lora_bundle(checkpoint.path)
        if not bundle.get("available"):
            raise HTTPException(status_code=409, detail=str(bundle.get("reason") or "lora_bundle_not_detected"))
        if payload.resources.gpu_count != 1 or len(payload.resources.gpu_ids) > 1:
            raise HTTPException(status_code=422, detail="lora_merge_requires_one_gpu")
        job = db.get(Job, checkpoint.job_id) if checkpoint.job_id else None
        if job is None:
            raise HTTPException(status_code=409, detail="checkpoint_job_not_found")
        job_root = Path(job.output_dir).resolve(strict=True)
        adapter_path = Path(str(bundle["adapter_path"]))
        action_path = Path(str(bundle["action_model_path"]))
        if any(path == job_root or not path.is_relative_to(job_root) for path in (adapter_path, action_path)):
            raise HTTPException(status_code=403, detail="lora_bundle_outside_job_output")
        output_path = Path(str(bundle["suggested_output_path"]))
        if payload.output_name:
            try:
                output_name = safe_output_name(payload.output_name)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            output_path = output_path.with_name(output_name)
        output_path = output_path.resolve(strict=False)
        if output_path.parent != adapter_path.parent or not output_path.is_relative_to(job_root):
            raise HTTPException(status_code=403, detail="lora_output_outside_job_output")
        if output_path.exists():
            raise HTTPException(status_code=409, detail="lora_output_already_exists")
        active_output = db.execute(
            select(UtilityRun.id).where(
                UtilityRun.output_path == str(output_path),
                UtilityRun.status.in_(UTILITY_ACTIVE_STATUSES | {"queued"}),
            )
        ).scalar_one_or_none()
        if active_output:
            raise HTTPException(status_code=409, detail="lora_output_already_queued")
        settings = SettingsService(db)
        command = build_lora_merge_command(
            python_executable=sys.executable,
            model=payload.model,
            adapter_path=adapter_path,
            action_model_path=action_path,
            output_path=output_path,
        )
        run = utility_manager.create_run(
            owner_id=user.id,
            kind="merge_lora_checkpoint",
            resource_id=checkpoint.id,
            command=command,
            cwd=runtime.repo_root,
            output_path=str(output_path),
            parameters={
                "checkpoint_id": checkpoint.id,
                "model": payload.model,
                "adapter_path": str(adapter_path),
                "action_model_path": str(action_path),
            },
            environment={
                str(key): str(value)
                for key, value in settings.get("environment", {}).items()
            },
            queue_class="gpu",
            requested_gpu_count=1,
            requested_gpu_ids=list(payload.resources.gpu_ids),
            db_session=db,
        )
        add_audit(
            db,
            "checkpoint.merge_lora",
            actor_id=user.id,
            target_type="checkpoint",
            target_id=checkpoint.id,
            detail={"utility_run_id": run.id, "model": payload.model, "output_path": str(output_path)},
        )
        return serialize_utility(run)

    @app.post("/api/v1/model-publications")
    def create_model_publication(
        payload: ModelPublicationRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        checkpoint = db.get(Checkpoint, payload.checkpoint_id)
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="checkpoint_not_found")
        experiment = db.get(Experiment, checkpoint.experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail="experiment_not_found")
        assert_owner_or_admin(user, experiment.owner_id)
        if not checkpoint.is_complete:
            raise HTTPException(status_code=409, detail="checkpoint_incomplete")
        raw_source = Path(checkpoint.path)
        if raw_source.is_symlink():
            raise HTTPException(status_code=409, detail="checkpoint_source_invalid")
        try:
            source = raw_source.resolve(strict=True)
        except OSError as exc:
            raise HTTPException(status_code=409, detail="checkpoint_not_found_on_disk") from exc
        if not (source.is_file() or source.is_dir()):
            raise HTTPException(status_code=409, detail="checkpoint_source_invalid")
        publication_roots: list[Path] = []
        for value in SettingsService(db).get("results_roots", ["results"]):
            root = Path(str(value)).expanduser()
            if not root.is_absolute():
                root = runtime.repo_root / root
            publication_roots.append(root.resolve(strict=False))
        if not any(source != root and source.is_relative_to(root) for root in publication_roots):
            raise HTTPException(status_code=403, detail="checkpoint_path_outside_results_roots")
        if not hf_secret_store.user_status(user.id)["configured"]:
            raise HTTPException(status_code=409, detail="huggingface_publish_token_not_configured")

        publication = ModelPublication(
            id=new_id(),
            owner_id=user.id,
            checkpoint_id=checkpoint.id,
            provider="huggingface",
            repo_id=payload.repo_id,
            revision=payload.revision,
            private=payload.private,
            status="queued",
            source_path=str(source),
            metadata_json={
                "checkpoint_name": checkpoint.name,
                "checkpoint_step": checkpoint.step,
                "checkpoint_size_bytes": checkpoint.size_bytes,
            },
        )
        db.add(publication)
        db.flush()
        command = [
            sys.executable,
            "-m",
            "alphabrain_ui.utility_tasks",
            "publish-huggingface",
            "--source",
            str(source),
            "--repo-id",
            payload.repo_id,
            "--revision",
            payload.revision,
        ]
        if payload.private:
            command.append("--private")
        utility = utility_manager.create_run(
            owner_id=user.id,
            kind="publish_huggingface",
            resource_id=payload.repo_id,
            command=command,
            cwd=runtime.repo_root,
            output_path=f"https://huggingface.co/{payload.repo_id}/tree/{payload.revision}",
            parameters={
                "hf_auth": "user_publish",
                "hf_user_id": user.id,
                "publication_id": publication.id,
                "checkpoint_id": checkpoint.id,
            },
            queue_class="cpu",
            db_session=db,
        )
        publication.utility_run_id = utility.id
        add_audit(
            db,
            "model.publish",
            actor_id=user.id,
            target_type="checkpoint",
            target_id=checkpoint.id,
            detail={
                "publication_id": publication.id,
                "provider": "huggingface",
                "repo_id": payload.repo_id,
                "revision": payload.revision,
                "private": payload.private,
            },
        )
        return _serialize_model_publication(publication)

    @app.get("/api/v1/model-publications")
    def list_model_publications(
        checkpoint_id: str | None = None,
        limit: int = Query(100, ge=1, le=500),
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        statement = select(ModelPublication).order_by(ModelPublication.created_at.desc()).limit(limit)
        if checkpoint_id:
            statement = statement.where(ModelPublication.checkpoint_id == checkpoint_id)
        rows = db.execute(statement).scalars().all()
        for row in rows:
            if row.utility_run_id and row.status not in {"completed", "failed", "cancelled", "stopped"}:
                utility_manager.sync_publication_status(row.utility_run_id)
        db.expire_all()
        return [_serialize_model_publication(db.get(ModelPublication, row.id) or row) for row in rows]

    @app.get("/api/v1/model-publications/{publication_id}")
    def model_publication_detail(
        publication_id: str,
        _user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        publication = db.get(ModelPublication, publication_id)
        if publication is None:
            raise HTTPException(status_code=404, detail="model_publication_not_found")
        if publication.utility_run_id:
            utility_manager.sync_publication_status(publication.utility_run_id)
            db.expire_all()
            publication = db.get(ModelPublication, publication_id) or publication
        return _serialize_model_publication(publication)

    @app.delete("/api/v1/checkpoints/{checkpoint_id}")
    def delete_checkpoint(
        checkpoint_id: str,
        payload: DeleteRequest,
        user: User = Depends(current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        row = db.get(Checkpoint, checkpoint_id)
        if row is None:
            raise HTTPException(status_code=404, detail="checkpoint_not_found")
        experiment = db.get(Experiment, row.experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail="experiment_not_found")
        assert_owner_or_admin(user, experiment.owner_id)
        if payload.confirmation != experiment.name:
            raise HTTPException(status_code=422, detail="confirmation_does_not_match_experiment_name")
        active = db.execute(
            select(func.count(Job.id)).where(
                Job.experiment_id == experiment.id,
                Job.status.in_(ACTIVE_STATUSES | {"queued", "blocked"}),
            )
        ).scalar_one()
        if active:
            raise HTTPException(status_code=409, detail="cannot_delete_checkpoint_of_active_experiment")
        references = db.execute(
            select(func.count(Job.id)).where(
                Job.input_checkpoint_path == row.path,
                Job.status.in_(ACTIVE_STATUSES | {"queued", "blocked"}),
            )
        ).scalar_one()
        if references:
            raise HTTPException(status_code=409, detail="checkpoint_referenced_by_active_job")
        deployment_references = db.execute(
            select(func.count(ModelDeployment.id)).where(
                ModelDeployment.checkpoint_path == row.path,
                ModelDeployment.status.in_(DEPLOYMENT_ACTIVE_STATUSES | {"queued"}),
            )
        ).scalar_one()
        if deployment_references:
            raise HTTPException(status_code=409, detail="checkpoint_referenced_by_active_deployment")
        evaluation_references = db.execute(
            select(func.count(EvaluationRun.id)).where(
                EvaluationRun.checkpoint_path == row.path,
                EvaluationRun.status.in_({"queued", "starting", "running", "stopping"}),
            )
        ).scalar_one()
        if evaluation_references:
            raise HTTPException(status_code=409, detail="checkpoint_referenced_by_active_evaluation")
        raw_path = Path(row.path)
        job = db.get(Job, row.job_id) if row.job_id else None
        if job is None:
            raise HTTPException(status_code=409, detail="checkpoint_job_not_found")
        job_root = Path(job.output_dir).resolve()
        if raw_path == job_root or not raw_path.is_relative_to(job_root):
            raise HTTPException(status_code=403, detail="checkpoint_path_outside_job_output")
        settings = SettingsService(db)
        allowed_roots = []
        for value in settings.get("results_roots", ["results"]):
            root = Path(value).expanduser()
            if not root.is_absolute():
                root = runtime.repo_root / root
            allowed_roots.append(root.resolve())
        if any(raw_path == root for root in allowed_roots):
            raise HTTPException(status_code=403, detail="cannot_delete_results_root")
        if not any(raw_path.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(status_code=403, detail="checkpoint_path_outside_results_roots")
        if raw_path.is_symlink():
            raw_path.unlink()
        else:
            path = raw_path.resolve()
            if path == job_root or not path.is_relative_to(job_root):
                raise HTTPException(status_code=403, detail="checkpoint_path_outside_job_output")
            if any(path == root for root in allowed_roots):
                raise HTTPException(status_code=403, detail="cannot_delete_results_root")
            if not any(path.is_relative_to(root) for root in allowed_roots):
                raise HTTPException(status_code=403, detail="checkpoint_path_outside_results_roots")
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        db.delete(row)
        add_audit(db, "checkpoint.delete", actor_id=user.id, target_type="checkpoint", target_id=checkpoint_id)
        return {"ok": True}

    @app.get("/api/v1/dashboard")
    def dashboard(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
        counts = {
            status_name: db.execute(select(func.count(Job.id)).where(Job.status == status_name)).scalar_one()
            for status_name in (
                "queued",
                "blocked",
                "running",
                "completed",
                "failed",
                "dependency_failed",
                "interrupted",
                "stopped",
            )
        }
        counts["failed"] += counts["dependency_failed"] + counts["interrupted"]
        recent = (
            db.execute(
                select(Experiment)
                .options(selectinload(Experiment.stages).selectinload(ExperimentStage.jobs))
                .order_by(Experiment.created_at.desc())
                .limit(8)
            )
            .scalars()
            .all()
        )
        reservations = _gpu_reservations(db)
        settings = SettingsService(db)
        remote_training = RemoteTrainingConfig.from_settings(settings.all())
        storage_rows = [
            storage_snapshot(
                configured_storage_monitor_path(settings),
                float(settings.get("disk_min_free_gib", 100)),
                float(settings.get("disk_min_free_percent", 10)),
            )
        ]
        return {
            "user": _serialize_user(user),
            "job_counts": counts,
            "gpus": [gpu.to_dict() for gpu in gpu_monitor.snapshot(reservations)],
            "system_metrics": collect_system_metrics(),
            "remote_training": {
                "enabled": remote_training.enabled,
                "target": remote_training.target if remote_training.enabled else "",
            },
            "recent_experiments": [_serialize_experiment(row) for row in recent],
            "checkpoint_count": db.execute(select(func.count(Checkpoint.id))).scalar_one(),
            "storage": storage_rows,
            "alerts": [
                {
                    "level": "error" if row["error"] else "warning",
                    "code": "storage_unavailable" if row["error"] else "low_disk_space",
                    "path": row["path"],
                    "message": row["error"],
                }
                for row in storage_rows
                if row["low_space"]
            ],
        }

    @app.get("/api/v1/audit")
    def audit(
        limit: int = Query(200, ge=1, le=1000),
        _admin: User = Depends(admin_user),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        rows = db.execute(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)).scalars().all()
        return [
            {
                "id": row.id,
                "actor_id": row.actor_id,
                "action": row.action,
                "target_type": row.target_type,
                "target_id": row.target_id,
                "detail": row.detail,
                "created_at": row.created_at,
            }
            for row in rows
        ]

    if runtime.frontend_dist.is_dir() and (runtime.frontend_dist / "index.html").is_file():
        app.mount("/", SPAStaticFiles(directory=runtime.frontend_dist, html=True), name="frontend")
    else:

        @app.get("/")
        def frontend_missing() -> JSONResponse:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "frontend_not_built",
                    "hint": "Run: cd ui/frontend && npm install && npm run build",
                    "api_docs": "/api/docs",
                },
            )

    return app
