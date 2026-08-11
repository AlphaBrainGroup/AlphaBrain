"""Seed data and lightweight preflight helpers for the explicit demo mode."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

from .database import (
    Checkpoint,
    Database,
    DatasetRegistration,
    ExperimentTemplate,
    ModelDeployment,
    User,
)
from .dataset_registry import inspect_dataset
from .launchers import LaunchStage
from .runtime import RuntimeConfig
from .schemas import DeploymentRequest, EvaluationRequest, ExperimentRequest
from .services import AuthService, SettingsService

DEMO_USER_ID = "00000000-0000-4000-8000-000000000001"
DEMO_DATASET_ID = "00000000-0000-4000-8000-000000000002"
DEMO_TEMPLATE_ID = "00000000-0000-4000-8000-000000000003"


def _json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _seed_dataset(root: Path) -> dict[str, Any]:
    info = {
        "codebase_version": "v2.0",
        "robot_type": "demo_arm",
        "total_episodes": 2,
        "total_frames": 12,
        "total_tasks": 2,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7], "names": ["joint"]},
            "action": {"dtype": "float32", "shape": [7], "names": ["joint"]},
            "observation.images.front": {"dtype": "image", "shape": [224, 224, 3], "names": ["height", "width", "channel"]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    modality = {
        "state": {"joint_position": {"start": 0, "end": 7}},
        "action": {"joint_command": {"start": 0, "end": 7}},
        "video": {"front": {"original_key": "observation.images.front"}},
        "annotation": {"task": {"original_key": "task_index"}},
    }
    _json_file(root / "meta" / "info.json", info)
    _json_file(root / "meta" / "modality.json", modality)
    _json_file(
        root / "meta" / "stats_gr00t.json",
        {"observation.state": {"mean": [0.0] * 7, "std": [1.0] * 7}, "action": {"mean": [0.0] * 7, "std": [1.0] * 7}},
    )
    (root / "meta" / "tasks.jsonl").write_text(
        '{"task_index":0,"task":"pick up the red block"}\n'
        '{"task_index":1,"task":"place the red block in the tray"}\n',
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        '{"episode_index":0,"length":6,"tasks":["pick up the red block"]}\n'
        '{"episode_index":1,"length":6,"tasks":["place the red block in the tray"]}\n',
        encoding="utf-8",
    )
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        (data / f"episode_{index:06d}.parquet").write_bytes(b"PAR1DEMO-DATA-PAR1")
    return inspect_dataset(root)


def demo_template_spec(registration_id: str = DEMO_DATASET_ID) -> dict[str, Any]:
    return {
        "spec_version": 2,
        "architecture": {"backbone": "toy", "action_head": "mlp_regression"},
        "training": {"method": "imitation_learning", "family": "imitation_learning"},
        "dataset": {
            "id": "libero",
            "mix": "libero_goal",
            "registration_id": registration_id,
        },
        "parameters": {
            "run_id": "demo-toy-vla",
            "per_device_batch_size": 1,
            "learning_rate": 0.001,
            "max_train_steps": 10,
            "save_interval": 5,
            "seed": 42,
        },
        "resources": {"allocation": "auto", "num_gpus": 1, "gpu_ids": []},
        "workflow": {"id": "standard", "schema_version": 1, "config": {}},
        "resume": {"mode": "none", "checkpoint": None},
        "wandb": {"enabled": False, "mode": "disabled", "categories": ["metrics", "config"]},
        "expert_overrides": {},
        "metadata": {
            "description": "Local end-to-end demo. Training, GPU work, and model outputs are simulated."
        },
    }


def prepare_demo_environment(runtime: RuntimeConfig, database: Database) -> None:
    if not runtime.demo_mode:
        return
    dataset_root = runtime.state_dir / "demo-assets" / "libero-mini"
    report = _seed_dataset(dataset_root)
    pretrained_root = runtime.state_dir / "demo-assets" / "pretrained"
    pretrained_root.mkdir(parents=True, exist_ok=True)
    results_root = runtime.state_dir / "demo-results"
    results_root.mkdir(parents=True, exist_ok=True)

    with database.session() as db:
        user = db.get(User, DEMO_USER_ID)
        if user is None:
            user = db.execute(select(User).where(User.is_local.is_(True))).scalar_one_or_none()
        if user is None:
            user = User(
                id=DEMO_USER_ID,
                username="demo",
                display_name="AlphaBrain Demo",
                password_hash=AuthService(db).hash_password("demo-local-only"),
                role="administrator",
                language="zh-CN",
                theme="light",
                experimental_enabled=True,
                is_active=True,
                is_local=True,
            )
            db.add(user)
            db.flush()
        settings = SettingsService(db)
        environment = dict(settings.get("environment", {}) or {})
        environment.update(
            {
                "LIBERO_HOME": str(dataset_root),
                "LIBERO_DATA_ROOT": str(dataset_root),
                "LEROBOT_LIBERO_DATA_DIR": str(dataset_root),
                "LIBERO_PYTHON": sys.executable,
                "PRETRAINED_MODELS_DIR": str(pretrained_root),
            }
        )
        settings.update(
            {
                "initialized": True,
                "deployment_mode": "personal",
                "experimental_globally_enabled": True,
                "environment": environment,
                "results_roots": [str(results_root)],
                "dataset_roots": [str(runtime.state_dir / "demo-assets")],
                "managed_dataset_root": str(runtime.state_dir / "demo-datasets"),
                "pretrained_root": str(pretrained_root),
                "model_server_python": sys.executable,
                "disk_min_free_gib": 0.0,
                "disk_min_free_percent": 0.0,
            },
            actor_id=user.id,
        )
        registration = db.get(DatasetRegistration, DEMO_DATASET_ID)
        values = {
            "owner_id": user.id,
            "name": "LIBERO Mini · Demo",
            "description": "Two tiny local episodes for the end-to-end demo.",
            "path": str(dataset_root.resolve()),
            "source_path": str(dataset_root.resolve()),
            "storage_mode": "reference",
            "visibility": "shared",
            "format": str(report.get("format", "LeRobot v2.0")),
            "status": "ready" if report.get("valid") else "invalid",
            "dataset_id": "libero",
            "dataset_mix": "libero_goal",
            "fingerprint": str(report.get("fingerprint", "")),
            "size_bytes": int(report.get("size_bytes", 0)),
            "episode_count": int(report.get("episode_count", 0)),
            "step_count": int(report.get("step_count", 0)),
            "validation": report,
            "metadata_json": {"demo": True, "simulated": True},
            "stats_status": "ready",
        }
        if registration is None:
            registration = DatasetRegistration(id=DEMO_DATASET_ID, **values)
            db.add(registration)
        else:
            for key, value in values.items():
                setattr(registration, key, value)
        template = db.get(ExperimentTemplate, DEMO_TEMPLATE_ID)
        if template is None:
            db.add(
                ExperimentTemplate(
                    id=DEMO_TEMPLATE_ID,
                    owner_id=user.id,
                    name="Demo · Toy VLA 全流程",
                    description="一键体验训练、日志、Checkpoint、部署、推理和评测；不会加载真实模型。",
                    visibility="shared",
                    spec=demo_template_spec(),
                )
            )


def _demo_issue(code: str, zh: str, en: str) -> dict[str, Any]:
    return {
        "level": "info",
        "code": code,
        "message": en,
        "message_i18n": {"zh-CN": zh, "en-US": en},
        "detail": {"demo": True, "simulated": True},
    }


def demo_experiment_preflight(
    runtime: RuntimeConfig,
    payload: ExperimentRequest,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[dict[str, Any]]]:
    resolved = copy.deepcopy(payload.spec)
    resolved.setdefault("training", {}).setdefault("family", "imitation_learning")
    resolved.setdefault("parameters", {}).setdefault("run_id", "demo-toy-vla")
    resolved["output_root"] = str(runtime.state_dir / "demo-results" / "training")
    resources = resolved.setdefault("resources", {})
    resources["num_gpus"] = int(resources.get("num_gpus", resources.get("gpu_count", 1)) or 1)
    resources.setdefault("gpu_ids", [])
    preview = [
        {
            "name": "Simulated VLA training",
            "phase": "train",
            "command": [sys.executable, "-m", "alphabrain_ui.demo_worker", "train"],
            "environment": {"ALPHABRAIN_DEMO": "1"},
            "output_dir": str(runtime.state_dir / "demo-results" / "training"),
            "requested_gpu_count": resources["num_gpus"],
            "requested_gpu_ids": resources["gpu_ids"],
        }
    ]
    return (
        resolved,
        "verified",
        [
            _demo_issue(
                "demo_simulation_active",
                "演示模式已启用：队列、日志与产物链路真实运行，训练计算由本地模拟器完成。",
                "Demo mode is active: queue, logs, and artifacts are real; training compute is simulated locally.",
            )
        ],
        preview,
    )


def demo_launch_plan(
    runtime: RuntimeConfig,
    payload: ExperimentRequest,
    resolved: dict[str, Any],
    experiment_id: str,
) -> list[LaunchStage]:
    run_id = str(resolved.get("parameters", {}).get("run_id") or "demo-toy-vla")
    output_dir = runtime.state_dir / "demo-results" / "training" / f"{experiment_id}-{run_id}"
    metrics_path = output_dir / "metrics.jsonl"
    config_path = runtime.state_dir / "configs" / experiment_id / "demo-resolved.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    resources = dict(resolved.get("resources") or {})
    gpu_count = int(resources.get("num_gpus", resources.get("gpu_count", 1)) or 1)
    return [
        LaunchStage(
            name="Simulated VLA training",
            phase="train",
            command=[
                sys.executable,
                "-m",
                "alphabrain_ui.demo_worker",
                "train",
                "--output-dir",
                str(output_dir),
                "--metrics-path",
                str(metrics_path),
            ],
            environment={"ALPHABRAIN_DEMO": "1"},
            cwd=str(runtime.repo_root),
            output_dir=str(output_dir),
            metrics_path=str(metrics_path),
            requested_gpu_count=gpu_count,
            requested_gpu_ids=[int(value) for value in resources.get("gpu_ids", [])],
            config_snapshot_path=str(config_path),
        )
    ]


def _resolve_checkpoint(db, kind: str, checkpoint_id: str | None, path: str | None) -> tuple[str | None, str]:
    if kind == "indexed":
        row = db.get(Checkpoint, str(checkpoint_id or ""))
        if row is None or not row.is_complete:
            raise ValueError("checkpoint_not_found")
        return row.id, row.path
    resolved = Path(str(path or "")).expanduser().resolve(strict=False)
    if not resolved.exists():
        raise ValueError("checkpoint_not_found")
    return None, str(resolved)


def demo_deployment_preflight(payload: DeploymentRequest, db) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    try:
        checkpoint_id, checkpoint_path = _resolve_checkpoint(
            db,
            payload.checkpoint_source.kind,
            payload.checkpoint_source.checkpoint_id,
            payload.checkpoint_source.path,
        )
    except ValueError:
        return {"ok": False, "resolved": {}, "issues": [{"level": "error", "code": "checkpoint_not_found", "message": "Checkpoint not found.", "detail": {}}]}
    return {
        "ok": True,
        "compatibility": "verified",
        "resolved": {
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": checkpoint_path,
            "combination_id": payload.combination_id or "deploy_qwen2_5_oft",
            "adapter_id": "base_framework_websocket",
            "backbone_id": "qwen2_5_vl",
            "action_head_id": "mlp_regression",
            "resources": {
                "gpu_count": payload.resources.gpu_count,
                "gpu_ids": payload.resources.gpu_ids,
            },
            "endpoint": {
                "scope": "local",
                "bind_host": "127.0.0.1",
                "advertised_host": "127.0.0.1",
                "port": payload.endpoint.port,
                "idle_timeout_seconds": payload.endpoint.idle_timeout_seconds,
            },
            "parameters": payload.parameters,
            "environment": {"ALPHABRAIN_DEMO": "1"},
            "startup_timeout_seconds": 15,
        },
        "issues": [
            _demo_issue(
                "demo_model_server",
                "模型服务将使用确定性的本地模拟策略，但 WebSocket、鉴权和健康检查均为真实链路。",
                "The model server uses a deterministic local policy while WebSocket, authentication, and health checks remain real.",
            )
        ],
    }


def demo_checkpoint_inspection(checkpoint_id: str | None = None) -> dict[str, Any]:
    candidate = {"combination_id": "deploy_qwen2_5_oft", "benchmark_ids": ["libero", "libero_plus"]}
    return {
        "ok": True,
        "valid": True,
        "detected": {
            "framework": "DemoBaseFramework",
            "backbone": "qwen2_5_vl",
            "action_head": "mlp_regression",
            "adapter_id": "base_framework_websocket",
            "combination_id": "deploy_qwen2_5_oft",
            "candidate_combination_ids": ["deploy_qwen2_5_oft"],
            "checkpoint_id": checkpoint_id,
        },
        "candidates": [candidate],
        "evaluation_candidates": [candidate],
        "compatible_benchmark_ids": ["libero", "libero_plus"],
        "issues": [
            _demo_issue(
                "demo_checkpoint_detected",
                "已识别演示 Checkpoint，可用于模拟部署与评测。",
                "The demo checkpoint is ready for simulated deployment and evaluation.",
            )
        ],
    }


def demo_evaluation_preflight(
    runtime: RuntimeConfig,
    payload: EvaluationRequest,
    db,
) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    try:
        if payload.source_kind == "managed_deployment":
            deployment = db.get(ModelDeployment, str(payload.deployment_id or ""))
            if deployment is None or deployment.status != "running":
                raise ValueError("deployment_not_running")
            checkpoint_id, checkpoint_path = deployment.checkpoint_id, deployment.checkpoint_path
            combination_id = deployment.combination_id
        else:
            source = payload.checkpoint_source
            if source is None:
                raise ValueError("checkpoint_not_found")
            checkpoint_id, checkpoint_path = _resolve_checkpoint(db, source.kind, source.checkpoint_id, source.path)
            combination_id = payload.combination_id or "deploy_qwen2_5_oft"
    except ValueError as exc:
        return {"ok": False, "can_submit": False, "resolved": {}, "issues": [{"level": "error", "code": str(exc), "message": str(exc), "detail": {}}]}
    resolved = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": checkpoint_path,
        "combination_id": combination_id,
        "adapter_id": "base_framework_websocket",
        "backbone_id": "qwen2_5_vl",
        "action_head_id": "mlp_regression",
        "benchmark_id": payload.benchmark_id,
        "benchmark_status": "verified",
        "preset": payload.preset,
        "suite": payload.suite or "libero_goal",
        "task_set": payload.task_set or "",
        "split": payload.split or "demo",
        "parameters": {**payload.parameters, "demo": True},
        "server_parameters": payload.model_parameters,
        "resources": {
            "gpu_count": payload.resources.gpu_count,
            "gpu_ids": payload.resources.gpu_ids,
        },
        "environment": {"ALPHABRAIN_DEMO": "1"},
        "output_root": str(runtime.state_dir / "demo-results" / "evaluation"),
        "wandb": payload.wandb.model_dump(mode="json"),
    }
    issues = [
        _demo_issue(
            "demo_evaluation",
            "评测任务会生成真实进度流和标准结果文件；机器人仿真回合由确定性模拟器代替。",
            "The evaluation writes real progress and result artifacts; deterministic simulation replaces robot episodes.",
        )
    ]
    return {
        "ok": True,
        "can_submit": True,
        "compatibility": "verified",
        "resolved": resolved,
        "issues": issues,
        "items": issues,
        "command_preview": [[sys.executable, "-m", "alphabrain_ui.demo_worker", "evaluate"]],
        "diff": {},
    }
