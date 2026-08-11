"""Managed evaluation process lifecycle and optional W&B result publishing."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import signal
import socket
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import psutil
import yaml
from pydantic import ValidationError
from sqlalchemy import delete, select

from .database import (
    DeploymentGPUReservation,
    EvaluationGroup,
    EvaluationGPUReservation,
    EvaluationRun,
    GPUReservation,
    ModelDeployment,
    UtilityGPUReservation,
    utcnow,
)
from .deployment_registry import build_adapter_command, get_deployment_catalog
from .evaluation_registry import get_benchmark
from .gpu import GPUMonitor
from .preflight import read_dotenv
from .process_control import (
    FORCE_KILL_SIGNAL,
    new_process_group_kwargs,
    process_group_matches,
    signal_process_group,
)
from .runtime import RuntimeConfig
from .schemas import EvaluationResultV1, EvaluationResultV2
from .secrets import SecureSecretStore
from .services import SettingsService
from .specialized_evaluation import upgrade_v1_result, write_result_v2
from .wandb import WANDB_API_KEY_ENV, WandbSecretStore

EVALUATION_ACTIVE_STATUSES = {"starting", "running", "stopping"}
EVALUATION_TERMINAL_STATUSES = {"completed", "failed", "stopped", "cancelled", "interrupted"}
POLICY_API_KEY_ENV = "ALPHABRAIN_POLICY_API_KEY"
SPECIALIZED_EVALUATION_KINDS = {
    "cl_matrix",
    "rl_iterations",
    "online_stdp_baseline",
    "online_stdp_adapted",
}


def _valid_controller_key(value: str | None) -> bool:
    return bool(
        value
        and len(value) >= 32
        and not any(character.isspace() for character in value)
    )


def build_evaluation_output_dir(
    output_root: str | Path,
    evaluation_id: str,
    name: str,
) -> Path:
    """Return the documented non-overwriting UI evaluation directory."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-._")[:64] or "evaluation"
    return (
        Path(output_root).expanduser().resolve(strict=False)
        / utcnow().strftime("%Y%m%d")
        / f"{evaluation_id}-{slug}"
    )


def _merged_reservations(db) -> dict[int, str]:  # type: ignore[no-untyped-def]
    result = {row.gpu_index: row.job_id for row in db.execute(select(GPUReservation)).scalars().all()}
    result.update(
        {
            row.gpu_index: f"deployment:{row.deployment_id}"
            for row in db.execute(select(DeploymentGPUReservation)).scalars().all()
        }
    )
    result.update(
        {
            row.gpu_index: f"evaluation:{row.evaluation_id}"
            for row in db.execute(select(EvaluationGPUReservation)).scalars().all()
        }
    )
    result.update(
        {
            row.gpu_index: f"utility:{row.utility_run_id}"
            for row in db.execute(select(UtilityGPUReservation)).scalars().all()
        }
    )
    return result


def read_evaluation_result(path: str | Path) -> dict[str, Any]:
    """Load and validate the benchmark-neutral result artifact."""

    with Path(path).open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if raw.get("schema_version") == "evaluation-result-v2":
        return EvaluationResultV2.model_validate(raw).model_dump(mode="json")
    return EvaluationResultV1.model_validate(raw).model_dump(mode="json")


def read_run_evaluation_result(evaluation: EvaluationRun) -> dict[str, Any]:
    """Read a run result, upgrading WM predicted-video v1 output to v2."""

    try:
        return read_evaluation_result(evaluation.result_path)
    except OSError:
        if evaluation.evaluation_kind != "world_model_video":
            raise
    legacy_path = Path(evaluation.output_dir) / "evaluation-result-v1.json"
    legacy = read_evaluation_result(legacy_path)
    output_root = Path(evaluation.output_dir).resolve()
    artifacts: list[dict[str, Any]] = []
    for path in output_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".mp4", ".webm", ".mkv", ".avi", ".json", ".log"}:
            continue
        artifacts.append(
            {
                "kind": "video" if suffix in {".mp4", ".webm", ".mkv", ".avi"} else suffix.lstrip("."),
                "path": path.relative_to(output_root).as_posix(),
                "name": path.name,
            }
        )
    upgraded = upgrade_v1_result(legacy, kind="world_model_video", artifacts=artifacts)
    write_result_v2(evaluation.result_path, upgraded)
    return EvaluationResultV2.model_validate(upgraded).model_dump(mode="json")


def refresh_evaluation_group(db, group_id: str | None) -> EvaluationGroup | None:  # type: ignore[no-untyped-def]
    """Recompute one group's durable status and v2 aggregate result."""

    if not group_id:
        return None
    group = db.get(EvaluationGroup, group_id)
    if group is None:
        return None
    runs = list(
        db.execute(
            select(EvaluationRun)
            .where(EvaluationRun.group_id == group_id)
            .order_by(EvaluationRun.position)
        ).scalars()
    )
    if not runs:
        group.status = "failed"
        group.error = "Evaluation group has no child runs."
        group.finished_at = utcnow()
        return group
    statuses = {run.status for run in runs}
    if any(status in EVALUATION_ACTIVE_STATUSES for status in statuses):
        group.status = "running"
    elif "queued" in statuses:
        group.status = "queued" if statuses == {"queued"} else "running"
    elif statuses == {"completed"}:
        group.status = "completed"
    elif "completed" in statuses:
        group.status = "partial"
    elif statuses <= {"cancelled"}:
        group.status = "cancelled"
    elif statuses <= {"stopped", "cancelled"}:
        group.status = "stopped"
    else:
        group.status = "failed"
    started = [run.started_at for run in runs if run.started_at]
    finished = [run.finished_at for run in runs if run.finished_at]
    group.started_at = min(started) if started else None
    terminal = all(run.status in EVALUATION_TERMINAL_STATUSES for run in runs)
    group.finished_at = max(finished) if terminal and finished else None
    summaries = [dict(run.result_summary or {}) for run in runs]
    episodes = sum(int(item.get("num_episodes", 0) or 0) for item in summaries)
    successes = sum(int(item.get("num_successes", 0) or 0) for item in summaries)
    rates = [float(item["success_rate"]) for item in summaries if item.get("success_rate") is not None]
    success_rate = successes / episodes if episodes else (sum(rates) / len(rates) if rates else 0.0)
    group.result_summary = {
        "num_runs": len(runs),
        "num_completed": sum(run.status == "completed" for run in runs),
        "num_failed": sum(run.status in {"failed", "interrupted"} for run in runs),
        "num_episodes": episodes,
        "num_successes": successes,
        "success_rate": success_rate,
    }
    group.error = "; ".join(run.error for run in runs if run.error)[:10_000]
    if terminal and group.result_path:
        child_results: dict[str, dict[str, Any]] = {}
        for run in runs:
            with contextlib.suppress(OSError, UnicodeError, json.JSONDecodeError, ValidationError):
                child_results[run.id] = read_run_evaluation_result(run)
        children = [
            {
                "evaluation_id": run.id,
                "name": run.name,
                "position": run.position,
                "kind": run.evaluation_kind,
                "status": run.status,
                "checkpoint": run.checkpoint_path,
                "suite": run.suite,
                "result_path": run.result_path,
                "summary": dict(run.result_summary or {}),
                "error": run.error or None,
            }
            for run in runs
        ]
        comparisons: list[dict[str, Any]] = []
        matrices: list[dict[str, Any]] = [
            {**dict(matrix), "metadata": {**dict(matrix.get("metadata") or {}), "evaluation_id": run.id}}
            for run in runs
            for matrix in child_results.get(run.id, {}).get("matrices", [])
            if isinstance(matrix, Mapping)
        ]
        series: list[dict[str, Any]] = [
            {**dict(item), "metadata": {**dict(item.get("metadata") or {}), "evaluation_id": run.id}}
            for run in runs
            for item in child_results.get(run.id, {}).get("series", [])
            if isinstance(item, Mapping)
        ]
        if group.kind == "batch":
            row_values = list(dict.fromkeys(run.checkpoint_path for run in runs))
            column_values = list(dict.fromkeys(run.suite or run.task_set or run.split for run in runs))
            if row_values and column_values:
                by_cell = {
                    (run.checkpoint_path, run.suite or run.task_set or run.split):
                    dict(run.result_summary or {}).get("success_rate")
                    for run in runs
                }
                matrices.append(
                    {
                        "id": "checkpoint_suite_success_rate",
                        "name": "Checkpoint x suite success rate",
                        "row_labels": [Path(value).name or value for value in row_values],
                        "column_labels": column_values,
                        "values": [
                            [by_cell.get((checkpoint, suite)) for suite in column_values]
                            for checkpoint in row_values
                        ],
                        "value_format": "rate",
                        "metadata": {},
                    }
                )
        if group.kind == "online_stdp" and len(runs) >= 2:
            baseline = next((run for run in runs if run.evaluation_kind.endswith("baseline")), runs[0])
            adapted = next((run for run in runs if run.evaluation_kind.endswith("adapted")), runs[-1])
            base_rate = dict(baseline.result_summary or {}).get("success_rate")
            adapted_rate = dict(adapted.result_summary or {}).get("success_rate")
            delta = None
            if base_rate is not None and adapted_rate is not None:
                delta = float(adapted_rate) - float(base_rate)
            comparisons.append(
                {
                    "id": "online_stdp_baseline_pair",
                    "name": "Baseline vs online STDP",
                    "baseline_evaluation_id": baseline.id,
                    "adapted_evaluation_id": adapted.id,
                    "baseline_success_rate": base_rate,
                    "adapted_success_rate": adapted_rate,
                    "delta": delta,
                }
            )
            series.append(
                {
                    "id": "online_stdp_success_rate",
                    "name": "Baseline vs online STDP",
                    "x_label": "Variant",
                    "y_label": "Success rate",
                    "points": [
                        {"x": "Baseline", "y": base_rate},
                        {"x": "Online STDP", "y": adapted_rate},
                    ],
                    "metadata": {},
                }
            )
        payload = {
            "schema_version": "evaluation-result-v2",
            "kind": group.kind,
            "status": "completed" if group.status == "completed" else "partial" if group.status == "partial" else "failed",
            "benchmark": runs[0].benchmark_id,
            "checkpoint": "",
            "suite": {"name": "group"},
            "started_at": group.started_at,
            "finished_at": group.finished_at,
            "duration_seconds": (
                max(0.0, (group.finished_at - group.started_at).total_seconds())
                if group.started_at and group.finished_at
                else None
            ),
            "summary": dict(group.result_summary),
            "tasks": [],
            "episodes": [],
            "videos": [],
            "artifacts": [],
            "matrices": matrices,
            "series": series,
            "comparisons": comparisons,
            "children": children,
            "parameters": {},
            "metadata": {"group_id": group.id},
            "error": group.error or None,
        }
        with contextlib.suppress(OSError):
            write_result_v2(group.result_path, payload)
    return group


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(dict(value), stream, allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _wandb_upload_worker(payload: Mapping[str, Any]) -> dict[str, str]:
    """Upload one completed result from an isolated process."""

    import wandb

    config = dict(payload.get("wandb") or {})
    result = read_evaluation_result(str(payload["result_path"]))
    categories = {str(value) for value in config.get("categories", [])}
    init_config: dict[str, Any] | None = None
    if "config" in categories:
        init_config = {
            "evaluation_id": payload.get("evaluation_id"),
            "benchmark": result["benchmark"],
            "checkpoint": result["checkpoint"],
            "suite": result["suite"],
            "parameters": result["parameters"],
            "model": payload.get("model", {}),
        }
    run = wandb.init(
        project=config.get("project") or "AlphaBrain",
        entity=config.get("entity") or None,
        name=config.get("run_name") or payload.get("name") or None,
        group=config.get("group") or None,
        job_type="evaluation",
        tags=list(config.get("tags") or []),
        notes=config.get("notes") or None,
        config=init_config,
        settings=wandb.Settings(silent=True),
    )
    if run is None:
        raise RuntimeError("W&B did not create a run")
    try:
        metrics: dict[str, Any] = {}
        if "summary" in categories:
            summary = result["summary"]
            metrics.update(
                {
                    "evaluation/success_rate": summary.get("success_rate", 0.0),
                    "evaluation/num_tasks": summary.get("num_tasks", 0),
                    "evaluation/num_episodes": summary.get("num_episodes", 0),
                    "evaluation/num_successes": summary.get("num_successes", 0),
                    "evaluation/duration_seconds": result.get("duration_seconds") or 0.0,
                }
            )
        if "tasks" in categories:
            table = wandb.Table(
                columns=["task_id", "task_name", "num_episodes", "num_successes", "success_rate"]
            )
            for task in result["tasks"]:
                table.add_data(
                    task["id"], task["name"], task["num_episodes"], task["num_successes"], task["success_rate"]
                )
                metrics[f"tasks/{task['id']}/success_rate"] = task["success_rate"]
            metrics["evaluation/tasks"] = table
        if "videos" in categories:
            output_dir = Path(str(payload["output_dir"])).resolve()
            videos: list[Any] = []
            for item in result["videos"]:
                path = Path(str(item["path"]))
                path = path.resolve() if path.is_absolute() else (output_dir / path).resolve()
                try:
                    path.relative_to(output_dir)
                except ValueError:
                    continue
                if path.is_file():
                    videos.append(wandb.Video(str(path), format=path.suffix.lstrip(".") or "mp4"))
            if videos:
                metrics["evaluation/videos"] = videos
        if metrics:
            run.log(metrics)
        run_url = str(getattr(run, "url", "") or "")
    finally:
        run.finish()
    return {"run_url": run_url}


class EvaluationManager:
    def __init__(
        self,
        config: RuntimeConfig,
        database,
        gpu_monitor: GPUMonitor,
        *,
        lock: asyncio.Lock | None = None,
        graceful_stop_timeout: float | None = None,
        controller_secret_store: SecureSecretStore | None = None,
        source_catalog_provider: Callable[[], Mapping[str, Any]] | None = None,
    ):
        self.config = config
        self.database = database
        self.gpu_monitor = gpu_monitor
        self.lock = lock or asyncio.Lock()
        self._stop = asyncio.Event()
        self._waiters: dict[str, asyncio.Task] = {}
        self._escalations: dict[str, asyncio.Task] = {}
        self._uploads: dict[str, asyncio.Task] = {}
        configured_timeout = config.stop_grace_seconds if graceful_stop_timeout is None else graceful_stop_timeout
        self._graceful_stop_timeout = max(0.0, float(configured_timeout))
        self._wandb_lock = asyncio.Lock()
        self.controller_secret_store = controller_secret_store or SecureSecretStore(
            config.state_dir,
            "deployment-controller",
        )
        self.source_catalog_provider = source_catalog_provider

    async def start(self) -> None:
        self._stop.clear()
        await self._reconcile_existing()

    async def shutdown(self) -> None:
        self._stop.set()
        tasks = [*self._waiters.values(), *self._escalations.values(), *self._uploads.values()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._waiters.clear()
        self._escalations.clear()
        self._uploads.clear()

    async def start_queued_unlocked(self, evaluation_id: str) -> None:
        """Reserve one GPU and start a queued evaluation.

        The caller owns the shared scheduler lock; this method intentionally
        does not acquire it so jobs, deployments, and evaluations can share a
        single FIFO decision.
        """

        with self.database.session() as db:
            evaluation = db.get(EvaluationRun, evaluation_id)
            if evaluation is None or evaluation.status != "queued":
                return
            managed = evaluation.source_kind == "managed_deployment"
            if managed:
                deployment = (
                    db.get(ModelDeployment, evaluation.deployment_id)
                    if evaluation.deployment_id
                    else None
                )
                failure = ""
                chosen: list[int] = []
                if deployment is None:
                    failure = "The managed deployment no longer exists."
                elif deployment.status != "running":
                    failure = "The managed deployment is no longer running."
                elif not 1 <= int(deployment.port or 0) <= 65535:
                    failure = "The managed deployment has no valid server port."
                else:
                    try:
                        chosen = [int(value) for value in deployment.assigned_gpu_ids]
                    except (TypeError, ValueError):
                        chosen = []
                    reserved = {
                        int(row.gpu_index)
                        for row in db.execute(
                            select(DeploymentGPUReservation).where(
                                DeploymentGPUReservation.deployment_id == deployment.id
                            )
                        ).scalars()
                    }
                    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen).issubset(reserved):
                        failure = "The managed deployment GPU assignment is invalid."
                    else:
                        try:
                            controller_key = self.controller_secret_store.read(deployment.id)
                        except (OSError, RuntimeError, PermissionError, ValueError):
                            controller_key = None
                        if not _valid_controller_key(controller_key):
                            failure = "The managed deployment UI controller credential is unavailable."
                if failure:
                    evaluation.status = "failed"
                    evaluation.error = failure
                    evaluation.finished_at = utcnow()
                    refresh_evaluation_group(db, evaluation.group_id)
                    return
                evaluation.server_port = int(deployment.port)
                evaluation.requested_gpu_count = 0
                evaluation.requested_gpu_ids = []
            else:
                snapshot = self.gpu_monitor.snapshot(_merged_reservations(db))
                if not snapshot:
                    return
                by_index = {gpu.index: gpu for gpu in snapshot}
                requested_count = max(1, int(evaluation.requested_gpu_count or 1))
                if requested_count > 1 and evaluation.evaluation_kind not in {"cl_matrix", "rl_iterations"}:
                    evaluation.status = "failed"
                    evaluation.error = "Only CL matrix and RL iteration evaluations may reserve multiple GPUs."
                    evaluation.finished_at = utcnow()
                    refresh_evaluation_group(db, evaluation.group_id)
                    return
                if evaluation.requested_gpu_ids:
                    chosen = [int(value) for value in evaluation.requested_gpu_ids]
                    if len(chosen) != requested_count or any(index not in by_index for index in chosen):
                        evaluation.status = "failed"
                        evaluation.error = "The requested evaluation GPU is no longer visible."
                        evaluation.finished_at = utcnow()
                        refresh_evaluation_group(db, evaluation.group_id)
                        return
                    selected = [by_index[index] for index in chosen]
                    if any(gpu.error for gpu in selected):
                        evaluation.status = "failed"
                        evaluation.error = "One or more requested evaluation GPUs cannot be inspected."
                        evaluation.finished_at = utcnow()
                        refresh_evaluation_group(db, evaluation.group_id)
                        return
                    if any(not gpu.available for gpu in selected):
                        return
                else:
                    healthy = [gpu for gpu in snapshot if not gpu.error]
                    if not healthy:
                        evaluation.status = "failed"
                        evaluation.error = "No inspectable GPU is visible."
                        evaluation.finished_at = utcnow()
                        return
                    available = [gpu.index for gpu in healthy if gpu.available]
                    if len(available) < requested_count:
                        return
                    chosen = available[:requested_count]

                active_ports = {
                    int(value)
                    for value in db.execute(
                        select(ModelDeployment.port).where(
                            ModelDeployment.status.in_({"starting", "running", "stopping"}),
                            ModelDeployment.port.is_not(None),
                        )
                    ).scalars()
                    if value is not None
                }
                active_ports.update(
                    int(value)
                    for value in db.execute(
                        select(EvaluationRun.server_port).where(
                            EvaluationRun.status.in_(EVALUATION_ACTIVE_STATUSES),
                            EvaluationRun.id != evaluation.id,
                            EvaluationRun.server_port.is_not(None),
                        )
                    ).scalars()
                    if value is not None
                )
                evaluation.server_port = (
                    None
                    if evaluation.evaluation_kind in SPECIALIZED_EVALUATION_KINDS
                    else self._allocate_port(active_ports)
                )
                for gpu_index in chosen:
                    db.add(EvaluationGPUReservation(gpu_index=gpu_index, evaluation_id=evaluation.id))
            evaluation.assigned_gpu_ids = chosen
            evaluation.status = "starting"
            evaluation.started_at = utcnow()
            evaluation.finished_at = None
            evaluation.stop_requested_at = None
            evaluation.exit_code = None
            evaluation.error = ""
            refresh_evaluation_group(db, evaluation.group_id)
        await self._spawn(evaluation_id)

    @staticmethod
    def _allocate_port(excluded: set[int]) -> int:
        for _attempt in range(32):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = int(sock.getsockname()[1])
            if port not in excluded:
                return port
        raise RuntimeError("Unable to allocate a free evaluation model-server port")

    def _runtime_config(
        self,
        evaluation: EvaluationRun,
        *,
        model_python: str,
    ) -> tuple[dict[str, Any], list[str]]:
        evaluation_kind = str(getattr(evaluation, "evaluation_kind", "standard"))
        source_kind = str(getattr(evaluation, "source_kind", "temporary_checkpoint"))
        parameters = dict(evaluation.parameters or {})
        suite_name = evaluation.suite or evaluation.task_set or str(parameters.get("env_name", ""))
        benchmark = get_benchmark(evaluation.benchmark_id, include_experimental=True) or {}
        client_entrypoint = str(benchmark.get("client_entrypoint", ""))
        if evaluation.benchmark_id == "libero" and evaluation.adapter_id == "cosmos_policy_websocket":
            client_entrypoint = "benchmarks/LIBERO/eval/eval_libero_cosmos.py"
        if source_kind == "managed_deployment":
            if not evaluation.assigned_gpu_ids or not 1 <= int(evaluation.server_port or 0) <= 65535:
                raise ValueError("Managed evaluation endpoint is incomplete.")
            mode = {
                "type": "eval",
                "reuse_server": True,
                "checkpoint": evaluation.checkpoint_path,
                "benchmark": evaluation.benchmark_id,
                "task_suite": suite_name,
                "task_set": evaluation.task_set,
                "split": evaluation.split,
                "num_trials": int(parameters.get("num_trials", parameters.get("n_episodes", 1))),
                "host": "127.0.0.1",
                "port": int(evaluation.server_port),
                "gpu_id": int(evaluation.assigned_gpu_ids[0]),
                "use_bf16": True,
                # The runner uses this interpreter only for progress/result
                # helpers.  It never starts a model server in reuse mode.
                "server_python": model_python,
                "server_entrypoint": "",
                "server_args": [],
                "client_python": "",
                "client_entrypoint": client_entrypoint,
                "env_name": str(parameters.get("env_name", "")),
                "n_episodes": int(parameters.get("n_episodes", parameters.get("num_trials", 1))),
                "n_envs": int(parameters.get("n_envs", 1)),
                "max_episode_steps": int(parameters.get("max_episode_steps", 1440)),
                "n_action_steps": int(parameters.get("n_action_steps", 16)),
                "task_ids": str(parameters.get("task_ids", "")),
                "task_list": str(parameters.get("task_list", "")),
                "task_limit": int(parameters.get("task_limit", 0)),
                "sort_tasks": bool(parameters.get("sort_tasks", True)),
                "seed": int(parameters.get("seed", 7)),
                "num_views": int(parameters.get("num_views", 2)),
                "output_dir": evaluation.output_dir,
                "numba_disable_jit": int(parameters.get("numba_disable_jit", 1)),
                "mujoco_gl": str(parameters.get("mujoco_gl", "egl")),
                "pyopengl_platform": str(parameters.get("pyopengl_platform", "egl")),
                "predict_video": evaluation_kind == "world_model_video"
                or bool(parameters.get("predict_video", False)),
            }
            return {"modes": {"ui_evaluation": mode}}, []

        model_parameters = dict(evaluation.model_parameters or {})
        model_parameters["port"] = int(evaluation.server_port or 0)
        model_parameters.setdefault("idle_timeout_seconds", -1)
        resolved_server = build_adapter_command(
            evaluation.adapter_id,
            evaluation.checkpoint_path,
            combination_id=evaluation.combination_id,
            backbone_id=evaluation.backbone_id,
            action_head_id=evaluation.action_head_id,
            python_executable=model_python,
            repo_root=self.config.repo_root,
            parameters=model_parameters,
            include_experimental=True,
            source_catalog=(
                self.source_catalog_provider()
                if self.source_catalog_provider is not None
                else None
            ),
        )
        if not resolved_server.get("valid") or len(resolved_server.get("command", [])) < 2:
            codes = sorted(
                {
                    str(item.get("code", "invalid_server_command"))
                    for item in resolved_server.get("issues", [])
                    if isinstance(item, Mapping)
                }
            )
            raise ValueError(f"Evaluation model-server command is invalid ({', '.join(codes)}).")
        command = [str(value) for value in resolved_server["command"]]
        # Temporary evaluation services are unauthenticated legacy endpoints
        # and must never listen on the laboratory LAN.  Rewrite adapter-provided
        # values too, so a future registry entry cannot weaken this invariant.
        host_argument_found = False
        index = 0
        while index < len(command):
            argument = command[index]
            if argument == "--host":
                if index + 1 >= len(command):
                    raise ValueError("Evaluation model-server --host requires a value.")
                command[index + 1] = "127.0.0.1"
                host_argument_found = True
                index += 2
                continue
            if argument.startswith("--host="):
                command[index] = "--host=127.0.0.1"
                host_argument_found = True
            index += 1
        if not host_argument_found:
            command.extend(["--host", "127.0.0.1"])
        entrypoint = command[1]
        try:
            entrypoint = str(Path(entrypoint).resolve().relative_to(self.config.repo_root.resolve()))
        except ValueError:
            raise ValueError("Evaluation model-server entrypoint is outside the repository.") from None
        mode = {
            "type": "eval",
            "reuse_server": False,
            "checkpoint": evaluation.checkpoint_path,
            "benchmark": evaluation.benchmark_id,
            "task_suite": suite_name,
            "task_set": evaluation.task_set,
            "split": evaluation.split,
            "num_trials": int(parameters.get("num_trials", parameters.get("n_episodes", 1))),
            "host": "127.0.0.1",
            "port": int(evaluation.server_port or 0),
            # run_eval explicitly sets CUDA_VISIBLE_DEVICES for its server and
            # client commands, so this must remain the physical NVML index.
            "gpu_id": int(evaluation.assigned_gpu_ids[0]),
            "use_bf16": str(model_parameters.get("precision", "bf16")) != "fp32",
            "server_python": model_python,
            "server_entrypoint": entrypoint,
            "server_args": command[2:],
            "client_python": "",
            "client_entrypoint": client_entrypoint,
            "env_name": str(parameters.get("env_name", "")),
            "n_episodes": int(parameters.get("n_episodes", parameters.get("num_trials", 1))),
            "n_envs": int(parameters.get("n_envs", 1)),
            "max_episode_steps": int(parameters.get("max_episode_steps", 1440)),
            "n_action_steps": int(parameters.get("n_action_steps", 16)),
            "task_ids": str(parameters.get("task_ids", "")),
            "task_list": str(parameters.get("task_list", "")),
            "task_limit": int(parameters.get("task_limit", 0)),
            "sort_tasks": bool(parameters.get("sort_tasks", True)),
            "seed": int(parameters.get("seed", 7)),
            "num_views": int(parameters.get("num_views", 2)),
            "output_dir": evaluation.output_dir,
            "numba_disable_jit": int(parameters.get("numba_disable_jit", 1)),
            "mujoco_gl": str(parameters.get("mujoco_gl", "egl")),
            "pyopengl_platform": str(parameters.get("pyopengl_platform", "egl")),
            "predict_video": evaluation_kind == "world_model_video"
            or bool(parameters.get("predict_video", False)),
        }
        return {"modes": {"ui_evaluation": mode}}, command

    async def _spawn(self, evaluation_id: str) -> None:
        with self.database.session() as db:
            evaluation = db.get(EvaluationRun, evaluation_id)
            if evaluation is None or evaluation.status != "starting":
                return
            settings = SettingsService(db)
            managed = evaluation.source_kind == "managed_deployment"
            controller_key: str | None = None
            if managed:
                deployment = (
                    db.get(ModelDeployment, evaluation.deployment_id)
                    if evaluation.deployment_id
                    else None
                )
                try:
                    controller_key = (
                        self.controller_secret_store.read(deployment.id)
                        if deployment is not None
                        else None
                    )
                except (OSError, RuntimeError, PermissionError, ValueError):
                    controller_key = None
                if (
                    deployment is None
                    or deployment.status != "running"
                    or not 1 <= int(deployment.port or 0) <= 65535
                    or not deployment.assigned_gpu_ids
                    or not _valid_controller_key(controller_key)
                ):
                    evaluation.status = "failed"
                    evaluation.error = "The managed deployment cannot be reused safely."
                    evaluation.finished_at = utcnow()
                    refresh_evaluation_group(db, evaluation.group_id)
                    return
                evaluation.server_port = int(deployment.port)
                evaluation.assigned_gpu_ids = [int(value) for value in deployment.assigned_gpu_ids]
                model_python = sys.executable
            else:
                model_python = str(settings.get("model_server_python", "") or "").strip()
                model_python = (
                    model_python
                    or os.environ.get("ALPHABRAIN_PYTHON", "").strip()
                    or sys.executable
                )
            output_dir = Path(evaluation.output_dir).expanduser().resolve(strict=False)
            evaluation.output_dir = str(output_dir)
            evaluation.config_path = str(output_dir / "evaluation-config.yaml")
            evaluation.log_path = str(output_dir / "runner.log")
            evaluation.progress_path = str(output_dir / "progress.jsonl")
            evaluation.result_schema_version = (
                "evaluation-result-v2"
                if evaluation.evaluation_kind in SPECIALIZED_EVALUATION_KINDS | {"world_model_video"}
                else "evaluation-result-v1"
            )
            evaluation.result_path = str(output_dir / f"{evaluation.result_schema_version}.json")
            log_handle = None
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
                direct = evaluation.evaluation_kind in SPECIALIZED_EVALUATION_KINDS
                if self.config.demo_mode:
                    config_value = {
                        "kind": evaluation.evaluation_kind,
                        "benchmark": evaluation.benchmark_id,
                        "checkpoint": evaluation.checkpoint_path,
                        "suite": evaluation.suite,
                        "parameters": dict(evaluation.parameters or {}),
                        "gpu_ids": list(evaluation.assigned_gpu_ids or []),
                        "demo": True,
                    }
                elif direct:
                    config_value = {
                        "kind": evaluation.evaluation_kind,
                        "checkpoint": evaluation.checkpoint_path,
                        "suite": evaluation.suite,
                        "parameters": dict(evaluation.parameters or {}),
                        "gpu_ids": list(evaluation.assigned_gpu_ids or []),
                    }
                else:
                    config_value, _server_command = self._runtime_config(
                        evaluation, model_python=model_python
                    )
                _atomic_yaml(Path(evaluation.config_path), config_value)
                Path(evaluation.config_path).chmod(0o444)
                if self.config.demo_mode:
                    command = [
                        model_python,
                        "-m",
                        "alphabrain_ui.demo_worker",
                        "evaluate",
                        "--result-path",
                        evaluation.result_path,
                        "--progress-path",
                        evaluation.progress_path,
                        "--schema-version",
                        evaluation.result_schema_version,
                        "--kind",
                        evaluation.evaluation_kind,
                        "--benchmark",
                        evaluation.benchmark_id,
                        "--checkpoint",
                        evaluation.checkpoint_path,
                        "--suite",
                        evaluation.suite,
                    ]
                elif direct:
                    command = [
                        model_python,
                        "-m",
                        "alphabrain_ui.specialized_evaluation",
                        "--kind",
                        evaluation.evaluation_kind,
                        "--repo-root",
                        str(self.config.repo_root),
                        "--output-dir",
                        evaluation.output_dir,
                        "--result-path",
                        evaluation.result_path,
                        "--checkpoint",
                        evaluation.checkpoint_path,
                        "--suite",
                        evaluation.suite,
                        "--gpu-ids",
                        ",".join(str(value) for value in evaluation.assigned_gpu_ids),
                        "--parameters-json",
                        json.dumps(dict(evaluation.parameters or {}), ensure_ascii=False),
                    ]
                else:
                    command = [
                        "bash",
                        str(self.config.repo_root / "scripts/run_eval.sh"),
                        "ui_evaluation",
                        evaluation.config_path,
                    ]
                evaluation.command = command
                env = read_dotenv(self.config.repo_root / ".env")
                env.update(os.environ)
                env.update(
                    {
                        str(key): str(value)
                        for key, value in dict(settings.get("environment", {}) or {}).items()
                    }
                )
                env.update({str(key): str(value) for key, value in dict(evaluation.environment or {}).items()})
                # Evaluation itself never receives the W&B credential; upload
                # happens later in an isolated worker using WandbSecretStore.
                env.pop(WANDB_API_KEY_ENV, None)
                # Never inherit or persist a controller credential.  Managed
                # reuse injects the exact deployment key only into this child.
                env.pop(POLICY_API_KEY_ENV, None)
                env.update(
                    {
                        "ALPHABRAIN_UI_MANAGED": "1",
                        "EVAL_OUTPUT_DIR": evaluation.output_dir,
                        "EVAL_CONFIG_FILE": evaluation.config_path,
                        "EVAL_PROGRESS_PATH": evaluation.progress_path,
                        "EVAL_TASK_LIMIT": str(int(dict(evaluation.parameters or {}).get("task_limit", 0))),
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": str(self.config.repo_root) + os.pathsep + env.get("PYTHONPATH", ""),
                    }
                )
                if direct:
                    # CL/RL launchers shard physical NVML IDs themselves.
                    env.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(
                        str(value) for value in evaluation.assigned_gpu_ids
                    )
                if managed and controller_key is not None:
                    env[POLICY_API_KEY_ENV] = controller_key
                python_path = Path(model_python).expanduser()
                if python_path.parent != Path("."):
                    env["PATH"] = str(python_path.parent) + os.pathsep + env.get("PATH", "")
                evaluation.environment = {
                    str(key): str(value)
                    for key, value in dict(evaluation.environment or {}).items()
                    if str(key) not in {WANDB_API_KEY_ENV, POLICY_API_KEY_ENV}
                }
                log_path = Path(evaluation.log_path)
                log_handle = log_path.open("ab", buffering=0)
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(self.config.repo_root),
                    env=env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    **new_process_group_kwargs(),
                )
            except FileExistsError:
                if log_handle is not None:
                    log_handle.close()
                evaluation.status = "failed"
                evaluation.error = "Evaluation output directory already exists."
                evaluation.finished_at = utcnow()
                self._release_reservations(db, evaluation.id)
                refresh_evaluation_group(db, evaluation.group_id)
                return
            except Exception:
                if log_handle is not None:
                    log_handle.close()
                evaluation.status = "failed"
                evaluation.error = "Failed to prepare or start the evaluation runner."
                evaluation.finished_at = utcnow()
                self._release_reservations(db, evaluation.id)
                refresh_evaluation_group(db, evaluation.group_id)
                return
            evaluation.pid = process.pid
            evaluation.pgid = process.pid
            evaluation.status = "running"
            with contextlib.suppress(psutil.Error):
                evaluation.process_created_at = psutil.Process(process.pid).create_time()
            refresh_evaluation_group(db, evaluation.group_id)
        waiter = asyncio.create_task(
            self._monitor_process(evaluation_id, process, log_handle),
            name=f"evaluation-{evaluation_id}",
        )
        self._waiters[evaluation_id] = waiter

    async def _monitor_process(
        self,
        evaluation_id: str,
        process: asyncio.subprocess.Process,
        log_handle,
    ) -> None:
        upload_after = False
        try:
            try:
                exit_code = await process.wait()
            finally:
                log_handle.close()
            self._cancel_escalation(evaluation_id)
            with self.database.session() as db:
                evaluation = db.get(EvaluationRun, evaluation_id)
                if evaluation is None:
                    return
                evaluation.exit_code = exit_code
                evaluation.pid = None
                evaluation.pgid = None
                evaluation.process_created_at = None
                parsed: dict[str, Any] | None = None
                try:
                    parsed = read_run_evaluation_result(evaluation)
                except (OSError, json.JSONDecodeError, ValidationError):
                    parsed = None
                if parsed is not None:
                    evaluation.result_summary = dict(parsed.get("summary") or {})
                if evaluation.stop_requested_at is not None:
                    evaluation.status = "stopped"
                elif exit_code != 0:
                    evaluation.status = "failed"
                    if not evaluation.error:
                        evaluation.error = f"Evaluation runner exited with code {exit_code}."
                elif parsed is None:
                    evaluation.status = "failed"
                    evaluation.error = (
                        f"Evaluation completed without a valid {evaluation.result_schema_version}.json artifact."
                    )
                elif parsed.get("status") != "completed":
                    evaluation.status = "failed"
                    evaluation.error = str(parsed.get("error") or "The benchmark reported an incomplete result.")
                else:
                    evaluation.status = "completed"
                    evaluation.error = ""
                    wandb_config = dict(evaluation.wandb or {})
                    if wandb_config.get("enabled") and wandb_config.get("mode") != "disabled":
                        evaluation.wandb_status = "pending"
                        upload_after = True
                    else:
                        evaluation.wandb_status = "disabled"
                evaluation.finished_at = utcnow()
                self._release_reservations(db, evaluation.id)
                refresh_evaluation_group(db, evaluation.group_id)
            if upload_after:
                self._schedule_upload(evaluation_id)
        finally:
            current = self._waiters.get(evaluation_id)
            if current is asyncio.current_task():
                self._waiters.pop(evaluation_id, None)

    @staticmethod
    def _process_matches_identity(pid: int, pgid: int, created_at: float | None) -> bool:
        try:
            process = psutil.Process(pid)
            if created_at is not None and abs(process.create_time() - created_at) > 1:
                return False
            if not process_group_matches(pid, pgid):
                return False
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (OSError, psutil.Error):
            return False

    @staticmethod
    def _matches_process(evaluation: EvaluationRun) -> bool:
        if not evaluation.pid:
            return False
        return EvaluationManager._process_matches_identity(
            int(evaluation.pid),
            int(evaluation.pgid or evaluation.pid),
            evaluation.process_created_at,
        )

    @staticmethod
    def _release_reservations(db, evaluation_id: str) -> None:  # type: ignore[no-untyped-def]
        db.execute(
            delete(EvaluationGPUReservation).where(EvaluationGPUReservation.evaluation_id == evaluation_id)
        )

    def _cancel_escalation(self, evaluation_id: str) -> None:
        task = self._escalations.pop(evaluation_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    def _schedule_escalation(
        self,
        evaluation_id: str,
        *,
        pid: int,
        pgid: int,
        created_at: float | None,
        delay: float | None = None,
    ) -> bool:
        current = self._escalations.get(evaluation_id)
        if current is not None and not current.done():
            return False
        task = asyncio.create_task(
            self._escalate_after_timeout(
                evaluation_id,
                pid=pid,
                pgid=pgid,
                created_at=created_at,
                delay=self._graceful_stop_timeout if delay is None else max(0.0, delay),
            ),
            name=f"kill-evaluation-{evaluation_id}",
        )
        self._escalations[evaluation_id] = task

        def cleanup(done: asyncio.Task) -> None:
            if self._escalations.get(evaluation_id) is done:
                self._escalations.pop(evaluation_id, None)

        task.add_done_callback(cleanup)
        return True

    async def _escalate_after_timeout(
        self,
        evaluation_id: str,
        *,
        pid: int,
        pgid: int,
        created_at: float | None,
        delay: float,
    ) -> None:
        await asyncio.sleep(delay)
        with self.database.session() as db:
            evaluation = db.get(EvaluationRun, evaluation_id)
            if (
                evaluation is None
                or evaluation.status not in EVALUATION_ACTIVE_STATUSES
                or evaluation.pid != pid
                or evaluation.pgid != pgid
                or not self._process_matches_identity(pid, pgid, created_at)
            ):
                return
            with contextlib.suppress(ProcessLookupError, PermissionError):
                signal_process_group(pid, pgid, FORCE_KILL_SIGNAL)

    def _remaining_grace_period(self, evaluation: EvaluationRun) -> float:
        if evaluation.stop_requested_at is None:
            return self._graceful_stop_timeout
        elapsed = max(0.0, (utcnow() - evaluation.stop_requested_at).total_seconds())
        return max(0.0, self._graceful_stop_timeout - elapsed)

    async def _reconcile_existing(self) -> None:
        watch_specs: list[tuple[str, int, int, float | None]] = []
        escalation_specs: list[tuple[str, int, int, float | None, float]] = []
        upload_ids: set[str] = set()
        with self.database.session() as db:
            rows = db.execute(
                select(EvaluationRun).where(EvaluationRun.status.in_(EVALUATION_ACTIVE_STATUSES))
            ).scalars().all()
            for evaluation in rows:
                if not self._matches_process(evaluation):
                    was_stopping = evaluation.stop_requested_at is not None
                    parsed = None
                    with contextlib.suppress(
                        OSError,
                        UnicodeError,
                        json.JSONDecodeError,
                        ValidationError,
                    ):
                        parsed = read_run_evaluation_result(evaluation)
                    if parsed is not None:
                        evaluation.result_summary = dict(parsed.get("summary") or {})
                    if was_stopping:
                        evaluation.status = "stopped"
                    elif parsed is not None and parsed.get("status") == "completed":
                        evaluation.status = "completed"
                        evaluation.error = ""
                        wandb_config = dict(evaluation.wandb or {})
                        if wandb_config.get("enabled") and wandb_config.get("mode") != "disabled":
                            evaluation.wandb_status = "pending"
                            upload_ids.add(evaluation.id)
                        else:
                            evaluation.wandb_status = "disabled"
                    elif parsed is not None:
                        evaluation.status = "failed"
                        evaluation.error = str(
                            parsed.get("error") or "The benchmark reported an incomplete result."
                        )
                    else:
                        evaluation.status = "interrupted"
                        evaluation.error = evaluation.error or (
                            "The evaluation process was no longer running when the UI restarted."
                        )
                    evaluation.pid = None
                    evaluation.pgid = None
                    evaluation.process_created_at = None
                    evaluation.finished_at = utcnow()
                    self._release_reservations(db, evaluation.id)
                    refresh_evaluation_group(db, evaluation.group_id)
                    continue
                if evaluation.source_kind != "managed_deployment":
                    for index in evaluation.assigned_gpu_ids:
                        if db.get(EvaluationGPUReservation, int(index)) is None:
                            db.add(EvaluationGPUReservation(gpu_index=int(index), evaluation_id=evaluation.id))
                pid = int(evaluation.pid)
                pgid = int(evaluation.pgid or evaluation.pid)
                watch_specs.append((evaluation.id, pid, pgid, evaluation.process_created_at))
                if evaluation.status == "stopping":
                    escalation_specs.append(
                        (
                            evaluation.id,
                            pid,
                            pgid,
                            evaluation.process_created_at,
                            self._remaining_grace_period(evaluation),
                        )
                    )

            # A clean shutdown changes an in-flight upload back to pending,
            # but a process crash can leave it marked uploading.  Completed
            # evaluation uploads are idempotent from the UI's perspective, so
            # normalize both recoverable states and resume them after commit.
            upload_rows = db.execute(
                select(EvaluationRun).where(
                    EvaluationRun.status == "completed",
                    EvaluationRun.wandb_status.in_(("pending", "uploading")),
                )
            ).scalars().all()
            for evaluation in upload_rows:
                wandb_config = dict(evaluation.wandb or {})
                if wandb_config.get("enabled") and wandb_config.get("mode") != "disabled":
                    evaluation.wandb_status = "pending"
                    upload_ids.add(evaluation.id)
                else:
                    evaluation.wandb_status = "disabled"
                    evaluation.wandb_error = ""
        for evaluation_id, pid, pgid, created_at in watch_specs:
            self._waiters[evaluation_id] = asyncio.create_task(
                self._watch_detached(evaluation_id, pid, pgid, created_at),
                name=f"detached-evaluation-{evaluation_id}",
            )
        for evaluation_id, pid, pgid, created_at, delay in escalation_specs:
            self._schedule_escalation(
                evaluation_id,
                pid=pid,
                pgid=pgid,
                created_at=created_at,
                delay=delay,
            )
        for evaluation_id in sorted(upload_ids):
            self._schedule_upload(evaluation_id)

    async def _watch_detached(
        self,
        evaluation_id: str,
        pid: int,
        pgid: int,
        created_at: float | None,
    ) -> None:
        try:
            while not self._stop.is_set():
                if not self._process_matches_identity(pid, pgid, created_at):
                    self._cancel_escalation(evaluation_id)
                    upload_after = False
                    with self.database.session() as db:
                        evaluation = db.get(EvaluationRun, evaluation_id)
                        if (
                            evaluation is not None
                            and evaluation.status in EVALUATION_ACTIVE_STATUSES
                            and evaluation.pid == pid
                        ):
                            was_stopping = evaluation.stop_requested_at is not None
                            parsed = None
                            with contextlib.suppress(
                                OSError,
                                UnicodeError,
                                json.JSONDecodeError,
                                ValidationError,
                            ):
                                parsed = read_run_evaluation_result(evaluation)
                            if parsed is not None:
                                evaluation.result_summary = dict(parsed.get("summary") or {})
                            if was_stopping:
                                evaluation.status = "stopped"
                            elif parsed is not None and parsed.get("status") == "completed":
                                evaluation.status = "completed"
                                evaluation.error = ""
                                wandb_config = dict(evaluation.wandb or {})
                                if wandb_config.get("enabled") and wandb_config.get("mode") != "disabled":
                                    evaluation.wandb_status = "pending"
                                    upload_after = True
                                else:
                                    evaluation.wandb_status = "disabled"
                            elif parsed is not None:
                                evaluation.status = "failed"
                                evaluation.error = str(
                                    parsed.get("error")
                                    or "The benchmark reported an incomplete result."
                                )
                            else:
                                evaluation.status = "interrupted"
                                evaluation.error = evaluation.error or (
                                    "The detached evaluation exited while the UI was offline."
                                )
                            evaluation.pid = None
                            evaluation.pgid = None
                            evaluation.process_created_at = None
                            evaluation.finished_at = utcnow()
                            self._release_reservations(db, evaluation.id)
                            refresh_evaluation_group(db, evaluation.group_id)
                    if upload_after:
                        self._schedule_upload(evaluation_id)
                    return
                await asyncio.sleep(self.config.scheduler_interval)
        finally:
            current = self._waiters.get(evaluation_id)
            if current is asyncio.current_task():
                self._waiters.pop(evaluation_id, None)

    async def cancel(self, evaluation_id: str) -> EvaluationRun:
        async with self.lock:
            with self.database.session() as db:
                evaluation = self._get(db, evaluation_id)
                if evaluation.status != "queued":
                    raise ValueError("Only queued evaluations can be cancelled")
                evaluation.status = "cancelled"
                evaluation.finished_at = utcnow()
                refresh_evaluation_group(db, evaluation.group_id)
                return evaluation

    async def stop(self, evaluation_id: str, *, force: bool = False) -> EvaluationRun:
        async with self.lock:
            signal_spec: tuple[int, int, float | None] | None = None
            previous_state: tuple[str, Any] | None = None
            with self.database.session() as db:
                evaluation = self._get(db, evaluation_id)
                if evaluation.status not in EVALUATION_ACTIVE_STATUSES:
                    raise ValueError("Evaluation is not running")
                if not evaluation.pid or not evaluation.pgid:
                    evaluation.status = "stopped"
                    evaluation.stop_requested_at = utcnow()
                    evaluation.finished_at = utcnow()
                    self._release_reservations(db, evaluation.id)
                    refresh_evaluation_group(db, evaluation.group_id)
                    return evaluation
                if not self._matches_process(evaluation):
                    evaluation.status = "interrupted"
                    evaluation.error = "The recorded evaluation process identity no longer matches."
                    evaluation.finished_at = utcnow()
                    evaluation.pid = None
                    evaluation.pgid = None
                    evaluation.process_created_at = None
                    self._release_reservations(db, evaluation.id)
                    refresh_evaluation_group(db, evaluation.group_id)
                    return evaluation
                previous_state = (evaluation.status, evaluation.stop_requested_at)
                evaluation.stop_requested_at = utcnow()
                evaluation.status = "stopping"
                signal_spec = (int(evaluation.pid), int(evaluation.pgid), evaluation.process_created_at)
                result = evaluation
            assert signal_spec is not None and previous_state is not None
            pid, pgid, created_at = signal_spec
            escalation_created = False
            if not force:
                escalation_created = self._schedule_escalation(
                    evaluation_id, pid=pid, pgid=pgid, created_at=created_at
                )
            try:
                signal_process_group(pid, pgid, FORCE_KILL_SIGNAL if force else signal.SIGTERM)
            except ProcessLookupError:
                self._cancel_escalation(evaluation_id)
                with self.database.session() as db:
                    current = self._get(db, evaluation_id)
                    if current.pid == pid and current.pgid == pgid:
                        current.status = "stopped"
                        current.finished_at = utcnow()
                        current.pid = None
                        current.pgid = None
                        current.process_created_at = None
                        self._release_reservations(db, current.id)
                        refresh_evaluation_group(db, current.group_id)
                    result = current
            except PermissionError:
                if escalation_created:
                    self._cancel_escalation(evaluation_id)
                restored = False
                with self.database.session() as db:
                    current = self._get(db, evaluation_id)
                    if current.pid == pid and current.pgid == pgid:
                        current.status, current.stop_requested_at = previous_state
                        restored = True
                    result = current
                if restored:
                    raise ValueError("The evaluation process could not be signalled.") from None
            else:
                if force:
                    self._cancel_escalation(evaluation_id)
            return result

    async def retry_upload(
        self,
        evaluation_id: str,
        secret_store: WandbSecretStore | None = None,
    ) -> EvaluationRun:
        """Upload completed metrics without changing the evaluation outcome."""

        async with self._wandb_lock:
            with self.database.session() as db:
                evaluation = self._get(db, evaluation_id)
                if evaluation.status != "completed":
                    raise ValueError("Only completed evaluations can be uploaded")
                if evaluation.wandb_status == "uploaded":
                    return evaluation
                wandb_config = dict(evaluation.wandb or {})
                if not wandb_config.get("enabled") or wandb_config.get("mode") == "disabled":
                    evaluation.wandb_status = "disabled"
                    evaluation.wandb_error = ""
                    return evaluation
                store = secret_store or WandbSecretStore(self.config.state_dir)
                try:
                    api_key = store.read_api_key() if wandb_config.get("mode") == "online" else None
                except (OSError, RuntimeError, PermissionError) as error:
                    evaluation.wandb_status = "failed"
                    evaluation.wandb_error = f"Unable to read the W&B credential ({type(error).__name__})."
                    return evaluation
                if wandb_config.get("mode") == "online" and not api_key:
                    evaluation.wandb_status = "failed"
                    evaluation.wandb_error = "The global W&B API key is not configured."
                    return evaluation
                evaluation.wandb_status = "uploading"
                evaluation.wandb_error = ""
                configured_environment = dict(SettingsService(db).get("environment", {}) or {})
                worker_payload = {
                    "evaluation_id": evaluation.id,
                    "name": evaluation.name,
                    "result_path": evaluation.result_path,
                    "output_dir": evaluation.output_dir,
                    "wandb": wandb_config,
                    "model": {
                        "combination_id": evaluation.combination_id,
                        "backbone_id": evaluation.backbone_id,
                        "action_head_id": evaluation.action_head_id,
                    },
                }
            child_environment = os.environ.copy()
            child_environment.pop(WANDB_API_KEY_ENV, None)
            child_environment.update(
                {
                    "WANDB_MODE": str(wandb_config.get("mode", "online")),
                    "WANDB_SILENT": "true",
                    "WANDB_CONSOLE": "off",
                    "WANDB_DISABLE_CODE": "true",
                    "WANDB_DISABLE_GIT": "true",
                    "WANDB_DIR": str(Path(str(worker_payload["output_dir"])) / "wandb"),
                }
            )
            if configured_environment.get("WANDB_BASE_URL"):
                child_environment["WANDB_BASE_URL"] = str(configured_environment["WANDB_BASE_URL"])
            if api_key:
                child_environment[WANDB_API_KEY_ENV] = api_key
            process: asyncio.subprocess.Process | None = None
            try:
                Path(child_environment["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "alphabrain_ui.evaluations",
                    "--wandb-upload-worker",
                    cwd=str(self.config.repo_root),
                    env=child_environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    **new_process_group_kwargs(),
                )
                stdout, _stderr = await asyncio.wait_for(
                    process.communicate(json.dumps(worker_payload, ensure_ascii=False).encode("utf-8")),
                    timeout=300,
                )
                if process.returncode != 0:
                    raise RuntimeError("W&B upload worker failed")
                marker = "__ALPHABRAIN_WANDB_UPLOAD__="
                line = next(
                    (value for value in reversed(stdout.decode("utf-8", errors="replace").splitlines()) if value.startswith(marker)),
                    "",
                )
                if not line:
                    raise RuntimeError("W&B upload worker returned no result")
                upload_result = json.loads(line.removeprefix(marker))
                if not isinstance(upload_result, dict):
                    raise RuntimeError("W&B upload worker returned an invalid result")
            except asyncio.CancelledError:
                if process is not None and process.returncode is None:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        signal_process_group(process.pid, process.pid, FORCE_KILL_SIGNAL)
                    with contextlib.suppress(Exception):
                        await process.wait()
                with self.database.session() as db:
                    evaluation = self._get(db, evaluation_id)
                    if evaluation.wandb_status == "uploading":
                        evaluation.wandb_status = "pending"
                raise
            except Exception as error:
                if process is not None and process.returncode is None:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        signal_process_group(process.pid, process.pid, FORCE_KILL_SIGNAL)
                    with contextlib.suppress(Exception):
                        await process.wait()
                safe_error = type(error).__name__
                with self.database.session() as db:
                    evaluation = self._get(db, evaluation_id)
                    evaluation.wandb_status = "failed"
                    evaluation.wandb_error = f"W&B upload failed ({safe_error})."
                    return evaluation
            finally:
                child_environment.pop(WANDB_API_KEY_ENV, None)
                if api_key:
                    api_key = None
            with self.database.session() as db:
                evaluation = self._get(db, evaluation_id)
                evaluation.wandb_status = "uploaded"
                evaluation.wandb_error = ""
                evaluation.wandb_run_url = str(upload_result.get("run_url", ""))
                return evaluation

    def request_upload(
        self,
        evaluation_id: str,
        secret_store: WandbSecretStore | None = None,
    ) -> EvaluationRun:
        """Validate and enqueue a W&B retry without waiting for the worker."""

        current = self._uploads.get(evaluation_id)
        upload_active = current is not None and not current.done()
        should_schedule = False
        with self.database.session() as db:
            evaluation = self._get(db, evaluation_id)
            if evaluation.status != "completed":
                raise ValueError("Only completed evaluations can be uploaded")
            if evaluation.wandb_status == "uploaded" or upload_active:
                return evaluation
            wandb_config = dict(evaluation.wandb or {})
            if not wandb_config.get("enabled") or wandb_config.get("mode") == "disabled":
                evaluation.wandb_status = "disabled"
                evaluation.wandb_error = ""
                return evaluation
            evaluation.wandb_status = "pending"
            evaluation.wandb_error = ""
            should_schedule = True
            result = evaluation
        if should_schedule:
            self._schedule_upload(evaluation_id, secret_store)
        return result

    def _schedule_upload(
        self,
        evaluation_id: str,
        secret_store: WandbSecretStore | None = None,
    ) -> None:
        current = self._uploads.get(evaluation_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self.retry_upload(evaluation_id, secret_store),
            name=f"wandb-evaluation-{evaluation_id}",
        )
        self._uploads[evaluation_id] = task

        def cleanup(done: asyncio.Task) -> None:
            if self._uploads.get(evaluation_id) is done:
                self._uploads.pop(evaluation_id, None)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                done.result()

        task.add_done_callback(cleanup)

    @staticmethod
    def _get(db, evaluation_id: str) -> EvaluationRun:  # type: ignore[no-untyped-def]
        evaluation = db.get(EvaluationRun, evaluation_id)
        if evaluation is None:
            raise KeyError(evaluation_id)
        return evaluation


def _main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wandb-upload-worker", action="store_true")
    args, _unknown = parser.parse_known_args()
    if not args.wandb_upload_worker:
        return 2
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid payload")
        result = _wandb_upload_worker(payload)
    except Exception as error:
        print(
            "__ALPHABRAIN_WANDB_UPLOAD__=" + json.dumps({"error": type(error).__name__}),
            flush=True,
        )
        return 1
    print("__ALPHABRAIN_WANDB_UPLOAD__=" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "EVALUATION_ACTIVE_STATUSES",
    "EVALUATION_TERMINAL_STATUSES",
    "EvaluationManager",
    "build_evaluation_output_dir",
    "read_evaluation_result",
]
