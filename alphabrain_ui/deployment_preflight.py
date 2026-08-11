"""Deployment request validation shared by preflight and submission APIs.

The UI process only inspects checkpoint metadata and filesystem structure.  It
never imports a checkpoint or deserializes model weights; model imports are
probed in the separately configured model-server Python process.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import select

from .database import (
    Checkpoint,
    DeploymentGPUReservation,
    EvaluationGPUReservation,
    GPUReservation,
    ModelDeployment,
)
from .deployment_registry import get_deployment_catalog, resolve_deployment
from .deployments import DEPLOYMENT_ACTIVE_STATUSES, model_server_python
from .schemas import DeploymentRequest
from .services import SettingsService

_REQUIRED_MODEL_SERVER_MODULES = (
    "torch",
    "transformers",
    "websockets",
    "msgpack",
    "numpy",
    "AlphaBrain",
)
_PROBE_CACHE_SECONDS = 30.0
_probe_cache: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}


def _issue(
    level: str,
    code: str,
    field: str,
    zh: str,
    en: str,
    **detail: Any,
) -> dict[str, Any]:
    return {
        "severity": level,
        "level": level,
        "code": code,
        "field": field,
        "path": field,
        "message": zh,
        "message_i18n": {"zh-CN": zh, "en-US": en},
        "detail": detail,
    }


def resolve_python_executable(value: str) -> str | None:
    """Resolve a configured executable without invoking a shell."""

    raw = str(value or "").strip()
    if not raw:
        return None
    expanded = Path(raw).expanduser()
    if expanded.is_absolute() or os.sep in raw:
        candidate = expanded.resolve(strict=False)
        return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    resolved = shutil.which(raw)
    return str(Path(resolved).resolve()) if resolved else None


def probe_model_server_python(
    python_executable: str,
    repo_root: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Import runtime dependencies in the model-server interpreter.

    Results are cached briefly because importing Torch repeatedly can be
    expensive.  The cache is intentionally short so installing a missing
    package while the UI is running is detected without a restart.
    """

    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (environment or {}).items()})
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    # Environment values can affect imports (especially PYTHONPATH), but must
    # never be retained verbatim in the cache key or returned probe details.
    environment_fingerprint = hashlib.sha256(
        json.dumps(sorted(env.items()), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_key = (python_executable, str(repo_root.resolve()), environment_fingerprint)
    cached = _probe_cache.get(cache_key)
    now = time.monotonic()
    if cached is not None and now - cached[0] < _PROBE_CACHE_SECONDS:
        return json.loads(json.dumps(cached[1]))

    marker = "__ALPHABRAIN_MODEL_SERVER_PROBE__="
    script = (
        "import importlib,json\n"
        f"modules={list(_REQUIRED_MODEL_SERVER_MODULES)!r}\n"
        "errors={}\n"
        "for name in modules:\n"
        "  try: importlib.import_module(name)\n"
        "  except Exception as exc: errors[name]=type(exc).__name__\n"
        f"print({marker!r}+json.dumps(errors,ensure_ascii=False),flush=True)\n"
        "raise SystemExit(1 if errors else 0)\n"
    )
    try:
        completed = subprocess.run(
            [python_executable, "-c", script],
            cwd=str(repo_root),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Exception messages may echo environment values from wrappers or
        # import hooks.  The type is sufficient to distinguish launch errors
        # from missing modules without reflecting secrets into the API.
        result = {"ok": False, "errors": {"python": type(exc).__name__}}
        _probe_cache[cache_key] = (now, result)
        return json.loads(json.dumps(result))

    payload: dict[str, str] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            try:
                parsed = json.loads(line.removeprefix(marker))
                if isinstance(parsed, dict):
                    payload = {}
                    for module_name in _REQUIRED_MODEL_SERVER_MODULES:
                        if module_name not in parsed:
                            continue
                        # A module name plus a fixed status is sufficient for
                        # remediation and cannot reflect a forged marker value.
                        payload[module_name] = "ImportFailed"
                else:
                    payload = None
            except json.JSONDecodeError:
                payload = None
            break
    if payload is None:
        # stdout/stderr belongs to imported third-party modules and can contain
        # arbitrary environment-derived text.  Never reflect it into API data.
        payload = {"python": "ProbeResultMissing"}
    elif completed.returncode != 0 and not payload:
        payload = {"python": "ProbeExitedNonZero"}
    result = {"ok": completed.returncode == 0 and not payload, "errors": payload}
    _probe_cache[cache_key] = (now, result)
    return json.loads(json.dumps(result))


def _checkpoint_index(db) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    return [
        {
            "id": row.id,
            "checkpoint_id": row.id,
            "path": row.path,
            "is_complete": row.is_complete,
        }
        for row in db.execute(select(Checkpoint)).scalars().all()
    ]


def _merged_reservations(db) -> dict[int, str]:  # type: ignore[no-untyped-def]
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
    return reservations


def _port_available(port: int, bind_host: str) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
        return True
    except OSError:
        return False


def _advertised_host(payload: DeploymentRequest, runtime_host: str) -> tuple[str, str, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    scope = payload.endpoint.scope
    bind_host = "127.0.0.1" if scope == "local" else "0.0.0.0"
    supplied = str(payload.endpoint.advertised_host or "").strip()
    if scope == "local":
        advertised = "127.0.0.1"
        if supplied and supplied not in {"127.0.0.1", "localhost"}:
            issues.append(
                _issue(
                    "warning",
                    "local_endpoint_host_normalized",
                    "endpoint.advertised_host",
                    "仅本机服务的访问地址已规范化为 127.0.0.1。",
                    "The advertised host for a local-only service was normalized to 127.0.0.1.",
                )
            )
    else:
        fallback = (
            runtime_host
            if runtime_host not in {"", "0.0.0.0", "::", "127.0.0.1", "localhost"}
            else socket.gethostname()
        )
        advertised = supplied or fallback
        if advertised in {"0.0.0.0", "::", "127.0.0.1", "localhost"}:
            issues.append(
                _issue(
                    "error",
                    "lan_advertised_host_unreachable",
                    "endpoint.advertised_host",
                    "局域网部署必须填写其他机器可访问的主机名或 IPv4 地址。",
                    "A LAN deployment needs a hostname or IPv4 address reachable by other machines.",
                )
            )
    if any(token in advertised for token in ("://", "/", " ", "\t", "\n", ":")):
        issues.append(
            _issue(
                "error",
                "advertised_host_invalid",
                "endpoint.advertised_host",
                "访问地址只填写主机名或 IPv4 地址，不要包含协议、端口或路径。",
                "Advertised host must be a hostname or IPv4 address without a scheme, port, or path.",
            )
        )
    return bind_host, advertised, issues


def validate_deployment_request(
    payload: DeploymentRequest,
    *,
    db,
    runtime,
    gpu_monitor,
    include_experimental: bool,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Return a safe, fully resolved deployment preflight result."""

    issues: list[dict[str, Any]] = []
    settings = SettingsService(db)
    # Canonicalise by discriminator.  In particular, an indexed request must
    # never be allowed to smuggle a client-supplied path alongside a valid ID.
    if payload.checkpoint_source.kind == "indexed":
        source = {"kind": "indexed", "checkpoint_id": payload.checkpoint_source.checkpoint_id}
    else:
        source = {"kind": "local", "path": payload.checkpoint_source.path}
    indexed_checkpoint: Checkpoint | None = None
    if payload.checkpoint_source.kind == "indexed":
        checkpoint_id = str(payload.checkpoint_source.checkpoint_id or "")
        indexed_checkpoint = db.get(Checkpoint, checkpoint_id) if checkpoint_id else None
        if indexed_checkpoint is None:
            issues.append(
                _issue(
                    "error",
                    "checkpoint_index_not_found",
                    "checkpoint_source.checkpoint_id",
                    "索引中的 Checkpoint 不存在。",
                    "The indexed checkpoint does not exist.",
                    checkpoint_id=checkpoint_id,
                )
            )
        elif not indexed_checkpoint.is_complete:
            issues.append(
                _issue(
                    "error",
                    "checkpoint_incomplete",
                    "checkpoint_source.checkpoint_id",
                    "该 Checkpoint 未通过完整性标记，不能部署。",
                    "The selected checkpoint is not marked complete and cannot be deployed.",
                    checkpoint_id=checkpoint_id,
                )
            )

    python_value = model_server_python(settings)
    python_executable = resolve_python_executable(python_value)
    if python_executable is None:
        issues.append(
            _issue(
                "error",
                "model_server_python_not_found",
                "settings.model_server_python",
                "模型服务 Python 不存在或不可执行，请在设置中配置正确的环境。",
                "The model-server Python does not exist or is not executable; configure it in Settings.",
                configured_value=python_value,
            )
        )

    runtime_parameters = dict(payload.parameters)
    runtime_parameters["port"] = int(payload.endpoint.port or 10093)
    runtime_parameters["idle_timeout_seconds"] = int(payload.endpoint.idle_timeout_seconds)
    resolved = resolve_deployment(
        source,
        combination_id=payload.combination_id,
        checkpoint_index=_checkpoint_index(db),
        repo_root=runtime.repo_root,
        environment={str(key): str(value) for key, value in settings.get("environment", {}).items()},
        python_executable=python_executable or python_value,
        parameters=runtime_parameters,
        include_experimental=include_experimental,
        source_catalog=source_catalog,
    )
    issues.extend(resolved.get("issues", []))

    full_catalog = get_deployment_catalog(
        include_experimental=True,
        source_catalog=source_catalog,
    )
    selected_combination = next(
        (row for row in full_catalog["deployment_combinations"] if row.get("id") == payload.combination_id),
        None,
    )
    if selected_combination and selected_combination.get("status") == "experimental":
        if not include_experimental:
            issues.append(
                _issue(
                    "error",
                    "experimental_deployment_not_enabled",
                    "combination_id",
                    "该实验性部署组合尚未在系统和当前用户设置中同时启用。",
                    "This experimental deployment combination is not enabled for both the system and current user.",
                )
            )
        elif not payload.acknowledge_experimental:
            issues.append(
                _issue(
                    "error",
                    "experimental_risk_not_acknowledged",
                    "acknowledge_experimental",
                    "请先确认实验性部署可能需要自行补充代码或配置。",
                    "Acknowledge that the experimental deployment may require custom code or configuration.",
                )
            )

    bind_host, advertised_host, endpoint_issues = _advertised_host(payload, runtime.host)
    issues.extend(endpoint_issues)
    if payload.endpoint.port is not None:
        reserved_port = db.execute(
            select(ModelDeployment.id).where(
                ModelDeployment.port == payload.endpoint.port,
                ModelDeployment.status.in_(DEPLOYMENT_ACTIVE_STATUSES | {"queued"}),
            )
        ).scalars().first()
        if reserved_port is not None:
            issues.append(
                _issue(
                    "error",
                    "deployment_port_reserved",
                    "endpoint.port",
                    "该端口已被另一个排队中或运行中的部署占用。",
                    "The port is reserved by another queued or active deployment.",
                    deployment_id=reserved_port,
                    port=payload.endpoint.port,
                )
            )
        elif not _port_available(payload.endpoint.port, bind_host):
            issues.append(
                _issue(
                    "error",
                    "deployment_port_unavailable",
                    "endpoint.port",
                    "该端口当前无法绑定。",
                    "The requested port cannot currently be bound.",
                    port=payload.endpoint.port,
                )
            )

    reservations = _merged_reservations(db)
    snapshot = gpu_monitor.snapshot(reservations)
    visible = {int(gpu.index): gpu for gpu in snapshot}
    healthy = {index: gpu for index, gpu in visible.items() if not gpu.error}
    requested_ids = [int(value) for value in payload.resources.gpu_ids]
    if payload.resources.strategy == "fixed":
        if len(requested_ids) != payload.resources.gpu_count or len(set(requested_ids)) != len(requested_ids):
            issues.append(
                _issue(
                    "error",
                    "fixed_gpu_selection_invalid",
                    "resources.gpu_ids",
                    "固定 GPU 的数量必须与申请数量一致，且不能重复。",
                    "Fixed GPU IDs must be unique and match the requested GPU count.",
                )
            )
        missing = sorted(set(requested_ids) - set(visible))
        failed = sorted(index for index in requested_ids if index in visible and visible[index].error)
        if missing or failed:
            issues.append(
                _issue(
                    "error",
                    "requested_gpus_unavailable",
                    "resources.gpu_ids",
                    "一个或多个指定 GPU 不可见或无法检查。",
                    "One or more requested GPUs are not visible or cannot be inspected.",
                    missing=missing,
                    inspection_failed=failed,
                )
            )
    elif payload.resources.gpu_count > len(healthy):
        issues.append(
            _issue(
                "error",
                "insufficient_visible_gpus",
                "resources.gpu_count",
                "当前可检查的 GPU 数量少于部署申请数量。",
                "Fewer inspectable GPUs are visible than the deployment requests.",
                requested=payload.resources.gpu_count,
                visible=len(healthy),
            )
        )
    busy_ids = sorted(
        index
        for index in requested_ids
        if index in reservations or (index in healthy and not healthy[index].available)
    )
    available_count = sum(1 for gpu in healthy.values() if gpu.available)
    if busy_ids or (payload.resources.strategy == "auto" and available_count < payload.resources.gpu_count):
        issues.append(
            _issue(
                "info",
                "deployment_will_queue_for_gpu",
                "resources",
                "所需 GPU 当前正忙；部署会进入全局 FIFO 队列等待。",
                "The requested GPUs are busy; the deployment will wait in the global FIFO queue.",
                busy_gpu_ids=busy_ids,
            )
        )

    if python_executable is not None:
        probe = probe_model_server_python(
            python_executable,
            runtime.repo_root,
            {str(key): str(value) for key, value in settings.get("environment", {}).items()},
        )
        if not probe["ok"]:
            issues.append(
                _issue(
                    "error",
                    "model_server_dependencies_missing",
                    "settings.model_server_python",
                    "模型服务环境缺少 AlphaBrain 模型加载或 WebSocket 协议所需依赖。",
                    (
                        "The model-server environment is missing dependencies required for "
                        "AlphaBrain model loading or its WebSocket protocol."
                    ),
                    errors=probe["errors"],
                )
            )

    # Registry parameters are the single source of truth. Endpoint-owned
    # fields remain top-level so they cannot drift from the eventual process.
    resolved_parameters = dict(resolved.get("resolved_parameters", {}))
    resolved_parameters.pop("port", None)
    resolved_parameters.pop("idle_timeout_seconds", None)
    checkpoint_path = str(resolved.get("checkpoint_path", ""))
    safe_resolved = {
        "checkpoint_id": indexed_checkpoint.id if indexed_checkpoint is not None else None,
        "checkpoint_path": checkpoint_path,
        "combination_id": resolved.get("combination_id") or payload.combination_id,
        "adapter_id": resolved.get("adapter_id"),
        "backbone_id": resolved.get("backbone_id"),
        "action_head_id": resolved.get("action_head_id"),
        "parameters": resolved_parameters,
        "environment": resolved.get("environment", {}),
        "startup_timeout_seconds": resolved.get("startup_timeout_seconds", 900),
        "python_executable": python_executable,
        "command_preview": resolved.get("command", []),
        "resources": {
            "strategy": payload.resources.strategy,
            "gpu_count": payload.resources.gpu_count,
            "gpu_ids": requested_ids if payload.resources.strategy == "fixed" else [],
        },
        "endpoint": {
            "scope": payload.endpoint.scope,
            "bind_host": bind_host,
            "advertised_host": advertised_host,
            "port": payload.endpoint.port,
            "idle_timeout_seconds": payload.endpoint.idle_timeout_seconds,
        },
    }
    ok = bool(resolved.get("valid")) and not any(item.get("level", item.get("severity")) == "error" for item in issues)
    if ok:
        issues.append(
            _issue(
                "success",
                "deployment_preflight_passed",
                "deployment",
                "部署预检通过，可以提交到全局 GPU 队列。",
                "Deployment preflight passed and can be submitted to the global GPU queue.",
            )
        )
    return {
        "ok": ok,
        "can_submit": ok,
        "items": issues,
        "issues": issues,
        "resolved": safe_resolved,
    }
