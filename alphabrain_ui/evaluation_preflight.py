"""Static validation and safe launch resolution for managed evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
from .deployment_preflight import probe_model_server_python, resolve_python_executable
from .deployment_registry import get_deployment_catalog, resolve_deployment
from .evaluation_registry import (
    compatible_benchmark_ids,
    get_benchmark,
    resolve_benchmark_parameters,
)
from .gpu import storage_snapshot
from .preflight import effective_environment
from .schemas import EvaluationRequest
from .secrets import SecureSecretStore
from .services import SettingsService
from .wandb import WandbSecretStore

_BENCHMARK_IMPORTS = {
    "libero": ("libero", "tyro", "imageio", "requests"),
    "libero_plus": ("libero", "tyro", "imageio", "requests"),
    "robocasa365": ("robocasa", "robosuite", "gymnasium", "tyro"),
    "robocasa_tabletop": ("robocasa", "robosuite", "gymnasium", "tyro"),
}
_BENCHMARK_PYTHON_KEYS = {
    "libero": "LIBERO_PYTHON",
    "libero_plus": "LIBERO_PLUS_PYTHON",
    "robocasa365": "ROBOCASA365_PYTHON",
    "robocasa_tabletop": "ROBOCASA_TABLETOP_PYTHON",
}
_PROBE_CACHE_SECONDS = 30.0
_SAFE_PROBE_ERROR_TYPES = frozenset(
    {
        "AssertionError",
        "AttributeError",
        "EOFError",
        "FileNotFoundError",
        "ImportError",
        "KeyError",
        "ModuleNotFoundError",
        "NameError",
        "NotADirectoryError",
        "OSError",
        "PermissionError",
        "RuntimeError",
        "SubprocessError",
        "SyntaxError",
        "SystemExit",
        "TimeoutExpired",
        "TypeError",
        "Unsupported",
        "ValueError",
    }
)
_probe_cache: dict[
    tuple[str, tuple[str, ...], str, str],
    tuple[float, dict[str, Any]],
] = {}


def _issue(
    level: str,
    code: str,
    field: str | None,
    zh: str,
    en: str,
    **detail: Any,
) -> dict[str, Any]:
    return {
        "level": level,
        "severity": level,
        "code": code,
        "field": field,
        "path": field,
        "message": zh,
        "message_i18n": {"zh-CN": zh, "en-US": en},
        "detail": detail,
    }


def _checkpoint_index(db) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    return [
        {"id": row.id, "path": row.path, "name": row.name, "is_complete": row.is_complete}
        for row in db.execute(select(Checkpoint)).scalars().all()
    ]


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
    return result


def _result_root(runtime, settings: SettingsService) -> Path:  # type: ignore[no-untyped-def]
    configured = list(settings.get("results_roots", ["results"]) or ["results"])[0]
    root = Path(str(configured)).expanduser()
    if not root.is_absolute():
        root = runtime.repo_root / root
    return root.resolve(strict=False) / "evaluation" / "ui"


def _configured_executable(value: str, repo_root: Path) -> str | None:
    value = str(value).strip()
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and (repo_root / candidate).is_file():
        candidate = repo_root / candidate
    if candidate.parent != Path(".") or candidate.is_absolute():
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return None
        return str(resolved) if resolved.is_file() and os.access(resolved, os.X_OK) else None
    return shutil.which(value)


def probe_benchmark_python(
    python_executable: str,
    modules: tuple[str, ...],
    repo_root: Path,
    environment: Mapping[str, str] | None = None,
    *,
    entrypoint: str | Path | None = None,
    pythonpath_entries: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Load the real benchmark client without reflecting third-party output.

    The probe runs in the simulator's Python interpreter and imports the exact
    client file that the launcher will execute.  Importing in a subprocess
    keeps module-level simulator state out of the UI process, while loading the
    entrypoint catches adapter and sibling-module failures that coarse package
    imports alone cannot detect.
    """

    repo_root = repo_root.expanduser().resolve(strict=False)
    entrypoint_path = (
        Path(entrypoint).expanduser().resolve(strict=False)
        if entrypoint is not None
        else None
    )
    if entrypoint_path is not None:
        try:
            entrypoint_path.relative_to(repo_root)
        except ValueError:
            return {"ok": False, "errors": {"entrypoint": "ValueError"}}
        if not entrypoint_path.is_file() or entrypoint_path.suffix != ".py":
            return {"ok": False, "errors": {"entrypoint": "FileNotFoundError"}}
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (environment or {}).items()})
    search_paths = [str(repo_root)]
    if entrypoint_path is not None:
        search_paths.append(str(entrypoint_path.parent))
    search_paths.extend(
        str(Path(value).expanduser().resolve(strict=False))
        for value in pythonpath_entries
    )
    search_paths.extend(value for value in env.get("PYTHONPATH", "").split(os.pathsep) if value)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(search_paths))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    fingerprint = hashlib.sha256(
        json.dumps(sorted(env.items()), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_key = (python_executable, modules, str(entrypoint_path or ""), fingerprint)
    now = time.monotonic()
    cached = _probe_cache.get(cache_key)
    if cached is not None and now - cached[0] < _PROBE_CACHE_SECONDS:
        return json.loads(json.dumps(cached[1]))
    marker = "__ALPHABRAIN_BENCHMARK_PROBE__="
    script = (
        "import importlib,importlib.util,json,pathlib,sys\n"
        f"modules={list(modules)!r}\n"
        f"repo_root=pathlib.Path({str(repo_root)!r}).resolve()\n"
        f"entrypoint={str(entrypoint_path)!r}\n"
        "errors={}\n"
        "if sys.version_info < (3,10): errors['python_version']='Unsupported'\n"
        "for name in modules:\n"
        "  try: importlib.import_module(name)\n"
        "  except BaseException as exc: errors[name]=type(exc).__name__\n"
        "if entrypoint and not errors:\n"
        "  try:\n"
        "    path=pathlib.Path(entrypoint).resolve(strict=True)\n"
        "    path.relative_to(repo_root)\n"
        "    if not path.is_file() or path.suffix != '.py': raise ValueError()\n"
        "    for value in (str(path.parent),str(repo_root)):\n"
        "      if value not in sys.path: sys.path.insert(0,value)\n"
        "    spec=importlib.util.spec_from_file_location('_alphabrain_benchmark_probe',path)\n"
        "    if spec is None or spec.loader is None: raise ImportError()\n"
        "    module=importlib.util.module_from_spec(spec)\n"
        "    sys.modules[spec.name]=module\n"
        "    try: spec.loader.exec_module(module)\n"
        "    finally: sys.modules.pop(spec.name,None)\n"
        "  except BaseException as exc: errors['entrypoint']=type(exc).__name__\n"
        f"print({marker!r}+json.dumps(errors),flush=True)\n"
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
    except (OSError, subprocess.SubprocessError) as error:
        error_type = type(error).__name__
        if error_type not in _SAFE_PROBE_ERROR_TYPES:
            error_type = "SubprocessError"
        result = {"ok": False, "errors": {"python": error_type}}
        _probe_cache[cache_key] = (now, result)
        return json.loads(json.dumps(result))
    parsed: dict[str, str] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if not line.startswith(marker):
            continue
        try:
            raw = json.loads(line.removeprefix(marker))
        except json.JSONDecodeError:
            break
        if isinstance(raw, dict):
            parsed = {}
            for module in (*modules, "entrypoint", "python_version"):
                raw_error = str(raw.get(module, ""))
                if raw_error:
                    parsed[module] = (
                        raw_error
                        if raw_error in _SAFE_PROBE_ERROR_TYPES
                        else "ImportFailed"
                    )
        break
    if parsed is None:
        parsed = {"python": "ProbeResultMissing"}
    elif completed.returncode != 0 and not parsed:
        parsed = {"python": "ProbeExitedNonZero"}
    result = {"ok": completed.returncode == 0 and not parsed, "errors": parsed}
    _probe_cache[cache_key] = (now, result)
    return json.loads(json.dumps(result))


def _translate_registry_issues(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    for item in items:
        code = str(item.get("code", "benchmark_parameter_invalid"))
        field = item.get("field")
        detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
        translated.append(
            _issue(
                "error",
                code,
                str(field) if field else "parameters",
                "评测参数无效，请检查专家参数。",
                str(item.get("message") or "The benchmark parameters are invalid."),
                **detail,
            )
        )
    return translated


def validate_evaluation_request(
    payload: EvaluationRequest,
    *,
    db,
    runtime,
    gpu_monitor,
    include_experimental: bool,
    wandb_configured: bool | None = None,
    controller_secret_store: SecureSecretStore | None = None,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Return a credential-free, fully resolved evaluation preflight."""

    issues: list[dict[str, Any]] = []
    settings = SettingsService(db)
    configured_environment = {
        str(key): str(value) for key, value in dict(settings.get("environment", {}) or {}).items()
    }
    environment = effective_environment(runtime.repo_root, configured_environment)

    managed = payload.source_kind == "managed_deployment"
    managed_row: ModelDeployment | None = None
    deployment_valid = True
    indexed_checkpoint: Checkpoint | None = None
    assigned_deployment_gpu_ids: list[int] = []
    if managed:
        managed_row = db.get(ModelDeployment, str(payload.deployment_id or ""))
        if managed_row is None:
            deployment_valid = False
            issues.append(
                _issue(
                    "error", "managed_deployment_not_found", "deployment_id",
                    "所选托管部署不存在。", "The selected managed deployment does not exist.",
                )
            )
        else:
            indexed_checkpoint = (
                db.get(Checkpoint, managed_row.checkpoint_id)
                if managed_row.checkpoint_id
                else None
            )
            if managed_row.status != "running":
                deployment_valid = False
                issues.append(
                    _issue(
                        "error", "managed_deployment_not_running", "deployment_id",
                        "所选托管部署当前未运行。", "The selected managed deployment is not running.",
                        status=managed_row.status,
                    )
                )
            port = int(managed_row.port or 0)
            if not 1 <= port <= 65535:
                deployment_valid = False
                issues.append(
                    _issue(
                        "error", "managed_deployment_port_missing", "deployment_id",
                        "托管部署没有有效的服务端口。", "The managed deployment has no valid server port.",
                    )
                )
            try:
                assigned_deployment_gpu_ids = [int(value) for value in managed_row.assigned_gpu_ids]
            except (TypeError, ValueError):
                assigned_deployment_gpu_ids = []
            if not assigned_deployment_gpu_ids or len(set(assigned_deployment_gpu_ids)) != len(
                assigned_deployment_gpu_ids
            ):
                deployment_valid = False
                issues.append(
                    _issue(
                        "error", "managed_deployment_gpu_missing", "deployment_id",
                        "托管部署没有有效的 GPU 分配。", "The managed deployment has no valid GPU assignment.",
                    )
                )
            else:
                reserved_gpu_ids = {
                    int(row.gpu_index)
                    for row in db.execute(
                        select(DeploymentGPUReservation).where(
                            DeploymentGPUReservation.deployment_id == managed_row.id
                        )
                    ).scalars()
                }
                if not set(assigned_deployment_gpu_ids).issubset(reserved_gpu_ids):
                    deployment_valid = False
                    issues.append(
                        _issue(
                            "error", "managed_deployment_gpu_reservation_missing", "deployment_id",
                            "托管部署的 GPU 占用记录不完整。",
                            "The managed deployment GPU reservation is incomplete.",
                        )
                    )
            secret_store = controller_secret_store or SecureSecretStore(
                runtime.state_dir, "deployment-controller"
            )
            try:
                controller_key = secret_store.read(managed_row.id)
                controller_configured = bool(
                    controller_key
                    and len(controller_key) >= 32
                    and not any(character.isspace() for character in controller_key)
                )
            except (OSError, RuntimeError, PermissionError, ValueError):
                controller_configured = False
            if not controller_configured:
                deployment_valid = False
                issues.append(
                    _issue(
                        "error", "managed_deployment_controller_key_missing", "deployment_id",
                        "托管部署缺少 UI 控制凭据，无法安全复用。",
                        "The managed deployment has no UI controller credential and cannot be reused safely.",
                    )
                )
            if payload.combination_id and payload.combination_id != managed_row.combination_id:
                deployment_valid = False
                issues.append(
                    _issue(
                        "error", "managed_deployment_combination_mismatch", "combination_id",
                        "所选模型组合与托管部署不一致。",
                        "The selected model combination does not match the managed deployment.",
                        deployment_combination_id=managed_row.combination_id,
                    )
                )
        deployment: dict[str, Any] = {
            "valid": deployment_valid and managed_row is not None,
            "checkpoint_path": managed_row.checkpoint_path if managed_row is not None else "",
            "combination_id": managed_row.combination_id if managed_row is not None else payload.combination_id,
            "adapter_id": managed_row.adapter_id if managed_row is not None else "",
            "backbone_id": managed_row.backbone_id if managed_row is not None else "",
            "action_head_id": managed_row.action_head_id if managed_row is not None else "",
            "resolved_parameters": {
                str(key): value
                for key, value in dict(managed_row.parameters if managed_row is not None else {}).items()
                if not str(key).startswith("_")
            },
            "issues": [],
        }
    else:
        checkpoint_source = payload.checkpoint_source
        if checkpoint_source is None:  # guarded by the schema; keeps this function total for direct callers
            raise ValueError("checkpoint_source is required for a temporary evaluation")
        if checkpoint_source.kind == "indexed":
            checkpoint_id = str(checkpoint_source.checkpoint_id or "")
            indexed_checkpoint = db.get(Checkpoint, checkpoint_id) if checkpoint_id else None
            source: dict[str, Any] = {"kind": "indexed", "checkpoint_id": checkpoint_id}
            if indexed_checkpoint is None:
                issues.append(
                    _issue(
                        "error", "checkpoint_index_not_found", "checkpoint_source.checkpoint_id",
                        "索引中的 Checkpoint 不存在。", "The indexed checkpoint does not exist.",
                        checkpoint_id=checkpoint_id,
                    )
                )
            elif not indexed_checkpoint.is_complete:
                issues.append(
                    _issue(
                        "error", "checkpoint_incomplete", "checkpoint_source.checkpoint_id",
                        "该 Checkpoint 未通过完整性标记，不能评测。",
                        "The selected checkpoint is not marked complete and cannot be evaluated.",
                        checkpoint_id=checkpoint_id,
                    )
                )
        else:
            source = {"kind": "local", "path": checkpoint_source.path}

        model_parameters = dict(payload.model_parameters)
        model_parameters.setdefault("port", 10093)
        model_parameters.setdefault("idle_timeout_seconds", -1)
        model_python_value = str(settings.get("model_server_python", "") or "").strip()
        model_python_value = model_python_value or os.environ.get("ALPHABRAIN_PYTHON", "").strip() or "python"
        model_python = resolve_python_executable(model_python_value)
        if model_python is None:
            issues.append(
                _issue(
                    "error", "model_server_python_not_found", "settings.model_server_python",
                    "模型服务 Python 不存在或不可执行，请先在设置中配置。",
                    "The model-server Python is missing or not executable; configure it in Settings.",
                    configured_value=model_python_value,
                )
            )
        else:
            model_probe = probe_model_server_python(model_python, runtime.repo_root, environment)
            if not model_probe["ok"]:
                issues.append(
                    _issue(
                        "error", "model_server_dependencies_missing", "settings.model_server_python",
                        "模型服务环境缺少 AlphaBrain 模型加载或 WebSocket 协议所需依赖。",
                        "The model-server environment is missing AlphaBrain model-loading or WebSocket dependencies.",
                        errors=model_probe["errors"],
                    )
                )
        deployment = resolve_deployment(
            source,
            combination_id=payload.combination_id,
            checkpoint_index=_checkpoint_index(db),
            repo_root=runtime.repo_root,
            environment=environment,
            python_executable=model_python or model_python_value,
            parameters=model_parameters,
            include_experimental=include_experimental,
            source_catalog=source_catalog,
        )
        issues.extend(deployment.get("issues", []))

    benchmark = get_benchmark(payload.benchmark_id, include_experimental=True)
    if benchmark is None:
        issues.append(
            _issue(
                "error", "benchmark_unknown", "benchmark_id",
                "评测平台不存在或尚未接入。", "The benchmark is unknown or not wired.",
            )
        )
        benchmark = {}
    benchmark_status = str(benchmark.get("status", "unsupported"))
    if benchmark_status == "experimental":
        if not include_experimental:
            issues.append(
                _issue(
                    "error", "experimental_benchmark_not_enabled", "benchmark_id",
                    "该实验性评测平台尚未在系统和当前用户设置中同时启用。",
                    "This experimental benchmark is not enabled for both the system and current user.",
                )
            )
        elif not payload.acknowledge_experimental:
            issues.append(
                _issue(
                    "error", "experimental_risk_not_acknowledged", "acknowledge_experimental",
                    "请确认实验性评测可能需要自行补充环境、代码或配置。",
                    "Acknowledge that experimental evaluation may require custom environment, code, or configuration.",
                )
            )

    full_deployment_catalog = get_deployment_catalog(
        include_experimental=True,
        source_catalog=source_catalog,
    )
    selected_combination_id = str(deployment.get("combination_id") or payload.combination_id or "")
    selected_combination = next(
        (
            item for item in full_deployment_catalog["deployment_combinations"]
            if item.get("id") == selected_combination_id
        ),
        None,
    )
    if managed and managed_row is not None and selected_combination is None:
        deployment_valid = False
        issues.append(
            _issue(
                "error", "managed_deployment_combination_unknown", "deployment_id",
                "托管部署使用的模型组合已不在 UI 注册表中。",
                "The managed deployment model combination is no longer present in the UI registry.",
                combination_id=selected_combination_id,
            )
        )
    adapter_id = str(deployment.get("adapter_id") or "")
    if (
        managed
        and selected_combination
        and selected_combination.get("status") == "experimental"
        and not include_experimental
    ):
        deployment_valid = False
        issues.append(
            _issue(
                "error", "experimental_model_not_enabled", "deployment_id",
                "该托管部署使用实验性模型组合，请先在设置中启用实验性功能。",
                "This managed deployment uses an experimental model combination; enable experimental features first.",
            )
        )
    if (
        selected_combination
        and selected_combination.get("status") == "experimental"
        and include_experimental
        and not payload.acknowledge_experimental
        and not any(item.get("code") == "experimental_risk_not_acknowledged" for item in issues)
    ):
        issues.append(
            _issue(
                "error",
                "experimental_risk_not_acknowledged",
                "acknowledge_experimental",
                "请确认实验性模型组合可能需要自行补充环境、代码或配置。",
                "Acknowledge that the experimental model combination may require custom environment, code, or configuration.",
            )
        )
    compatible = compatible_benchmark_ids(selected_combination or {})
    if benchmark and payload.benchmark_id not in compatible:
        issues.append(
            _issue(
                "error", "checkpoint_benchmark_incompatible", "benchmark_id",
                "所选模型组合尚未接通该评测平台。",
                "The selected model combination is not wired to this benchmark.",
                combination_id=selected_combination_id,
                compatible_benchmark_ids=compatible,
            )
        )

    specialized_parameter_keys = {
        "run_id", "run_dir", "model", "benchmark", "trials", "last_only",
        "n_eps", "num_workers", "bottleneck_dim", "encoder_layers",
        "encoder_heads", "actor_hidden_dim", "ref_dropout", "fixed_std", "iter",
        "predict_video", "stdp_lr", "stdp_warmup", "stdp_max_deviation",
        "stdp_rollback_shrink", "stdp_reset_per_task",
    }
    benchmark_parameters = {
        key: value for key, value in payload.parameters.items()
        if key not in specialized_parameter_keys
    }
    resolved_parameters, parameter_issues = resolve_benchmark_parameters(
        payload.benchmark_id,
        payload.preset,
        benchmark_parameters,
        include_experimental=True,
    )
    resolved_parameters.update(
        {key: value for key, value in payload.parameters.items() if key in specialized_parameter_keys}
    )
    if adapter_id == "cosmos_policy_websocket" and "seed" not in payload.parameters:
        # Preserve the Cosmos Policy evaluation client's established default
        # while still allowing researchers to override it explicitly.
        resolved_parameters["seed"] = 0
    issues.extend(_translate_registry_issues(parameter_issues))
    if payload.kind == "cl_matrix" and not str(payload.parameters.get("run_id", "")).strip():
        issues.append(
            _issue(
                "error", "cl_run_id_required", "parameters.run_id",
                "持续学习矩阵评测需要 CL run ID。",
                "CL matrix evaluation requires parameters.run_id.",
            )
        )
    if payload.kind == "rl_iterations" and not str(payload.parameters.get("run_dir", "")).strip():
        issues.append(
            _issue(
                "error", "rl_run_dir_required", "parameters.run_dir",
                "RL iteration 评测需要训练运行目录。",
                "RL iteration evaluation requires parameters.run_dir.",
            )
        )

    suite = payload.suite or str(benchmark.get("default_suite", ""))
    task_set = payload.task_set or str(benchmark.get("default_task_set", ""))
    split = payload.split or str(benchmark.get("default_split", ""))
    allowed_suites = {str(item["id"]) for item in benchmark.get("suites", [])}
    if allowed_suites and suite not in allowed_suites:
        issues.append(
            _issue(
                "error", "benchmark_suite_invalid", "suite",
                "所选任务套件不受该评测平台支持。", "The selected suite is not supported by this benchmark.",
                allowed=sorted(allowed_suites),
            )
        )
    allowed_task_sets = {str(item["id"]) for item in benchmark.get("task_sets", [])}
    selected_task_sets = {value.strip() for value in task_set.split(",") if value.strip()}
    invalid_task_sets = sorted(selected_task_sets - allowed_task_sets)
    if allowed_task_sets and not str(resolved_parameters.get("task_list", "")).strip() and (
        not selected_task_sets or invalid_task_sets
    ):
        issues.append(
            _issue(
                "error", "benchmark_task_set_invalid", "task_set",
                "所选 Task Set 不受当前 UI 注册表支持。", "The selected task set is not supported by the UI registry.",
                allowed=sorted(allowed_task_sets), invalid=invalid_task_sets,
            )
        )
    allowed_splits = {str(item["id"]) for item in benchmark.get("splits", [])}
    if allowed_splits and split not in allowed_splits:
        issues.append(
            _issue(
                "error", "benchmark_split_invalid", "split",
                "所选数据划分不受该评测平台支持。", "The selected split is not supported by this benchmark.",
                allowed=sorted(allowed_splits),
            )
        )

    runner_path = runtime.repo_root / "scripts/run_eval.sh"
    client_entrypoint = str(benchmark.get("client_entrypoint", ""))
    if payload.benchmark_id == "libero" and adapter_id == "cosmos_policy_websocket":
        # run_eval.sh selects this client whenever the Cosmos server adapter is
        # selected, so preflight must probe the same file rather than the base
        # AlphaBrain LIBERO client declared by the benchmark catalog.
        client_entrypoint = "benchmarks/LIBERO/eval/eval_libero_cosmos.py"
    client_path = runtime.repo_root / client_entrypoint
    adapter = next(
        (item for item in full_deployment_catalog["deployment_adapters"] if item.get("id") == adapter_id),
        None,
    )
    server_entrypoint = str((adapter or {}).get("entrypoint", ""))
    server_path = runtime.repo_root / server_entrypoint
    required_entrypoints = [
        ("runner", runner_path, "evaluation_runner_missing", "统一评测启动脚本不存在。", "The evaluation runner is missing."),
        ("benchmark_id", client_path, "benchmark_client_missing", "评测客户端入口不存在。", "The benchmark client entrypoint is missing."),
    ]
    if not managed:
        required_entrypoints.append(
            ("combination_id", server_path, "model_server_entrypoint_missing", "模型服务入口不存在。", "The model-server entrypoint is missing.")
        )
    for field, path, code, zh, en in required_entrypoints:
        if not path.is_file():
            issues.append(_issue("error", code, field, zh, en, path=str(path)))

    runtime_environment: dict[str, str] = {}
    for requirement in benchmark.get("environment", []):
        key = str(requirement["key"])
        value = str(environment.get(key) or requirement.get("default") or "").strip()
        if not value:
            if requirement.get("required"):
                issues.append(
                    _issue(
                        "error", "benchmark_environment_missing", f"settings.environment.{key}",
                        f"评测环境变量 {key} 尚未设置。", f"Benchmark environment variable {key} is not configured.",
                        environment_key=key,
                    )
                )
            continue
        if requirement.get("kind") == "directory":
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = runtime.repo_root / path
            path = path.resolve(strict=False)
            if not path.is_dir():
                issues.append(
                    _issue(
                        "error", "benchmark_environment_path_invalid", f"settings.environment.{key}",
                        f"{key} 指向的目录不存在。", f"The directory configured by {key} does not exist.",
                        environment_key=key, path=str(path),
                    )
                )
            runtime_environment[key] = str(path)
        elif requirement.get("kind") == "executable":
            executable = _configured_executable(value, runtime.repo_root)
            if executable is None:
                issues.append(
                    _issue(
                        "error", "benchmark_python_not_found", f"settings.environment.{key}",
                        f"{key} 指向的 Python 不存在或不可执行。",
                        f"The Python executable configured by {key} is missing or not executable.",
                        environment_key=key, configured_value=value,
                    )
                )
            else:
                runtime_environment[key] = executable

    client_key = _BENCHMARK_PYTHON_KEYS.get(payload.benchmark_id)
    client_python = runtime_environment.get(client_key or "")
    if client_key and client_python and client_path.is_file():
        probe_environment = dict(environment)
        pythonpath_entries: list[str | Path] = []
        if payload.benchmark_id == "libero":
            libero_root = runtime_environment.get("LIBERO_HOME", "")
            if libero_root:
                probe_environment["LIBERO_HOME"] = libero_root
                probe_environment["LIBERO_CONFIG_PATH"] = str(Path(libero_root) / "libero")
                pythonpath_entries.append(libero_root)
        elif payload.benchmark_id == "libero_plus":
            libero_root = runtime_environment.get("LIBERO_PLUS_HOME", "")
            if libero_root:
                # This exactly mirrors run_eval.sh: LIBERO-plus uses its own
                # checkout for both LIBERO_HOME and LIBERO_PLUS_HOME.
                probe_environment["LIBERO_PLUS_HOME"] = libero_root
                probe_environment["LIBERO_HOME"] = libero_root
                probe_environment["LIBERO_CONFIG_PATH"] = str(Path(libero_root) / "libero")
                pythonpath_entries.append(libero_root)
        client_probe = probe_benchmark_python(
            client_python,
            _BENCHMARK_IMPORTS[payload.benchmark_id],
            runtime.repo_root,
            probe_environment,
            entrypoint=client_path,
            pythonpath_entries=tuple(pythonpath_entries),
        )
        if not client_probe["ok"]:
            issues.append(
                _issue(
                    "error", "benchmark_dependencies_missing", f"settings.environment.{client_key}",
                    "仿真评测 Python 缺少所选平台的运行依赖。",
                    "The simulation Python is missing dependencies required by the selected benchmark.",
                    errors=client_probe["errors"],
                )
            )

    wandb = payload.wandb.model_dump(mode="json")
    if wandb["enabled"] and wandb["mode"] == "online":
        if wandb_configured is None:
            try:
                wandb_configured = WandbSecretStore(runtime.state_dir).status()["configured"]
            except (OSError, RuntimeError, PermissionError):
                wandb_configured = False
        if not wandb_configured:
            issues.append(
                _issue(
                    "error", "wandb_api_key_missing", "wandb",
                    "W&B 在线上传已开启，但全局 API Key 尚未配置。",
                    "Online W&B upload is enabled, but the global API key is not configured.",
                )
            )

    requested_ids: list[int] = []
    healthy: dict[int, Any] = {}
    if not managed:
        reservations = _merged_reservations(db)
        snapshot = gpu_monitor.snapshot(reservations)
        visible = {int(gpu.index): gpu for gpu in snapshot}
        healthy = {index: gpu for index, gpu in visible.items() if not gpu.error}
        if not gpu_monitor.available:
            issues.append(
                _issue(
                    "error", "nvml_unavailable", "resources",
                    "NVIDIA NVML 不可用，无法安全分配评测 GPU。",
                    "NVIDIA NVML is unavailable, so an evaluation GPU cannot be allocated safely.",
                    error=gpu_monitor.error,
                )
            )
        elif not healthy:
            issues.append(
                _issue(
                    "error", "no_visible_gpu", "resources",
                    "没有可检查的 GPU。", "No inspectable GPU is visible.",
                )
            )
        requested_ids = [int(value) for value in payload.resources.gpu_ids]
        if payload.resources.strategy == "fixed":
            missing = sorted(set(requested_ids) - set(healthy))
            if missing:
                issues.append(
                    _issue(
                        "error", "requested_gpus_unavailable", "resources.gpu_ids",
                        "指定 GPU 不可见或无法检查。", "The requested GPU is not visible or cannot be inspected.",
                        gpu_ids=missing,
                    )
                )
        busy = [
            index for index in requested_ids
            if index in reservations or (index in healthy and not healthy[index].available)
        ]
        available_count = sum(bool(gpu.available) for gpu in healthy.values())
        if busy or (
            payload.resources.strategy == "auto"
            and available_count < payload.resources.gpu_count
        ):
            issues.append(
                _issue(
                    "info", "evaluation_will_queue_for_gpu", "resources",
                    "GPU 当前正忙，评测会进入全局 FIFO 队列。",
                    "The GPU is busy; the evaluation will wait in the global FIFO queue.",
                    busy_gpu_ids=busy,
                )
            )

    output_root = _result_root(runtime, settings)
    disk = storage_snapshot(
        output_root,
        float(settings.get("disk_min_free_gib", 100)),
        float(settings.get("disk_min_free_percent", 10)),
    )
    if disk["error"]:
        issues.append(
            _issue(
                "error", "evaluation_storage_unavailable", "output_dir",
                "评测结果目录不可用。", "The evaluation results directory is unavailable.", **disk,
            )
        )
    elif disk["low_space"]:
        issues.append(
            _issue(
                "warning", "low_disk_space", "output_dir",
                "评测结果目录剩余空间低于设置阈值。", "The evaluation results directory is below its free-space threshold.",
                **disk,
            )
        )

    resolved_model_parameters = dict(deployment.get("resolved_parameters", {}))
    resolved_model_parameters.pop("port", None)
    resolved_model_parameters.pop("idle_timeout_seconds", None)
    checkpoint_path = str(deployment.get("checkpoint_path", ""))
    resolved_resources = (
        {
            "strategy": "managed_deployment",
            "gpu_count": 0,
            "gpu_ids": [],
            "assigned_gpu_ids": assigned_deployment_gpu_ids,
            "single_gpu_mode": len(assigned_deployment_gpu_ids) == 1,
            "visible_gpu_ids": assigned_deployment_gpu_ids,
            "reuse_deployment": True,
        }
        if managed
        else {
            "strategy": payload.resources.strategy,
            "gpu_count": payload.resources.gpu_count,
            "gpu_ids": requested_ids if payload.resources.strategy == "fixed" else [],
            "single_gpu_mode": len(healthy) == 1,
            "visible_gpu_ids": sorted(healthy),
            "reuse_deployment": False,
        }
    )
    safe_resolved = {
        "checkpoint_id": (
            managed_row.checkpoint_id
            if managed_row is not None
            else indexed_checkpoint.id if indexed_checkpoint is not None else None
        ),
        "kind": payload.kind,
        "source_kind": payload.source_kind,
        "deployment_id": payload.deployment_id,
        "checkpoint_path": checkpoint_path,
        "combination_id": selected_combination_id,
        "adapter_id": adapter_id,
        "backbone_id": deployment.get("backbone_id"),
        "action_head_id": deployment.get("action_head_id"),
        "server_entrypoint": server_entrypoint,
        "server_parameters": resolved_model_parameters,
        "model_parameters": resolved_model_parameters,
        "benchmark_id": payload.benchmark_id,
        "client_entrypoint": client_entrypoint,
        "benchmark_status": benchmark_status,
        "preset": payload.preset,
        "suite": suite,
        "task_set": task_set,
        "split": split,
        "parameters": resolved_parameters,
        "wandb": wandb,
        "resources": resolved_resources,
        "reuse_server": managed,
        "managed_server": (
            {
                "host": "127.0.0.1",
                "port": int(managed_row.port or 0),
                "assigned_gpu_ids": assigned_deployment_gpu_ids,
            }
            if managed_row is not None
            else None
        ),
        "environment": runtime_environment,
        "output_root": str(output_root),
        "runner_mode": str(benchmark.get("runner_mode", "ui_evaluation")),
        "command": (
            ["python", "-m", "alphabrain_ui.specialized_evaluation", "--kind", payload.kind]
            if payload.kind in {"cl_matrix", "rl_iterations", "online_stdp"}
            else ["bash", str(runner_path), "ui_evaluation", "${EVAL_CONFIG_FILE}"]
        ),
    }
    compatibility = "experimental" if benchmark_status == "experimental" or (
        selected_combination and selected_combination.get("status") == "experimental"
    ) else "verified"
    can_submit = deployment_valid and bool(deployment.get("valid")) and not any(
        item.get("level", item.get("severity")) == "error" for item in issues
    )
    if can_submit:
        issues.append(
            _issue(
                "info", "evaluation_preflight_passed", "evaluation",
                (
                    "评测预检通过，将复用当前运行中的托管部署。"
                    if managed
                    else "评测预检通过，可以提交到全局 GPU 队列。"
                ),
                (
                    "Evaluation preflight passed and will reuse the running managed deployment."
                    if managed
                    else "Evaluation preflight passed and can be submitted to the global GPU queue."
                ),
            )
        )
    preview_environment = {
        "ALPHABRAIN_UI_MANAGED": "1",
        "EVAL_REUSE_SERVER": "1" if managed else "0",
        "EVAL_OUTPUT_DIR": "${EVAL_OUTPUT_DIR}",
        "EVAL_CONFIG_FILE": "${EVAL_CONFIG_FILE}",
        "EVAL_PROGRESS_PATH": "${EVAL_PROGRESS_PATH}",
    }
    return {
        "ok": can_submit,
        "can_submit": can_submit,
        "compatibility": compatibility,
        "items": issues,
        "issues": issues,
        "resolved": safe_resolved,
        "command_preview": [
            {
                "command": safe_resolved["command"],
                "cwd": str(runtime.repo_root),
                "environment": preview_environment,
            }
        ],
        "diff": {},
    }


__all__ = ["validate_evaluation_request"]
