from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

from .gpu import GPUMonitor, storage_snapshot
from .datasets import validate_dataset_directory
from .launchers import build_launch_plan, normalize_family

ISSUE_MESSAGES_ZH = {
    "config_resolution_failed": "配置解析失败。",
    "experimental_disabled": "实验性组合尚未启用。",
    "experimental_combination": "此组合尚未验证，可能无法正常运行。",
    "python_version": "AlphaBrain UI 需要 Python 3.10 或更高版本。",
    "launch_plan_failed": "无法生成启动计划。",
    "launcher_missing": "启动脚本不存在。",
    "output_exists": "输出目录已存在且非空，禁止覆盖。",
    "output_unreadable": "输出路径不可访问。",
    "pretrained_root_missing": "尚未配置预训练模型根目录。",
    "pretrained_root_not_found": "预训练模型根目录不存在。",
    "dataset_not_found": "数据集路径不存在。",
    "dataset_root_missing": "尚未配置可用的 LIBERO 数据集目录。",
    "checkpoint_required": "该训练方法需要已有 VLA checkpoint。",
    "checkpoint_not_found": "Checkpoint 不存在。",
    "nvml_unavailable": "NVIDIA NVML 不可用。",
    "gpu_count": "可见 GPU 数量不足。",
    "gpu_ids_not_visible": "指定的 GPU 编号当前不可见。",
    "gpu_inspection_failed": "无法读取所申请 GPU 的状态。",
    "gpu_queue": "任务到达队首时会再次检查 GPU 可用性。",
    "low_disk_space": "结果目录剩余空间低于阈值。",
    "storage_unavailable": "结果目录不可访问。",
}


def issue(level: str, code: str, message: str, field: str | None = None, **detail: Any) -> dict[str, Any]:
    return {
        "level": level,
        "code": code,
        "message": message,
        "message_i18n": {"zh-CN": ISSUE_MESSAGES_ZH.get(code, message), "en-US": message},
        "field": field,
        "detail": detail,
    }


def read_dotenv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


def effective_environment(repo_root: Path, configured: dict[str, str] | None = None) -> dict[str, str]:
    values = read_dotenv(repo_root / ".env")
    values.update({key: value for key, value in os.environ.items() if isinstance(value, str)})
    values.update(configured or {})
    return values


def run_preflight(
    *,
    repo_root: Path,
    state_dir: Path,
    name: str,
    spec: dict[str, Any],
    configured_environment: dict[str, str] | None,
    gpu_monitor: GPUMonitor,
    results_roots: list[str],
    disk_min_free_gib: float,
    disk_min_free_percent: float,
    include_experimental: bool,
    source_catalog: Mapping[str, Any] | None = None,
    remote_gpu_ids: list[int] | None = None,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    family = normalize_family(spec)
    resolved: dict[str, Any] = dict(spec)
    compatibility = str(spec.get("compatibility", "verified"))
    env = effective_environment(repo_root, configured_environment)
    capability_static_issues: list[dict[str, Any]] = []

    # Use the capability subsystem when present, while keeping preflight useful
    # for hand-authored expert specs.
    try:
        from .capabilities import resolve_experiment, validate_compatibility

        resolved = resolve_experiment(
            spec,
            repo_root=repo_root,
            environment=env,
            source_catalog=source_catalog,
        )
        cap_issues = validate_compatibility(spec, source_catalog=source_catalog)
        for item in cap_issues:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif not isinstance(item, dict):
                item = {"level": "warning", "code": "compatibility", "message": str(item)}
            item.setdefault("field", None)
            item.setdefault("detail", {})
            issues.append(item)
        compatibility = str(resolved.get("compatibility", compatibility))
        family = normalize_family(resolved)
        try:
            from .configuration import preflight_static

            capability_static_issues = preflight_static(
                spec,
                repo_root,
                resolved=resolved,
                environment=env,
                source_catalog=source_catalog,
            )
        except ImportError:
            pass
    except ImportError:
        pass
    except Exception as exc:
        issues.append(issue("error", "config_resolution_failed", str(exc), "spec"))

    existing_codes = {item.get("code") for item in issues}
    for item in capability_static_issues:
        if item.get("code") not in existing_codes:
            issues.append(item)
            existing_codes.add(item.get("code"))

    if compatibility == "experimental" and not include_experimental:
        issues.append(issue("error", "experimental_disabled", "Experimental combinations are disabled."))
    elif compatibility == "experimental":
        issues.append(issue("warning", "experimental_combination", "This combination is not verified and may fail."))

    dataset_node = spec.get("dataset", {}) if isinstance(spec.get("dataset", {}), dict) else {}
    imported_root = dataset_node.get("root") or dataset_node.get("data_root")
    # Dataset Center references are resolved server-side into a trusted
    # ``mixture_spec`` after their registration status, visibility and
    # fingerprints have been checked.  Running the legacy named-mixture
    # validator as well would incorrectly require a registered dataset leaf
    # directory to have the built-in LIBERO/Robocasa parent layout.
    if imported_root and not dataset_node.get("mixture_spec"):
        report = validate_dataset_directory(
            str(imported_root),
            str(dataset_node.get("id") or ""),
            str(dataset_node.get("mix") or dataset_node.get("dataset_mix") or "") or None,
            base_dir=repo_root,
        )
        issues.extend(report["issues"])

    if sys.version_info < (3, 10):
        issues.append(issue("error", "python_version", "AlphaBrain UI requires Python 3.10 or newer."))

    try:
        stages = build_launch_plan(repo_root, state_dir, name, resolved, environment=configured_environment)
    except Exception as exc:
        stages = []
        issues.append(issue("error", "launch_plan_failed", str(exc), "spec"))

    for stage in stages:
        executable_or_script = next((arg for arg in stage.command if arg.endswith((".sh", ".py"))), None)
        if executable_or_script and not (repo_root / executable_or_script).exists():
            issues.append(issue("error", "launcher_missing", f"Launcher does not exist: {executable_or_script}"))
        output = Path(stage.output_dir)
        try:
            occupied = output.exists() and (not output.is_dir() or any(output.iterdir()))
        except OSError as exc:
            issues.append(issue("error", "output_unreadable", f"Output path is not readable: {output}", error=str(exc)))
        else:
            if occupied:
                issues.append(
                    issue("error", "output_exists", f"Output path already exists and cannot be overwritten: {output}")
                )

    if family in {"rl_token", "vla_ppo"}:
        checkpoint = resolved.get("checkpoint") or resolved.get("training", {}).get("checkpoint")
        if not checkpoint:
            issues.append(issue("error", "checkpoint_required", "RL training requires a VLA checkpoint."))
        elif not Path(str(checkpoint)).expanduser().exists():
            issues.append(issue("error", "checkpoint_not_found", f"Checkpoint does not exist: {checkpoint}"))

    requested = max((stage.requested_gpu_count for stage in stages), default=1)
    if remote_gpu_ids is not None:
        visible_gpu_ids = set(remote_gpu_ids)
        if requested > len(visible_gpu_ids):
            issues.append(
                issue(
                    "error",
                    "gpu_count",
                    f"Requested {requested} GPUs, but only {len(visible_gpu_ids)} remote GPU IDs are configured.",
                )
            )
        invalid_gpu_ids = sorted(
            {gpu_id for stage in stages for gpu_id in stage.requested_gpu_ids if gpu_id not in visible_gpu_ids}
        )
        if invalid_gpu_ids:
            issues.append(
                issue(
                    "error",
                    "gpu_ids_not_visible",
                    f"Requested GPU IDs are not configured for remote training: {invalid_gpu_ids}",
                    "resources.gpu_ids",
                    visible_gpu_ids=sorted(visible_gpu_ids),
                    invalid_gpu_ids=invalid_gpu_ids,
                )
            )
        issues.append(
            issue(
                "info",
                "gpu_queue",
                "Remote GPU availability is checked against this platform's queue when the job reaches the front.",
            )
        )
    else:
        gpus = gpu_monitor.snapshot()
        healthy_gpus = [gpu for gpu in gpus if not gpu.error]
        if not gpu_monitor.available:
            issues.append(issue("error", "nvml_unavailable", f"NVIDIA NVML is unavailable: {gpu_monitor.error}"))
        elif requested > len(healthy_gpus):
            issues.append(
                issue(
                    "error",
                    "gpu_count",
                    f"Requested {requested} GPUs, but only {len(healthy_gpus)} can be inspected.",
                    gpu_errors={gpu.index: gpu.error for gpu in gpus if gpu.error},
                )
            )
        else:
            visible_gpu_ids = {gpu.index for gpu in gpus}
            invalid_gpu_ids = sorted(
                {gpu_id for stage in stages for gpu_id in stage.requested_gpu_ids if gpu_id not in visible_gpu_ids}
            )
            if invalid_gpu_ids:
                issues.append(
                    issue(
                        "error",
                        "gpu_ids_not_visible",
                        f"Requested GPU IDs are not visible: {invalid_gpu_ids}",
                        "resources.gpu_ids",
                        visible_gpu_ids=sorted(visible_gpu_ids),
                        invalid_gpu_ids=invalid_gpu_ids,
                    )
                )
            requested_with_errors = sorted(
                {
                    gpu_id
                    for stage in stages
                    for gpu_id in stage.requested_gpu_ids
                    if gpu_id in {gpu.index for gpu in gpus if gpu.error}
                }
            )
            if requested_with_errors:
                issues.append(
                    issue(
                        "error",
                        "gpu_inspection_failed",
                        f"Unable to inspect requested GPU IDs: {requested_with_errors}",
                        "resources.gpu_ids",
                        gpu_ids=requested_with_errors,
                    )
                )
            issues.append(
                issue(
                    "info",
                    "gpu_queue",
                    "GPU availability is checked again when the queued job reaches the front.",
                )
            )

    for configured_root in results_roots or ["results"]:
        root = Path(configured_root).expanduser()
        if not root.is_absolute():
            root = repo_root / root
        disk = storage_snapshot(root, disk_min_free_gib, disk_min_free_percent)
        if disk["error"]:
            issues.append(issue("error", "storage_unavailable", f"Results directory is unavailable: {root}", **disk))
        elif disk["low_space"]:
            issues.append(issue("warning", "low_disk_space", f"Low disk space under {root}.", **disk))

    previews = [stage.redacted_preview() for stage in stages]
    return resolved, compatibility, issues, previews
