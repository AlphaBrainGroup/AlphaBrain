"""Declarative benchmark capabilities for the AlphaBrain research console.

The registry deliberately describes only benchmark clients that exist in this
checkout.  Deployment compatibility remains owned by ``deployment_registry``;
this module maps its wired combination metadata onto benchmark choices.
"""

from __future__ import annotations

from copy import deepcopy
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .deployment_registry import get_deployment_catalog, inspect_checkpoint
from .preflight import effective_environment

_PRESETS: list[dict[str, Any]] = [
    {
        "id": "quick",
        "label": {"zh-CN": "快速检查", "en-US": "Quick check"},
        "description": {
            "zh-CN": "只运行第一个任务的 1 个 episode，用于验证环境和模型链路。",
            "en-US": "Run one episode on the first task to validate the environment and model path.",
        },
        "episodes_per_task": 1,
        "task_limit": 1,
    },
    {
        "id": "standard",
        "label": {"zh-CN": "标准评测", "en-US": "Standard"},
        "description": {
            "zh-CN": "运行全部所选任务，每个任务 10 个 episode。",
            "en-US": "Run every selected task with 10 episodes per task.",
        },
        "episodes_per_task": 10,
        "task_limit": 0,
    },
    {
        "id": "full",
        "label": {"zh-CN": "完整评测", "en-US": "Full benchmark"},
        "description": {
            "zh-CN": "运行全部所选任务，每个任务 50 个 episode。",
            "en-US": "Run every selected task with 50 episodes per task.",
        },
        "episodes_per_task": 50,
        "task_limit": 0,
    },
    {
        "id": "custom",
        "label": {"zh-CN": "自定义", "en-US": "Custom"},
        "description": {
            "zh-CN": "自行设置任务范围、episode 数量和专家参数。",
            "en-US": "Choose the task scope, episode count, and expert parameters.",
        },
    },
]

_COMMON_ROBOCASA_PROPERTIES: dict[str, Any] = {
    "n_episodes": {
        "type": "integer",
        "minimum": 1,
        "maximum": 1000,
        "default": 1,
        "title": {"zh-CN": "每任务 Episode 数", "en-US": "Episodes per task"},
    },
    "n_envs": {
        "type": "integer",
        "minimum": 1,
        "maximum": 64,
        "default": 1,
        "title": {"zh-CN": "并行环境数", "en-US": "Parallel environments"},
    },
    "max_episode_steps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100000,
        "default": 1440,
        "title": {"zh-CN": "Episode 最大步数", "en-US": "Maximum episode steps"},
    },
    "n_action_steps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 256,
        "default": 16,
        "title": {"zh-CN": "Action chunk 长度", "en-US": "Action chunk length"},
    },
    "seed": {
        "type": "integer",
        "minimum": 0,
        "maximum": 2147483647,
        "default": 7,
        "title": {"zh-CN": "随机种子", "en-US": "Seed"},
    },
    "task_limit": {
        "type": "integer",
        "minimum": 0,
        "maximum": 100000,
        "default": 1,
        "title": {"zh-CN": "最多任务数（0 为全部）", "en-US": "Task limit (0 means all)"},
    },
}

_BENCHMARKS: list[dict[str, Any]] = [
    {
        "id": "libero",
        "status": "verified",
        "label": {"zh-CN": "LIBERO", "en-US": "LIBERO"},
        "description": {
            "zh-CN": "AlphaBrain 当前最成熟的仿真评测链路，支持四个官方任务套件。",
            "en-US": "AlphaBrain's most mature simulation evaluation path with all four official suites.",
        },
        "runner_mode": "ui_evaluation",
        "runner_benchmark": "libero",
        "client_entrypoint": "benchmarks/LIBERO/eval/eval_libero.py",
        "deployment_benchmark_ids": ["libero"],
        "suites": [
            {"id": "libero_goal", "label": {"zh-CN": "Goal", "en-US": "Goal"}},
            {"id": "libero_spatial", "label": {"zh-CN": "Spatial", "en-US": "Spatial"}},
            {"id": "libero_object", "label": {"zh-CN": "Object", "en-US": "Object"}},
            {"id": "libero_10", "label": {"zh-CN": "Long", "en-US": "Long"}},
            {"id": "libero_all", "label": {"zh-CN": "全部套件", "en-US": "All suites"}},
        ],
        "default_suite": "libero_goal",
        "environment": [
            {"key": "LIBERO_HOME", "kind": "directory", "required": True},
            {"key": "LIBERO_PYTHON", "kind": "executable", "required": False, "default": "python"},
        ],
        "parameter_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "num_trials": {
                    "type": "integer", "minimum": 1, "maximum": 50, "default": 1,
                    "title": {"zh-CN": "每任务 Trial 数", "en-US": "Trials per task"},
                },
                "num_views": {
                    "type": "integer", "enum": [1, 2], "default": 2,
                    "title": {"zh-CN": "相机视角数", "en-US": "Camera views"},
                },
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647, "default": 7},
                "task_limit": {"type": "integer", "minimum": 0, "maximum": 1000, "default": 1},
                "task_ids": {
                    "type": "string", "maxLength": 4096, "default": "",
                    "title": {"zh-CN": "任务 ID（逗号分隔）", "en-US": "Task IDs (comma separated)"},
                },
            },
        },
    },
    {
        "id": "robocasa365",
        "status": "verified",
        "label": {"zh-CN": "RoboCasa365", "en-US": "RoboCasa365"},
        "description": {
            "zh-CN": "支持 task set、split、自定义任务列表和并行环境的 RoboCasa365 评测。",
            "en-US": "RoboCasa365 evaluation with task sets, splits, custom task lists, and vector environments.",
        },
        "runner_mode": "ui_evaluation",
        "runner_benchmark": "robocasa365",
        "client_entrypoint": "benchmarks/Robocasa365/eval/simulation_env.py",
        "deployment_benchmark_ids": ["robocasa365"],
        "task_sets": [
            {"id": "atomic_seen", "label": {"zh-CN": "Atomic Seen", "en-US": "Atomic Seen"}},
            {"id": "atomic_unseen", "label": {"zh-CN": "Atomic Unseen", "en-US": "Atomic Unseen"}},
            {"id": "composite_seen", "label": {"zh-CN": "Composite Seen", "en-US": "Composite Seen"}},
            {"id": "composite_unseen", "label": {"zh-CN": "Composite Unseen", "en-US": "Composite Unseen"}},
            {"id": "target50", "label": {"zh-CN": "Target 50", "en-US": "Target 50"}},
        ],
        "default_task_set": "target50",
        "splits": [
            {"id": "target", "label": {"zh-CN": "Target", "en-US": "Target"}},
            {"id": "train", "label": {"zh-CN": "Train", "en-US": "Train"}},
        ],
        "default_split": "target",
        "environment": [
            {"key": "ROBOCASA365_PYTHON", "kind": "executable", "required": False, "default": "python"},
            {"key": "ROBOCASA365_DATA_ROOT", "kind": "directory", "required": False},
        ],
        "parameter_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **_COMMON_ROBOCASA_PROPERTIES,
                "task_list": {"type": "string", "maxLength": 20000, "default": ""},
                "sort_tasks": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "id": "robocasa_tabletop",
        "status": "experimental",
        "label": {"zh-CN": "RoboCasa Tabletop（实验性）", "en-US": "RoboCasa Tabletop (experimental)"},
        "description": {
            "zh-CN": "按完整环境 ID 运行 Tabletop 评测；环境和模型映射仍在持续验证。",
            "en-US": "Run a Tabletop environment by its full ID; environment/model mappings remain under validation.",
        },
        "runner_mode": "ui_evaluation",
        "runner_benchmark": "robocasa_tabletop",
        "client_entrypoint": "benchmarks/Robocasa_tabletop/eval/simulation_env.py",
        "deployment_benchmark_ids": ["robocasa"],
        "default_suite": "",
        "environment": [
            {"key": "ROBOCASA_TABLETOP_PYTHON", "kind": "executable", "required": False, "default": "python"},
            {"key": "ROBOCASA_TABLETOP_DATA_ROOT", "kind": "directory", "required": False},
        ],
        "parameter_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["env_name"],
            "properties": {
                **_COMMON_ROBOCASA_PROPERTIES,
                "env_name": {"type": "string", "minLength": 1, "maxLength": 1024},
            },
        },
    },
    {
        "id": "libero_plus",
        "status": "experimental",
        "label": {"zh-CN": "LIBERO-plus（实验性）", "en-US": "LIBERO-plus (experimental)"},
        "description": {
            "zh-CN": "LIBERO 分布偏移的零样本评测；任务规模很大，运行前需单独配置环境。",
            "en-US": "Zero-shot LIBERO distribution-shift evaluation; its large task set requires a separate environment.",
        },
        "runner_mode": "ui_evaluation",
        "runner_benchmark": "libero_plus",
        "client_entrypoint": "benchmarks/LIBERO-plus/eval/eval_libero.py",
        "deployment_benchmark_ids": ["libero"],
        "suites": [
            {"id": "libero_goal", "label": {"zh-CN": "Goal", "en-US": "Goal"}},
            {"id": "libero_spatial", "label": {"zh-CN": "Spatial", "en-US": "Spatial"}},
            {"id": "libero_object", "label": {"zh-CN": "Object", "en-US": "Object"}},
            {"id": "libero_10", "label": {"zh-CN": "Long", "en-US": "Long"}},
            {"id": "libero_all", "label": {"zh-CN": "全部套件", "en-US": "All suites"}},
        ],
        "default_suite": "libero_goal",
        "environment": [
            {"key": "LIBERO_PLUS_HOME", "kind": "directory", "required": True},
            {"key": "LIBERO_PLUS_PYTHON", "kind": "executable", "required": True},
        ],
        "parameter_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "num_trials": {"type": "integer", "minimum": 1, "maximum": 50, "default": 1},
                "num_views": {"type": "integer", "enum": [1, 2], "default": 2},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647, "default": 7},
                "task_limit": {"type": "integer", "minimum": 0, "maximum": 10030, "default": 1},
                "task_ids": {
                    "type": "string", "maxLength": 4096, "default": "",
                    "title": {"zh-CN": "任务 ID（逗号分隔）", "en-US": "Task IDs (comma separated)"},
                },
            },
        },
    },
]


def _allowed(status: str, include_experimental: bool) -> bool:
    return status == "verified" or (status == "experimental" and include_experimental)


def get_benchmark(benchmark_id: str, *, include_experimental: bool = False) -> dict[str, Any] | None:
    return next(
        (
            deepcopy(item)
            for item in _BENCHMARKS
            if item["id"] == benchmark_id and _allowed(str(item["status"]), include_experimental)
        ),
        None,
    )


def compatible_benchmark_ids(combination: Mapping[str, Any]) -> list[str]:
    declared = {str(value) for value in combination.get("benchmarks", [])}
    return [
        str(item["id"])
        for item in _BENCHMARKS
        if declared.intersection(str(value) for value in item.get("deployment_benchmark_ids", []))
    ]


def get_evaluation_catalog(
    include_experimental: bool = False,
    *,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    deployment = get_deployment_catalog(
        include_experimental=include_experimental,
        source_catalog=source_catalog,
    )
    combinations: list[dict[str, Any]] = []
    for combination in deployment["deployment_combinations"]:
        benchmark_ids = compatible_benchmark_ids(combination)
        combinations.append(
            {
                "id": combination["id"],
                "status": combination.get("status", "verified"),
                "adapter_id": combination.get("adapter"),
                "backbone_id": combination.get("backbone"),
                "action_head_id": combination.get("action_head"),
                "benchmark_ids": [
                    value
                    for value in benchmark_ids
                    if get_benchmark(value, include_experimental=include_experimental) is not None
                ],
            }
        )
    benchmarks = [
        deepcopy(item) for item in _BENCHMARKS if _allowed(str(item["status"]), include_experimental)
    ]
    for benchmark in benchmarks:
        benchmark["compatibility"] = {
            "combination_ids": [
                item["id"]
                for item in combinations
                if benchmark["id"] in item["benchmark_ids"]
                and _allowed(str(item["status"]), include_experimental)
            ]
        }
    return {
        "schema_version": 1,
        "registry_version": deployment.get("registry_version", ""),
        "benchmarks": benchmarks,
        "presets": deepcopy(_PRESETS),
        "combinations": combinations,
        "filters": {"include_experimental": include_experimental, "unsupported_hidden": True},
    }


def evaluate_catalog_readiness(
    catalog: Mapping[str, Any],
    settings: Any,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Annotate a catalog copy with non-secret environment readiness data."""

    result = deepcopy(dict(catalog))
    repo = Path(repo_root).expanduser().resolve()
    if hasattr(settings, "get"):
        configured = settings.get("environment", {})
    else:
        configured = {}
    configured = dict(configured or {}) if isinstance(configured, Mapping) else {}
    environment = effective_environment(repo, {str(key): str(value) for key, value in configured.items()})
    for benchmark in result.get("benchmarks", []):
        readiness: list[dict[str, Any]] = []
        required_ready = True
        for raw in benchmark.get("environment", []):
            item = deepcopy(raw)
            key = str(item["key"])
            value = str(environment.get(key) or item.get("default") or "").strip()
            item["configured"] = bool(value)
            available = False
            resolved_value = ""
            if value and item.get("kind") == "directory":
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = repo / path
                path = path.resolve(strict=False)
                available = path.is_dir()
                resolved_value = str(path)
            elif value and item.get("kind") == "executable":
                candidate = Path(value).expanduser()
                if not candidate.is_absolute() and (repo / candidate).is_file():
                    candidate = repo / candidate
                if candidate.parent != Path(".") or candidate.is_absolute():
                    try:
                        path = candidate.resolve(strict=True)
                    except OSError:
                        pass
                    else:
                        available = path.is_file() and os.access(path, os.X_OK)
                        resolved_value = str(path)
                else:
                    located = shutil.which(value)
                    available = located is not None
                    resolved_value = located or ""
            if not value and not item.get("required"):
                available = True
            item["available"] = available
            item["resolved_value"] = resolved_value
            if available:
                zh = "已就绪" if value else "使用默认环境"
                en = "Ready" if value else "Using the default environment"
            elif value:
                zh = "已配置，但路径不可用"
                en = "Configured, but the path is unavailable"
            else:
                zh = "尚未配置"
                en = "Not configured"
            item["message"] = {"zh-CN": zh, "en-US": en}
            readiness.append(item)
            if item.get("required") and not available:
                required_ready = False
        client = repo / str(benchmark.get("client_entrypoint", ""))
        benchmark["environment"] = readiness
        benchmark["readiness"] = {
            "ready": required_ready and client.is_file(),
            "client_entrypoint_available": client.is_file(),
        }
    return result


def resolve_benchmark_parameters(
    benchmark_id: str,
    preset: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    include_experimental: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = get_benchmark(benchmark_id, include_experimental=include_experimental)
    if benchmark is None:
        return {}, [{"code": "benchmark_unknown", "field": "benchmark_id", "message": "Unknown benchmark."}]
    schema = benchmark.get("parameter_schema", {})
    properties = schema.get("properties", {})
    resolved = {
        str(key): deepcopy(value.get("default"))
        for key, value in properties.items()
        if isinstance(value, Mapping) and "default" in value
    }
    selected_preset = next((item for item in _PRESETS if item["id"] == preset), None)
    issues: list[dict[str, Any]] = []
    if selected_preset is None:
        issues.append({"code": "evaluation_preset_unknown", "field": "preset", "message": "Unknown preset."})
    elif preset != "custom":
        episodes = int(selected_preset["episodes_per_task"])
        resolved["num_trials" if benchmark_id in {"libero", "libero_plus"} else "n_episodes"] = episodes
        resolved["task_limit"] = int(selected_preset["task_limit"])
    supplied = dict(parameters or {})
    unknown = sorted(set(supplied) - set(properties))
    if unknown and not bool(schema.get("additionalProperties", True)):
        issues.append(
            {
                "code": "benchmark_parameter_unknown",
                "field": "parameters",
                "message": "One or more benchmark parameters are not supported.",
                "detail": {"unknown_parameters": unknown},
            }
        )
    resolved.update({str(key): deepcopy(value) for key, value in supplied.items() if key in properties})
    for key, rule in properties.items():
        if key not in resolved:
            continue
        value = resolved[key]
        kind = rule.get("type")
        valid_type = (
            (kind == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (kind == "string" and isinstance(value, str))
            or (kind == "boolean" and isinstance(value, bool))
        )
        if kind and not valid_type:
            issues.append(
                {"code": "benchmark_parameter_type", "field": f"parameters.{key}", "message": "Invalid parameter type."}
            )
            continue
        if "enum" in rule and value not in rule["enum"]:
            issues.append(
                {"code": "benchmark_parameter_value", "field": f"parameters.{key}", "message": "Invalid parameter value."}
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rule and value < rule["minimum"]:
                issues.append(
                    {"code": "benchmark_parameter_minimum", "field": f"parameters.{key}", "message": "Parameter is too small."}
                )
            if "maximum" in rule and value > rule["maximum"]:
                issues.append(
                    {"code": "benchmark_parameter_maximum", "field": f"parameters.{key}", "message": "Parameter is too large."}
                )
        if isinstance(value, str):
            if len(value) < int(rule.get("minLength", 0)):
                issues.append(
                    {"code": "benchmark_parameter_required", "field": f"parameters.{key}", "message": "Parameter is required."}
                )
            if "maxLength" in rule and len(value) > int(rule["maxLength"]):
                issues.append(
                    {"code": "benchmark_parameter_too_long", "field": f"parameters.{key}", "message": "Parameter is too long."}
                )
    for key in schema.get("required", []):
        if resolved.get(key) in (None, ""):
            issues.append(
                {"code": "benchmark_parameter_required", "field": f"parameters.{key}", "message": "Parameter is required."}
            )
    task_ids = str(resolved.get("task_ids", "")).strip()
    if task_ids:
        tokens = [value.strip() for value in task_ids.split(",")]
        if any(not value.isdigit() for value in tokens):
            issues.append(
                {
                    "code": "benchmark_task_ids_invalid",
                    "field": "parameters.task_ids",
                    "message": "Task IDs must be comma-separated non-negative integers.",
                }
            )
        else:
            resolved["task_ids"] = ",".join(dict.fromkeys(tokens))
    return resolved, issues


def inspect_evaluation_checkpoint(
    source: str | Path | Mapping[str, Any],
    *,
    checkpoint_index: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    repo_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    combination_id: str | None = None,
    include_experimental: bool = False,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inspection = inspect_checkpoint(
        source,
        checkpoint_index=checkpoint_index,
        repo_root=repo_root,
        environment=environment,
        combination_id=combination_id,
        include_experimental=include_experimental,
        source_catalog=source_catalog,
    )
    candidates = []
    for combination in inspection.get("candidates", []):
        candidates.append(
            {
                "combination_id": combination.get("id"),
                "benchmark_ids": [
                    value
                    for value in compatible_benchmark_ids(combination)
                    if get_benchmark(value, include_experimental=include_experimental) is not None
                ],
            }
        )
    result = deepcopy(inspection)
    result["evaluation_candidates"] = candidates
    selected = inspection.get("detected", {}).get("combination_id")
    result["compatible_benchmark_ids"] = next(
        (item["benchmark_ids"] for item in candidates if item["combination_id"] == selected), []
    )
    return result


benchmark_for_checkpoint = inspect_evaluation_checkpoint

__all__ = [
    "benchmark_for_checkpoint",
    "compatible_benchmark_ids",
    "evaluate_catalog_readiness",
    "get_benchmark",
    "get_evaluation_catalog",
    "inspect_evaluation_checkpoint",
    "resolve_benchmark_parameters",
]
