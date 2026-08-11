"""Capability discovery and compatibility validation for AlphaBrain UI.

The public functions in this module only return plain JSON-serialisable data;
they are safe to expose from FastAPI without importing any training modules.
The source of truth is ``registry/catalog.yaml``.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .registry import load_catalog
from .workflows import validate_workflow_spec, workflow_schema_for

SELECTABLE_STATUSES = frozenset({"verified", "experimental"})
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

_ALIASES: dict[str, dict[str, str]] = {
    "backbone": {
        "qwen": "qwen2_5_vl",
        "qwen2.5-vl": "qwen2_5_vl",
        "qwen2_5": "qwen2_5_vl",
        "qwen3": "qwen3_vl",
        "qwen3-vl": "qwen3_vl",
        "paligemma3b": "paligemma",
        "llama": "llama3_2_vision",
        "llama3.2": "llama3_2_vision",
        "cosmos2.5": "cosmos2_5",
        "cosmos25": "cosmos2_5",
        "v-jepa2": "vjepa2",
        "wan2.2": "wan2_2",
    },
    "action_head": {
        "mlp": "mlp_regression",
        "oft": "mlp_regression",
        "dit": "flow_matching_dit",
        "flow_matching": "flow_matching_dit",
        "groot": "flow_matching_dit",
        "pi05": "pi05_action_expert",
        "pi0.5": "pi05_action_expert",
        "spiking": "snn",
    },
    "method": {
        "il": "imitation_learning",
        "supervised": "imitation_learning",
        "stdp": "r_stdp",
        "er": "continual_er",
        "mir": "continual_mir",
        "ewc": "continual_ewc",
        "rlt": "rlt_td3",
        "rlt_a": "rlt_a_td3",
        "ppo": "rlt_a_ppo",
        "grpo": "rlt_a_grpo",
    },
    "dataset": {
        "libero-plus": "libero",
        "libero_plus": "libero",
        "robocasa_tabletop": "robocasa",
        "robo_casa": "robocasa",
        "robo_casa365": "robocasa365",
    },
}


def _component_index(catalog: Mapping[str, Any], group: str) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in catalog.get("components", {}).get(group, [])}


def _catalog_copy(source_catalog: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return an isolated catalog, defaulting to the packaged registry.

    Callers such as the UI registry-overlay service can inject an effective
    catalog explicitly.  Keeping the fallback here preserves the standalone
    module API and avoids mutating either the cached built-in catalog or an
    application-owned effective catalog while it is filtered for a response.
    """

    return deepcopy(dict(source_catalog)) if source_catalog is not None else load_catalog()


def _canonical(kind: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    return _ALIASES.get(kind, {}).get(normalized.lower(), normalized)


def _nested_id(node: Any, *names: str) -> Any:
    if isinstance(node, str):
        return node
    if not isinstance(node, Mapping):
        return None
    for name in names:
        value = node.get(name)
        if isinstance(value, Mapping):
            value = value.get("id")
        if value not in (None, ""):
            return value
    return None


def selection_from_spec(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract and canonicalise the four compatibility axes from a UI spec.

    Both concise values (``architecture.backbone: qwen``) and API-shaped
    objects (``architecture.backbone: {id: qwen2_5_vl}``) are accepted.
    """

    spec = spec or {}
    architecture = spec.get("architecture", {})
    training = spec.get("training", {})
    dataset = spec.get("dataset", {})
    return {
        "backbone": _canonical(
            "backbone",
            _nested_id(architecture, "backbone", "backbone_id", "vlm_backbone", "world_model_backbone")
            or _nested_id(spec, "backbone_id", "backbone"),
        ),
        "action_head": _canonical(
            "action_head",
            _nested_id(architecture, "action_head", "action_head_id", "head")
            or _nested_id(spec, "action_head_id", "action_head"),
        ),
        "method": _canonical(
            "method",
            _nested_id(training, "method", "method_id", "training_method") or _nested_id(spec, "method_id", "method"),
        ),
        "dataset": _canonical(
            "dataset",
            _nested_id(dataset, "id", "dataset", "dataset_id", "name") or _nested_id(spec, "dataset_id", "dataset"),
        ),
    }


def _status_allowed(status: str, include_experimental: bool) -> bool:
    return status == "verified" or (include_experimental and status == "experimental")


def resolve_pretrained_directory_status(
    component: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve a backbone's packaged local-model path without loading weights.

    The result is deliberately small and JSON serialisable so it can be shown
    next to a backbone in the first builder step.  A missing default means the
    component does not require a local pretrained directory (for example the
    debug-only ToyVLA backbone).
    """

    raw_path = component.get("default_pretrained")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {
            "required": False,
            "configured": True,
            "path": None,
            "exists": True,
            "issue": None,
        }

    env = os.environ if environment is None else environment
    missing_variables: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        value = env.get(name)
        if value in (None, ""):
            if name not in missing_variables:
                missing_variables.append(name)
            return match.group(0)
        return str(value)

    expanded = _ENV_REFERENCE_RE.sub(substitute, raw_path.strip())
    if missing_variables:
        variables = ", ".join(missing_variables)
        return {
            "required": True,
            "configured": False,
            "path": raw_path,
            "exists": False,
            "issue": {
                "code": "missing_environment_variable",
                "variables": missing_variables,
                "message_i18n": {
                    "zh-CN": f"环境变量 {variables} 未设置。",
                    "en-US": f"Environment variable {variables} is not set.",
                },
            },
        }

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    candidate = Path(expanded).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)
    is_directory = candidate.is_dir()
    issue: dict[str, Any] | None = None
    if not is_directory:
        code = "pretrained_directory_not_found" if not candidate.exists() else "pretrained_path_not_directory"
        issue = {
            "code": code,
            "message_i18n": {
                "zh-CN": f"预训练模型目录不存在：{candidate}"
                if code == "pretrained_directory_not_found"
                else f"预训练模型路径不是目录：{candidate}",
                "en-US": f"Pretrained model directory does not exist: {candidate}"
                if code == "pretrained_directory_not_found"
                else f"Pretrained model path is not a directory: {candidate}",
            },
        }
    return {
        "required": True,
        "configured": True,
        "path": str(candidate),
        "exists": is_directory,
        "issue": issue,
    }


def _matching_combinations(
    selection: Mapping[str, Any],
    *,
    include_unsupported: bool = True,
    source_catalog: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for combo in _catalog_copy(source_catalog).get("combinations", []):
        if not include_unsupported and combo.get("status") == "unsupported":
            continue
        if selection.get("backbone") and combo.get("backbone") != selection["backbone"]:
            continue
        if selection.get("action_head") and combo.get("action_head") != selection["action_head"]:
            continue
        if selection.get("method") and combo.get("method") != selection["method"]:
            continue
        if selection.get("dataset") and selection["dataset"] not in combo.get("datasets", []):
            continue
        matches.append(combo)
    return matches


def get_catalog(
    include_experimental: bool = False,
    *,
    repo_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the capability catalog filtered for a normal or expert UI.

    Unsupported combinations are never returned.  Experimental components are
    returned only after the caller has applied the administrator and per-user
    opt-in checks.  Components with no remaining combination are pruned so a
    selection control cannot offer a dead end.
    """

    catalog = _catalog_copy(source_catalog)
    combinations = [
        combo
        for combo in catalog.get("combinations", [])
        if _status_allowed(combo.get("status", "unsupported"), include_experimental)
    ]
    for combination in combinations:
        combination["workflow"] = workflow_schema_for(combination, catalog)
    catalog["combinations"] = combinations

    references = {
        "backbones": {combo["backbone"] for combo in combinations},
        "action_heads": {combo["action_head"] for combo in combinations},
        "training_methods": {combo["method"] for combo in combinations},
        "datasets": {dataset for combo in combinations for dataset in combo.get("datasets", [])},
    }
    for group, items in catalog.get("components", {}).items():
        catalog["components"][group] = [
            item
            for item in items
            if item.get("id") in references.get(group, set())
            and _status_allowed(item.get("status", "unsupported"), include_experimental)
        ]

    catalog["filters"] = {
        "include_experimental": include_experimental,
        "unsupported_hidden": True,
    }
    frontend_groups = {
        "backbones": "backbone",
        "action_heads": "action_head",
        "training_methods": "method",
        "datasets": "dataset",
    }

    def frontend_parameters() -> list[dict[str, Any]]:
        properties = catalog.get("parameter_schemas", {}).get("common", {}).get("properties", {})
        values: list[dict[str, Any]] = []
        for key, schema in properties.items():
            # These fields already have dedicated controls in the experiment
            # form; only expose additional module fields dynamically.
            if key in {
                "run_id",
                "output_root_dir",
                "seed",
                "per_device_batch_size",
                "max_train_steps",
                "save_interval",
                "learning_rate",
            }:
                continue
            labels = schema.get("title", {}) if isinstance(schema.get("title"), Mapping) else {}
            item: dict[str, Any] = {
                "key": key,
                "type": (
                    "select"
                    if schema.get("enum")
                    else ("path" if key.endswith(("_root", "_checkpoint")) else schema.get("type", "string"))
                ),
                "label": labels.get("en-US", key),
                "label_zh": labels.get("zh-CN", labels.get("en-US", key)),
                "required": key == "run_id",
            }
            if "default" in schema:
                item["default"] = schema["default"]
            if "minimum" in schema:
                item["min"] = schema["minimum"]
            if "maximum" in schema:
                item["max"] = schema["maximum"]
            if schema.get("enum"):
                item["options"] = [{"label": str(value), "value": value} for value in schema["enum"]]
            values.append(item)
        return values

    common_parameters = frontend_parameters()
    flat: list[dict[str, Any]] = []
    for group, kind in frontend_groups.items():
        converted: list[dict[str, Any]] = []
        for component in catalog["components"].get(group, []):
            component_id = component["id"]
            if kind == "backbone":
                component["pretrained_directory"] = resolve_pretrained_directory_status(
                    component,
                    repo_root=repo_root,
                    environment=environment,
                )
            related = [
                combo
                for combo in combinations
                if component_id
                in {
                    combo.get("backbone"),
                    combo.get("action_head"),
                    combo.get("method"),
                    *combo.get("datasets", []),
                }
            ]
            compatible_ids = sorted(
                {
                    value
                    for combo in related
                    for value in (
                        combo.get("backbone"),
                        combo.get("action_head"),
                        combo.get("method"),
                        *combo.get("datasets", []),
                    )
                    if value and value != component_id
                }
            )
            labels = component.get("label", {})
            descriptions = component.get("description", {})
            ui_item = deepcopy(component)
            ui_item.update(
                {
                    "kind": kind,
                    "name": labels.get("en-US", component_id),
                    "name_zh": labels.get("zh-CN", labels.get("en-US", component_id)),
                    "description": descriptions.get("en-US", ""),
                    "description_zh": descriptions.get("zh-CN", descriptions.get("en-US", "")),
                    "compatible_with": compatible_ids,
                    "recommended_gpu_count": min((int(combo.get("min_gpus", 1)) for combo in related), default=1),
                }
            )
            if kind == "backbone":
                ui_item["category"] = component.get("kind", "vlm")
            if kind == "method":
                parameters = deepcopy(common_parameters)
                if not component.get("requires_checkpoint"):
                    parameters = [item for item in parameters if item["key"] != "pretrained_checkpoint"]
                ui_item["parameters"] = parameters
            converted.append(ui_item)
        key = "methods" if group == "training_methods" else group
        catalog[key] = converted
        flat.extend(converted)
    catalog["capabilities"] = flat
    catalog["experimental_allowed"] = include_experimental
    return catalog


def get_compatible_options(
    selection: Mapping[str, Any] | None = None,
    *,
    include_experimental: bool = False,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return component options reachable from a partial model selection."""

    raw_selection = selection_from_spec(selection or {})
    combos = [
        combo
        for combo in _matching_combinations(
            raw_selection,
            include_unsupported=False,
            source_catalog=source_catalog,
        )
        if _status_allowed(combo.get("status", "unsupported"), include_experimental)
    ]
    catalog = _catalog_copy(source_catalog)
    wanted = {
        "backbones": {combo["backbone"] for combo in combos},
        "action_heads": {combo["action_head"] for combo in combos},
        "training_methods": {combo["method"] for combo in combos},
        "datasets": {value for combo in combos for value in combo.get("datasets", [])},
    }
    return {
        group: [deepcopy(item) for item in catalog["components"][group] if item["id"] in ids]
        for group, ids in wanted.items()
    }


def _issue(
    severity: str,
    code: str,
    path: str,
    zh: str,
    en: str,
    **metadata: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "severity": severity,
        "level": severity,
        "code": code,
        "path": path,
        "field": path,
        "message": zh,
        "message_i18n": {"zh-CN": zh, "en-US": en},
        "detail": {},
    }
    result.update(metadata)
    result["detail"].update(metadata)
    return result


def _experimental_flags(spec: Mapping[str, Any]) -> tuple[bool, bool]:
    node = spec.get("experimental", {})
    if isinstance(node, bool):
        return node, node
    if not isinstance(node, Mapping):
        node = {}
    enabled = bool(node.get("enabled", spec.get("allow_experimental", spec.get("experimental_confirmed", False))))
    acknowledged = bool(
        node.get(
            "risk_acknowledged",
            node.get("acknowledged", spec.get("experimental_acknowledged", spec.get("experimental_confirmed", False))),
        )
    )
    return enabled, acknowledged


def validate_compatibility(
    spec: Mapping[str, Any] | None,
    *,
    source_catalog: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate component compatibility and static field constraints.

    The function never reads datasets, model weights or GPUs.  Filesystem
    checks belong to :func:`alphabrain_ui.configuration.preflight_static`.
    """

    if not isinstance(spec, Mapping):
        return [
            _issue(
                "error",
                "invalid_spec",
                "$",
                "实验配置必须是对象。",
                "Experiment spec must be an object.",
            )
        ]

    catalog = _catalog_copy(source_catalog)
    selection = selection_from_spec(spec)
    issues: list[dict[str, Any]] = []
    groups = {
        "backbone": ("backbones", "architecture.backbone", "Backbone"),
        "action_head": ("action_heads", "architecture.action_head", "Action Head"),
        "method": ("training_methods", "training.method", "Training Method"),
        "dataset": ("datasets", "dataset.id", "Dataset"),
    }
    for key, (group, path, label) in groups.items():
        value = selection.get(key)
        if not value:
            issues.append(_issue("error", f"missing_{key}", path, f"请选择 {label}。", f"Select a {label}."))
            continue
        if value not in _component_index(catalog, group):
            issues.append(
                _issue(
                    "error",
                    f"unknown_{key}",
                    path,
                    f"未知的 {label}：{value}",
                    f"Unknown {label}: {value}",
                    value=value,
                )
            )

    if any(issue["code"].startswith(("missing_", "unknown_")) for issue in issues):
        return issues

    matches = _matching_combinations(selection, source_catalog=catalog)
    if not matches:
        issues.append(
            _issue(
                "error",
                "incompatible_combination",
                "architecture",
                "该 Backbone、Action Head、训练方法与数据集组合尚未接通。",
                "This Backbone, Action Head, training method, and dataset combination is not wired.",
                selection=selection,
            )
        )
        return issues

    selectable = [combo for combo in matches if combo.get("status") in SELECTABLE_STATUSES]
    if not selectable:
        combo = matches[0]
        reason = combo.get("reason", {})
        issues.append(
            _issue(
                "error",
                "unsupported_combination",
                "architecture",
                reason.get("zh-CN", "该组合在当前代码中明确不受支持。"),
                reason.get("en-US", "This combination is explicitly unsupported by the current code."),
                combination_id=combo.get("id"),
                selection=selection,
            )
        )
        return issues

    combo = selectable[0]
    enabled, acknowledged = _experimental_flags(spec)
    if combo.get("status") == "experimental":
        if not enabled:
            issues.append(
                _issue(
                    "error",
                    "experimental_disabled",
                    "experimental.enabled",
                    "该组合仅在实验性功能开启后可用。",
                    "This combination is available only when experimental features are enabled.",
                    combination_id=combo["id"],
                )
            )
        elif not acknowledged:
            issues.append(
                _issue(
                    "error",
                    "experimental_acknowledgement_required",
                    "experimental.risk_acknowledged",
                    "启动实验性组合前必须确认风险。",
                    "Risk acknowledgement is required before using an experimental combination.",
                    combination_id=combo["id"],
                )
            )
        else:
            risk = combo.get("risk", {})
            issues.append(
                _issue(
                    "warning",
                    "experimental_combination",
                    "architecture",
                    risk.get("zh-CN", "该组合未经完整验证，可能需要补充代码或配置。"),
                    risk.get("en-US", "This combination is not fully verified and may require code or config changes."),
                    combination_id=combo["id"],
                )
            )

    training = spec.get("training", {}) if isinstance(spec.get("training", {}), Mapping) else {}
    mode = training.get("template_mode") or training.get("mode")
    if mode and combo.get("modes") and mode not in combo["modes"]:
        issues.append(
            _issue(
                "error",
                "invalid_template_mode",
                "training.template_mode",
                f"模板模式 {mode} 不属于所选组合。",
                f"Template mode {mode} does not belong to the selected combination.",
                allowed=combo["modes"],
            )
        )

    dataset_node = spec.get("dataset", {}) if isinstance(spec.get("dataset", {}), Mapping) else {}
    mix = dataset_node.get("mix") or dataset_node.get("dataset_mix")
    dataset_info = _component_index(catalog, "datasets").get(selection["dataset"], {})
    allowed_mixes = dataset_info.get("mixes", [])
    if mix and allowed_mixes and mix not in allowed_mixes:
        issues.append(
            _issue(
                "error",
                "invalid_dataset_mix",
                "dataset.mix",
                f"数据混合 {mix} 不属于 {selection['dataset']}。",
                f"Dataset mix {mix} is not supported by {selection['dataset']}.",
                allowed=allowed_mixes,
            )
        )
    combo_mixes = combo.get("dataset_mixes", [])
    if mix and combo_mixes and mix not in combo_mixes:
        issues.append(
            _issue(
                "error",
                "dataset_mix_not_wired",
                "dataset.mix",
                f"所选模型与训练方法尚未接通数据混合 {mix}。",
                f"Dataset mix {mix} is not wired for the selected model and training method.",
                allowed=combo_mixes,
            )
        )

    resources = spec.get("resources", {}) if isinstance(spec.get("resources", {}), Mapping) else {}
    num_gpus = resources.get("num_gpus", resources.get("gpu_count", 1))
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        issues.append(
            _issue(
                "error",
                "invalid_num_gpus",
                "resources.num_gpus",
                "GPU 数量必须是正整数。",
                "GPU count must be a positive integer.",
            )
        )
    gpu_ids = resources.get("gpu_ids", [])
    if gpu_ids and (
        not isinstance(gpu_ids, list)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        issues.append(
            _issue(
                "error",
                "invalid_gpu_ids",
                "resources.gpu_ids",
                "GPU ID 必须是无重复的非负整数列表。",
                "GPU IDs must be a unique list of non-negative integers.",
            )
        )
    elif (
        resources.get("allocation", resources.get("strategy")) == "fixed"
        and isinstance(num_gpus, int)
        and len(gpu_ids) != num_gpus
    ):
        issues.append(
            _issue(
                "error",
                "gpu_count_mismatch",
                "resources.gpu_ids",
                "固定 GPU 列表长度必须等于 GPU 数量。",
                "The fixed GPU list length must equal num_gpus.",
            )
        )
    port = resources.get("main_process_port", 29500)
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        issues.append(
            _issue(
                "error",
                "invalid_port",
                "resources.main_process_port",
                "主进程端口必须在 1024 到 65535 之间。",
                "Main-process port must be between 1024 and 65535.",
            )
        )

    parameters = spec.get("parameters", {}) if isinstance(spec.get("parameters", {}), Mapping) else {}
    run_id = parameters.get("run_id", spec.get("run_id", spec.get("name")))
    if run_id is not None and (not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id)):
        issues.append(
            _issue(
                "error",
                "invalid_run_id",
                "parameters.run_id",
                "实验名称只能包含字母、数字、点、下划线和连字符，且最长 128 字符。",
                "Run ID may contain only letters, numbers, dots, underscores, and hyphens (max 128 characters).",
            )
        )
    positive_integer_parameters = (
        "per_device_batch_size",
        "batch_size",
        "gradient_accumulation_steps",
        "max_train_steps",
        "max_steps",
        "save_interval",
        "eval_interval",
    )
    for key in positive_integer_parameters:
        value = parameters.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            issues.append(
                _issue(
                    "error",
                    "invalid_positive_integer",
                    f"parameters.{key}",
                    f"{key} 必须是正整数。",
                    f"{key} must be a positive integer.",
                )
            )
    learning_rate = parameters.get("learning_rate")
    if learning_rate is not None and not isinstance(learning_rate, Mapping):
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            issues.append(
                _issue(
                    "error",
                    "invalid_learning_rate",
                    "parameters.learning_rate",
                    "学习率必须大于 0。",
                    "Learning rate must be greater than zero.",
                )
            )

    expert = spec.get("expert_overrides", {})
    if expert is not None and not isinstance(expert, Mapping):
        issues.append(
            _issue(
                "error",
                "invalid_expert_overrides",
                "expert_overrides",
                "专家配置必须是对象。",
                "Expert overrides must be an object.",
            )
        )
    issues.extend(validate_workflow_spec(spec, combo))
    return issues


def find_combination(
    spec: Mapping[str, Any],
    *,
    require_enabled: bool = True,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the unique registry combination selected by ``spec``.

    ``ValueError`` carries the validation issue list in its first argument so
    callers can map it directly to an HTTP 422 response.
    """

    catalog = _catalog_copy(source_catalog)
    issues = validate_compatibility(spec, source_catalog=catalog)
    blocking = [issue for issue in issues if issue["severity"] == "error"]
    if require_enabled and blocking:
        raise ValueError(blocking)
    selection = selection_from_spec(spec)
    matches = [
        combo
        for combo in _matching_combinations(selection, source_catalog=catalog)
        if combo.get("status") in SELECTABLE_STATUSES
    ]
    if not matches:
        raise ValueError(blocking or [{"code": "incompatible_combination", "selection": selection}])
    return deepcopy(matches[0])


def resolve_experiment(
    spec: Mapping[str, Any],
    repo_root: Any,
    *,
    source_catalog: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lazy compatibility export for the primary configuration resolver.

    Keeping the import inside the function avoids a module cycle because the
    resolver itself uses compatibility validation from this module.
    """

    from .configuration import resolve_experiment as _resolve

    return _resolve(spec, repo_root, source_catalog=source_catalog, **kwargs)


__all__ = [
    "find_combination",
    "get_catalog",
    "get_compatible_options",
    "resolve_pretrained_directory_status",
    "resolve_experiment",
    "selection_from_spec",
    "validate_compatibility",
]
