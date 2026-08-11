"""Versioned training-workflow contracts for the UI and configuration resolver.

The capability registry owns field presentation and resolved-config paths.  This
module supplies the small amount of behaviour that data alone cannot express:
legacy-spec migration, conditional validation, and deterministic override
generation.  It deliberately has no dependency on training libraries.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, MutableMapping

from .registry import load_catalog

SPEC_VERSION = 2
RESUME_MODES = frozenset({"none", "weights_only", "full_state"})
_TASK_IDS_RE = re.compile(r"^\s*\d+(?:\s*,\s*\d+)*\s*$")


def _issue(severity: str, code: str, path: str, zh: str, en: str, **detail: Any) -> dict[str, Any]:
    return {
        "severity": severity,
        "level": severity,
        "code": code,
        "path": path,
        "field": path,
        "message": zh,
        "message_i18n": {"zh-CN": zh, "en-US": en},
        "detail": detail,
    }


def _set_path(target: MutableMapping[str, Any], dotted_path: str, value: Any) -> None:
    node = target
    parts = [part for part in dotted_path.split(".") if part]
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            node[part] = child
        node = child
    if parts:
        node[parts[-1]] = deepcopy(value)


def workflow_schema_for(
    combination: Mapping[str, Any], catalog: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return an isolated schema referenced by a registry combination."""

    raw = combination.get("workflow", "standard")
    if isinstance(raw, Mapping):
        return deepcopy(dict(raw))
    catalog = catalog or load_catalog()
    schema = catalog.get("workflow_schemas", {}).get(str(raw))
    if not isinstance(schema, Mapping):
        raise KeyError(f"Unknown workflow schema: {raw}")
    return deepcopy(dict(schema))


def field_is_active(
    field: Mapping[str, Any],
    values: Mapping[str, Any],
    combination: Mapping[str, Any],
) -> bool:
    methods = field.get("methods")
    if isinstance(methods, list) and combination.get("method") not in methods:
        return False
    algorithms = field.get("algorithms")
    if isinstance(algorithms, list) and combination.get("rl_algorithm") not in algorithms:
        return False
    condition = field.get("visible_when")
    if not isinstance(condition, Mapping):
        return True
    current = values.get(str(condition.get("key", "")))
    if "equals" in condition:
        return current == condition["equals"]
    if isinstance(condition.get("in"), list):
        return current in condition["in"]
    return True


def _workflow_defaults(schema: Mapping[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for field in schema.get("fields", []):
        if isinstance(field, Mapping) and "default" in field:
            defaults[str(field["key"])] = deepcopy(field["default"])
    return defaults


def _legacy_workflow_values(spec: Mapping[str, Any], workflow_id: str) -> dict[str, Any]:
    training = spec.get("training", {}) if isinstance(spec.get("training"), Mapping) else {}
    values: dict[str, Any] = {}
    if workflow_id == "rl_token":
        encoder = training.get("encoder_checkpoint")
        values["encoder_mode"] = "reuse" if encoder else "train"
        if encoder:
            values["encoder_checkpoint"] = encoder
        if training.get("all_tasks"):
            values["task_scope"] = "all"
        elif training.get("task_ids") not in (None, ""):
            values.update({"task_scope": "subset", "task_ids": str(training["task_ids"])})
        elif training.get("task_id") is not None:
            values.update({"task_scope": "single", "task_id": training["task_id"]})
    elif workflow_id == "neurovla_pretrain" and training.get("pipeline_mode"):
        values["pipeline_mode"] = training["pipeline_mode"]
    return values


def normalise_workflow_spec(spec: Mapping[str, Any], combination: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate a legacy experiment mapping to the canonical spec-v2 shape."""

    result = deepcopy(dict(spec))
    schema = workflow_schema_for(combination)
    raw_workflow = result.get("workflow")
    raw_config: Mapping[str, Any] = {}
    if isinstance(raw_workflow, Mapping) and isinstance(raw_workflow.get("config"), Mapping):
        raw_config = raw_workflow["config"]
    values = {
        **_workflow_defaults(schema),
        **_legacy_workflow_values(result, str(schema["id"])),
        **deepcopy(dict(raw_config)),
    }
    result["spec_version"] = SPEC_VERSION
    result["workflow"] = {
        "id": str(schema["id"]),
        "schema_version": int(schema.get("schema_version", 1)),
        "config": values,
    }

    raw_resume = result.get("resume")
    if isinstance(raw_resume, Mapping):
        mode = str(raw_resume.get("mode", "none"))
        checkpoint = raw_resume.get("checkpoint")
    else:
        training = result.get("training", {}) if isinstance(result.get("training"), Mapping) else {}
        parameters = result.get("parameters", {}) if isinstance(result.get("parameters"), Mapping) else {}
        explicit = training.get("resume_checkpoint") or parameters.get("resume_checkpoint")
        legacy_checkpoint = training.get("checkpoint") or parameters.get("pretrained_checkpoint")
        if explicit:
            mode, checkpoint = "full_state", explicit
        elif training.get("is_resume") and legacy_checkpoint:
            mode, checkpoint = "full_state", legacy_checkpoint
        else:
            mode, checkpoint = "none", None
    result["resume"] = {"mode": mode, "checkpoint": checkpoint or None}
    return result


def _valid_type(value: Any, field_type: str) -> bool:
    if field_type == "boolean":
        return isinstance(value, bool)
    if field_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if field_type in {"string", "path", "textarea", "password", "select"}:
        return isinstance(value, str)
    return True


def validate_workflow_spec(spec: Mapping[str, Any], combination: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate workflow and resume nodes without touching the filesystem."""

    issues: list[dict[str, Any]] = []
    version = spec.get("spec_version")
    if version is not None and version not in {1, SPEC_VERSION}:
        issues.append(
            _issue(
                "error",
                "unsupported_spec_version",
                "spec_version",
                f"不支持实验配置版本：{version}",
                f"Unsupported experiment spec version: {version}",
                supported=[1, SPEC_VERSION],
            )
        )

    schema = workflow_schema_for(combination)
    raw_workflow = spec.get("workflow")
    if raw_workflow is None:
        raw_config: Mapping[str, Any] = {}
    elif not isinstance(raw_workflow, Mapping):
        return issues + [
            _issue("error", "invalid_workflow", "workflow", "Workflow 必须是对象。", "Workflow must be an object.")
        ]
    else:
        workflow_id = raw_workflow.get("id")
        if workflow_id not in (None, schema["id"]):
            issues.append(
                _issue(
                    "error",
                    "workflow_mismatch",
                    "workflow.id",
                    f"所选组合要求 workflow {schema['id']}，不能使用 {workflow_id}。",
                    f"The selected combination requires workflow {schema['id']}, not {workflow_id}.",
                    expected=schema["id"],
                )
            )
        raw_config = raw_workflow.get("config", {})
        if not isinstance(raw_config, Mapping):
            issues.append(
                _issue(
                    "error",
                    "invalid_workflow_config",
                    "workflow.config",
                    "Workflow 配置必须是对象。",
                    "Workflow config must be an object.",
                )
            )
            raw_config = {}

    values = {**_workflow_defaults(schema), **dict(raw_config)}
    fields = {str(item["key"]): item for item in schema.get("fields", []) if isinstance(item, Mapping)}
    unknown = sorted(set(raw_config) - set(fields))
    if unknown:
        issues.append(
            _issue(
                "error",
                "unknown_workflow_fields",
                "workflow.config",
                f"Workflow 包含未知字段：{', '.join(unknown)}",
                f"Workflow contains unknown fields: {', '.join(unknown)}",
                fields=unknown,
            )
        )
    for key, field in fields.items():
        if not field_is_active(field, values, combination):
            continue
        value = values.get(key)
        path = f"workflow.config.{key}"
        if field.get("required") and (value is None or (isinstance(value, str) and not value.strip())):
            issues.append(
                _issue("error", "required_workflow_field", path, f"请填写 {key}。", f"Provide {key}.")
            )
            continue
        if value is None:
            continue
        field_type = str(field.get("type", "string"))
        if not _valid_type(value, field_type):
            issues.append(
                _issue(
                    "error",
                    "invalid_workflow_field_type",
                    path,
                    f"{key} 的类型应为 {field_type}。",
                    f"{key} must have type {field_type}.",
                    expected=field_type,
                )
            )
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if field.get("min") is not None and value < field["min"]:
                issues.append(_issue("error", "workflow_value_too_small", path, f"{key} 小于最小值。", f"{key} is below its minimum."))
            if field.get("max") is not None and value > field["max"]:
                issues.append(_issue("error", "workflow_value_too_large", path, f"{key} 大于最大值。", f"{key} exceeds its maximum."))
        options = field.get("options")
        if isinstance(options, list):
            allowed = [item.get("value") for item in options if isinstance(item, Mapping)]
            if value not in allowed:
                issues.append(
                    _issue("error", "invalid_workflow_option", path, f"{key} 不是有效选项。", f"{key} is not an allowed option.", allowed=allowed)
                )
    if schema.get("id") == "rl_token" and values.get("task_scope") == "subset":
        task_ids = values.get("task_ids")
        if isinstance(task_ids, str) and task_ids and not _TASK_IDS_RE.fullmatch(task_ids):
            issues.append(
                _issue(
                    "error",
                    "invalid_task_ids",
                    "workflow.config.task_ids",
                    "任务 ID 必须是逗号分隔的非负整数。",
                    "Task IDs must be comma-separated non-negative integers.",
                )
            )

    resume = spec.get("resume")
    if resume is not None:
        if not isinstance(resume, Mapping):
            issues.append(_issue("error", "invalid_resume", "resume", "Resume 必须是对象。", "Resume must be an object."))
        else:
            mode = resume.get("mode", "none")
            checkpoint = resume.get("checkpoint")
            if mode not in RESUME_MODES:
                issues.append(
                    _issue("error", "invalid_resume_mode", "resume.mode", "Resume 模式无效。", "Resume mode is invalid.", allowed=sorted(RESUME_MODES))
                )
            elif mode != "none" and (not isinstance(checkpoint, str) or not checkpoint.strip()):
                issues.append(
                    _issue(
                        "error",
                        "resume_checkpoint_required",
                        "resume.checkpoint",
                        "所选 Resume 模式需要 checkpoint 路径。",
                        "The selected resume mode requires a checkpoint path.",
                    )
                )
    return issues


def workflow_overrides(
    spec: Mapping[str, Any], combination: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return base and per-stage resolved-config overrides."""

    schema = workflow_schema_for(combination)
    workflow = spec.get("workflow", {}) if isinstance(spec.get("workflow"), Mapping) else {}
    values = workflow.get("config", {}) if isinstance(workflow.get("config"), Mapping) else {}
    base: dict[str, Any] = {}
    stages: dict[str, dict[str, Any]] = {}
    for field in schema.get("fields", []):
        if not isinstance(field, Mapping) or not field_is_active(field, values, combination):
            continue
        key = str(field["key"])
        path = field.get("config_path")
        if not path or key not in values or values[key] is None or values[key] == "":
            continue
        stage = field.get("stage")
        target = stages.setdefault(str(stage), {}) if stage else base
        _set_path(target, str(path), values[key])
    method = combination.get("method")
    if method == "continual_er":
        # ER's repository-backed configuration uses the legacy replay block;
        # mirror the structured workbench values there instead of creating an
        # algorithm block that would be ignored while replay.enabled is true.
        replay = {
            "enabled": True,
            "method": "experience_replay",
            "buffer_size_per_task": values.get("buffer_size_per_task", 1000),
            "replay_batch_ratio": values.get("replay_batch_ratio", 0.5),
            "balanced_sampling": values.get("balanced_sampling", True),
        }
        _set_path(base, "continual_learning.replay", replay)
        base.get("continual_learning", {}).pop("algorithm", None)
    elif method in {"continual_mir", "continual_ewc"}:
        _set_path(base, "continual_learning.replay.enabled", False)
        _set_path(base, "continual_learning.algorithm.name", method.removeprefix("continual_"))
    return base, stages


def resume_overrides(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Map the explicit spec-v2 resume contract to trainer configuration."""

    resume = spec.get("resume", {}) if isinstance(spec.get("resume"), Mapping) else {}
    mode = resume.get("mode", "none")
    checkpoint = resume.get("checkpoint")
    if mode == "weights_only":
        return {"trainer": {"pretrained_checkpoint": checkpoint, "is_resume": False, "resume_checkpoint": None}}
    if mode == "full_state":
        return {"trainer": {"pretrained_checkpoint": None, "is_resume": True, "resume_checkpoint": checkpoint}}
    return {}


__all__ = [
    "RESUME_MODES",
    "SPEC_VERSION",
    "field_is_active",
    "normalise_workflow_spec",
    "resume_overrides",
    "validate_workflow_spec",
    "workflow_overrides",
    "workflow_schema_for",
]
