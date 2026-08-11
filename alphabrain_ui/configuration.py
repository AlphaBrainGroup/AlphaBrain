"""Experiment-spec resolution and non-mutating static preflight checks.

Resolution mirrors AlphaBrain's documented merge order while keeping every
repository YAML read-only.  The returned snapshot is a plain dictionary that
can be persisted as YAML/JSON by the backend and passed to launcher adapters.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml
from pydantic import ValidationError

from .capabilities import (
    find_combination,
    selection_from_spec,
    validate_compatibility,
)
from .registry import load_catalog
from .workflows import normalise_workflow_spec, resume_overrides, workflow_overrides
from .wandb import (
    normalize_wandb_config,
    unsupported_wandb_categories,
    wandb_category_catalog,
    wandb_environment,
)

SNAPSHOT_PATH_TOKEN = "${RESOLVED_CONFIG_PATH}"
ALLOCATED_GPU_IDS_TOKEN = "${ALLOCATED_GPU_IDS}"
_BASH_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-(.*?))?\}")
_OMEGA_ENV_RE = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")
_ANY_ENV_RE = re.compile(r"\$\{(?:oc\.env:)?([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?(?:,[^}]*)?\}")


def _is_secret_environment_name(name: str) -> bool:
    upper = name.upper()
    return upper in {"TOKEN", "PASSWORD", "SECRET", "API_KEY", "APIKEY", "CREDENTIAL", "CREDENTIALS"} or upper.endswith(
        ("_TOKEN", "_PASSWORD", "_SECRET", "_API_KEY", "_APIKEY", "_CREDENTIAL", "_CREDENTIALS")
    )


def _safe_expansion_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Exclude credentials so their values can never enter persisted snapshots."""

    return {str(key): str(value) for key, value in environment.items() if not _is_secret_environment_name(str(key))}


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment cannot be resolved safely."""

    def __init__(self, issues: Sequence[Mapping[str, Any]]):
        self.issues = [dict(item) for item in issues]
        super().__init__(self.issues)


def _issue(severity: str, code: str, path: str, zh: str, en: str, **metadata: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "severity": severity,
        "level": severity,
        "code": code,
        "path": path,
        "field": path,
        "message": zh,
        "message_i18n": {"zh-CN": zh, "en-US": en},
        "detail": {},
    }
    value.update(metadata)
    value["detail"].update(metadata)
    return value


def _deep_merge(*nodes: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in nodes:
        if not node:
            continue
        for key, value in node.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ExperimentConfigurationError(
            [
                _issue(
                    "error",
                    "invalid_yaml_root",
                    str(path),
                    f"配置文件根节点必须是对象：{path}",
                    f"Configuration root must be an object: {path}",
                )
            ]
        )
    return value


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _source_record(repo_root: Path, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    try:
        display = str(path.relative_to(repo_root))
    except ValueError:
        display = str(path)
    return {"path": display, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def _expand_environment(value: Any, environment: Mapping[str, str]) -> Any:
    """Expand known shell/OmegaConf env references and preserve unknown ones."""

    if isinstance(value, str):

        def omega_replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in environment:
                return environment[name]
            if default is not None:
                return default.strip()
            return match.group(0)

        def bash_replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(3)
            if name in environment:
                return environment[name]
            if default is not None:
                return default
            return match.group(0)

        return _BASH_ENV_RE.sub(bash_replace, _OMEGA_ENV_RE.sub(omega_replace, value))
    if isinstance(value, Mapping):
        return {key: _expand_environment(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item, environment) for item in value]
    return value


def _set_path(target: MutableMapping[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part for part in dotted_path.lstrip("-").split(".") if part]
    if not parts:
        return
    node: MutableMapping[str, Any] = target
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = deepcopy(value)


def _parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _dotlist_to_mapping(items: Sequence[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    index = 0
    while index < len(items):
        raw = str(items[index])
        if "=" in raw:
            key, value = raw.lstrip("-").split("=", 1)
            index += 1
        elif raw.startswith("--") and index + 1 < len(items):
            key, value = raw[2:], items[index + 1]
            index += 2
        else:
            index += 1
            continue
        _set_path(result, key, _parse_scalar(value))
    return result


def _resolve_finetune_mode(repo_root: Path, mode: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    finetune_path = repo_root / "configs/finetune_config.yaml"
    finetune = _load_yaml(finetune_path)
    sources.append(_source_record(repo_root, finetune_path))
    modes = finetune.get("modes", {})
    if mode not in modes or modes[mode].get("type") == "eval":
        raise ExperimentConfigurationError(
            [
                _issue(
                    "error",
                    "unknown_training_mode",
                    "training.template_mode",
                    f"找不到训练模式：{mode}",
                    f"Training mode not found: {mode}",
                )
            ]
        )
    mode_node = deepcopy(modes[mode])
    defaults = finetune.get("defaults", {})

    layers: list[dict[str, Any]] = []
    model_key = mode_node.get("model") or defaults.get("model")
    dataset_key = mode_node.get("dataset") or defaults.get("dataset")
    trainer_key = mode_node.get("trainer_defaults") or defaults.get("trainer")
    for folder, key in (("models", model_key), ("datasets", dataset_key), ("trainer", trainer_key)):
        if not key:
            continue
        path = repo_root / "configs" / folder / f"{key}.yaml"
        if not path.is_file():
            raise ExperimentConfigurationError(
                [
                    _issue(
                        "error",
                        "missing_source_config",
                        str(path),
                        f"缺少源配置：{path}",
                        f"Missing source config: {path}",
                    )
                ]
            )
        layers.append(_load_yaml(path))
        sources.append(_source_record(repo_root, path))

    recipe_path = mode_node.get("config_yaml")
    if recipe_path:
        path = _repo_path(repo_root, recipe_path)
        if path.is_file():
            recipe = _load_yaml(path)
            legacy_model_path = recipe.pop("_model_config_", None)
            if legacy_model_path:
                legacy_path = _repo_path(repo_root, legacy_model_path)
                recipe = _deep_merge(_load_yaml(legacy_path), recipe)
                sources.append(_source_record(repo_root, legacy_path))
            layers.append(recipe)
            sources.append(_source_record(repo_root, path))

    global_overrides = {key: deepcopy(finetune[key]) for key in ("environment", "seed") if key in finetune}
    mapped: dict[str, Any] = {}
    if "run_id" in mode_node:
        mapped["run_id"] = mode_node["run_id"]
    if "output_root_dir" in mode_node:
        mapped["output_root_dir"] = mode_node["output_root_dir"]
    elif finetune.get("common", {}).get("output_root_dir"):
        mapped["output_root_dir"] = finetune["common"]["output_root_dir"]
    if mode_node.get("framework_name"):
        _set_path(mapped, "framework.name", mode_node["framework_name"])
    if mode_node.get("base_vlm"):
        base_vlm = str(mode_node["base_vlm"])
        if not Path(base_vlm).is_absolute() and not base_vlm.startswith(("./", "data/", "${")):
            base_vlm = "${PRETRAINED_MODELS_DIR}/" + base_vlm
        _set_path(mapped, "framework.qwenvl.base_vlm", base_vlm)
    if "data_root" in mode_node:
        _set_path(mapped, "datasets.vla_data.data_root_dir", mode_node["data_root"])
    if "dataset_mix" in mode_node:
        _set_path(mapped, "datasets.vla_data.dataset_mix", mode_node["dataset_mix"])
    training = mode_node.get("training", {})
    for key in (
        "gradient_accumulation_steps",
        "max_train_steps",
        "save_interval",
        "eval_interval",
        "freeze_modules",
        "pretrained_checkpoint",
    ):
        if key in training:
            _set_path(mapped, f"trainer.{key}", training[key])
    if "per_device_batch_size" in training:
        _set_path(mapped, "datasets.vla_data.per_device_batch_size", training["per_device_batch_size"])

    direct = {
        key: deepcopy(mode_node[key])
        for key in (
            "framework",
            "datasets",
            "trainer",
            "trackers",
            "wandb_project",
            "wandb_entity",
            "is_debug",
            "stdp",
            "lora",
        )
        if key in mode_node
    }
    extras = _dotlist_to_mapping(mode_node.get("extra_args", []))
    return _deep_merge(*layers, global_overrides, mapped, direct, extras)


def _resolve_direct_configs(
    repo_root: Path,
    combination: Mapping[str, Any],
    selection: Mapping[str, Any],
    sources: list[dict[str, Any]],
    *,
    source_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_index = {item["id"]: item for item in source_catalog["components"]["datasets"]}
    paths: list[str] = list(combination.get("config_paths", []))
    if combination.get("model_config"):
        paths.append(combination["model_config"])
    dataset_path = combination.get("dataset_config") or dataset_index.get(selection["dataset"], {}).get("config")
    if dataset_path and combination.get("launcher") not in {
        "continual_learning",
        "world_model",
        "rl_token",
        "vanilla_vla_ppo",
        "cotrain",
        "vlm_only",
    }:
        paths.append(dataset_path)
    if combination.get("launcher") == "resolved_config":
        paths.append("configs/trainer/default.yaml")
    layers: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for value in paths:
        path = _repo_path(repo_root, value)
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            raise ExperimentConfigurationError(
                [
                    _issue(
                        "error",
                        "missing_source_config",
                        str(path),
                        f"缺少源配置：{path}",
                        f"Missing source config: {path}",
                    )
                ]
            )
        layers.append(_load_yaml(path))
        sources.append(_source_record(repo_root, path))
    return _deep_merge(*layers)


def _normalise_spec(
    spec: Mapping[str, Any],
    combination: Mapping[str, Any],
    *,
    source_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    selection = selection_from_spec(spec)
    result = deepcopy(dict(spec))
    architecture = result.get("architecture") if isinstance(result.get("architecture"), dict) else {}
    architecture.update({"backbone": selection["backbone"], "action_head": selection["action_head"]})
    result["architecture"] = architecture
    training = result.get("training") if isinstance(result.get("training"), dict) else {}
    training["method"] = selection["method"]
    training["family"] = {
        "neuro_pretrain": "neurovla_pretrain",
        "neuro_stdp": "r_stdp",
        "continual_learning": "continual_learning",
        "world_model": "world_model",
        "cosmos_policy": "cosmos_policy",
        "rl_token": "rl_token",
        "vanilla_vla_ppo": "vla_ppo",
        "cotrain": "multimodal_cotrain",
        "vlm_only": "vlm_only",
    }.get(combination.get("launcher"), "imitation_learning")
    if combination.get("cl_model"):
        training.setdefault("model", combination["cl_model"])
        method_name = selection["method"]
        algorithm = method_name[len("continual_") :] if method_name.startswith("continual_") else method_name
        training.setdefault("algorithm", algorithm)
    if combination.get("default_mode") and not (training.get("template_mode") or training.get("mode")):
        training["template_mode"] = combination["default_mode"]
    if training.get("template_mode") and not training.get("mode"):
        training["mode"] = training["template_mode"]
    result["training"] = training
    dataset = result.get("dataset") if isinstance(result.get("dataset"), dict) else {}
    dataset["id"] = selection["dataset"]
    if not (dataset.get("mix") or dataset.get("dataset_mix")):
        datasets = {item["id"]: item for item in source_catalog["components"]["datasets"]}
        dataset["mix"] = datasets.get(selection["dataset"], {}).get("default_mix")
    elif "dataset_mix" in dataset and "mix" not in dataset:
        dataset["mix"] = dataset.pop("dataset_mix")
    result["dataset"] = dataset
    resources = result.get("resources") if isinstance(result.get("resources"), dict) else {}
    if "strategy" in resources and "allocation" not in resources:
        resources["allocation"] = resources["strategy"]
    if "gpu_count" in resources and "num_gpus" not in resources:
        resources["num_gpus"] = resources["gpu_count"]
    resources.setdefault("allocation", "auto")
    resources.setdefault("num_gpus", combination.get("min_gpus", 1))
    resources.setdefault("gpu_ids", [])
    resources.setdefault("main_process_port", 29500)
    resources.setdefault("deepspeed_config", "configs/deepspeed/accelerate_zero2.yaml")
    resources.setdefault("strategy", resources["allocation"])
    resources.setdefault("gpu_count", resources["num_gpus"])
    result["resources"] = resources
    result["parameters"] = (
        deepcopy(result.get("parameters", {})) if isinstance(result.get("parameters", {}), Mapping) else {}
    )
    if "batch_size" in result["parameters"] and "per_device_batch_size" not in result["parameters"]:
        result["parameters"]["per_device_batch_size"] = result["parameters"]["batch_size"]
    if "max_steps" in result["parameters"] and "max_train_steps" not in result["parameters"]:
        result["parameters"]["max_train_steps"] = result["parameters"]["max_steps"]
    if result.get("name") and not result["parameters"].get("run_id"):
        result["parameters"]["run_id"] = result["name"]
    if result["parameters"].get("pretrained_checkpoint") and not training.get("checkpoint"):
        training["checkpoint"] = result["parameters"]["pretrained_checkpoint"]
    if combination.get("launcher") == "world_model":
        training.setdefault("variant", combination.get("world_model_key"))
        paths = combination.get("config_paths", [])
        if paths:
            training.setdefault("config_file", paths[0])
    if combination.get("launcher") in {"rl_token", "vanilla_vla_ppo"}:
        training.setdefault("track", combination.get("track", "rlt"))
        training.setdefault(
            "algorithm",
            "vla_ppo" if combination.get("launcher") == "vanilla_vla_ppo" else combination.get("rl_algorithm", "td3"),
        )
        if training["algorithm"] == "td3":
            training["algorithm"] = "offpolicy"
        training.setdefault("backbone", "pi05" if selection["backbone"] == "paligemma" else "qwen")
        training.setdefault(
            "full_pipeline",
            not bool(training.get("encoder_checkpoint")) and combination.get("launcher") != "vanilla_vla_ppo",
        )
    result["expert_overrides"] = (
        deepcopy(result.get("expert_overrides", {})) if isinstance(result.get("expert_overrides", {}), Mapping) else {}
    )
    return normalise_workflow_spec(result, combination)


def _apply_method_overrides(
    config: dict[str, Any], spec: Mapping[str, Any], combination: Mapping[str, Any]
) -> dict[str, Any]:
    method = spec["training"]["method"]
    overrides: dict[str, Any] = {}
    if method == "continual_mir":
        overrides = {
            "continual_learning": {
                "replay": {"enabled": False},
                "algorithm": {
                    "name": "mir",
                    "buffer_size_per_task": 1000,
                    "replay_batch_ratio": 0.5,
                    "balanced_sampling": True,
                    "mir_refresh_interval": 50,
                    "mir_candidate_size": 16,
                    "mir_top_k": 8,
                    "mir_lora_only": True,
                },
            },
            "datasets": {"vla_data": {"per_device_batch_size": 4}},
        }
    elif method == "continual_ewc":
        overrides = {
            "continual_learning": {
                "replay": {"enabled": False},
                "algorithm": {"name": "ewc"},
            }
        }
    if combination.get("launcher") == "continual_learning":
        mix = spec["dataset"].get("mix")
        if mix in {"libero_long", "libero_10"}:
            overrides = _deep_merge(
                overrides,
                {
                    "datasets": {"vla_data": {"dataset_mix": "libero_long"}},
                    "continual_learning": {"task_sequence": "libero_long"},
                },
            )
    return _deep_merge(config, overrides)


_PARAMETER_PATHS = {
    "run_id": "run_id",
    "output_root_dir": "output_root_dir",
    "seed": "seed",
    "per_device_batch_size": "datasets.vla_data.per_device_batch_size",
    "batch_size": "datasets.vla_data.per_device_batch_size",
    "gradient_accumulation_steps": "trainer.gradient_accumulation_steps",
    "max_train_steps": "trainer.max_train_steps",
    "max_steps": "trainer.max_train_steps",
    "save_interval": "trainer.save_interval",
    "eval_interval": "trainer.eval_interval",
    "freeze_modules": "trainer.freeze_modules",
    "pretrained_checkpoint": "trainer.pretrained_checkpoint",
    "data_root": "datasets.vla_data.data_root_dir",
    "num_workers": "datasets.vla_data.num_workers",
    "action_dim": "framework.action_model.action_dim",
    "state_dim": "framework.action_model.state_dim",
    "action_horizon": "framework.action_model.action_horizon",
}


def _parameter_overrides(spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    result: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    parameters = spec.get("parameters", {})
    for key, value in parameters.items():
        if key in {"config", "overrides"} and isinstance(value, Mapping):
            result = _deep_merge(result, value)
            continue
        if key == "learning_rate":
            path = "trainer.learning_rate" if isinstance(value, Mapping) else "trainer.learning_rate.base"
        else:
            path = _PARAMETER_PATHS.get(key)
        if path:
            _set_path(result, path, value)
            provenance[path] = f"parameters.{key}"
    dataset = spec.get("dataset", {})
    if dataset.get("mix"):
        _set_path(result, "datasets.vla_data.dataset_mix", dataset["mix"])
        provenance["datasets.vla_data.dataset_mix"] = "dataset.mix"
    dataset_root = dataset.get("root") or dataset.get("data_root")
    if dataset_root:
        _set_path(result, "datasets.vla_data.data_root_dir", dataset_root)
        provenance["datasets.vla_data.data_root_dir"] = "dataset.data_root"
    mixture_spec = dataset.get("mixture_spec")
    if isinstance(mixture_spec, list) and mixture_spec:
        _set_path(result, "datasets.vla_data.mixture_spec", deepcopy(mixture_spec))
        provenance["datasets.vla_data.mixture_spec"] = "dataset.mixture_spec"
    return result, provenance


def _normalise_expert_overrides(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if "." in key:
            _set_path(result, key, item)
        elif isinstance(item, Mapping):
            result[key] = _normalise_expert_overrides(item)
        else:
            result[key] = deepcopy(item)
    return result


def _secret_paths(node: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            key_lower = str(key).lower()
            is_secret = key_lower in {
                "token",
                "api_key",
                "apikey",
                "secret",
                "password",
                "passwd",
            } or key_lower.endswith(("_api_key", "_access_token", "_auth_token", "_secret", "_password"))
            if is_secret and value not in (None, ""):
                found.append(path)
            found.extend(_secret_paths(value, path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_secret_paths(value, f"{prefix}[{index}]"))
    return found


def _rl_pipeline_config(spec: Mapping[str, Any], combination: Mapping[str, Any]) -> dict[str, Any]:
    params = spec.get("parameters", {})
    training = spec.get("training", {})
    workflow = spec.get("workflow", {}) if isinstance(spec.get("workflow"), Mapping) else {}
    workflow_config = workflow.get("config", {}) if isinstance(workflow.get("config"), Mapping) else {}
    encoder_checkpoint = workflow_config.get("encoder_checkpoint") or training.get("encoder_checkpoint")
    return {
        "run_id": params.get("run_id", combination["id"]),
        "output_root_dir": params.get("output_root_dir", "./results/rlt_training"),
        "seed": params.get("seed", 42),
        "ui_pipeline": {
            "family": "rl_token",
            "track": combination.get("track"),
            "algorithm": combination.get("rl_algorithm"),
            "vla_checkpoint": training.get("checkpoint") or params.get("pretrained_checkpoint"),
            "encoder_checkpoint": encoder_checkpoint,
            "encoder_mode": workflow_config.get("encoder_mode", "reuse" if encoder_checkpoint else "train"),
            "skip_encoder_pretrain": bool(encoder_checkpoint),
            "dataset": spec["dataset"],
            "parameters": deepcopy(params),
        },
    }


def _accelerate_argv(config: Mapping[str, Any], resources: Mapping[str, Any], entrypoint: str) -> list[str]:
    return [
        "python",
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(resources["deepspeed_config"]),
        "--num_processes",
        str(resources["num_gpus"]),
        "--main_process_port",
        "{main_process_port}",
        entrypoint,
        "--config_yaml",
        SNAPSHOT_PATH_TOKEN,
    ]


def _command_preview(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    combination: Mapping[str, Any],
    repo_root: Path,
    *,
    stage_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    launcher = combination["launcher"]
    resources = spec["resources"]
    gpu_env = ",".join(str(item) for item in resources.get("gpu_ids", [])) or ALLOCATED_GPU_IDS_TOKEN
    environment: dict[str, str] = {"CUDA_VISIBLE_DEVICES": gpu_env}
    environment.update(wandb_environment(spec.get("wandb"), config_path=SNAPSHOT_PATH_TOKEN))
    stages: list[dict[str, Any]] = []

    if launcher in {"unified_finetune", "resolved_config", "cosmos_policy"}:
        argv = _accelerate_argv(config, resources, "AlphaBrain/training/train_alphabrain.py")
    elif launcher == "neuro_pretrain":
        argv = _accelerate_argv(config, resources, "AlphaBrain/training/train_alphabrain.py")
        workflow_config = spec.get("workflow", {}).get("config", {})
        if workflow_config.get("pipeline_mode") == "pretrain_then_stdp":
            run_id = str(config.get("run_id", "neurovla"))
            output_root = Path(str(config.get("output_root_dir", "./results/training")))
            stdp_config = dict((stage_configs or {}).get("stdp", {}))
            stages = [
                {
                    "id": "pretrain",
                    "depends_on": [],
                    "environment": dict(environment),
                    "argv": argv,
                    "output_dir": str(output_root / run_id),
                    "requested_gpu_count": int(resources["num_gpus"]),
                },
                {
                    "id": "stdp",
                    "depends_on": ["pretrain"],
                    "environment": dict(environment),
                    "argv": _accelerate_argv(stdp_config or config, resources, "AlphaBrain/training/train_stdp.py"),
                    "output_dir": str(output_root / f"{run_id}-stdp"),
                    "requested_gpu_count": int(resources["num_gpus"]),
                    **({"resolved_config": stdp_config} if stdp_config else {}),
                },
            ]
    elif launcher == "neuro_stdp":
        argv = _accelerate_argv(config, resources, "AlphaBrain/training/train_stdp.py")
    elif launcher == "continual_learning":
        argv = _accelerate_argv(config, resources, "AlphaBrain/training/continual_learning/train.py")
    elif launcher == "cotrain":
        argv = _accelerate_argv(config, resources, "AlphaBrain/training/train_alphabrain_cotrain.py")
    elif launcher == "vlm_only":
        argv = _accelerate_argv(config, resources, "AlphaBrain/training/train_alphabrain_vlm.py")
    elif launcher == "world_model":
        environment.update(
            {
                "MODEL": str(combination["world_model_key"]),
                "CONFIG_YAML": SNAPSHOT_PATH_TOKEN,
                "NUM_GPUS": str(resources["num_gpus"]),
                "MASTER_PORT": "{main_process_port}",
                "DEEPSPEED_CONFIG": str(resources["deepspeed_config"]),
            }
        )
        argv = ["bash", "scripts/run_world_model/train/run_world_model.sh"]
    elif launcher == "vanilla_vla_ppo":
        pipeline = config.get("ui_pipeline", {})
        checkpoint = config.get("ui_pipeline", {}).get("vla_checkpoint") or "${VLA_CHECKPOINT}"
        environment.update(
            {
                "CKPT_PATH": str(checkpoint),
                "TASK_ID": str(pipeline.get("task_id", 0)),
                "G_PER_TASK": str(pipeline.get("episodes_per_task", 8)),
                "NUM_ENVS_PER_TASK": str(pipeline.get("envs_per_task", 4)),
                "MAX_ITER": str(pipeline.get("max_iterations", 500)),
                "EVAL_INTERVAL": str(pipeline.get("eval_interval", 10)),
            }
        )
        argv = ["bash", "scripts/run_rl_scripts/run_qwen_vla_ppo.sh", "0"]
    elif launcher == "rl_token":
        pipeline = config.get("ui_pipeline", {})
        track = str(combination["track"])
        checkpoint = pipeline.get("vla_checkpoint") or "${VLA_CHECKPOINT}"
        encoder = pipeline.get("encoder_checkpoint")
        encoder_mode = str(pipeline.get("encoder_mode", "reuse" if encoder else "train"))
        rollout_gpu_count = max(1, int(pipeline.get("rollout_gpu_count", 1)))
        train_gpu_role = max(0, int(pipeline.get("train_gpu_role", 0)))
        requested_gpu_count = max(rollout_gpu_count, train_gpu_role + 1)
        task_scope = str(pipeline.get("task_scope", "single"))
        base_env = {
            "TRACK": track,
            "BACKBONE": "pi05" if combination.get("backbone") == "paligemma" else "qwen",
            "CKPT_PATH": str(checkpoint),
            "TASK_SCOPE": task_scope,
            "TASK_ID": str(pipeline.get("task_id", 0)),
            "TASK_IDS": str(pipeline.get("task_ids", "")),
            "ROLLOUT_GPUS": ",".join(str(index) for index in range(rollout_gpu_count)),
            "TRAIN_GPU": str(train_gpu_role),
            "G_PER_TASK": str(pipeline.get("episodes_per_task", 8)),
            "NUM_ENVS_PER_TASK": str(pipeline.get("envs_per_task", 4)),
            "USE_STEPLOCK": "1" if pipeline.get("use_steplock", True) else "0",
            "BUFFER_CAPACITY": str(pipeline.get("buffer_capacity", 100000)),
            "BUFFER_WARMUP": str(pipeline.get("buffer_warmup", 512)),
            "TD_BATCH_SIZE": str(pipeline.get("td_batch_size", 256)),
            "TAU": str(pipeline.get("tau", 0.005)),
            "MAX_ITER": str(pipeline.get("max_iterations", 500)),
            "EVAL_INTERVAL": str(pipeline.get("eval_interval", 10)),
        }
        run_id = str(config.get("run_id", combination["id"]))
        output_root = Path(str(config.get("output_root_dir", "./results/rlt_training")))
        if encoder_mode == "train" and not encoder:
            stages.append(
                {
                    "id": "encoder_pretrain",
                    "depends_on": [],
                    "environment": {**environment, **base_env},
                    "argv": ["bash", "scripts/run_rl_scripts/run_rlt_pretrain.sh", "0"],
                    "output_dir": str(output_root / run_id / "pretrain"),
                    "requested_gpu_count": 1,
                }
            )
        second_env = {**environment, **base_env}
        if encoder:
            second_env["ENCODER_PATH"] = str(encoder)
        algorithm = combination.get("rl_algorithm")
        script = {
            "td3": "scripts/run_rl_scripts/run_rlt_rl.sh",
            "ppo": "scripts/run_rl_scripts/run_rlt_a_ppo.sh",
            "grpo": "scripts/run_rl_scripts/run_rlt_a_grpo.sh",
        }[algorithm]
        stages.append(
            {
                "id": "rl_offpolicy" if algorithm == "td3" else f"rl_{algorithm}",
                "depends_on": [] if encoder else ["encoder_pretrain"],
                "environment": second_env,
                "argv": ["bash", script, "0"],
                "output_dir": str(output_root / run_id / ("rl_offpolicy" if algorithm == "td3" else f"rl_{algorithm}")),
                "requested_gpu_count": requested_gpu_count if algorithm == "td3" else 1,
            }
        )
        argv = stages[0]["argv"]
    else:
        raise ExperimentConfigurationError(
            [
                _issue(
                    "error",
                    "missing_launcher",
                    "architecture",
                    "该组合没有启动适配器。",
                    "This combination has no launcher adapter.",
                )
            ]
        )

    display_parts = [f"{key}={shlex.quote(value)}" for key, value in environment.items()]
    display_parts.extend(shlex.quote(item) for item in argv)
    result = {
        "launcher_id": launcher,
        "working_directory": str(repo_root),
        "environment": environment,
        "argv": argv,
        "display": " ".join(display_parts),
        "config_path_token": SNAPSHOT_PATH_TOKEN,
    }
    if stages:
        result["stages"] = stages
    return result


def make_template_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Strip per-run allocation details while retaining scientific choices."""

    value = deepcopy(dict(spec))
    parameters = value.get("parameters")
    if isinstance(parameters, dict):
        parameters.pop("run_id", None)
        parameters.pop("output_root_dir", None)
    value.pop("run_id", None)
    resources = value.get("resources")
    if isinstance(resources, dict):
        resources.pop("gpu_ids", None)
        resources.pop("main_process_port", None)
        resources["allocation"] = "auto"
    return value


def resolve_experiment(
    spec: Mapping[str, Any],
    repo_root: Path,
    *,
    environment: Mapping[str, str] | None = None,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an editable experiment spec into an immutable launch snapshot."""

    repo_root = Path(repo_root).expanduser().resolve()
    catalog = deepcopy(dict(source_catalog)) if source_catalog is not None else load_catalog()
    compatibility = validate_compatibility(spec, source_catalog=catalog)
    blocking = [item for item in compatibility if item["severity"] == "error"]
    if blocking:
        raise ExperimentConfigurationError(blocking)
    try:
        combination = find_combination(spec, source_catalog=catalog)
    except ValueError as error:
        raise ExperimentConfigurationError(error.args[0]) from error
    normalized = _normalise_spec(spec, combination, source_catalog=catalog)
    wandb_config: dict[str, Any] | None = None
    raw_wandb = normalized.get("wandb")
    if raw_wandb is not None:
        if not isinstance(raw_wandb, Mapping):
            raise ExperimentConfigurationError(
                [
                    _issue(
                        "error",
                        "invalid_wandb_config",
                        "wandb",
                        "W&B 配置必须是对象。",
                        "W&B configuration must be an object.",
                    )
                ]
            )
        secret_fields = _secret_paths(raw_wandb, "wandb")
        if secret_fields:
            raise ExperimentConfigurationError(
                [
                    _issue(
                        "error",
                        "secret_in_wandb_config",
                        "wandb",
                        "W&B API Key 不能写入实验配置，请在设置中单独保存。",
                        "A W&B API key cannot be stored in an experiment; save it separately in Settings.",
                        keys=secret_fields,
                    )
                ]
            )
        try:
            wandb_config = normalize_wandb_config(raw_wandb)
        except ValidationError as error:
            raise ExperimentConfigurationError(
                [
                    _issue(
                        "error",
                        "invalid_wandb_config",
                        "wandb",
                        "W&B 配置包含无效字段或值。",
                        "W&B configuration contains an invalid field or value.",
                        errors=[
                            {key: value for key, value in item.items() if key not in {"input", "ctx"}}
                            for item in error.errors()
                        ],
                    )
                ]
            ) from error
        normalized["wandb"] = wandb_config
        unsupported = unsupported_wandb_categories(wandb_config, combination)
        if unsupported:
            raise ExperimentConfigurationError(
                [
                    _issue(
                        "error",
                        "wandb_category_not_supported",
                        "wandb.categories",
                        f"所选训练器尚未接通这些 W&B 内容：{', '.join(unsupported)}",
                        f"The selected trainer does not support these W&B categories: {', '.join(unsupported)}",
                        categories=unsupported,
                    )
                ]
            )
    selection = selection_from_spec(normalized)
    sources: list[dict[str, Any]] = []

    mode = normalized["training"].get("template_mode") or normalized["training"].get("mode")
    if combination["launcher"] in {"unified_finetune", "neuro_pretrain", "neuro_stdp", "cosmos_policy"} and mode:
        config = _resolve_finetune_mode(repo_root, mode, sources)
    elif combination["launcher"] in {"rl_token", "vanilla_vla_ppo"}:
        config = _rl_pipeline_config(normalized, combination)
    else:
        config = _resolve_direct_configs(
            repo_root,
            combination,
            selection,
            sources,
            source_catalog=catalog,
        )

    config = _apply_method_overrides(config, normalized, combination)
    parameter_overrides, parameter_sources = _parameter_overrides(normalized)
    config = _deep_merge(config, parameter_overrides)
    workflow_base_overrides, workflow_stage_overrides = workflow_overrides(normalized, combination)
    config = _deep_merge(config, workflow_base_overrides)
    if combination["launcher"] in {"cotrain", "vlm_only"}:
        config = _deep_merge(
            config,
            {"framework": {"qwenvl": {"base_vlm": "${PRETRAINED_MODELS_DIR}/Qwen2.5-VL-3B-Instruct"}}},
        )
    if not config.get("run_id"):
        config["run_id"] = combination["id"]
        parameter_sources["run_id"] = "registry.default"
    if not config.get("output_root_dir"):
        config["output_root_dir"] = (
            "./results/Checkpoints" if combination["launcher"] == "continual_learning" else "./results/training"
        )
        parameter_sources["output_root_dir"] = "registry.default"

    expert = _normalise_expert_overrides(normalized.get("expert_overrides", {}))
    reserved = sorted({"modes", "defaults"}.intersection(expert))
    if reserved:
        raise ExperimentConfigurationError(
            [
                _issue(
                    "error",
                    "reserved_expert_key",
                    "expert_overrides",
                    f"解析后的配置不能覆盖保留根键：{', '.join(reserved)}",
                    f"Resolved configs cannot override reserved root keys: {', '.join(reserved)}",
                )
            ]
        )
    secrets = _secret_paths(expert)
    if secrets:
        raise ExperimentConfigurationError(
            [
                _issue(
                    "error",
                    "secret_in_expert_config",
                    "expert_overrides",
                    "密钥不能写入实验配置，请在系统环境设置中配置。",
                    "Secrets cannot be stored in experiment config; configure them in system environment settings.",
                    keys=secrets,
                )
            ]
        )
    config = _deep_merge(config, expert)
    # Resume is a versioned structural contract rather than an expert tuning
    # knob.  Apply it last so weights-only and full-state cannot be mixed by
    # stale legacy fields in a template.
    config = _deep_merge(config, resume_overrides(normalized))

    stage_configs: dict[str, dict[str, Any]] = {}
    workflow_config = normalized.get("workflow", {}).get("config", {})
    if combination["launcher"] == "neuro_pretrain" and workflow_config.get("pipeline_mode") == "pretrain_then_stdp":
        stdp_config = _resolve_finetune_mode(repo_root, "neuro_vla_stdp", sources)
        stdp_config = _deep_merge(
            stdp_config,
            parameter_overrides,
            workflow_stage_overrides.get("stdp", {}),
            expert,
        )
        pretrain_run_id = str(config.get("run_id", combination["id"]))
        output_root = str(config.get("output_root_dir", "./results/training"))
        stdp_config["run_id"] = f"{pretrain_run_id}-stdp"
        stdp_config["output_root_dir"] = output_root
        stdp_config = _deep_merge(
            stdp_config,
            {
                "trainer": {
                    "pretrained_checkpoint": str(Path(output_root) / pretrain_run_id / "final_model"),
                    "is_resume": False,
                    "resume_checkpoint": None,
                }
            },
        )
        stage_configs["stdp"] = stdp_config

    if wandb_config is not None:
        enabled = bool(wandb_config["enabled"]) and wandb_config["mode"] != "disabled"
        mode = wandb_config["mode"] if enabled else "disabled"
        config["wandb_mode"] = mode
        config["wandb_project"] = wandb_config["project"]
        config["wandb_entity"] = wandb_config["entity"]
        config["wandb_run_name"] = wandb_config["run_name"]
        config["wandb_group"] = wandb_config["group"]
        config["wandb_job_type"] = wandb_config["job_type"]
        config["wandb_tags"] = deepcopy(wandb_config["tags"])
        config["wandb_notes"] = wandb_config["notes"]
        config["wandb_upload_categories"] = deepcopy(wandb_config["categories"] if enabled else [])
        environment_node = config.get("environment") if isinstance(config.get("environment"), Mapping) else {}
        config["environment"] = _deep_merge(
            environment_node,
            {
                "wandb_mode": mode,
                "wandb_project": wandb_config["project"],
                "wandb_entity": wandb_config["entity"],
            },
        )
        raw_trackers = config.get("trackers", ["jsonl"])
        if not isinstance(raw_trackers, Sequence) or isinstance(raw_trackers, (str, bytes)):
            raw_trackers = [raw_trackers]
        trackers = [str(item) for item in raw_trackers if item and str(item) != "wandb"]
        if enabled:
            trackers.append("wandb")
        config["trackers"] = trackers
        wandb_projection = {
            key: deepcopy(config[key])
            for key in (
                "wandb_mode",
                "wandb_project",
                "wandb_entity",
                "wandb_run_name",
                "wandb_group",
                "wandb_job_type",
                "wandb_tags",
                "wandb_notes",
                "wandb_upload_categories",
                "environment",
                "trackers",
            )
            if key in config
        }
        stage_configs = {key: _deep_merge(value, wandb_projection) for key, value in stage_configs.items()}

    env = _safe_expansion_environment(os.environ if environment is None else environment)
    config = _expand_environment(config, env)
    stage_configs = {key: _expand_environment(value, env) for key, value in stage_configs.items()}
    canonical = json.dumps(
        {"base": config, "stages": stage_configs},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    command = _command_preview(config, normalized, combination, repo_root, stage_configs=stage_configs)
    compatibility_status = combination["status"]
    snapshot = {
        "schema_version": 2,
        "registry_version": catalog["registry_version"],
        "compatibility": compatibility_status,
        "spec": normalized,
        "template_spec": make_template_spec(normalized),
        "combination": {
            key: deepcopy(combination[key])
            for key in ("id", "status", "backbone", "action_head", "method", "launcher", "min_gpus")
            if key in combination
        },
        "resolved_config": config,
        "stage_resolved_configs": stage_configs,
        "config_fingerprint": fingerprint,
        "sources": sources,
        "parameter_sources": parameter_sources,
        "command_preview": command,
        "compatibility_issues": compatibility,
    }
    # Keep the normalized axes at the top level as a compatibility surface for
    # launcher code that consumes an experiment spec directly.  The canonical
    # copies remain under ``spec`` and ``resolved_config``.
    for key in (
        "architecture",
        "training",
        "dataset",
        "resources",
        "parameters",
        "expert_overrides",
        "experimental",
        "wandb",
        "workflow",
        "resume",
        "spec_version",
    ):
        if key in normalized:
            snapshot[key] = deepcopy(normalized[key])
    if wandb_config is not None:
        snapshot["wandb_capabilities"] = wandb_category_catalog(combination)
    # Legacy launcher adapters use ``dataset.id`` as the task-mixture value;
    # expose that shape at the compatibility surface while the canonical spec
    # retains the dataset capability ID under ``snapshot['spec']``.
    if isinstance(snapshot.get("dataset"), dict) and snapshot["dataset"].get("mix"):
        snapshot["dataset"]["capability_id"] = snapshot["dataset"]["id"]
        snapshot["dataset"]["id"] = snapshot["dataset"]["mix"]
    snapshot["run_id"] = config.get("run_id")
    snapshot["output_root"] = config.get("output_root_dir")
    if normalized.get("training", {}).get("checkpoint"):
        snapshot["checkpoint"] = normalized["training"]["checkpoint"]
    return snapshot


def _walk_paths(node: Any, prefix: str = "") -> list[tuple[str, str]]:
    wanted = {
        "base_vlm",
        "checkpoint_path",
        "pretrained_dir",
        "text_encoder_path",
        "reason1_path",
        "pretrained_checkpoint",
        "data_root_dir",
        "data_dir",
        "t5_embeddings_path",
        "load_path",
        "encoder_checkpoint",
        "resume_checkpoint",
    }
    found: list[tuple[str, str]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in wanted and isinstance(value, str) and value:
                found.append((path, value))
            found.extend(_walk_paths(value, path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk_paths(value, f"{prefix}[{index}]"))
    return found


def preflight_static(
    spec: Mapping[str, Any],
    repo_root: Path,
    *,
    resolved: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    source_catalog: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run compatibility, source, path, and output-collision checks.

    GPU availability, queue state and live port binding are intentionally left
    to the scheduler.  This helper performs no writes and never downloads data.
    """

    repo_root = Path(repo_root).expanduser().resolve()
    issues = validate_compatibility(spec, source_catalog=source_catalog)
    if any(item["severity"] == "error" for item in issues):
        return issues
    if resolved is None:
        try:
            resolved = resolve_experiment(
                spec,
                repo_root,
                environment=environment,
                source_catalog=source_catalog,
            )
        except ExperimentConfigurationError as error:
            return issues + error.issues
    config = resolved.get("resolved_config", resolved)
    env = _safe_expansion_environment(os.environ if environment is None else environment)

    for source in resolved.get("sources", []):
        path = _repo_path(repo_root, source["path"])
        if not path.is_file():
            issues.append(
                _issue(
                    "error",
                    "missing_source_config",
                    source["path"],
                    f"缺少源配置：{path}",
                    f"Missing source config: {path}",
                )
            )

    resources = resolved.get("spec", spec).get("resources", {})
    deepspeed = resources.get("deepspeed_config")
    if deepspeed and not _repo_path(repo_root, deepspeed).is_file():
        issues.append(
            _issue(
                "error",
                "missing_deepspeed_config",
                "resources.deepspeed_config",
                f"找不到 Accelerate/DeepSpeed 配置：{deepspeed}",
                f"Accelerate/DeepSpeed config not found: {deepspeed}",
            )
        )
    requested = resources.get("num_gpus", 1)
    recommended = resolved.get("combination", {}).get("min_gpus", 1)
    if isinstance(requested, int) and requested < recommended:
        issues.append(
            _issue(
                "warning",
                "below_recommended_gpu_count",
                "resources.num_gpus",
                f"该配方建议至少 {recommended} 张 GPU；当前请求 {requested} 张。",
                f"This recipe recommends at least {recommended} GPUs; {requested} requested.",
                recommended=recommended,
                requested=requested,
            )
        )

    seen_missing_env: set[str] = set()
    seen_paths: set[tuple[str, str]] = set()
    for dotted, raw_value in _walk_paths(config):
        # Several framework configs keep a short ``qwenvl.base_vlm`` routing
        # marker next to the real PaliGemma/Llama/world-model path.  It is a
        # model discriminator (for example ``paligemma-3b``), not a filesystem
        # dependency, so only inspect the concrete sibling path.
        framework = config.get("framework", {}) if isinstance(config, Mapping) else {}
        if dotted == "framework.qwenvl.base_vlm" and any(
            isinstance(framework.get(key), Mapping) for key in ("paligemma", "llamavl", "world_model")
        ):
            continue
        expanded = _expand_environment(raw_value, env)
        missing_names = [name for name in _ANY_ENV_RE.findall(expanded) if name not in env]
        for name in missing_names:
            if name in seen_missing_env:
                continue
            seen_missing_env.add(name)
            issues.append(
                _issue(
                    "error",
                    "missing_environment_variable",
                    dotted,
                    f"环境变量 {name} 未设置。",
                    f"Environment variable {name} is not set.",
                    variable=name,
                )
            )
        if missing_names or not expanded or expanded.startswith(("http://", "https://")):
            continue
        candidate = _repo_path(repo_root, expanded)
        marker = (dotted, str(candidate))
        if marker in seen_paths:
            continue
        seen_paths.add(marker)
        if not candidate.exists():
            is_data = dotted.endswith("data_root_dir")
            code = "missing_dataset_path" if is_data else "missing_model_path"
            zh_kind = "数据路径" if is_data else "模型/权重路径"
            en_kind = "Dataset path" if is_data else "Model/weight path"
            issues.append(
                _issue(
                    "error",
                    code,
                    dotted,
                    f"{zh_kind}不存在：{candidate}",
                    f"{en_kind} does not exist: {candidate}",
                    value=str(candidate),
                )
            )

    method = selection_from_spec(spec).get("method")
    if method in {"r_stdp", "rlt_td3", "rlt_a_td3", "rlt_a_grpo", "vanilla_vla_ppo"}:
        params = spec.get("parameters", {}) if isinstance(spec.get("parameters", {}), Mapping) else {}
        training = spec.get("training", {}) if isinstance(spec.get("training", {}), Mapping) else {}
        checkpoint = training.get("checkpoint") or params.get("pretrained_checkpoint")
        if not checkpoint:
            issues.append(
                _issue(
                    "error",
                    "checkpoint_required",
                    "parameters.pretrained_checkpoint",
                    "该训练方法需要已有 VLA checkpoint。",
                    "This training method requires an existing VLA checkpoint.",
                )
            )
        else:
            checkpoint_path = _repo_path(repo_root, _expand_environment(checkpoint, env))
            if not checkpoint_path.exists():
                issues.append(
                    _issue(
                        "error",
                        "missing_checkpoint",
                        "parameters.pretrained_checkpoint",
                        f"Checkpoint 不存在：{checkpoint_path}",
                        f"Checkpoint does not exist: {checkpoint_path}",
                    )
                )

    run_id = config.get("run_id")
    output_root = config.get("output_root_dir")
    if run_id and output_root:
        output_path = _repo_path(repo_root, _expand_environment(output_root, env)) / str(run_id)
        resume = resolved.get("spec", spec).get("resume", {})
        full_state_resume = isinstance(resume, Mapping) and resume.get("mode") == "full_state"
        if output_path.exists() and not full_state_resume:
            issues.append(
                _issue(
                    "error",
                    "output_exists",
                    "parameters.run_id",
                    f"输出目录已存在，禁止覆盖：{output_path}",
                    f"Output directory already exists and will not be overwritten: {output_path}",
                    value=str(output_path),
                )
            )
    if not any(item["severity"] == "error" for item in issues):
        issues.append(
            _issue(
                "info",
                "static_preflight_passed",
                "$",
                "静态预检通过。",
                "Static preflight passed.",
            )
        )
    return issues


__all__ = [
    "ALLOCATED_GPU_IDS_TOKEN",
    "SNAPSHOT_PATH_TOKEN",
    "ExperimentConfigurationError",
    "make_template_spec",
    "preflight_static",
    "resolve_experiment",
]
