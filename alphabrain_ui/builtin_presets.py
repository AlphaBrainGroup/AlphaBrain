"""Curated, read-only UI examples backed by locally installed resources.

These records are intentionally not inserted into the experiment database:
they are starting points and local artifacts, not evidence that training or
evaluation has already run.  Stable IDs let the frontend link to them like
normal templates while keeping mutations disabled.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BUILTIN_TEMPLATE_PREFIX = "builtin-template:"
BUILTIN_CHECKPOINT_PREFIX = "builtin-checkpoint:"


_TEMPLATE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "slug": "qwen25-oft-libero",
        "name": {"zh-CN": "Qwen2.5-VL · LIBERO OFT 入门", "en-US": "Qwen2.5-VL · LIBERO OFT starter"},
        "description": {
            "zh-CN": "单 GPU 起步样例；使用本机 Qwen2.5-VL，适合先跑通 imitation learning 与 checkpoint 链路。",
            "en-US": "A single-GPU starter using the installed Qwen2.5-VL model to validate imitation learning and checkpoint output.",
        },
        "category": "baseline",
        "tags": ["Qwen2.5-VL", "OFT", "LIBERO", "1 GPU"],
        "model_directory": "Qwen2.5-VL-3B-Instruct",
        "dataset": "libero",
        "recommended_gpu_count": 1,
        "spec": {
            "spec_version": 2,
            "architecture": {"backbone": "qwen2_5_vl", "action_head": "mlp_regression"},
            "training": {"method": "imitation_learning"},
            "dataset": {"id": "libero", "mix": "libero_goal"},
            "parameters": {
                "per_device_batch_size": 4,
                "learning_rate": 0.0001,
                "max_train_steps": 10_000,
                "save_interval": 1_000,
                "seed": 42,
                "num_workers": 4,
            },
            "resources": {"allocation": "auto", "num_gpus": 1, "gpu_ids": []},
            "workflow": {"id": "standard", "schema_version": 1, "config": {}},
            "resume": {"mode": "none", "checkpoint": None},
            "wandb": {"enabled": False, "mode": "disabled", "categories": ["metrics", "config"]},
            "expert_overrides": {},
            "metadata": {"description": "Built-in Qwen2.5-VL LIBERO OFT starter"},
        },
    },
    {
        "slug": "qwen3-oft-libero",
        "name": {"zh-CN": "Qwen3-VL · LIBERO OFT", "en-US": "Qwen3-VL · LIBERO OFT"},
        "description": {
            "zh-CN": "使用本机 Qwen3-VL-4B 的单 GPU OFT 样例，可作为 Qwen2.5 基线的升级对照。",
            "en-US": "A single-GPU OFT example using the installed Qwen3-VL-4B model for comparison with the Qwen2.5 baseline.",
        },
        "category": "baseline",
        "tags": ["Qwen3-VL", "OFT", "LIBERO", "1 GPU"],
        "model_directory": "Qwen3-VL-4B-Instruct",
        "dataset": "libero",
        "recommended_gpu_count": 1,
        "spec": {
            "spec_version": 2,
            "architecture": {"backbone": "qwen3_vl", "action_head": "mlp_regression"},
            "training": {"method": "imitation_learning"},
            "dataset": {"id": "libero", "mix": "libero_goal"},
            "parameters": {
                "per_device_batch_size": 2,
                "learning_rate": 0.00005,
                "max_train_steps": 10_000,
                "save_interval": 1_000,
                "seed": 42,
                "num_workers": 4,
            },
            "resources": {"allocation": "auto", "num_gpus": 1, "gpu_ids": []},
            "workflow": {"id": "standard", "schema_version": 1, "config": {}},
            "resume": {"mode": "none", "checkpoint": None},
            "wandb": {"enabled": False, "mode": "disabled", "categories": ["metrics", "config"]},
            "expert_overrides": {},
            "metadata": {"description": "Built-in Qwen3-VL LIBERO OFT example"},
        },
    },
    {
        "slug": "neurovla-libero",
        "name": {"zh-CN": "NeuroVLA · LIBERO 预训练", "en-US": "NeuroVLA · LIBERO pretraining"},
        "description": {
            "zh-CN": "使用本机 Qwen2.5-VL 与 SNN Action Head 的 NeuroVLA 样例，保留仓库已接通的预训练 workflow。",
            "en-US": "A NeuroVLA example using the installed Qwen2.5-VL backbone, SNN action head, and the repository-backed pretraining workflow.",
        },
        "category": "brain_inspired",
        "tags": ["NeuroVLA", "SNN", "LIBERO", "1 GPU"],
        "model_directory": "Qwen2.5-VL-3B-Instruct",
        "dataset": "libero",
        "recommended_gpu_count": 1,
        "spec": {
            "spec_version": 2,
            "architecture": {"backbone": "qwen2_5_vl", "action_head": "snn"},
            "training": {"method": "imitation_learning"},
            "dataset": {"id": "libero", "mix": "libero_goal"},
            "parameters": {
                "per_device_batch_size": 2,
                "learning_rate": 0.0001,
                "max_train_steps": 20_000,
                "save_interval": 2_000,
                "seed": 42,
                "num_workers": 4,
            },
            "resources": {"allocation": "auto", "num_gpus": 1, "gpu_ids": []},
            "workflow": {"id": "neurovla_pretrain", "schema_version": 1, "config": {}},
            "resume": {"mode": "none", "checkpoint": None},
            "wandb": {"enabled": False, "mode": "disabled", "categories": ["metrics", "config"]},
            "expert_overrides": {},
            "metadata": {"description": "Built-in NeuroVLA LIBERO pretraining example"},
        },
    },
    {
        "slug": "vjepa2-world-model-libero",
        "name": {"zh-CN": "V-JEPA2 World Model · LIBERO", "en-US": "V-JEPA2 world model · LIBERO"},
        "description": {
            "zh-CN": "使用已登记 V-JEPA2 权重的 World Model 训练样例；默认按仓库配置申请 4 张 GPU。",
            "en-US": "A world-model training example backed by the registered V-JEPA2 weights; the repository recipe requests four GPUs.",
        },
        "category": "world_model",
        "tags": ["V-JEPA2", "World Model", "LIBERO", "4 GPU"],
        "resource_id": "world_model.vjepa2",
        "resource_fallback": "vjepa2/vjepa2_1_vitG_384.pt",
        "dataset": "libero",
        "recommended_gpu_count": 4,
        "spec": {
            "spec_version": 2,
            "architecture": {"backbone": "vjepa2", "action_head": "flow_matching_dit"},
            "training": {"method": "imitation_learning"},
            "dataset": {"id": "libero", "mix": "libero_all"},
            "parameters": {
                "per_device_batch_size": 1,
                "learning_rate": 0.0001,
                "max_train_steps": 20_000,
                "save_interval": 2_000,
                "seed": 42,
                "num_workers": 4,
            },
            "resources": {"allocation": "auto", "num_gpus": 4, "gpu_ids": []},
            "workflow": {"id": "world_model", "schema_version": 1, "config": {}},
            "resume": {"mode": "none", "checkpoint": None},
            "wandb": {"enabled": False, "mode": "disabled", "categories": ["metrics", "config", "videos"]},
            "expert_overrides": {},
            "metadata": {"description": "Built-in V-JEPA2 world-model example"},
        },
    },
    {
        "slug": "wan22-world-model-libero",
        "name": {"zh-CN": "Wan2.2 World Model · LIBERO", "en-US": "Wan2.2 world model · LIBERO"},
        "description": {
            "zh-CN": "使用本机 Wan2.2-TI2V-5B 的视频 World Model 样例；适合检查视频预测训练与回放链路。",
            "en-US": "A video world-model example using the installed Wan2.2-TI2V-5B resources to exercise prediction and replay workflows.",
        },
        "category": "world_model",
        "tags": ["Wan2.2", "World Model", "Video", "4 GPU"],
        "resource_id": "world_model.wan22",
        "resource_fallback": "Wan2.2-TI2V-5B",
        "dataset": "libero",
        "recommended_gpu_count": 4,
        "spec": {
            "spec_version": 2,
            "architecture": {"backbone": "wan2_2", "action_head": "flow_matching_dit"},
            "training": {"method": "imitation_learning"},
            "dataset": {"id": "libero", "mix": "libero_all"},
            "parameters": {
                "per_device_batch_size": 1,
                "learning_rate": 0.0001,
                "max_train_steps": 20_000,
                "save_interval": 2_000,
                "seed": 42,
                "num_workers": 4,
            },
            "resources": {"allocation": "auto", "num_gpus": 4, "gpu_ids": []},
            "workflow": {"id": "world_model", "schema_version": 1, "config": {}},
            "resume": {"mode": "none", "checkpoint": None},
            "wandb": {"enabled": False, "mode": "disabled", "categories": ["metrics", "config", "videos"]},
            "expert_overrides": {},
            "metadata": {"description": "Built-in Wan2.2 world-model example"},
        },
    },
)


_CHECKPOINT_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "slug": "cosmos-policy-libero-predict2-2b",
        "directory": "Cosmos-Policy-LIBERO-Predict2-2B",
        "name": {"zh-CN": "Cosmos Policy · LIBERO 预训练策略", "en-US": "Cosmos Policy · LIBERO pretrained policy"},
        "description": {
            "zh-CN": "可由当前 Cosmos Policy 部署适配器识别；启动时仍需填写 Cosmos Predict2 基础模型目录。",
            "en-US": "Recognized by the current Cosmos Policy adapter; deployment still needs a Cosmos Predict2 base-model directory.",
        },
        "required_files": (
            "Cosmos-Policy-LIBERO-Predict2-2B.pt",
            "config.json",
            "libero_dataset_statistics.json",
            "libero_t5_embeddings.pkl",
        ),
        "combination_id": "deploy_cosmos_policy",
        "missing_requirements": {
            "zh-CN": ["Cosmos Predict2 基础模型目录"],
            "en-US": ["Cosmos Predict2 base-model directory"],
        },
    },
    {
        "slug": "pi05-base",
        "directory": "pi05_base",
        "name": {"zh-CN": "Pi0.5 · LeRobot 格式", "en-US": "Pi0.5 · LeRobot format"},
        "description": {
            "zh-CN": "LeRobot 0.6.0 原生 Pi0.5 checkpoint；通过专用适配器加载，不转换为 OpenPI 或 AlphaBrain BaseFramework 格式。",
            "en-US": "Native LeRobot 0.6.0 Pi0.5 checkpoint loaded by its dedicated adapter without conversion to OpenPI or AlphaBrain BaseFramework format.",
        },
        "required_files": ("model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"),
        "combination_id": "deploy_lerobot_pi05",
        "checkpoint_family": "pi05",
        "checkpoint_format": "lerobot",
        "checkpoint_format_label": {"zh-CN": "LeRobot Pi0.5", "en-US": "LeRobot Pi0.5"},
        "missing_requirements": {
            "zh-CN": [],
            "en-US": [],
        },
    },
)


def _path_size(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.is_file())


def _resource_status(
    definition: Mapping[str, Any],
    *,
    pretrained_root: Path,
    resource_paths: Mapping[str, str],
    environment: Mapping[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    requirements: list[dict[str, Any]] = []
    if definition.get("model_directory"):
        model_path = pretrained_root / str(definition["model_directory"])
        requirements.append({"id": "model", "path": str(model_path), "ready": model_path.is_dir() and any(model_path.iterdir())})
    if definition.get("resource_id"):
        resource_id = str(definition["resource_id"])
        raw = str(resource_paths.get(resource_id) or "")
        if raw:
            model_path = Path(raw).expanduser().resolve(strict=False)
        elif definition.get("resource_fallback"):
            model_path = pretrained_root / str(definition["resource_fallback"])
        else:
            model_path = None
        requirements.append({"id": resource_id, "path": str(model_path) if model_path else "", "ready": bool(model_path and model_path.exists())})
    if definition.get("dataset") == "libero":
        raw = str(environment.get("LEROBOT_LIBERO_DATA_DIR") or environment.get("LIBERO_DATA_ROOT") or "")
        dataset_path = Path(raw).expanduser().resolve(strict=False) if raw else None
        requirements.append({"id": "dataset.libero", "path": str(dataset_path) if dataset_path else "", "ready": bool(dataset_path and dataset_path.is_dir())})
    ready = [bool(item["ready"]) for item in requirements]
    status = "ready" if ready and all(ready) else "partial" if any(ready) else "missing"
    return status, requirements


def builtin_templates(
    *,
    pretrained_root: str | Path,
    resource_paths: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    root = Path(pretrained_root).expanduser().resolve(strict=False)
    resources = resource_paths or {}
    env = environment or {}
    rows: list[dict[str, Any]] = []
    for definition in _TEMPLATE_DEFINITIONS:
        status, requirements = _resource_status(
            definition,
            pretrained_root=root,
            resource_paths=resources,
            environment=env,
        )
        rows.append(
            {
                "id": BUILTIN_TEMPLATE_PREFIX + str(definition["slug"]),
                "owner_id": None,
                "owner_name": "AlphaBrain",
                "name": definition["name"]["zh-CN"],
                "name_i18n": dict(definition["name"]),
                "description": definition["description"]["zh-CN"],
                "description_i18n": dict(definition["description"]),
                "visibility": "shared",
                "spec": definition["spec"],
                "version": 1,
                "created_at": None,
                "updated_at": None,
                "builtin": True,
                "category": definition["category"],
                "tags": list(definition["tags"]),
                "availability": status,
                "requirements": requirements,
                "recommended_gpu_count": int(definition["recommended_gpu_count"]),
            }
        )
    return rows


def builtin_checkpoint_candidates(pretrained_root: str | Path) -> list[dict[str, Any]]:
    root = Path(pretrained_root).expanduser().resolve(strict=False)
    rows: list[dict[str, Any]] = []
    for definition in _CHECKPOINT_DEFINITIONS:
        path = root / str(definition["directory"])
        if not path.is_dir():
            continue
        required_paths = [path / str(name) for name in definition["required_files"]]
        complete = all(item.is_file() for item in required_paths)
        modified = max((item.stat().st_mtime for item in required_paths if item.exists()), default=path.stat().st_mtime)
        rows.append(
            {
                "id": BUILTIN_CHECKPOINT_PREFIX + str(definition["slug"]),
                "experiment_id": None,
                "experiment_name": definition["name"]["zh-CN"],
                "name_i18n": dict(definition["name"]),
                "description": definition["description"]["zh-CN"],
                "description_i18n": dict(definition["description"]),
                "owner_name": "AlphaBrain",
                "job_id": None,
                "path": str(path.resolve()),
                "name": path.name,
                "step": None,
                "size_bytes": _path_size(required_paths),
                "is_complete": complete,
                "is_resumable": False,
                "complete": complete,
                "resumable": False,
                "kind": "builtin",
                "builtin": True,
                "important": True,
                "combination_id": definition["combination_id"],
                "checkpoint_family": definition.get("checkpoint_family"),
                "checkpoint_format": definition.get("checkpoint_format"),
                "checkpoint_format_label_i18n": dict(definition.get("checkpoint_format_label", {})),
                "missing_requirements_i18n": dict(definition["missing_requirements"]),
                "created_at": datetime.fromtimestamp(modified, timezone.utc),
                "can_delete": False,
                "metadata": {
                    "source": "local_pretrained_resource",
                    "builtin": True,
                    "description_i18n": dict(definition["description"]),
                    "missing_requirements_i18n": dict(definition["missing_requirements"]),
                },
            }
        )
    return rows


def is_builtin_template_id(value: str) -> bool:
    return value.startswith(BUILTIN_TEMPLATE_PREFIX)


def is_builtin_checkpoint_id(value: str) -> bool:
    return value.startswith(BUILTIN_CHECKPOINT_PREFIX)
