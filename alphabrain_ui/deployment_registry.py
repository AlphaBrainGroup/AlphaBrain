"""Data-only model-deployment discovery and checkpoint preflight.

This module intentionally does not import AlphaBrain, PyTorch, Transformers,
or either model-server entrypoint.  Checkpoint inspection is limited to file
names plus YAML/JSON metadata, so using it from the UI process can never
deserialize untrusted model weights.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from string import Template
from typing import Any

import yaml

from .registry import load_catalog

_STATUS_VERIFIED = "verified"
_STATUS_EXPERIMENTAL = "experimental"
_GENERIC_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.pt")
_COSMOS_WEIGHT_NAMES = (
    "cosmos_dit.pt",
    "Cosmos-Policy-LIBERO-Predict2-2B.pt",
    "pytorch_model.pt",
)
_RL_WEIGHT_NAMES = frozenset(
    {
        "actor.pt",
        "critic.pt",
        "encoder.pt",
        "value_head.pt",
        "vla_finetuned.pt",
        "vla_state_dict.pt",
    }
)
_VLM_BACKBONES = frozenset({"qwen2_5_vl", "qwen3_vl", "paligemma", "llama3_2_vision"})
_WORLD_MODEL_BACKBONES = frozenset({"cosmos2", "cosmos2_5", "vjepa2", "wan2_2"})
_CHECKPOINT_FORMAT_LABELS = {
    "openpi": {"zh-CN": "OpenPI Pi0.5", "en-US": "OpenPI Pi0.5"},
    "lerobot": {"zh-CN": "LeRobot Pi0.5", "en-US": "LeRobot Pi0.5"},
    "alphabrain": {
        "zh-CN": "AlphaBrain BaseFramework Pi0.5",
        "en-US": "AlphaBrain BaseFramework Pi0.5",
    },
    "unknown": {"zh-CN": "未知格式", "en-US": "Unknown format"},
}


def _checkpoint_format_metadata(origin: str = "unknown", *, family: str | None = None) -> dict[str, Any]:
    return {
        "checkpoint_family": family,
        "checkpoint_format": origin,
        "checkpoint_origin": origin,
        "checkpoint_format_label": deepcopy(_CHECKPOINT_FORMAT_LABELS.get(origin, _CHECKPOINT_FORMAT_LABELS["unknown"])),
    }


def _allowed_status(status: str, include_experimental: bool) -> bool:
    return status == _STATUS_VERIFIED or (include_experimental and status == _STATUS_EXPERIMENTAL)


def _plain_copy(value: Any) -> Any:
    """Copy through JSON to guarantee that the public result is JSON-only."""

    return json.loads(json.dumps(value, ensure_ascii=False))


def get_deployment_catalog(
    include_experimental: bool = False,
    *,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return only wired deployment adapters and model combinations.

    Unsupported entries are never exposed.  Experimental entries require an
    explicit caller opt-in.  Referenced backbone/action-head components are
    included so a schema-driven UI does not need to join against the training
    capability response.
    """

    source = deepcopy(dict(source_catalog)) if source_catalog is not None else load_catalog()
    combinations = [
        deepcopy(item)
        for item in source.get("deployment_combinations", [])
        if _allowed_status(str(item.get("status", "unsupported")), include_experimental)
    ]
    adapter_ids = {str(item.get("adapter", "")) for item in combinations}
    adapters = [
        deepcopy(item)
        for item in source.get("deployment_adapters", [])
        if item.get("id") in adapter_ids
        and _allowed_status(str(item.get("status", "unsupported")), include_experimental)
    ]
    available_adapter_ids = {str(item["id"]) for item in adapters}
    combinations = [item for item in combinations if item.get("adapter") in available_adapter_ids]

    referenced = {
        "backbones": {str(item.get("backbone")) for item in combinations},
        "action_heads": {str(item.get("action_head")) for item in combinations},
    }
    components: dict[str, list[dict[str, Any]]] = {}
    for group, ids in referenced.items():
        components[group] = [
            deepcopy(item)
            for item in source.get("components", {}).get(group, [])
            if item.get("id") in ids
            and _allowed_status(str(item.get("status", "unsupported")), include_experimental)
        ]

    result = {
        "schema_version": source.get("schema_version", 1),
        "registry_version": source.get("registry_version", ""),
        "deployment_adapters": adapters,
        "deployment_combinations": combinations,
        # Short aliases are convenient for API clients while preserving the
        # explicit registry keys above for direct catalog consumers.
        "adapters": deepcopy(adapters),
        "combinations": deepcopy(combinations),
        "components": components,
        "filters": {
            "include_experimental": bool(include_experimental),
            "unsupported_hidden": True,
        },
    }
    return _plain_copy(result)


def _issue(
    severity: str,
    code: str,
    field: str,
    zh: str,
    en: str,
    **detail: Any,
) -> dict[str, Any]:
    issue = {
        "severity": severity,
        "level": severity,
        "code": code,
        "field": field,
        "path": field,
        "message": zh,
        "message_i18n": {"zh-CN": zh, "en-US": en},
        "detail": _plain_copy(detail),
    }
    return issue


def _index_record(checkpoint_id: str, checkpoint_index: Any) -> Mapping[str, Any] | str | None:
    if checkpoint_index is None:
        return None
    if isinstance(checkpoint_index, Mapping):
        direct = checkpoint_index.get(checkpoint_id)
        if isinstance(direct, (Mapping, str)):
            return direct
        records = checkpoint_index.values()
    elif isinstance(checkpoint_index, Sequence) and not isinstance(checkpoint_index, (str, bytes)):
        records = checkpoint_index
    else:
        return None
    for record in records:
        if isinstance(record, Mapping) and str(record.get("id", record.get("checkpoint_id", ""))) == checkpoint_id:
            return record
    return None


def resolve_checkpoint_source(
    source: str | Path | Mapping[str, Any],
    *,
    checkpoint_index: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a local path or a checkpoint-index record without database imports."""

    issues: list[dict[str, Any]] = []
    checkpoint_id: str | None = None
    source_kind = "local_path"
    raw_path: Any = None

    if isinstance(source, Mapping):
        raw_path = source.get("path") or source.get("checkpoint_path")
        checkpoint_id_value = source.get("checkpoint_id", source.get("id"))
        if checkpoint_id_value not in (None, ""):
            checkpoint_id = str(checkpoint_id_value)
        if raw_path in (None, "") and checkpoint_id:
            source_kind = "checkpoint_index"
            record = _index_record(checkpoint_id, checkpoint_index)
            if isinstance(record, Mapping):
                raw_path = record.get("path") or record.get("checkpoint_path")
            elif isinstance(record, str):
                raw_path = record
        elif checkpoint_id:
            source_kind = "checkpoint_index"
    elif isinstance(source, (str, Path)):
        raw_path = str(source)
        candidate = Path(str(source)).expanduser()
        if not candidate.is_absolute() and repo_root is not None:
            candidate = Path(repo_root).expanduser() / candidate
        if not candidate.exists():
            record = _index_record(str(source), checkpoint_index)
            if record is not None:
                checkpoint_id = str(source)
                source_kind = "checkpoint_index"
                raw_path = record.get("path") if isinstance(record, Mapping) else record
    else:
        raw_path = None

    if raw_path in (None, ""):
        issues.append(
            _issue(
                "error",
                "checkpoint_source_missing",
                "checkpoint",
                "请选择 checkpoint 或填写本地 checkpoint 路径。",
                "Select a checkpoint or provide a local checkpoint path.",
                checkpoint_id=checkpoint_id,
            )
        )
        return {
            "valid": False,
            "kind": source_kind,
            "checkpoint_id": checkpoint_id,
            "path": "",
            "issues": issues,
        }

    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (Path(repo_root).expanduser() if repo_root is not None else Path.cwd()) / path
    path = path.resolve(strict=False)
    if not path.exists():
        issues.append(
            _issue(
                "error",
                "checkpoint_not_found",
                "checkpoint.path",
                f"Checkpoint 不存在：{path}",
                f"Checkpoint does not exist: {path}",
                checkpoint_id=checkpoint_id,
                checkpoint_path=str(path),
            )
        )
    elif not (path.is_file() or path.is_dir()):
        issues.append(
            _issue(
                "error",
                "checkpoint_path_unsupported",
                "checkpoint.path",
                f"Checkpoint 路径不是普通文件或目录：{path}",
                f"Checkpoint path is not a regular file or directory: {path}",
                checkpoint_path=str(path),
            )
        )
    return {
        "valid": not any(item["severity"] == "error" for item in issues),
        "kind": source_kind,
        "checkpoint_id": checkpoint_id,
        "path": str(path),
        "issues": issues,
    }


def _read_mapping(path: Path, *, kind: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream) if kind == "json" else yaml.safe_load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return None, _issue(
            "error",
            f"invalid_{kind}",
            str(path),
            f"无法解析 {path.name}：{exc}",
            f"Could not parse {path.name}: {exc}",
            file=str(path),
        )
    if not isinstance(value, Mapping):
        return None, _issue(
            "error",
            f"invalid_{kind}_root",
            str(path),
            f"{path.name} 的顶层必须是对象。",
            f"The root of {path.name} must be an object.",
            file=str(path),
        )
    return dict(value), None


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _framework_name(config: Mapping[str, Any]) -> str | None:
    framework = config.get("framework", {})
    if isinstance(framework, Mapping):
        value = framework.get("name") or framework.get("framework_py")
        if value not in (None, ""):
            return str(value)
    value = config.get("framework_name")
    return str(value) if value not in (None, "") else None


def _backbone_from_hint(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).lower().replace("_", "-")
    if "qwen3" in text:
        return "qwen3_vl"
    if "qwen2.5" in text or "qwen2-5" in text or "qwen2" in text:
        return "qwen2_5_vl"
    if "paligemma" in text:
        return "paligemma"
    if "llama-3.2" in text or "llama3.2" in text or "llama3-2" in text:
        return "llama3_2_vision"
    if "cosmos2.5" in text or "cosmos-2.5" in text or "cosmos2-5" in text:
        return "cosmos2_5"
    if "vjepa" in text or "v-jepa" in text:
        return "vjepa2"
    if "wan2.2" in text or "wan-2.2" in text or "wan2-2" in text:
        return "wan2_2"
    # cosmos2-diffusers is shared by current 2.0 and 2.5 recipes, so it is
    # intentionally not guessed without an additional explicit hint.
    if "cosmos2" in text and "diffusers" not in text:
        return "cosmos2"
    return None


def _embedded_backbone(checkpoint_dir: Path) -> str | None:
    for dirname in ("vlm_pretrained", "qwen_pretrained"):
        config_path = checkpoint_dir / dirname / "config.json"
        if not config_path.is_file():
            continue
        data, _error = _read_mapping(config_path, kind="json")
        if data:
            for hint in (data.get("model_type"), data.get("architectures"), data.get("name_or_path")):
                detected = _backbone_from_hint(hint)
                if detected:
                    return detected
    return None


def _detect_architecture(config: Mapping[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    framework = _framework_name(config)
    framework_block = config.get("framework", {})
    if not isinstance(framework_block, Mapping):
        framework_block = {}
    world_model = framework_block.get("world_model")
    is_world_model = isinstance(world_model, Mapping)

    backbone: str | None = None
    world_backend: str | None = None
    if is_world_model:
        assert isinstance(world_model, Mapping)
        world_backend_value = world_model.get("backend") or world_model.get("backbone")
        world_backend = str(world_backend_value) if world_backend_value not in (None, "") else None
        hints = (
            world_model.get("backbone"),
            _nested(framework_block, "qwenvl", "base_vlm"),
            world_model.get("backend"),
            world_model.get("checkpoint_path"),
            world_model.get("pretrained_dir"),
        )
        for hint in hints:
            backbone = _backbone_from_hint(hint)
            if backbone in _WORLD_MODEL_BACKBONES:
                break
            backbone = None
    else:
        hints = (
            _nested(framework_block, "qwenvl", "base_vlm"),
            _nested(framework_block, "paligemma", "base_vlm"),
            _nested(framework_block, "llamavl", "base_vlm"),
            _embedded_backbone(checkpoint_dir),
        )
        for hint in hints:
            backbone = _backbone_from_hint(hint)
            if backbone:
                break

    fixed: dict[str, tuple[str | None, str | None]] = {
        "PaliGemmaOFT": ("paligemma", "mlp_regression"),
        "PaliGemmaPi05": ("paligemma", "pi05_action_expert"),
        "LlamaOFT": ("llama3_2_vision", "mlp_regression"),
        "NeuroVLA": (backbone or "qwen2_5_vl", "snn"),
        "QwenOFT": (backbone, "mlp_regression"),
        "QwenGR00T": (backbone, "flow_matching_dit"),
        "WorldModelVLA": (backbone, "flow_matching_dit"),
        "CosmosPolicy": ("cosmos2", "cosmos_policy_dit"),
    }
    if framework in fixed:
        fixed_backbone, action_head = fixed[framework]
        backbone = fixed_backbone or backbone
    else:
        action_head = None
    return {
        "framework": framework,
        "backbone": backbone,
        "action_head": action_head,
        "is_world_model": is_world_model,
        "world_model_backend": world_backend,
    }


def _candidate_combinations(
    detected: Mapping[str, Any],
    *,
    include_experimental: bool,
    source_catalog: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    catalog = get_deployment_catalog(
        include_experimental=include_experimental,
        source_catalog=source_catalog,
    )
    candidates: list[dict[str, Any]] = []
    for combination in catalog["deployment_combinations"]:
        names = {str(value) for value in combination.get("framework_names", [])}
        if detected.get("framework") not in names:
            continue
        combination_is_world = bool(combination.get("world_model_backends"))
        if bool(detected.get("is_world_model")) != combination_is_world:
            continue
        if detected.get("backbone") and combination.get("backbone") != detected.get("backbone"):
            continue
        if detected.get("action_head") and combination.get("action_head") != detected.get("action_head"):
            continue
        candidates.append(deepcopy(combination))
    return candidates


def _resolve_configured_path(
    value: Any,
    *,
    repo_root: Path,
    environment: Mapping[str, str],
    checkpoint_root: Path | None = None,
) -> tuple[Path | None, str | None]:
    if value in (None, ""):
        return None, None
    expanded = Template(str(value)).safe_substitute(environment)
    if re.search(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", expanded):
        return None, "unresolved_environment"
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False), None

    # Training configs are often copied into an offline deployment bundle
    # while retaining a path relative to the original project root.  Search
    # from the checkpoint/config location through its parents before falling
    # back to the UI repository root.  No recursive filesystem scan is used.
    if checkpoint_root is not None:
        start = checkpoint_root if checkpoint_root.is_dir() else checkpoint_root.parent
        for base in (start, *start.parents):
            candidate = (base / path).resolve(strict=False)
            if candidate.exists():
                return candidate, None
    return (repo_root / path).resolve(strict=False), None


def _validate_embedded_vlm(path: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not (path / "config.json").is_file():
        issues.append(
            _issue(
                "error",
                "embedded_vlm_config_missing",
                "checkpoint.vlm_pretrained",
                f"内嵌 VLM 目录缺少 config.json：{path}",
                f"Embedded VLM directory is missing config.json: {path}",
                directory=str(path),
            )
        )
    processor_names = (
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    if not any((path / name).is_file() for name in processor_names):
        issues.append(
            _issue(
                "error",
                "embedded_vlm_processor_missing",
                "checkpoint.vlm_pretrained",
                f"内嵌 VLM 目录缺少 processor/tokenizer 配置：{path}",
                f"Embedded VLM directory is missing processor/tokenizer metadata: {path}",
                directory=str(path),
            )
        )
    return issues


def _validate_external_vlm(path: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not (path / "config.json").is_file():
        issues.append(
            _issue(
                "error",
                "pretrained_vlm_config_missing",
                "checkpoint.framework",
                f"基础 VLM 目录缺少 config.json：{path}",
                f"Base VLM directory is missing config.json: {path}",
                directory=str(path),
            )
        )
    processor_names = (
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    if not any((path / name).is_file() for name in processor_names):
        issues.append(
            _issue(
                "error",
                "pretrained_vlm_processor_missing",
                "checkpoint.framework",
                f"基础 VLM 目录缺少 processor/tokenizer 配置：{path}",
                f"Base VLM directory is missing processor/tokenizer metadata: {path}",
                directory=str(path),
            )
        )
    return issues


def _generic_runtime_issues(
    checkpoint_dir: Path,
    config: Mapping[str, Any],
    detected: Mapping[str, Any],
    *,
    repo_root: Path,
    environment: Mapping[str, str],
) -> tuple[list[dict[str, Any]], str | None]:
    issues: list[dict[str, Any]] = []
    dependency_path: str | None = None
    backbone = detected.get("backbone")
    if backbone in _VLM_BACKBONES:
        embedded: Path | None = None
        for dirname in ("vlm_pretrained", "qwen_pretrained"):
            candidate = checkpoint_dir / dirname
            if candidate.is_dir() and any(candidate.iterdir()):
                embedded = candidate
                break
        if embedded is not None:
            dependency_path = str(embedded.resolve())
            issues.extend(_validate_embedded_vlm(embedded))
        else:
            framework = config.get("framework", {})
            if not isinstance(framework, Mapping):
                framework = {}
            key_by_backbone = {
                "qwen2_5_vl": ("qwenvl", "base_vlm"),
                "qwen3_vl": ("qwenvl", "base_vlm"),
                "paligemma": ("paligemma", "base_vlm"),
                "llama3_2_vision": ("llamavl", "base_vlm"),
            }
            base_path = _nested(framework, *key_by_backbone[str(backbone)])
            resolved, error = _resolve_configured_path(
                base_path,
                repo_root=repo_root,
                environment=environment,
                checkpoint_root=checkpoint_dir,
            )
            if error:
                issues.append(
                    _issue(
                        "error",
                        "pretrained_path_unresolved",
                        "checkpoint.framework",
                        "Checkpoint 配置中的基础模型路径包含未设置的环境变量。",
                        "The base-model path in the checkpoint config contains an unset environment variable.",
                        configured_path=str(base_path),
                    )
                )
            elif resolved is None or not resolved.is_dir():
                issues.append(
                    _issue(
                        "error",
                        "pretrained_model_missing",
                        "checkpoint.framework",
                        (
                            "Checkpoint 未包含可用的 VLM processor/config，"
                            "配置中的本地基础模型目录也不可访问。"
                        ),
                        (
                            "The checkpoint has no usable VLM processor/config and its configured "
                            "local base-model directory is unavailable."
                        ),
                        configured_path=str(base_path or ""),
                        resolved_path=str(resolved or ""),
                    )
                )
            else:
                dependency_path = str(resolved)
                issues.extend(_validate_external_vlm(resolved))
    elif backbone in _WORLD_MODEL_BACKBONES:
        framework = config.get("framework", {})
        world_model = framework.get("world_model", {}) if isinstance(framework, Mapping) else {}
        configured_path = None
        if isinstance(world_model, Mapping):
            configured_path = world_model.get("checkpoint_path") or world_model.get("pretrained_dir")
        resolved, error = _resolve_configured_path(
            configured_path,
            repo_root=repo_root,
            environment=environment,
            checkpoint_root=checkpoint_dir,
        )
        if error:
            issues.append(
                _issue(
                    "error",
                    "world_model_path_unresolved",
                    "checkpoint.framework.world_model",
                    "World Model 路径包含未设置的环境变量。",
                    "The world-model path contains an unset environment variable.",
                    configured_path=str(configured_path),
                )
            )
        elif resolved is None or not resolved.exists():
            issues.append(
                _issue(
                    "error",
                    "world_model_missing",
                    "checkpoint.framework.world_model",
                    "Checkpoint 引用的本地 World Model 权重不可访问。",
                    "The local world-model weights referenced by the checkpoint are unavailable.",
                    configured_path=str(configured_path or ""),
                    resolved_path=str(resolved or ""),
                )
            )
        else:
            dependency_path = str(resolved)
    return issues, dependency_path


def _has_rl_signature(path: Path) -> bool:
    if path.is_file():
        return path.name in _RL_WEIGHT_NAMES
    try:
        names = {child.name for child in path.iterdir() if child.is_file()}
    except OSError:
        return False
    return bool(names & _RL_WEIGHT_NAMES)


def _inspect_cosmos_directory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    weight_path = next((path / name for name in _COSMOS_WEIGHT_NAMES if (path / name).is_file()), None)
    if weight_path is None:
        fallback = sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix == ".pt" and "optimizer" not in child.name.lower()
        )
        weight_path = fallback[0] if fallback else None
    required = {
        "weights": weight_path,
        "t5_embeddings": path / "libero_t5_embeddings.pkl",
        "dataset_statistics": path / "libero_dataset_statistics.json",
    }
    if weight_path is None:
        issues.append(
            _issue(
                "error",
                "cosmos_weights_missing",
                "checkpoint.weights",
                "Cosmos Policy 目录缺少可加载的 DiT .pt 权重。",
                "Cosmos Policy directory is missing a loadable DiT .pt checkpoint.",
            )
        )
    for key, candidate in required.items():
        if key == "weights" or (candidate is not None and candidate.is_file()):
            continue
        issues.append(
            _issue(
                "error",
                f"cosmos_{key}_missing",
                f"checkpoint.{key}",
                f"Cosmos Policy 目录缺少 {candidate.name}。",
                f"Cosmos Policy directory is missing {candidate.name}.",
                expected_path=str(candidate),
            )
        )
    stats = path / "libero_dataset_statistics.json"
    if stats.is_file():
        _data, parse_issue = _read_mapping(stats, kind="json")
        if parse_issue:
            issues.append(parse_issue)
    return {
        "format": "cosmos_policy",
        "root_path": str(path),
        "weights_path": str(weight_path) if weight_path else "",
        "config_path": str(path / "config.json") if (path / "config.json").is_file() else "",
        "statistics_path": str(stats) if stats.is_file() else "",
        "t5_embeddings_path": str(path / "libero_t5_embeddings.pkl")
        if (path / "libero_t5_embeddings.pkl").is_file()
        else "",
        "runtime_dependency_path": None,
    }, issues


def _is_lerobot_pi05_directory(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    config, _issue_value = _read_mapping(config_path, kind="json")
    return bool(config and str(config.get("type", "")).lower() == "pi05")


def _is_openpi_pi05_directory(path: Path) -> bool:
    """Recognize OpenPI Pi0.5 only when layout and metadata both say so.

    ``params/`` plus ``assets/`` is shared by more than one OpenPI policy, so
    the directory name is deliberately not used as evidence for Pi0.5.
    """

    if not (path.is_dir() and (path / "params").is_dir() and (path / "assets").is_dir()):
        return False

    def contains_pi05(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(contains_pi05(key) or contains_pi05(item) for key, item in value.items())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return any(contains_pi05(item) for item in value)
        if isinstance(value, str):
            normalized = value.lower().replace("π", "pi").replace("_", "").replace("-", "").replace(".", "")
            return "pi05" in normalized
        return False

    for name, kind in (
        ("config.json", "json"),
        ("metadata.json", "json"),
        ("checkpoint_metadata.json", "json"),
        ("config.yaml", "yaml"),
    ):
        candidate = path / name
        if not candidate.is_file():
            continue
        metadata, _parse_issue = _read_mapping(candidate, kind=kind)
        if metadata and contains_pi05(metadata):
            return True
    return False


def _inspect_lerobot_pi05_directory(
    path: Path,
    *,
    repo_root: Path,
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    for name in required:
        candidate = path / name
        if candidate.is_file():
            continue
        issues.append(
            _issue(
                "error",
                "lerobot_pi05_file_missing",
                f"checkpoint.{name}",
                f"LeRobot Pi0.5 checkpoint 缺少 {name}。",
                f"LeRobot Pi0.5 checkpoint is missing {name}.",
                expected_path=str(candidate),
            )
        )

    config, parse_issue = _read_mapping(path / "config.json", kind="json")
    if parse_issue:
        issues.append(parse_issue)
    elif str((config or {}).get("type", "")).lower() != "pi05":
        issues.append(
            _issue(
                "error",
                "lerobot_policy_type_mismatch",
                "checkpoint.config.type",
                "LeRobot checkpoint 的 policy type 不是 pi05。",
                "LeRobot checkpoint policy type is not pi05.",
            )
        )

    tokenizer_path: Path | None = None
    preprocessor, preprocessor_issue = _read_mapping(path / "policy_preprocessor.json", kind="json")
    if preprocessor_issue:
        issues.append(preprocessor_issue)
    elif preprocessor:
        for step in preprocessor.get("steps", []):
            if not isinstance(step, Mapping) or step.get("registry_name") != "tokenizer_processor":
                continue
            step_config = step.get("config", {})
            configured = step_config.get("tokenizer_name") if isinstance(step_config, Mapping) else None
            tokenizer_path, error = _resolve_configured_path(
                configured,
                repo_root=repo_root,
                environment=environment,
                checkpoint_root=path,
            )
            if error or tokenizer_path is None or not tokenizer_path.is_dir():
                issues.append(
                    _issue(
                        "error",
                        "lerobot_pi05_tokenizer_missing",
                        "checkpoint.policy_preprocessor.tokenizer_name",
                        "LeRobot Pi0.5 的本地 PaliGemma tokenizer 不可访问。",
                        "The local PaliGemma tokenizer for LeRobot Pi0.5 is unavailable.",
                        configured_path=str(configured or ""),
                        resolved_path=str(tokenizer_path or ""),
                    )
                )
            elif not (tokenizer_path / "tokenizer_config.json").is_file():
                issues.append(
                    _issue(
                        "error",
                        "lerobot_pi05_tokenizer_config_missing",
                        "checkpoint.policy_preprocessor.tokenizer_name",
                        "LeRobot Pi0.5 tokenizer 目录缺少 tokenizer_config.json。",
                        "LeRobot Pi0.5 tokenizer directory is missing tokenizer_config.json.",
                        resolved_path=str(tokenizer_path),
                    )
                )
            break
        else:
            issues.append(
                _issue(
                    "error",
                    "lerobot_pi05_tokenizer_step_missing",
                    "checkpoint.policy_preprocessor",
                    "LeRobot Pi0.5 preprocessor 缺少 tokenizer_processor。",
                    "LeRobot Pi0.5 preprocessor is missing tokenizer_processor.",
                )
            )

    postprocessor, postprocessor_issue = _read_mapping(path / "policy_postprocessor.json", kind="json")
    if postprocessor_issue:
        issues.append(postprocessor_issue)
    elif postprocessor is not None and not isinstance(postprocessor.get("steps"), list):
        issues.append(
            _issue(
                "error",
                "lerobot_pi05_postprocessor_invalid",
                "checkpoint.policy_postprocessor",
                "LeRobot Pi0.5 postprocessor steps 无效。",
                "LeRobot Pi0.5 postprocessor steps are invalid.",
            )
        )

    return {
        "format": "lerobot_pi05",
        "root_path": str(path),
        "weights_path": str(path / "model.safetensors") if (path / "model.safetensors").is_file() else "",
        "config_path": str(path / "config.json") if (path / "config.json").is_file() else "",
        "statistics_path": "",
        "t5_embeddings_path": "",
        "runtime_dependency_path": str(tokenizer_path) if tokenizer_path else None,
        **_checkpoint_format_metadata("lerobot", family="pi05"),
    }, issues


def _inspect_openpi_pi05_directory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return {
        "format": "openpi_pi05",
        "root_path": str(path),
        "weights_path": str(path / "params"),
        "config_path": "",
        "statistics_path": "",
        "t5_embeddings_path": "",
        "runtime_dependency_path": str(path / "assets"),
        **_checkpoint_format_metadata("openpi", family="pi05"),
    }, [
        _issue(
            "error",
            "openpi_pi05_adapter_not_wired",
            "checkpoint",
            "已识别 OpenPI Pi0.5 格式，但当前仅接通 LeRobot Pi0.5 部署适配器。",
            "OpenPI Pi0.5 format was detected, but only the LeRobot Pi0.5 adapter is currently wired.",
        )
    ]


def inspect_checkpoint(
    source: str | Path | Mapping[str, Any],
    *,
    checkpoint_index: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    repo_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    combination_id: str | None = None,
    include_experimental: bool = False,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Statically inspect a checkpoint and detect wired deployment candidates."""

    resolved_source = resolve_checkpoint_source(source, checkpoint_index=checkpoint_index, repo_root=repo_root)
    issues = list(resolved_source["issues"])
    empty = {
        "format": "unknown",
        "root_path": "",
        "weights_path": "",
        "config_path": "",
        "statistics_path": "",
        "t5_embeddings_path": "",
        "runtime_dependency_path": None,
        **_checkpoint_format_metadata(),
    }
    if not resolved_source["valid"]:
        return _plain_copy(
            {
                "valid": False,
                "deployable": False,
                "source": resolved_source,
                "checkpoint": empty,
                "detected": {
                    "framework": None,
                    "backbone": None,
                    "action_head": None,
                    "adapter_id": None,
                    "combination_id": None,
                    "candidate_combination_ids": [],
                },
                "candidates": [],
                "required_parameters": [],
                "issues": issues,
            }
        )

    source_path = Path(resolved_source["path"])
    if _has_rl_signature(source_path):
        issues.append(
            _issue(
                "error",
                "rl_checkpoint_not_deployable",
                "checkpoint",
                (
                    "该目录/文件是 RL Token actor、critic 或 encoder checkpoint，"
                    "当前模型服务没有对应推理适配器。"
                ),
                (
                    "This is an RL Token actor, critic, or encoder checkpoint; "
                    "no matching inference adapter is wired."
                ),
                checkpoint_path=str(source_path),
            )
        )
        return _plain_copy(
            {
                "valid": False,
                "deployable": False,
                "source": resolved_source,
                "checkpoint": empty,
                "detected": {
                    "framework": None,
                    "backbone": None,
                    "action_head": None,
                    "adapter_id": None,
                    "combination_id": None,
                    "candidate_combination_ids": [],
                },
                "candidates": [],
                "required_parameters": [],
                "issues": issues,
            }
        )

    repo = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()
    env = {key: str(value) for key, value in os.environ.items()}
    env.update({key: str(value) for key, value in (environment or {}).items()})

    config: dict[str, Any] | None = None
    checkpoint: dict[str, Any]
    cosmos_signature = False
    if source_path.is_dir():
        cosmos_config_path = source_path / "config.json"
        if cosmos_config_path.is_file():
            cosmos_config, _parse_issue = _read_mapping(cosmos_config_path, kind="json")
            cosmos_signature = bool(
                cosmos_config and str(cosmos_config.get("model_type", "")).lower() == "cosmos-policy"
            )
        cosmos_signature = cosmos_signature or any(
            (source_path / name).is_file()
            for name in ("cosmos_dit.pt", "libero_t5_embeddings.pkl", "libero_dataset_statistics.json")
        )
    if source_path.is_dir() and _is_lerobot_pi05_directory(source_path):
        checkpoint, pi05_issues = _inspect_lerobot_pi05_directory(
            source_path,
            repo_root=repo,
            environment=env,
        )
        issues.extend(pi05_issues)
        config = {"framework": {"name": "LeRobotPI05"}}
    elif source_path.is_dir() and _is_openpi_pi05_directory(source_path):
        checkpoint, openpi_issues = _inspect_openpi_pi05_directory(source_path)
        issues.extend(openpi_issues)
        config = {"framework": {"name": "OpenPIPI05"}}
    elif source_path.is_dir() and cosmos_signature:
        checkpoint, cosmos_issues = _inspect_cosmos_directory(source_path)
        issues.extend(cosmos_issues)
        framework_config = source_path / "framework_config.yaml"
        if framework_config.is_file():
            config, parse_issue = _read_mapping(framework_config, kind="yaml")
            if parse_issue:
                issues.append(parse_issue)
        if config is None:
            config = {"framework": {"name": "CosmosPolicy"}}
    else:
        root = source_path
        if source_path.is_dir() and not any((source_path / name).is_file() for name in _GENERIC_WEIGHT_NAMES):
            nested_vla = source_path / "vla"
            if nested_vla.is_dir() and not _has_rl_signature(source_path):
                root = nested_vla
        if source_path.is_file():
            if source_path.suffix not in {".pt", ".safetensors"}:
                issues.append(
                    _issue(
                        "error",
                        "checkpoint_weight_suffix_unsupported",
                        "checkpoint.path",
                        "旧式 checkpoint 只支持 .pt 或 .safetensors。",
                        "Legacy checkpoints must use .pt or .safetensors.",
                        checkpoint_path=str(source_path),
                    )
                )
            run_dir = source_path.parents[1] if len(source_path.parents) > 1 else source_path.parent
            config_path = run_dir / "config.yaml"
            statistics_path = run_dir / "dataset_statistics.json"
            checkpoint = {
                "format": "legacy_file",
                "root_path": str(run_dir),
                "weights_path": str(source_path),
                "config_path": str(config_path) if config_path.is_file() else "",
                "statistics_path": str(statistics_path) if statistics_path.is_file() else "",
                "t5_embeddings_path": "",
                "runtime_dependency_path": None,
                **_checkpoint_format_metadata(),
            }
        else:
            weight_path = next((root / name for name in _GENERIC_WEIGHT_NAMES if (root / name).is_file()), None)
            config_path = root / "framework_config.yaml"
            statistics_path = root / "dataset_statistics.json"
            checkpoint = {
                "format": "self_contained",
                "root_path": str(root),
                "weights_path": str(weight_path) if weight_path else "",
                "config_path": str(config_path) if config_path.is_file() else "",
                "statistics_path": str(statistics_path) if statistics_path.is_file() else "",
                "t5_embeddings_path": "",
                "runtime_dependency_path": None,
                **_checkpoint_format_metadata(),
            }
            if weight_path is None:
                issues.append(
                    _issue(
                        "error",
                        "checkpoint_weights_missing",
                        "checkpoint.weights",
                        "自包含 checkpoint 目录缺少 model.safetensors 或 pytorch_model.pt。",
                        "Self-contained checkpoint directory is missing model.safetensors or pytorch_model.pt.",
                    )
                )

        config_path_value = checkpoint["config_path"]
        if not config_path_value:
            expected = (
                Path(checkpoint["root_path"]) / "framework_config.yaml"
                if checkpoint["format"] == "self_contained"
                else Path(checkpoint["root_path"]) / "config.yaml"
            )
            issues.append(
                _issue(
                    "error",
                    "checkpoint_config_missing",
                    "checkpoint.config",
                    f"Checkpoint 缺少运行时配置：{expected}",
                    f"Checkpoint is missing its runtime configuration: {expected}",
                    expected_path=str(expected),
                )
            )
        else:
            config, parse_issue = _read_mapping(Path(config_path_value), kind="yaml")
            if parse_issue:
                issues.append(parse_issue)

        statistics_path_value = checkpoint["statistics_path"]
        if not statistics_path_value:
            issues.append(
                _issue(
                    "error",
                    "dataset_statistics_missing",
                    "checkpoint.statistics",
                    "Checkpoint 缺少 dataset_statistics.json，无法还原训练动作归一化。",
                    "Checkpoint is missing dataset_statistics.json required for action denormalization.",
                )
            )
        else:
            _stats, parse_issue = _read_mapping(Path(statistics_path_value), kind="json")
            if parse_issue:
                issues.append(parse_issue)

    detected = _detect_architecture(config or {}, Path(checkpoint["root_path"]))
    if checkpoint["format"] == "lerobot_pi05":
        detected = {
            "framework": "LeRobotPI05",
            "backbone": "paligemma",
            "action_head": "pi05_action_expert",
            "is_world_model": False,
            "world_model_backend": None,
        }
    elif checkpoint["format"] == "openpi_pi05":
        detected = {
            "framework": "OpenPIPI05",
            "backbone": "paligemma",
            "action_head": "pi05_action_expert",
            "is_world_model": False,
            "world_model_backend": None,
        }
    elif checkpoint["format"] == "cosmos_policy":
        detected = {
            "framework": "CosmosPolicy",
            "backbone": "cosmos2",
            "action_head": "cosmos_policy_dit",
            "is_world_model": False,
            "world_model_backend": None,
        }
    if not detected.get("framework"):
        issues.append(
            _issue(
                "error",
                "framework_not_detected",
                "checkpoint.config.framework",
                "无法从 checkpoint 配置中识别 AlphaBrain framework。",
                "Could not identify an AlphaBrain framework from the checkpoint config.",
            )
        )

    candidates = _candidate_combinations(
        detected,
        include_experimental=include_experimental,
        source_catalog=source_catalog,
    )
    selected: dict[str, Any] | None = None
    if combination_id:
        selected = next((item for item in candidates if item.get("id") == combination_id), None)
        if selected is None:
            issues.append(
                _issue(
                    "error",
                    "deployment_combination_mismatch",
                    "combination_id",
                    "选择的部署组合与 checkpoint 配置不兼容或尚未接通。",
                    "The selected deployment combination is incompatible with the checkpoint or is not wired.",
                    requested_combination_id=combination_id,
                    candidate_combination_ids=[item["id"] for item in candidates],
                )
            )
    elif len(candidates) == 1:
        selected = candidates[0]
    elif len(candidates) > 1:
        issues.append(
            _issue(
                "error",
                "deployment_combination_ambiguous",
                "combination_id",
                "Checkpoint 配置对应多个部署组合，请从兼容候选中确认一个。",
                "The checkpoint matches multiple deployment combinations; select one compatible candidate.",
                candidate_combination_ids=[item["id"] for item in candidates],
            )
        )
    elif detected.get("framework"):
        issues.append(
            _issue(
                "error",
                "deployment_adapter_not_wired",
                "checkpoint.config.framework",
                "该 framework/backbone/action head 尚未接通模型部署。",
                "This framework/backbone/action-head combination is not wired for deployment.",
                framework=detected.get("framework"),
                backbone=detected.get("backbone"),
                action_head=detected.get("action_head"),
            )
        )

    if selected is not None:
        # An explicit choice resolves intentionally ambiguous framework configs
        # (for example QwenOFT with a locally renamed base-model directory).
        detected["backbone"] = selected.get("backbone")
        detected["action_head"] = selected.get("action_head")

    if checkpoint["format"] not in {"cosmos_policy", "lerobot_pi05", "openpi_pi05"} and config is not None:
        runtime_issues, dependency = _generic_runtime_issues(
            Path(checkpoint["root_path"]),
            config,
            detected,
            repo_root=repo,
            environment=env,
        )
        issues.extend(runtime_issues)
        checkpoint["runtime_dependency_path"] = dependency

    if detected.get("framework") == "PaliGemmaPi05":
        checkpoint.update(_checkpoint_format_metadata("alphabrain", family="pi05"))

    adapter_ids = {str(item.get("adapter")) for item in candidates}
    adapter_id = str(selected["adapter"]) if selected else (next(iter(adapter_ids)) if len(adapter_ids) == 1 else None)
    required_parameters: list[str] = []
    if adapter_id:
        catalog = get_deployment_catalog(
            include_experimental=include_experimental,
            source_catalog=source_catalog,
        )
        adapter = next((item for item in catalog["deployment_adapters"] if item["id"] == adapter_id), None)
        required_parameters = list((adapter or {}).get("parameter_schema", {}).get("required", []))
    has_errors = any(item["severity"] == "error" for item in issues)
    detected_result = {
        **detected,
        "adapter_id": adapter_id,
        "combination_id": selected.get("id") if selected else None,
        "candidate_combination_ids": [item["id"] for item in candidates],
    }
    return _plain_copy(
        {
            "valid": not has_errors,
            "deployable": selected is not None and not has_errors,
            "source": resolved_source,
            "checkpoint": checkpoint,
            "detected": detected_result,
            "candidates": candidates,
            "required_parameters": required_parameters,
            "issues": issues,
        }
    )


def _adapter_by_id(
    adapter_id: str,
    include_experimental: bool,
    *,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    catalog = get_deployment_catalog(
        include_experimental=include_experimental,
        source_catalog=source_catalog,
    )
    return next((item for item in catalog["deployment_adapters"] if item["id"] == adapter_id), None)


def _parameter_defaults(adapter: Mapping[str, Any]) -> dict[str, Any]:
    properties = adapter.get("parameter_schema", {}).get("properties", {})
    if not isinstance(properties, Mapping):
        return {}
    return {
        str(key): deepcopy(schema["default"])
        for key, schema in properties.items()
        if isinstance(schema, Mapping) and "default" in schema
    }


def build_adapter_command(
    adapter_id: str,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    combination_id: str | None = None,
    backbone_id: str | None = None,
    action_head_id: str | None = None,
    python_executable: str = "python",
    repo_root: str | Path | None = None,
    parameters: Mapping[str, Any] | None = None,
    include_experimental: bool = False,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a model-server command as inert, persistable data.

    Authentication material is intentionally not accepted and can therefore
    never appear in the command preview or persisted resolved parameters.
    GPU visibility is also left to the deployment manager.
    """

    issues: list[dict[str, Any]] = []
    catalog = get_deployment_catalog(
        include_experimental=include_experimental,
        source_catalog=source_catalog,
    )
    adapter = next(
        (item for item in catalog["deployment_adapters"] if item.get("id") == adapter_id),
        None,
    )
    if adapter is None:
        issues.append(
            _issue(
                "error",
                "deployment_adapter_unknown",
                "adapter_id",
                "部署适配器不存在、未接通或未启用。",
                "Deployment adapter is unknown, unsupported, or not enabled.",
                adapter_id=adapter_id,
            )
        )
    selected_combination: Mapping[str, Any] | None = None
    if combination_id:
        selected_combination = next(
            (
                item
                for item in catalog["deployment_combinations"]
                if item.get("id") == combination_id
            ),
            None,
        )
        if selected_combination is None:
            issues.append(
                _issue(
                    "error",
                    "deployment_combination_unknown",
                    "combination_id",
                    "部署组合不存在、已禁用或尚未启用。",
                    "Deployment combination is unknown, disabled, or not enabled.",
                    combination_id=combination_id,
                )
            )
        else:
            mismatches = {
                field: {"expected": selected_combination.get(expected), "received": received}
                for field, expected, received in (
                    ("adapter_id", "adapter", adapter_id),
                    ("backbone_id", "backbone", backbone_id),
                    ("action_head_id", "action_head", action_head_id),
                )
                if received is not None and selected_combination.get(expected) != received
            }
            if mismatches:
                issues.append(
                    _issue(
                        "error",
                        "deployment_combination_metadata_mismatch",
                        "combination_id",
                        "部署组合与适配器或模型元数据不一致。",
                        "Deployment combination does not match the adapter or model metadata.",
                        mismatches=mismatches,
                    )
                )
    checkpoint_path = ""
    if isinstance(checkpoint, Mapping):
        checkpoint_path = str(
            checkpoint.get("path")
            or checkpoint.get("root_path")
            or checkpoint.get("checkpoint_path")
            or checkpoint.get("weights_path")
            or ""
        )
    else:
        checkpoint_path = str(checkpoint)
    if not checkpoint_path:
        issues.append(
            _issue(
                "error",
                "checkpoint_path_missing",
                "checkpoint",
                "缺少 checkpoint 路径。",
                "Checkpoint path is required.",
            )
        )

    resolved_parameters = _parameter_defaults(adapter or {})
    resolved_parameters.update(_plain_copy(dict(parameters or {})))
    # Accept the early boolean spelling as a compatibility alias while
    # persisting the schema-backed precision value used by the UI/manager.
    if "use_bf16" in resolved_parameters and "precision" not in resolved_parameters:
        resolved_parameters["precision"] = "bf16" if bool(resolved_parameters["use_bf16"]) else "fp32"
    resolved_parameters.pop("use_bf16", None)
    allowed_properties = (adapter or {}).get("parameter_schema", {}).get("properties", {})
    allowed_keys = set(allowed_properties) if isinstance(allowed_properties, Mapping) else set()
    sensitive = {
        key
        for key in resolved_parameters
        if any(token in key.lower() for token in ("key", "token", "secret", "password", "hash"))
    }
    if sensitive:
        for key in sorted(sensitive):
            resolved_parameters.pop(key, None)
        issues.append(
            _issue(
                "error",
                "sensitive_adapter_parameter_rejected",
                "parameters",
                "模型服务命令参数不得包含密钥、token、密码或哈希。",
                "Model-server command parameters must not contain keys, tokens, passwords, or hashes.",
                rejected_keys=sorted(sensitive),
            )
        )
    unknown = sorted(set(resolved_parameters) - allowed_keys)
    if unknown:
        issues.append(
            _issue(
                "error",
                "adapter_parameter_unknown",
                "parameters",
                "存在该部署适配器不支持的参数。",
                "One or more parameters are not supported by this deployment adapter.",
                unknown_parameters=unknown,
            )
        )
    required = set((adapter or {}).get("parameter_schema", {}).get("required", []))
    missing = sorted(key for key in required if resolved_parameters.get(key) in (None, ""))
    if missing:
        issues.append(
            _issue(
                "error",
                "adapter_parameter_required",
                "parameters",
                "缺少部署适配器必填参数。",
                "Required deployment-adapter parameters are missing.",
                missing_parameters=missing,
            )
        )

    port = resolved_parameters.get("port", 10093)
    idle_timeout = resolved_parameters.get("idle_timeout_seconds", 1800)
    if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
        issues.append(
            _issue(
                "error",
                "port_invalid",
                "parameters.port",
                "端口必须在 1024-65535 之间。",
                "Port must be between 1024 and 65535.",
            )
        )
    if not isinstance(idle_timeout, int) or isinstance(idle_timeout, bool) or idle_timeout < -1:
        issues.append(
            _issue(
                "error",
                "idle_timeout_invalid",
                "parameters.idle_timeout_seconds",
                "空闲时间必须为 -1 或非负整数秒。",
                "Idle timeout must be -1 or a non-negative integer number of seconds.",
            )
        )
    precision = str(resolved_parameters.get("precision", "bf16")).lower()
    if adapter_id == "base_framework_websocket" and precision not in {"bf16", "fp32"}:
        issues.append(
            _issue(
                "error",
                "precision_invalid",
                "parameters.precision",
                "BaseFramework 模型服务的推理精度只支持 bf16 或 fp32。",
                "BaseFramework model-server precision must be bf16 or fp32.",
            )
        )
    attention_backend = str(resolved_parameters.get("attention_backend", "auto")).lower()
    if adapter_id == "base_framework_websocket" and attention_backend not in {
        "auto",
        "sdpa",
        "flash_attention_2",
        "eager",
    }:
        issues.append(
            _issue(
                "error",
                "attention_backend_invalid",
                "parameters.attention_backend",
                "Attention 后端必须是 auto、sdpa、flash_attention_2 或 eager。",
                "Attention backend must be auto, sdpa, flash_attention_2, or eager.",
            )
        )

    repo = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()
    command: list[str] = []
    if adapter is not None and not any(item["severity"] == "error" for item in issues):
        entrypoint = repo / str(adapter["entrypoint"])
        command = [str(python_executable), str(entrypoint)]
        if adapter_id == "base_framework_websocket":
            base_vlm_value = str(resolved_parameters.get("base_vlm_path", "") or "").strip()
            if base_vlm_value:
                base_vlm = Path(base_vlm_value).expanduser()
                if not base_vlm.is_absolute():
                    base_vlm = repo / base_vlm
                base_vlm = base_vlm.resolve(strict=False)
                resolved_parameters["base_vlm_path"] = str(base_vlm)
                if not base_vlm.is_dir():
                    issues.append(
                        _issue(
                            "error",
                            "base_vlm_path_missing",
                            "parameters.base_vlm_path",
                            "基础 VLM 本地目录不存在或不是目录。",
                            "The local base VLM path does not exist or is not a directory.",
                            resolved_path=str(base_vlm),
                        )
                    )
                    command = []
            command.extend(
                [
                    "--ckpt_path",
                    checkpoint_path,
                    "--port",
                    str(port),
                    "--idle_timeout",
                    str(idle_timeout),
                ]
            )
            if precision == "bf16":
                command.append("--use_bf16")
            if base_vlm_value:
                command.extend(["--base-vlm-path", str(resolved_parameters["base_vlm_path"])])
            if "attention_backend" in resolved_parameters:
                command.extend(["--attention-backend", attention_backend])
        elif adapter_id == "lerobot_pi05_websocket":
            command.extend(
                [
                    "--ckpt_path",
                    checkpoint_path,
                    "--port",
                    str(port),
                    "--idle_timeout",
                    str(idle_timeout),
                ]
            )
        elif adapter_id == "cosmos_policy_websocket":
            pretrained = Path(str(resolved_parameters["pretrained_dir"])).expanduser()
            if not pretrained.is_absolute():
                pretrained = repo / pretrained
            pretrained = pretrained.resolve(strict=False)
            resolved_parameters["pretrained_dir"] = str(pretrained)
            if not pretrained.is_dir():
                issues.append(
                    _issue(
                        "error",
                        "cosmos_pretrained_dir_missing",
                        "parameters.pretrained_dir",
                        "Cosmos Predict2 基础模型目录不存在或不是目录。",
                        "Cosmos Predict2 base-model directory does not exist or is not a directory.",
                        resolved_path=str(pretrained),
                    )
                )
                command = []
            else:
                command.extend(
                    [
                        "--ckpt_dir",
                        checkpoint_path,
                        "--pretrained_dir",
                        str(pretrained),
                        "--port",
                        str(port),
                        "--idle_timeout",
                        str(idle_timeout),
                    ]
                )

    valid = not any(item["severity"] == "error" for item in issues)
    return _plain_copy(
        {
            "valid": valid,
            "command": command if valid else [],
            "environment": {},
            "startup_timeout_seconds": int((adapter or {}).get("startup_timeout_seconds", 900)),
            "checkpoint_path": checkpoint_path,
            "adapter_id": adapter_id,
            "combination_id": combination_id,
            "backbone_id": backbone_id,
            "action_head_id": action_head_id,
            "resolved_parameters": resolved_parameters,
            "issues": issues,
        }
    )


def resolve_deployment(
    source: str | Path | Mapping[str, Any],
    *,
    combination_id: str | None = None,
    checkpoint_index: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    repo_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    python_executable: str = "python",
    parameters: Mapping[str, Any] | None = None,
    include_experimental: bool = False,
    source_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect, select, and resolve a deployment into command data."""

    inspection = inspect_checkpoint(
        source,
        checkpoint_index=checkpoint_index,
        repo_root=repo_root,
        environment=environment,
        combination_id=combination_id,
        include_experimental=include_experimental,
        source_catalog=source_catalog,
    )
    selected_id = inspection["detected"].get("combination_id")
    adapter_id = inspection["detected"].get("adapter_id")
    if not inspection["deployable"] or not adapter_id:
        return _plain_copy(
            {
                "valid": False,
                "command": [],
                "environment": {},
                "startup_timeout_seconds": 900,
                "checkpoint_path": inspection["source"].get("path", ""),
                "adapter_id": adapter_id,
                "combination_id": selected_id,
                "backbone_id": inspection["detected"].get("backbone"),
                "action_head_id": inspection["detected"].get("action_head"),
                "resolved_parameters": _plain_copy(dict(parameters or {})),
                "inspection": inspection,
                "issues": inspection["issues"],
            }
        )

    checkpoint_path = (
        inspection["checkpoint"]["root_path"]
        if inspection["checkpoint"]["format"] != "legacy_file"
        else inspection["checkpoint"]["weights_path"]
    )
    command_parameters = _plain_copy(dict(parameters or {}))
    if (
        adapter_id == "base_framework_websocket"
        and not command_parameters.get("base_vlm_path")
        and inspection["checkpoint"].get("runtime_dependency_path")
        and (
            inspection["checkpoint"].get("format") == "legacy_file"
            or not any(
                (Path(inspection["checkpoint"]["root_path"]) / dirname).is_dir()
                for dirname in ("vlm_pretrained", "qwen_pretrained")
            )
        )
    ):
        command_parameters["base_vlm_path"] = inspection["checkpoint"]["runtime_dependency_path"]
    command = build_adapter_command(
        adapter_id,
        checkpoint_path,
        combination_id=selected_id,
        backbone_id=inspection["detected"].get("backbone"),
        action_head_id=inspection["detected"].get("action_head"),
        python_executable=python_executable,
        repo_root=repo_root,
        parameters=command_parameters,
        include_experimental=include_experimental,
        source_catalog=source_catalog,
    )
    combined_issues = list(inspection["issues"]) + list(command["issues"])
    command["inspection"] = inspection
    command["issues"] = combined_issues
    command["valid"] = inspection["deployable"] and command["valid"]
    if not command["valid"]:
        command["command"] = []
    return _plain_copy(command)


__all__ = [
    "build_adapter_command",
    "get_deployment_catalog",
    "inspect_checkpoint",
    "resolve_checkpoint_source",
    "resolve_deployment",
]
