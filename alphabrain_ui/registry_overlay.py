"""Safe, data-only capability-registry overlays and registry browsing.

An overlay cannot provide launch commands, filesystem paths, environment
variables, credentials, Python symbols, or configuration files. New entries
must derive from a checked-in entry; operational fields are copied from that
trusted base while the overlay may only change display metadata and component
references. Every added entry is forced to ``experimental``.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .registry import load_catalog

OVERLAY_SCHEMA = "alphabrain.registry-overlay"
OVERLAY_SCHEMA_VERSION = 1
MAX_OVERLAY_BYTES = 256 * 1024
COMPONENT_CATEGORIES = ("backbones", "action_heads", "training_methods", "datasets")
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,127}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._*-]{0,159}$")
_LANGUAGES = {"zh-CN", "en-US"}
_CREDENTIAL_PARTS = ("token", "secret", "password", "api_key", "apikey", "credential")


class RegistryOverlayError(ValueError):
    def __init__(self, code: str, path: str = "", detail: Any = None):
        self.code = code
        self.path = path
        self.detail = detail
        super().__init__(code)

    def as_issue(self) -> dict[str, Any]:
        return {"code": self.code, "path": self.path, "detail": self.detail}


def empty_overlay(version: str = "draft") -> dict[str, Any]:
    return {
        "schema": OVERLAY_SCHEMA,
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "overlay_version": version,
        "additions": {
            "components": {category: [] for category in COMPONENT_CATEGORIES},
            "combinations": [],
            "deployment_combinations": [],
        },
        "disables": {
            "components": {category: [] for category in COMPONENT_CATEGORIES},
            "combinations": [],
            "deployment_combinations": [],
        },
    }


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryOverlayError("overlay_mapping_required", path)
    return {str(key): item for key, item in value.items()}


def _only_keys(value: Mapping[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        lowered = " ".join(unknown).lower()
        code = (
            "overlay_credentials_forbidden"
            if any(part in lowered for part in _CREDENTIAL_PARTS)
            else "overlay_field_forbidden"
        )
        raise RegistryOverlayError(code, path, unknown)


def _identifier(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise RegistryOverlayError("overlay_invalid_id", path)
    return value


def _short_text(value: Any, *, path: str, limit: int, multiline: bool = False) -> str:
    if not isinstance(value, str):
        raise RegistryOverlayError("overlay_invalid_text", path)
    value = value.strip()
    if not value or len(value) > limit or "\x00" in value:
        raise RegistryOverlayError("overlay_invalid_text", path)
    if not multiline and any(char in value for char in "\r\n"):
        raise RegistryOverlayError("overlay_invalid_text", path)
    return value


def _localized(value: Any, *, path: str, required: bool) -> dict[str, str] | None:
    if value is None and not required:
        return None
    row = _mapping(value, path=path)
    _only_keys(row, _LANGUAGES, path=path)
    if not row:
        raise RegistryOverlayError("overlay_localized_text_required", path)
    return {
        language: _short_text(text, path=f"{path}.{language}", limit=2000, multiline=True)
        for language, text in row.items()
    }


def _list(value: Any, *, path: str, maximum: int = 256) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise RegistryOverlayError("overlay_invalid_list", path)
    return value


def _base_indexes(catalog: Mapping[str, Any]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    components_node = _mapping(catalog.get("components", {}), path="catalog.components")
    components: dict[str, dict[str, dict[str, Any]]] = {}
    for category in COMPONENT_CATEGORIES:
        rows = components_node.get(category, [])
        if not isinstance(rows, list):
            raise RegistryOverlayError("base_catalog_invalid", f"catalog.components.{category}")
        components[category] = {
            str(row.get("id")): row for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
    combinations = {
        str(row.get("id")): row
        for row in catalog.get("combinations", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    return components, combinations


def validate_registry_overlay(
    document: Mapping[str, Any],
    *,
    base_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize an untrusted overlay without writing anything."""

    base = deepcopy(dict(base_catalog or load_catalog()))
    base_components, base_combinations = _base_indexes(base)
    base_adapters = {
        str(row.get("id")): row
        for row in base.get("deployment_adapters", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    base_deployments = {
        str(row.get("id")): row
        for row in base.get("deployment_combinations", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    root = _mapping(document, path="")
    _only_keys(root, {"schema", "schema_version", "overlay_version", "additions", "disables"}, path="")
    if root.get("schema") != OVERLAY_SCHEMA:
        raise RegistryOverlayError("overlay_schema_mismatch", "schema")
    if root.get("schema_version") != OVERLAY_SCHEMA_VERSION:
        raise RegistryOverlayError("overlay_schema_version_unsupported", "schema_version")
    version = root.get("overlay_version")
    if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
        raise RegistryOverlayError("overlay_invalid_version", "overlay_version")

    additions = _mapping(root.get("additions", {}), path="additions")
    _only_keys(
        additions,
        {"components", "combinations", "deployment_combinations"},
        path="additions",
    )
    component_additions = _mapping(additions.get("components", {}), path="additions.components")
    _only_keys(component_additions, set(COMPONENT_CATEGORIES), path="additions.components")
    normalized_components: dict[str, list[dict[str, Any]]] = {category: [] for category in COMPONENT_CATEGORIES}
    added_ids: dict[str, set[str]] = {category: set() for category in COMPONENT_CATEGORIES}
    for category in COMPONENT_CATEGORIES:
        rows = _list(component_additions.get(category, []), path=f"additions.components.{category}")
        for index, raw in enumerate(rows):
            path = f"additions.components.{category}[{index}]"
            row = _mapping(raw, path=path)
            _only_keys(row, {"id", "based_on", "label", "description"}, path=path)
            identifier = _identifier(row.get("id"), path=f"{path}.id")
            based_on = _identifier(row.get("based_on"), path=f"{path}.based_on")
            if identifier in base_components[category] or identifier in added_ids[category]:
                raise RegistryOverlayError("overlay_id_conflict", f"{path}.id", identifier)
            if based_on not in base_components[category]:
                raise RegistryOverlayError("overlay_untrusted_base", f"{path}.based_on", based_on)
            added_ids[category].add(identifier)
            normalized = {
                "id": identifier,
                "based_on": based_on,
                "label": _localized(row.get("label"), path=f"{path}.label", required=True),
            }
            description = _localized(row.get("description"), path=f"{path}.description", required=False)
            if description:
                normalized["description"] = description
            normalized_components[category].append(normalized)

    combination_rows = _list(additions.get("combinations", []), path="additions.combinations")
    normalized_combinations: list[dict[str, Any]] = []
    combination_ids: set[str] = set()
    allowed_combination_fields = {
        "id", "based_on", "label", "description", "backbone", "action_head", "method",
        "datasets", "dataset_mixes", "min_gpus",
    }
    known_axes = {
        "backbone": set(base_components["backbones"]) | added_ids["backbones"],
        "action_head": set(base_components["action_heads"]) | added_ids["action_heads"],
        "method": set(base_components["training_methods"]) | added_ids["training_methods"],
    }
    known_datasets = set(base_components["datasets"]) | added_ids["datasets"]
    for index, raw in enumerate(combination_rows):
        path = f"additions.combinations[{index}]"
        row = _mapping(raw, path=path)
        _only_keys(row, allowed_combination_fields, path=path)
        identifier = _identifier(row.get("id"), path=f"{path}.id")
        based_on = _identifier(row.get("based_on"), path=f"{path}.based_on")
        if identifier in base_combinations or identifier in combination_ids:
            raise RegistryOverlayError("overlay_id_conflict", f"{path}.id", identifier)
        if based_on not in base_combinations:
            raise RegistryOverlayError("overlay_untrusted_base", f"{path}.based_on", based_on)
        combination_ids.add(identifier)
        normalized: dict[str, Any] = {"id": identifier, "based_on": based_on}
        for field in ("label", "description"):
            localized = _localized(row.get(field), path=f"{path}.{field}", required=field == "label")
            if localized:
                normalized[field] = localized
        for field in ("backbone", "action_head", "method"):
            value = row.get(field, base_combinations[based_on].get(field))
            value = _identifier(value, path=f"{path}.{field}")
            if value not in known_axes[field]:
                raise RegistryOverlayError("overlay_unknown_component_reference", f"{path}.{field}", value)
            normalized[field] = value
        datasets = _list(
            row.get("datasets", base_combinations[based_on].get("datasets", [])),
            path=f"{path}.datasets",
            maximum=64,
        )
        normalized_datasets = [_identifier(value, path=f"{path}.datasets") for value in datasets]
        if not normalized_datasets or len(set(normalized_datasets)) != len(normalized_datasets):
            raise RegistryOverlayError("overlay_invalid_dataset_references", f"{path}.datasets")
        unknown_datasets = sorted(set(normalized_datasets) - known_datasets)
        if unknown_datasets:
            raise RegistryOverlayError("overlay_unknown_component_reference", f"{path}.datasets", unknown_datasets)
        normalized["datasets"] = normalized_datasets
        if "dataset_mixes" in row:
            mixes = _list(row["dataset_mixes"], path=f"{path}.dataset_mixes", maximum=128)
            if any(not isinstance(value, str) or not _MIX_RE.fullmatch(value) for value in mixes):
                raise RegistryOverlayError("overlay_invalid_dataset_mix", f"{path}.dataset_mixes")
            normalized["dataset_mixes"] = list(dict.fromkeys(mixes))
        min_gpus = row.get("min_gpus", base_combinations[based_on].get("min_gpus", 1))
        if isinstance(min_gpus, bool) or not isinstance(min_gpus, int) or not 1 <= min_gpus <= 64:
            raise RegistryOverlayError("overlay_invalid_gpu_count", f"{path}.min_gpus")
        normalized["min_gpus"] = min_gpus
        normalized_combinations.append(normalized)

    deployment_rows = _list(
        additions.get("deployment_combinations", []),
        path="additions.deployment_combinations",
    )
    normalized_deployments: list[dict[str, Any]] = []
    deployment_ids: set[str] = set()
    allowed_deployment_fields = {
        "id", "based_on", "label", "description", "backbone", "action_head", "adapter",
        "source_combinations", "benchmarks", "recommended_gpu_count",
    }
    for index, raw in enumerate(deployment_rows):
        path = f"additions.deployment_combinations[{index}]"
        row = _mapping(raw, path=path)
        _only_keys(row, allowed_deployment_fields, path=path)
        identifier = _identifier(row.get("id"), path=f"{path}.id")
        based_on = _identifier(row.get("based_on"), path=f"{path}.based_on")
        if identifier in base_deployments or identifier in deployment_ids:
            raise RegistryOverlayError("overlay_id_conflict", f"{path}.id", identifier)
        if based_on not in base_deployments:
            raise RegistryOverlayError("overlay_untrusted_base", f"{path}.based_on", based_on)
        deployment_ids.add(identifier)
        normalized = {
            "id": identifier,
            "based_on": based_on,
            "label": _localized(row.get("label"), path=f"{path}.label", required=True),
        }
        description = _localized(
            row.get("description"), path=f"{path}.description", required=False
        )
        if description:
            normalized["description"] = description
        trusted = base_deployments[based_on]
        for field, known_values in (
            ("backbone", known_axes["backbone"]),
            ("action_head", known_axes["action_head"]),
            ("adapter", set(base_adapters)),
        ):
            value = _identifier(row.get(field, trusted.get(field)), path=f"{path}.{field}")
            if value not in known_values:
                raise RegistryOverlayError(
                    "overlay_unknown_operational_reference", f"{path}.{field}", value
                )
            normalized[field] = value
        sources = _list(
            row.get("source_combinations", trusted.get("source_combinations", [])),
            path=f"{path}.source_combinations",
            maximum=128,
        )
        source_ids = [
            _identifier(value, path=f"{path}.source_combinations") for value in sources
        ]
        unknown_sources = sorted(
            set(source_ids) - (set(base_combinations) | combination_ids)
        )
        if not source_ids or unknown_sources:
            raise RegistryOverlayError(
                "overlay_unknown_operational_reference",
                f"{path}.source_combinations",
                unknown_sources,
            )
        normalized["source_combinations"] = list(dict.fromkeys(source_ids))
        benchmarks = _list(
            row.get("benchmarks", trusted.get("benchmarks", [])),
            path=f"{path}.benchmarks",
            maximum=64,
        )
        normalized["benchmarks"] = list(
            dict.fromkeys(
                _identifier(value, path=f"{path}.benchmarks") for value in benchmarks
            )
        )
        recommended = row.get(
            "recommended_gpu_count", trusted.get("recommended_gpu_count", 1)
        )
        if isinstance(recommended, bool) or not isinstance(recommended, int) or not 1 <= recommended <= 64:
            raise RegistryOverlayError(
                "overlay_invalid_gpu_count", f"{path}.recommended_gpu_count"
            )
        normalized["recommended_gpu_count"] = recommended
        normalized_deployments.append(normalized)

    disables = _mapping(root.get("disables", {}), path="disables")
    _only_keys(
        disables,
        {"components", "combinations", "deployment_combinations"},
        path="disables",
    )
    disabled_components = _mapping(disables.get("components", {}), path="disables.components")
    _only_keys(disabled_components, set(COMPONENT_CATEGORIES), path="disables.components")
    normalized_disabled: dict[str, list[str]] = {}
    for category in COMPONENT_CATEGORIES:
        values = [
            _identifier(value, path=f"disables.components.{category}")
            for value in _list(
                disabled_components.get(category, []),
                path=f"disables.components.{category}",
            )
        ]
        unknown = sorted(set(values) - (set(base_components[category]) | added_ids[category]))
        if unknown:
            raise RegistryOverlayError("overlay_unknown_disable_target", f"disables.components.{category}", unknown)
        normalized_disabled[category] = list(dict.fromkeys(values))
    disabled_combinations = [
        _identifier(value, path="disables.combinations")
        for value in _list(disables.get("combinations", []), path="disables.combinations")
    ]
    unknown_combinations = sorted(set(disabled_combinations) - (set(base_combinations) | combination_ids))
    if unknown_combinations:
        raise RegistryOverlayError("overlay_unknown_disable_target", "disables.combinations", unknown_combinations)
    disabled_deployments = [
        _identifier(value, path="disables.deployment_combinations")
        for value in _list(
            disables.get("deployment_combinations", []),
            path="disables.deployment_combinations",
        )
    ]
    unknown_deployments = sorted(
        set(disabled_deployments) - (set(base_deployments) | deployment_ids)
    )
    if unknown_deployments:
        raise RegistryOverlayError(
            "overlay_unknown_disable_target",
            "disables.deployment_combinations",
            unknown_deployments,
        )

    return {
        "schema": OVERLAY_SCHEMA,
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "overlay_version": version,
        "additions": {
            "components": normalized_components,
            "combinations": normalized_combinations,
            "deployment_combinations": normalized_deployments,
        },
        "disables": {
            "components": normalized_disabled,
            "combinations": list(dict.fromkeys(disabled_combinations)),
            "deployment_combinations": list(dict.fromkeys(disabled_deployments)),
        },
    }


def apply_registry_overlay(base_catalog: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_registry_overlay(overlay, base_catalog=base_catalog)
    result = deepcopy(dict(base_catalog))
    base_components, base_combinations = _base_indexes(base_catalog)
    components = result.setdefault("components", {})
    for category, additions in normalized["additions"]["components"].items():
        target = components.setdefault(category, [])
        for addition in additions:
            clone = deepcopy(base_components[category][addition["based_on"]])
            clone["id"] = addition["id"]
            clone["status"] = "experimental"
            clone["label"] = deepcopy(addition["label"])
            if "description" in addition:
                clone["description"] = deepcopy(addition["description"])
            clone["risk"] = {
                "zh-CN": "此条目由管理员 overlay 添加，复用已接通实现，仍需研究人员验证。",
                "en-US": (
                    "This administrator overlay reuses a wired implementation and still requires "
                    "researcher validation."
                ),
            }
            target.append(clone)

    for addition in normalized["additions"]["combinations"]:
        clone = deepcopy(base_combinations[addition["based_on"]])
        clone.update(
            {
                key: deepcopy(value)
                for key, value in addition.items()
                if key not in {"based_on", "label", "description"}
            }
        )
        clone["id"] = addition["id"]
        clone["status"] = "experimental"
        if "label" in addition:
            clone["label"] = deepcopy(addition["label"])
        if "description" in addition:
            clone["description"] = deepcopy(addition["description"])
        clone["risk"] = {
            "zh-CN": (
                "此组合来自管理员 overlay，启动器和配置继承自内置组合，"
                "请在使用前验证。"
            ),
            "en-US": (
                "This combination comes from an administrator overlay; its launcher and configs are "
                "inherited from a built-in combination and require validation."
            ),
        }
        result.setdefault("combinations", []).append(clone)

    base_deployments = {
        str(row.get("id")): row
        for row in base_catalog.get("deployment_combinations", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    for addition in normalized["additions"]["deployment_combinations"]:
        clone = deepcopy(base_deployments[addition["based_on"]])
        clone.update(
            {
                key: deepcopy(value)
                for key, value in addition.items()
                if key not in {"based_on", "label", "description"}
            }
        )
        clone["id"] = addition["id"]
        clone["status"] = "experimental"
        clone["label"] = deepcopy(addition["label"])
        if "description" in addition:
            clone["description"] = deepcopy(addition["description"])
        clone["risk"] = {
            "zh-CN": "此部署组合来自管理员 overlay，仅引用内置 adapter。",
            "en-US": (
                "This deployment combination comes from an administrator overlay and only "
                "references a built-in adapter."
            ),
        }
        result.setdefault("deployment_combinations", []).append(clone)

    disabled = normalized["disables"]
    disabled_sets = {category: set(values) for category, values in disabled["components"].items()}
    for category, identifiers in disabled_sets.items():
        components[category] = [row for row in components.get(category, []) if row.get("id") not in identifiers]
    disabled_combinations = set(disabled["combinations"])
    result["combinations"] = [
        row
        for row in result.get("combinations", [])
        if row.get("id") not in disabled_combinations
        and row.get("backbone") not in disabled_sets["backbones"]
        and row.get("action_head") not in disabled_sets["action_heads"]
        and row.get("method") not in disabled_sets["training_methods"]
        and not set(row.get("datasets", [])) & disabled_sets["datasets"]
    ]
    disabled_deployments = set(disabled["deployment_combinations"])
    surviving_combinations = {str(row.get("id")) for row in result["combinations"]}
    result["deployment_combinations"] = [
        row
        for row in result.get("deployment_combinations", [])
        if row.get("id") not in disabled_deployments
        and row.get("backbone") not in disabled_sets["backbones"]
        and row.get("action_head") not in disabled_sets["action_heads"]
        and set(row.get("source_combinations", [])) <= surviving_combinations
    ]
    result["registry_version"] = f"{result.get('registry_version', 'unknown')}+overlay.{normalized['overlay_version']}"
    return result


def build_registry_view(catalog: Mapping[str, Any], overlay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build an admin-facing, credential-free inventory from an effective catalog."""

    overlay_ids = {category: set() for category in COMPONENT_CATEGORIES}
    overlay_combinations: set[str] = set()
    overlay_deployments: set[str] = set()
    if overlay:
        normalized = validate_registry_overlay(overlay)
        for category, rows in normalized["additions"]["components"].items():
            overlay_ids[category] = {row["id"] for row in rows}
        overlay_combinations = {row["id"] for row in normalized["additions"]["combinations"]}
        overlay_deployments = {
            row["id"] for row in normalized["additions"]["deployment_combinations"]
        }
    components: list[dict[str, Any]] = []
    known: dict[str, set[str]] = {}
    for category in COMPONENT_CATEGORIES:
        rows = catalog.get("components", {}).get(category, [])
        known[category] = {str(row.get("id")) for row in rows if isinstance(row, dict)}
        for row in rows:
            components.append(
                {
                    "id": row.get("id"),
                    "category": category,
                    "kind": row.get("kind"),
                    "status": row.get("status"),
                    "label": deepcopy(row.get("label", {})),
                    "description": deepcopy(row.get("description", {})),
                    "origin": "overlay" if row.get("id") in overlay_ids[category] else "built_in",
                    "config_reference": row.get("config") if category == "datasets" else None,
                }
            )
    warnings: list[dict[str, Any]] = []
    combinations: list[dict[str, Any]] = []
    for row in catalog.get("combinations", []):
        missing: list[str] = []
        for field, category in (
            ("backbone", "backbones"),
            ("action_head", "action_heads"),
            ("method", "training_methods"),
        ):
            if row.get(field) not in known[category]:
                missing.append(f"{field}:{row.get(field)}")
        for dataset in row.get("datasets", []):
            if dataset not in known["datasets"]:
                missing.append(f"dataset:{dataset}")
        if missing:
            warnings.append(
                {
                    "code": "registry_combination_missing_reference",
                    "id": row.get("id"),
                    "references": missing,
                }
            )
        combinations.append(
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "origin": "overlay" if row.get("id") in overlay_combinations else "built_in",
                "backbone": row.get("backbone"),
                "action_head": row.get("action_head"),
                "method": row.get("method"),
                "datasets": list(row.get("datasets", [])),
                "launcher": row.get("launcher"),
                "workflow": row.get("workflow", "standard"),
                "configuration_references": [
                    value
                    for value in (
                        row.get("model_config"),
                        row.get("dataset_config"),
                        *(row.get("config_paths", []) if isinstance(row.get("config_paths"), list) else []),
                    )
                    if isinstance(value, str)
                ],
                "min_gpus": row.get("min_gpus", 1),
                "valid_references": not missing,
            }
        )
    deployment_adapters = [
        {
            "id": row.get("id"),
            "status": row.get("status"),
            "label": deepcopy(row.get("label", {})),
            "entrypoint": row.get("entrypoint"),
            "transport": row.get("protocol", {}).get("transport"),
            "request_format": row.get("protocol", {}).get("request_format"),
        }
        for row in catalog.get("deployment_adapters", [])
        if isinstance(row, dict)
    ]
    adapter_ids = {str(row["id"]) for row in deployment_adapters}
    deployment_combinations: list[dict[str, Any]] = []
    for row in catalog.get("deployment_combinations", []):
        sources = list(row.get("source_combinations", []))
        valid = row.get("adapter") in adapter_ids and set(sources) <= {
            str(item.get("id")) for item in catalog.get("combinations", [])
        }
        if not valid:
            warnings.append(
                {
                    "code": "registry_deployment_missing_reference",
                    "id": row.get("id"),
                }
            )
        deployment_combinations.append(
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "origin": "overlay" if row.get("id") in overlay_deployments else "built_in",
                "adapter": row.get("adapter"),
                "backbone": row.get("backbone"),
                "action_head": row.get("action_head"),
                "source_combinations": sources,
                "benchmarks": list(row.get("benchmarks", [])),
                "recommended_gpu_count": row.get("recommended_gpu_count", 1),
                "valid_references": valid,
            }
        )
    return {
        "schema": "alphabrain.registry-view",
        "schema_version": 1,
        "registry_version": catalog.get("registry_version"),
        "components": components,
        "combinations": combinations,
        "deployment_adapters": deployment_adapters,
        "deployment_combinations": deployment_combinations,
        "workflow_ids": sorted(catalog.get("workflow_schemas", {})),
        "warnings": warnings,
        "summary": {
            "component_count": len(components),
            "combination_count": len(combinations),
            "overlay_component_count": sum(len(values) for values in overlay_ids.values()),
            "overlay_combination_count": len(overlay_combinations),
            "deployment_adapter_count": len(deployment_adapters),
            "deployment_combination_count": len(deployment_combinations),
            "overlay_deployment_combination_count": len(overlay_deployments),
        },
    }


class RegistryOverlayStore:
    filename = "overlay.yaml"

    def __init__(self, state_dir: str | Path, *, base_catalog: Mapping[str, Any] | None = None):
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.directory = self.state_dir / "registry"
        self.path = self.directory / self.filename
        self.base_catalog = deepcopy(dict(base_catalog)) if base_catalog is not None else None

    def _base(self) -> dict[str, Any]:
        return deepcopy(self.base_catalog) if self.base_catalog is not None else load_catalog()

    def _ensure_directory(self) -> None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = os.lstat(self.directory)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("registry_overlay_directory_invalid")
        os.chmod(self.directory, 0o700)

    def _read_bytes(self) -> bytes | None:
        try:
            descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RuntimeError("registry_overlay_open_failed") from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
                raise RuntimeError("registry_overlay_permissions_invalid")
            data = os.read(descriptor, MAX_OVERLAY_BYTES + 1)
            if len(data) > MAX_OVERLAY_BYTES:
                raise RegistryOverlayError("overlay_too_large")
            return data
        finally:
            os.close(descriptor)

    def load(self) -> dict[str, Any] | None:
        data = self._read_bytes()
        if data is None:
            return None
        try:
            document = yaml.safe_load(data.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise RegistryOverlayError("overlay_invalid_yaml") from exc
        return validate_registry_overlay(document, base_catalog=self._base())

    def preview(self, document: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_registry_overlay(document, base_catalog=self._base())
        effective = apply_registry_overlay(self._base(), normalized)
        return {
            "overlay": normalized,
            "effective_catalog": effective,
            "view": build_registry_view(effective, normalized),
        }

    def save(self, document: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_registry_overlay(document, base_catalog=self._base())
        encoded = yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False).encode("utf-8")
        if len(encoded) > MAX_OVERLAY_BYTES:
            raise RegistryOverlayError("overlay_too_large")
        self._ensure_directory()
        descriptor, temporary_name = tempfile.mkstemp(prefix=".overlay-", suffix=".yaml", dir=self.directory)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
            directory_fd = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        return self.status()

    def delete(self) -> dict[str, Any]:
        try:
            info = os.lstat(self.path)
        except FileNotFoundError:
            return self.status()
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("registry_overlay_path_invalid")
        self.path.unlink()
        return self.status()

    def status(self) -> dict[str, Any]:
        data = self._read_bytes()
        if data is None:
            return {"configured": False, "path": str(self.path), "sha256": None, "overlay_version": None}
        document = self.load()
        assert document is not None
        return {
            "configured": True,
            "path": str(self.path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "overlay_version": document["overlay_version"],
            "component_additions": sum(len(rows) for rows in document["additions"]["components"].values()),
            "combination_additions": len(document["additions"]["combinations"]),
            "deployment_combination_additions": len(
                document["additions"]["deployment_combinations"]
            ),
            "disabled_entries": sum(
                len(rows) for rows in document["disables"]["components"].values()
            )
            + len(document["disables"]["combinations"])
            + len(document["disables"]["deployment_combinations"]),
        }

    def effective_catalog(self) -> dict[str, Any]:
        document = self.load()
        return self._base() if document is None else apply_registry_overlay(self._base(), document)

    def browser_view(self) -> dict[str, Any]:
        document = self.load()
        return build_registry_view(self.effective_catalog(), document)
