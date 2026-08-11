from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from alphabrain_ui.registry import load_catalog
from alphabrain_ui.registry_overlay import (
    RegistryOverlayError,
    RegistryOverlayStore,
    apply_registry_overlay,
    build_registry_view,
    empty_overlay,
    validate_registry_overlay,
)


def valid_overlay() -> dict:
    document = empty_overlay("lab-v1")
    document["additions"]["components"]["backbones"] = [
        {
            "id": "lab_qwen_variant",
            "based_on": "qwen2_5_vl",
            "label": {"zh-CN": "实验室 Qwen 变体", "en-US": "Lab Qwen variant"},
            "description": {"zh-CN": "复用内置 Qwen 实现。", "en-US": "Reuses the built-in Qwen implementation."},
        }
    ]
    document["additions"]["combinations"] = [
        {
            "id": "lab_qwen_oft_libero",
            "based_on": "qwen_oft_libero",
            "label": {"zh-CN": "实验室 Qwen OFT", "en-US": "Lab Qwen OFT"},
            "backbone": "lab_qwen_variant",
            "action_head": "mlp_regression",
            "method": "imitation_learning",
            "datasets": ["libero"],
            "dataset_mixes": ["libero_goal"],
            "min_gpus": 2,
        }
    ]
    document["additions"]["deployment_combinations"] = [
        {
            "id": "lab_deploy_qwen_oft",
            "based_on": "deploy_qwen2_5_oft",
            "label": {"zh-CN": "实验室部署", "en-US": "Lab deployment"},
            "backbone": "lab_qwen_variant",
            "action_head": "mlp_regression",
            "adapter": "base_framework_websocket",
            "source_combinations": ["lab_qwen_oft_libero"],
            "benchmarks": ["libero"],
            "recommended_gpu_count": 2,
        }
    ]
    document["disables"]["combinations"] = ["toy_libero_debug"]
    return document


def test_overlay_additions_are_experimental_and_inherit_only_trusted_operations() -> None:
    base = load_catalog()
    normalized = validate_registry_overlay(valid_overlay(), base_catalog=base)
    effective = apply_registry_overlay(base, normalized)

    backbone = next(row for row in effective["components"]["backbones"] if row["id"] == "lab_qwen_variant")
    trusted_backbone = next(row for row in base["components"]["backbones"] if row["id"] == "qwen2_5_vl")
    assert backbone["status"] == "experimental"
    assert backbone["default_pretrained"] == trusted_backbone["default_pretrained"]
    combination = next(row for row in effective["combinations"] if row["id"] == "lab_qwen_oft_libero")
    trusted_combination = next(row for row in base["combinations"] if row["id"] == "qwen_oft_libero")
    assert combination["status"] == "experimental"
    assert combination["launcher"] == trusted_combination["launcher"]
    assert combination["model_config"] == trusted_combination["model_config"]
    assert combination["backbone"] == "lab_qwen_variant"
    assert combination["min_gpus"] == 2
    deployment = next(
        row
        for row in effective["deployment_combinations"]
        if row["id"] == "lab_deploy_qwen_oft"
    )
    trusted_deployment = next(
        row
        for row in base["deployment_combinations"]
        if row["id"] == "deploy_qwen2_5_oft"
    )
    assert deployment["status"] == "experimental"
    assert deployment["adapter"] == "base_framework_websocket"
    assert deployment["framework_names"] == trusted_deployment["framework_names"]
    assert deployment["source_combinations"] == ["lab_qwen_oft_libero"]
    assert not any(row["id"] == "toy_libero_debug" for row in effective["combinations"])
    # A verified built-in definition is never edited in place by an overlay.
    built_in = next(row for row in effective["combinations"] if row["id"] == "qwen_oft_libero")
    assert built_in == trusted_combination


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("command", ["bash", "-c", "curl example"], "overlay_field_forbidden"),
        ("launcher", "arbitrary_shell", "overlay_field_forbidden"),
        ("model_config", "/tmp/evil.yaml", "overlay_field_forbidden"),
        ("config_paths", ["../../evil.yaml"], "overlay_field_forbidden"),
        ("environment", {"PYTHONPATH": "/tmp"}, "overlay_field_forbidden"),
        ("api_key", "secret-value", "overlay_credentials_forbidden"),
        ("token", "secret-value", "overlay_credentials_forbidden"),
        ("status", "verified", "overlay_field_forbidden"),
    ],
)
def test_combination_overlay_rejects_operational_path_command_and_credentials(
    field: str, value, expected: str
) -> None:
    document = valid_overlay()
    document["additions"]["combinations"][0][field] = value
    with pytest.raises(RegistryOverlayError) as error:
        validate_registry_overlay(document)
    assert error.value.code == expected


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("default_pretrained", "/tmp/model", "overlay_field_forbidden"),
        ("config", "../../dataset.yaml", "overlay_field_forbidden"),
        ("password", "secret", "overlay_credentials_forbidden"),
        ("id", "../../escape", "overlay_invalid_id"),
        ("based_on", "missing_backbone", "overlay_untrusted_base"),
    ],
)
def test_component_overlay_rejects_unsafe_or_untrusted_fields(field: str, value, expected: str) -> None:
    document = valid_overlay()
    document["additions"]["components"]["backbones"][0][field] = value
    with pytest.raises(RegistryOverlayError) as error:
        validate_registry_overlay(document)
    assert error.value.code == expected


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("adapter", "arbitrary_adapter", "overlay_unknown_operational_reference"),
        ("entrypoint", "/tmp/server.py", "overlay_field_forbidden"),
        ("command", ["python", "/tmp/server.py"], "overlay_field_forbidden"),
        ("config_path", "../../evil.yaml", "overlay_field_forbidden"),
        ("hf_token", "secret", "overlay_credentials_forbidden"),
    ],
)
def test_deployment_overlay_can_only_reference_builtin_adapters(
    field: str, value, expected: str
) -> None:
    document = valid_overlay()
    document["additions"]["deployment_combinations"][0][field] = value
    with pytest.raises(RegistryOverlayError) as error:
        validate_registry_overlay(document)
    assert error.value.code == expected


def test_overlay_cannot_replace_builtin_ids_and_disable_cascades_combinations() -> None:
    conflict = valid_overlay()
    conflict["additions"]["components"]["backbones"][0]["id"] = "qwen2_5_vl"
    with pytest.raises(RegistryOverlayError) as error:
        validate_registry_overlay(conflict)
    assert error.value.code == "overlay_id_conflict"

    disable = empty_overlay("disable-qwen3")
    disable["disables"]["components"]["backbones"] = ["qwen3_vl"]
    effective = apply_registry_overlay(load_catalog(), disable)
    assert not any(row["id"] == "qwen3_vl" for row in effective["components"]["backbones"])
    assert not any(row.get("backbone") == "qwen3_vl" for row in effective["combinations"])


def test_overlay_store_is_atomic_owner_only_and_returns_isolated_values(tmp_path: Path) -> None:
    store = RegistryOverlayStore(tmp_path)
    status = store.save(valid_overlay())
    assert status["configured"] is True
    assert status["overlay_version"] == "lab-v1"
    assert len(status["sha256"]) == 64
    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert "curl" not in store.path.read_text(encoding="utf-8")

    loaded = store.load()
    assert loaded is not None
    loaded["overlay_version"] = "mutated"
    assert store.load()["overlay_version"] == "lab-v1"
    effective = store.effective_catalog()
    assert effective["registry_version"].endswith("+overlay.lab-v1")
    assert store.delete()["configured"] is False

    # Never follow or unlink an attacker-controlled overlay symlink.
    victim = tmp_path / "victim.yaml"
    victim.write_text("safe", encoding="utf-8")
    store.directory.mkdir(parents=True, exist_ok=True)
    store.path.symlink_to(victim)
    with pytest.raises(RuntimeError):
        store.load()
    with pytest.raises(RuntimeError):
        store.delete()
    assert victim.read_text(encoding="utf-8") == "safe"


def test_registry_browser_reports_origin_references_and_never_emits_credentials() -> None:
    overlay = validate_registry_overlay(valid_overlay())
    view = build_registry_view(apply_registry_overlay(load_catalog(), overlay), overlay)
    added = next(row for row in view["combinations"] if row["id"] == "lab_qwen_oft_libero")
    assert added["origin"] == "overlay"
    assert added["launcher"] == "unified_finetune"
    assert added["valid_references"] is True
    assert added["configuration_references"]
    assert view["summary"]["overlay_component_count"] == 1
    assert view["summary"]["overlay_combination_count"] == 1
    assert view["summary"]["overlay_deployment_combination_count"] == 1
    deployment = next(
        row
        for row in view["deployment_combinations"]
        if row["id"] == "lab_deploy_qwen_oft"
    )
    assert deployment["origin"] == "overlay"
    assert deployment["adapter"] == "base_framework_websocket"
    assert deployment["valid_references"] is True
    serialized = json.dumps(view).lower()
    assert "api_key" not in serialized
    assert "password" not in serialized
    assert "credential" not in serialized
