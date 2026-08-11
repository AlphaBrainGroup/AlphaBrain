from __future__ import annotations

import json
from pathlib import Path

import yaml

from alphabrain_ui.capabilities import (
    get_catalog,
    get_compatible_options,
    validate_compatibility,
)
from alphabrain_ui.configuration import preflight_static, resolve_experiment
from alphabrain_ui.deployment_registry import (
    build_adapter_command,
    get_deployment_catalog,
    inspect_checkpoint,
    resolve_deployment,
)
from alphabrain_ui.evaluation_registry import (
    get_evaluation_catalog,
    inspect_evaluation_checkpoint,
)
from alphabrain_ui.gpu import GPUInfo
from alphabrain_ui.preflight import run_preflight
from alphabrain_ui.registry import load_catalog
from alphabrain_ui.registry_overlay import (
    apply_registry_overlay,
    empty_overlay,
    validate_registry_overlay,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _effective_catalog() -> dict:
    overlay = empty_overlay("effective-test")
    overlay["additions"]["components"]["backbones"] = [
        {
            "id": "lab_qwen_variant",
            "based_on": "qwen2_5_vl",
            "label": {"zh-CN": "实验室 Qwen", "en-US": "Lab Qwen"},
        }
    ]
    overlay["additions"]["combinations"] = [
        {
            "id": "lab_qwen_oft_libero",
            "based_on": "qwen_oft_libero",
            "label": {"zh-CN": "实验室 Qwen OFT", "en-US": "Lab Qwen OFT"},
            "backbone": "lab_qwen_variant",
            "action_head": "mlp_regression",
            "method": "imitation_learning",
            "datasets": ["libero"],
            "dataset_mixes": ["libero_goal"],
            "min_gpus": 1,
        }
    ]
    overlay["additions"]["deployment_combinations"] = [
        {
            "id": "lab_deploy_qwen_oft",
            "based_on": "deploy_qwen2_5_oft",
            "label": {"zh-CN": "实验室部署", "en-US": "Lab deployment"},
            "backbone": "lab_qwen_variant",
            "action_head": "mlp_regression",
            "adapter": "base_framework_websocket",
            "source_combinations": ["lab_qwen_oft_libero"],
            "benchmarks": ["libero"],
            "recommended_gpu_count": 1,
        }
    ]
    overlay["disables"]["combinations"] = ["toy_libero_debug"]
    overlay["disables"]["deployment_combinations"] = ["deploy_qwen3_oft"]
    base = load_catalog()
    return apply_registry_overlay(
        base,
        validate_registry_overlay(overlay, base_catalog=base),
    )


def _experimental_spec(output_root: Path) -> dict:
    return {
        "architecture": {
            "backbone": "lab_qwen_variant",
            "action_head": "mlp_regression",
        },
        "training": {"method": "imitation_learning"},
        "dataset": {"id": "libero", "mix": "libero_goal"},
        "resources": {"allocation": "auto", "num_gpus": 1},
        "parameters": {
            "run_id": "effective-catalog-test",
            "output_root_dir": str(output_root),
        },
        "experimental": {"enabled": True, "risk_acknowledged": True},
        "expert_overrides": {},
    }


def _codes(issues: list[dict]) -> set[str]:
    return {str(item.get("code")) for item in issues}


def test_incomplete_calvin_and_robotwin_chains_remain_hidden_in_expert_mode() -> None:
    catalog = get_catalog(include_experimental=True)
    assert {"qwen_pi_calvin", "qwen_oft_robotwin"}.isdisjoint(
        {row["id"] for row in catalog["combinations"]}
    )
    assert {"calvin", "robotwin"}.isdisjoint(
        {row["id"] for row in catalog["components"]["datasets"]}
    )


def test_effective_catalog_filters_and_resolves_overlay_training(tmp_path: Path) -> None:
    effective = _effective_catalog()
    normal = get_catalog(source_catalog=effective)
    expert = get_catalog(include_experimental=True, source_catalog=effective)

    assert "lab_qwen_oft_libero" not in {row["id"] for row in normal["combinations"]}
    assert "lab_qwen_oft_libero" in {row["id"] for row in expert["combinations"]}
    assert "toy_libero_debug" not in {row["id"] for row in expert["combinations"]}
    assert "toy" not in {row["id"] for row in expert["components"]["backbones"]}

    options = get_compatible_options(
        {
            "architecture": {
                "backbone": "lab_qwen_variant",
                "action_head": "mlp_regression",
            }
        },
        include_experimental=True,
        source_catalog=effective,
    )
    assert {row["id"] for row in options["training_methods"]} == {"imitation_learning"}
    assert {row["id"] for row in options["datasets"]} == {"libero"}

    spec = _experimental_spec(tmp_path / "results")
    compatibility = validate_compatibility(spec, source_catalog=effective)
    assert _codes(compatibility) == {"experimental_combination"}
    snapshot = resolve_experiment(
        spec,
        REPO_ROOT,
        environment={},
        source_catalog=effective,
    )
    assert snapshot["combination"]["id"] == "lab_qwen_oft_libero"
    assert snapshot["registry_version"].endswith("+overlay.effective-test")

    disabled = {
        **spec,
        "architecture": {"backbone": "toy", "action_head": "mlp_regression"},
    }
    assert "incompatible_combination" in _codes(
        validate_compatibility(disabled, source_catalog=effective)
    )


def test_effective_catalog_reaches_static_and_full_preflight(tmp_path: Path) -> None:
    class Monitor:
        available = True
        error = ""

        def snapshot(self, reservations=None):  # type: ignore[no-untyped-def]
            return [
                GPUInfo(
                    index=0,
                    uuid="GPU-0",
                    name="GPU 0",
                    memory_total_bytes=24 * 1024**3,
                    memory_used_bytes=0,
                    memory_free_bytes=24 * 1024**3,
                    utilization_percent=0,
                    temperature_c=30,
                    processes=[],
                    available=True,
                )
            ]

    effective = _effective_catalog()
    pretrained = tmp_path / "pretrained"
    (pretrained / "Qwen2.5-VL-3B-Instruct").mkdir(parents=True)
    data = tmp_path / "libero"
    data.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    environment = {
        "PRETRAINED_MODELS_DIR": str(pretrained),
        "LIBERO_DATA_ROOT": str(data),
    }
    spec = _experimental_spec(results)
    snapshot = resolve_experiment(
        spec,
        REPO_ROOT,
        environment=environment,
        source_catalog=effective,
    )
    static_issues = preflight_static(
        spec,
        REPO_ROOT,
        resolved=snapshot,
        environment=environment,
        source_catalog=effective,
    )
    assert not {
        "unknown_backbone",
        "incompatible_combination",
        "config_resolution_failed",
    } & _codes(static_issues)

    resolved, compatibility, issues, _previews = run_preflight(
        repo_root=REPO_ROOT,
        state_dir=tmp_path / "state",
        name="effective-catalog-test",
        spec=spec,
        configured_environment=environment,
        gpu_monitor=Monitor(),
        results_roots=[str(results)],
        disk_min_free_gib=0,
        disk_min_free_percent=0,
        include_experimental=True,
        source_catalog=effective,
    )
    assert resolved["combination"]["id"] == "lab_qwen_oft_libero"
    assert compatibility == "experimental"
    assert not {
        "unknown_backbone",
        "incompatible_combination",
        "config_resolution_failed",
    } & _codes(issues)


def test_effective_catalog_drives_deployment_command_and_evaluation() -> None:
    effective = _effective_catalog()
    normal_deployment = get_deployment_catalog(source_catalog=effective)
    expert_deployment = get_deployment_catalog(
        include_experimental=True,
        source_catalog=effective,
    )
    assert "lab_deploy_qwen_oft" not in {
        row["id"] for row in normal_deployment["deployment_combinations"]
    }
    assert "lab_deploy_qwen_oft" in {
        row["id"] for row in expert_deployment["deployment_combinations"]
    }
    assert "deploy_qwen3_oft" not in {
        row["id"] for row in expert_deployment["deployment_combinations"]
    }

    command = build_adapter_command(
        "base_framework_websocket",
        "/checkpoints/lab-qwen",
        combination_id="lab_deploy_qwen_oft",
        backbone_id="lab_qwen_variant",
        action_head_id="mlp_regression",
        include_experimental=True,
        source_catalog=effective,
    )
    assert command["valid"] is True
    disabled = build_adapter_command(
        "base_framework_websocket",
        "/checkpoints/qwen3",
        combination_id="deploy_qwen3_oft",
        backbone_id="qwen3_vl",
        action_head_id="mlp_regression",
        include_experimental=True,
        source_catalog=effective,
    )
    assert disabled["valid"] is False
    assert "deployment_combination_unknown" in _codes(disabled["issues"])

    normal_evaluation = get_evaluation_catalog(source_catalog=effective)
    expert_evaluation = get_evaluation_catalog(
        include_experimental=True,
        source_catalog=effective,
    )
    assert "lab_deploy_qwen_oft" not in {
        row["id"] for row in normal_evaluation["combinations"]
    }
    added = next(
        row
        for row in expert_evaluation["combinations"]
        if row["id"] == "lab_deploy_qwen_oft"
    )
    assert set(added["benchmark_ids"]) == {"libero", "libero_plus"}
    assert "deploy_qwen3_oft" not in {
        row["id"] for row in expert_evaluation["combinations"]
    }
    assert expert_evaluation["registry_version"].endswith("+overlay.effective-test")


def test_disabled_deployment_is_rejected_by_inspect_and_resolve(
    tmp_path: Path,
) -> None:
    effective = _effective_catalog()
    checkpoint = tmp_path / "qwen3-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"never-deserialized")
    (checkpoint / "framework_config.yaml").write_text(
        yaml.safe_dump(
            {"framework": {"name": "QwenOFT", "qwenvl": {}}, "trainer": {}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (checkpoint / "dataset_statistics.json").write_text("{}", encoding="utf-8")
    embedded = checkpoint / "vlm_pretrained"
    embedded.mkdir()
    (embedded / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (embedded / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    inspected = inspect_checkpoint(
        checkpoint,
        repo_root=tmp_path,
        include_experimental=True,
        source_catalog=effective,
    )
    resolved = resolve_deployment(
        checkpoint,
        repo_root=tmp_path,
        include_experimental=True,
        source_catalog=effective,
    )
    evaluation = inspect_evaluation_checkpoint(
        checkpoint,
        repo_root=tmp_path,
        include_experimental=True,
        source_catalog=effective,
    )

    for result in (inspected, resolved, evaluation):
        assert result["valid"] is False
        assert "deployment_adapter_not_wired" in _codes(result["issues"])
    assert inspected["candidates"] == []
    assert evaluation["evaluation_candidates"] == []
