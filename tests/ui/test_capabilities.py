from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from alphabrain_ui.capabilities import (
    get_catalog,
    get_compatible_options,
    resolve_pretrained_directory_status,
    selection_from_spec,
    validate_compatibility,
)
from alphabrain_ui.configuration import (
    ExperimentConfigurationError,
    make_template_spec,
    preflight_static,
    resolve_experiment,
)
from alphabrain_ui.gpu import GPUInfo
from alphabrain_ui.preflight import run_preflight
from alphabrain_ui.registry import catalog_for_runtime, load_catalog

REPO_ROOT = Path(__file__).resolve().parents[2]


def spec_for(
    backbone: str = "qwen2_5_vl",
    head: str = "mlp_regression",
    method: str = "imitation_learning",
    dataset: str = "libero",
) -> dict:
    return {
        "architecture": {"backbone": backbone, "action_head": head},
        "training": {"method": method},
        "dataset": {"id": dataset, "mix": "libero_goal"},
        "resources": {"allocation": "auto", "num_gpus": 1},
        "parameters": {"run_id": "ui-test"},
        "expert_overrides": {},
    }


def codes(issues: list[dict]) -> set[str]:
    return {item["code"] for item in issues}


def test_catalog_hides_experimental_and_unsupported_by_default() -> None:
    normal = get_catalog()
    expert = get_catalog(include_experimental=True)

    assert normal["filters"]["unsupported_hidden"] is True
    assert all(item["status"] == "verified" for item in normal["combinations"])
    assert all(item["status"] != "unsupported" for item in expert["combinations"])
    assert len(expert["combinations"]) > len(normal["combinations"])
    assert "continual_ewc" not in {item["id"] for item in normal["components"]["training_methods"]}
    assert "continual_ewc" in {item["id"] for item in expert["components"]["training_methods"]}
    normal_methods = {item["id"] for item in normal["components"]["training_methods"]}
    expert_methods = {item["id"] for item in expert["components"]["training_methods"]}
    assert "rlt_a_ppo" not in normal_methods | expert_methods
    assert {"rlt_a_grpo", "vanilla_vla_ppo", "cotrain", "vlm_only"} <= expert_methods
    assert {"rlt_a_grpo", "vanilla_vla_ppo", "cotrain", "vlm_only"}.isdisjoint(normal_methods)

    # The same catalog also exposes an API/UI-friendly flattened view.
    assert normal["capabilities"]
    assert all({"id", "kind", "name", "name_zh", "compatible_with"} <= item.keys() for item in normal["capabilities"])
    component_ids = {group: {item["id"] for item in values} for group, values in normal["components"].items()}
    for combo in normal["combinations"]:
        assert {"id", "schema_version", "fields"} <= combo["workflow"].keys()
        assert combo["backbone"] in component_ids["backbones"]
        assert combo["action_head"] in component_ids["action_heads"]
        assert combo["method"] in component_ids["training_methods"]
        assert set(combo["datasets"]) <= component_ids["datasets"]


def test_regular_runtime_hides_demo_only_registry_entries() -> None:
    packaged = load_catalog()
    regular = catalog_for_runtime(packaged, demo_mode=False)
    demo = catalog_for_runtime(packaged, demo_mode=True)

    assert "toy" not in {
        item["id"] for item in regular["components"]["backbones"]
    }
    assert "toy_libero_debug" not in {
        item["id"] for item in regular["combinations"]
    }
    assert "toy" in {item["id"] for item in demo["components"]["backbones"]}
    assert "toy_libero_debug" in {item["id"] for item in demo["combinations"]}
    assert "toy" in {item["id"] for item in packaged["components"]["backbones"]}


def test_catalog_reports_backbone_pretrained_directory_status(tmp_path: Path) -> None:
    pretrained_root = tmp_path / "pretrained"
    qwen_directory = pretrained_root / "Qwen2.5-VL-3B-Instruct"
    qwen_directory.mkdir(parents=True)

    configured = get_catalog(
        repo_root=REPO_ROOT,
        environment={"PRETRAINED_MODELS_DIR": str(pretrained_root)},
    )
    backbones = {item["id"]: item for item in configured["backbones"]}
    assert backbones["qwen2_5_vl"]["pretrained_directory"] == {
        "required": True,
        "configured": True,
        "path": str(qwen_directory),
        "exists": True,
        "issue": None,
    }
    assert backbones["qwen3_vl"]["pretrained_directory"]["configured"] is True
    assert backbones["qwen3_vl"]["pretrained_directory"]["exists"] is False
    assert backbones["qwen3_vl"]["pretrained_directory"]["issue"]["code"] == "pretrained_directory_not_found"
    assert backbones["toy"]["pretrained_directory"] == {
        "required": False,
        "configured": True,
        "path": None,
        "exists": True,
        "issue": None,
    }

    missing = get_catalog(repo_root=REPO_ROOT, environment={})
    qwen = next(item for item in missing["backbones"] if item["id"] == "qwen2_5_vl")
    status = qwen["pretrained_directory"]
    assert status["configured"] is False
    assert status["exists"] is False
    assert status["path"] == "${PRETRAINED_MODELS_DIR}/Qwen2.5-VL-3B-Instruct"
    assert status["issue"]["code"] == "missing_environment_variable"
    assert status["issue"]["variables"] == ["PRETRAINED_MODELS_DIR"]


def test_pretrained_directory_status_rejects_a_file_path(tmp_path: Path) -> None:
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"weights")
    status = resolve_pretrained_directory_status(
        {"default_pretrained": "${MODEL_ROOT}"},
        repo_root=tmp_path,
        environment={"MODEL_ROOT": str(model_file)},
    )
    assert status["configured"] is True
    assert status["exists"] is False
    assert status["issue"]["code"] == "pretrained_path_not_directory"


def test_partial_selection_returns_only_reachable_options() -> None:
    options = get_compatible_options(
        {"architecture": {"backbone": "llama", "action_head": "mlp"}},
        include_experimental=False,
    )
    method_ids = {item["id"] for item in options["training_methods"]}
    dataset_ids = {item["id"] for item in options["datasets"]}
    assert {"imitation_learning", "continual_er", "continual_mir"} <= method_ids
    assert "rlt_a_td3" not in method_ids
    assert dataset_ids == {"libero"}


def test_flat_ui_spec_and_aliases_are_canonicalised() -> None:
    spec = {
        "name": "flat-ui-run",
        "backbone_id": "qwen",
        "action_head_id": "mlp",
        "method_id": "il",
        "dataset_id": "libero",
        "parameters": {"batch_size": 4, "max_steps": 10},
        "resources": {"strategy": "auto", "gpu_count": 1},
    }
    assert selection_from_spec(spec) == {
        "backbone": "qwen2_5_vl",
        "action_head": "mlp_regression",
        "method": "imitation_learning",
        "dataset": "libero",
    }
    assert validate_compatibility(spec) == []

    resolved = resolve_experiment(spec, REPO_ROOT, environment={})
    assert resolved["spec"]["parameters"]["run_id"] == "flat-ui-run"
    assert resolved["resolved_config"]["datasets"]["vla_data"]["per_device_batch_size"] == 4
    assert resolved["resolved_config"]["trainer"]["max_train_steps"] == 10

    spec["dataset"] = {"id": "libero", "root": "/datasets/local-libero"}
    resolved_with_local_data = resolve_experiment(spec, REPO_ROOT, environment={})
    assert resolved_with_local_data["resolved_config"]["datasets"]["vla_data"]["data_root_dir"] == "/datasets/local-libero"


def test_incompatible_and_explicitly_unsupported_combinations_are_errors() -> None:
    impossible = spec_for("llama3_2_vision", "snn", "imitation_learning")
    assert "incompatible_combination" in codes(validate_compatibility(impossible))

    pi_rlt_a = spec_for("paligemma", "pi05_action_expert", "rlt_a_td3")
    assert "unsupported_combination" in codes(validate_compatibility(pi_rlt_a))


def test_experimental_combination_requires_opt_in_and_acknowledgement() -> None:
    experimental = spec_for("qwen2_5_vl", "mlp_regression", "imitation_learning", "arx")
    experimental["dataset"]["mix"] = "arx_x5"
    assert "experimental_disabled" in codes(validate_compatibility(experimental))

    experimental["experimental"] = {"enabled": True}
    assert "experimental_acknowledgement_required" in codes(validate_compatibility(experimental))

    experimental["experimental"]["risk_acknowledged"] = True
    issues = validate_compatibility(experimental)
    assert codes(issues) == {"experimental_combination"}
    assert issues[0]["severity"] == issues[0]["level"] == "warning"


def test_resolve_preserves_sources_and_applies_precedence_without_editing_yaml() -> None:
    source = REPO_ROOT / "configs/finetune_config.yaml"
    before = source.read_bytes()
    spec = spec_for("paligemma", "mlp_regression")
    spec["resources"]["num_gpus"] = 2
    spec["training"]["template_mode"] = "paligemma_oft_goal"
    spec["parameters"].update(
        {
            "max_train_steps": 123,
            "learning_rate": 2e-5,
            "output_root_dir": "/tmp/alphabrain-ui-tests",
        }
    )
    spec["expert_overrides"] = {
        "trainer.save_interval": 17,
        "framework": {"action_model": {"action_dim": 8}},
    }

    first = resolve_experiment(spec, REPO_ROOT, environment={})
    second = resolve_experiment(deepcopy(spec), REPO_ROOT, environment={})
    cfg = first["resolved_config"]
    assert cfg["trainer"]["max_train_steps"] == 123
    assert cfg["trainer"]["learning_rate"]["base"] == 2e-5
    assert cfg["trainer"]["save_interval"] == 17
    assert cfg["framework"]["action_model"]["action_dim"] == 8
    assert first["config_fingerprint"] == second["config_fingerprint"]
    assert first["sources"] and all(len(item["sha256"]) == 64 for item in first["sources"])
    assert source.read_bytes() == before


def test_template_spec_removes_run_allocation_but_keeps_scientific_parameters() -> None:
    spec = spec_for()
    spec["parameters"].update({"max_train_steps": 99, "output_root_dir": "/tmp/out"})
    spec["resources"].update({"allocation": "fixed", "gpu_ids": [2], "main_process_port": 29999})
    template = make_template_spec(spec)
    assert "run_id" not in template["parameters"]
    assert "output_root_dir" not in template["parameters"]
    assert template["parameters"]["max_train_steps"] == 99
    assert template["resources"]["allocation"] == "auto"
    assert "gpu_ids" not in template["resources"]
    assert "main_process_port" not in template["resources"]


def test_continual_mir_resolves_algorithm_and_rejects_unwired_mix() -> None:
    spec = spec_for("qwen2_5_vl", "flow_matching_dit", "continual_mir")
    spec["resources"]["num_gpus"] = 2
    resolved = resolve_experiment(spec, REPO_ROOT, environment={})
    cl = resolved["resolved_config"]["continual_learning"]
    assert cl["replay"]["enabled"] is False
    assert cl["algorithm"]["name"] == "mir"
    assert resolved["training"]["family"] == "continual_learning"

    spec["dataset"]["mix"] = "libero_spatial"
    assert "dataset_mix_not_wired" in codes(validate_compatibility(spec))


def test_rl_resolution_exposes_dependency_stages_and_can_skip_pretrain() -> None:
    spec = spec_for("qwen2_5_vl", "mlp_regression", "rlt_td3")
    spec["training"]["checkpoint"] = "/models/qwen-vla"
    resolved = resolve_experiment(spec, REPO_ROOT, environment={})
    stages = resolved["command_preview"]["stages"]
    assert [stage["id"] for stage in stages] == ["encoder_pretrain", "rl_offpolicy"]
    assert stages[1]["depends_on"] == ["encoder_pretrain"]

    spec["training"]["encoder_checkpoint"] = "/models/encoder.pt"
    skipped = resolve_experiment(spec, REPO_ROOT, environment={})
    stages = skipped["command_preview"]["stages"]
    assert [stage["id"] for stage in stages] == ["rl_offpolicy"]
    assert stages[0]["depends_on"] == []


def test_reserved_expert_roots_are_rejected() -> None:
    spec = spec_for()
    spec["expert_overrides"] = {"modes": {"injected": {}}}
    with pytest.raises(ExperimentConfigurationError) as raised:
        resolve_experiment(spec, REPO_ROOT)
    assert raised.value.issues[0]["code"] == "reserved_expert_key"


def test_secrets_are_not_allowed_in_persisted_expert_config() -> None:
    spec = spec_for()
    spec["expert_overrides"] = {"wandb": {"api_key": "do-not-persist"}}
    with pytest.raises(ExperimentConfigurationError) as raised:
        resolve_experiment(spec, REPO_ROOT)
    assert raised.value.issues[0]["code"] == "secret_in_expert_config"


def test_secret_environment_references_are_not_expanded_into_snapshots() -> None:
    spec = spec_for()
    spec["expert_overrides"] = {"research_note": "${HF_TOKEN}"}
    resolved = resolve_experiment(spec, REPO_ROOT, environment={"HF_TOKEN": "super-secret-value"})
    serialized = json.dumps(resolved)
    assert "super-secret-value" not in serialized
    assert resolved["resolved_config"]["research_note"] == "${HF_TOKEN}"


def test_preflight_rejects_fixed_gpu_ids_that_are_not_visible(tmp_path: Path) -> None:
    class Monitor:
        available = True
        error = ""

        def snapshot(self, reservations=None):
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

    pretrained = tmp_path / "pretrained"
    (pretrained / "Qwen2.5-VL-3B-Instruct").mkdir(parents=True)
    data = tmp_path / "libero"
    data.mkdir()
    spec = spec_for()
    spec["resources"] = {"allocation": "fixed", "num_gpus": 1, "gpu_ids": [99]}
    _resolved, _compatibility, issues, _previews = run_preflight(
        repo_root=REPO_ROOT,
        state_dir=tmp_path / "state",
        name="invalid-gpu",
        spec=spec,
        configured_environment={"PRETRAINED_MODELS_DIR": str(pretrained), "LIBERO_DATA_ROOT": str(data)},
        gpu_monitor=Monitor(),
        results_roots=[str(tmp_path / "results")],
        disk_min_free_gib=0,
        disk_min_free_percent=0,
        include_experimental=False,
    )
    assert "gpu_ids_not_visible" in codes(issues)


def test_robocasa_preflight_does_not_require_unrelated_libero_paths(tmp_path: Path) -> None:
    class Monitor:
        available = True
        error = ""

        def snapshot(self, reservations=None):
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

    model = tmp_path / "qwen3"
    model.mkdir()
    data = tmp_path / "robocasa"
    data.mkdir()
    spec = spec_for("qwen3_vl", "mlp_regression", "imitation_learning", "robocasa")
    spec["dataset"]["mix"] = "fourier_gr1_unified_1000"
    spec["expert_overrides"] = {"base_vlm": str(model)}
    _resolved, _compatibility, issues, _previews = run_preflight(
        repo_root=REPO_ROOT,
        state_dir=tmp_path / "state",
        name="robocasa",
        spec=spec,
        configured_environment={"ROBOCASA_TABLETOP_DATA_ROOT": str(data)},
        gpu_monitor=Monitor(),
        results_roots=[str(tmp_path / "results")],
        disk_min_free_gib=0,
        disk_min_free_percent=0,
        include_experimental=False,
    )
    assert "dataset_root_missing" not in codes(issues)
    assert "pretrained_root_missing" not in codes(issues)


def test_missing_rl_encoder_checkpoint_blocks_submission(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vla"
    checkpoint.mkdir()
    spec = spec_for("qwen2_5_vl", "mlp_regression", "rlt_td3")
    spec["training"].update(
        {
            "checkpoint": str(checkpoint),
            "encoder_checkpoint": str(tmp_path / "missing-encoder.pt"),
        }
    )
    resolved = resolve_experiment(spec, REPO_ROOT, environment={})
    issues = preflight_static(spec, REPO_ROOT, resolved=resolved, environment={})
    assert "missing_model_path" in codes(issues)


def test_spec_v2_resume_modes_map_explicitly_and_legacy_specs_migrate() -> None:
    legacy = spec_for()
    legacy_resolved = resolve_experiment(legacy, REPO_ROOT, environment={})
    assert legacy_resolved["spec"]["spec_version"] == 2
    assert legacy_resolved["spec"]["workflow"]["id"] == "standard"
    assert legacy_resolved["spec"]["resume"] == {"mode": "none", "checkpoint": None}

    weights = spec_for()
    weights["resume"] = {"mode": "weights_only", "checkpoint": "/models/base"}
    weights_cfg = resolve_experiment(weights, REPO_ROOT, environment={})["resolved_config"]["trainer"]
    assert weights_cfg["pretrained_checkpoint"] == "/models/base"
    assert weights_cfg["is_resume"] is False
    assert weights_cfg["resume_checkpoint"] is None

    full = spec_for()
    full["resume"] = {"mode": "full_state", "checkpoint": "/runs/checkpoints/steps_100"}
    full_cfg = resolve_experiment(full, REPO_ROOT, environment={})["resolved_config"]["trainer"]
    assert full_cfg["pretrained_checkpoint"] is None
    assert full_cfg["is_resume"] is True
    assert full_cfg["resume_checkpoint"] == "/runs/checkpoints/steps_100"


def test_structured_cl_stdp_and_world_model_workflows_override_real_config_paths() -> None:
    cl = spec_for("qwen2_5_vl", "flow_matching_dit", "continual_mir")
    cl["workflow"] = {
        "id": "continual_learning",
        "config": {
            "task_sequence": "libero_10",
            "steps_per_task": 321,
            "mir_refresh_interval": 7,
            "mir_candidate_size": 12,
            "mir_top_k": 4,
        },
    }
    cl_cfg = resolve_experiment(cl, REPO_ROOT, environment={})["resolved_config"]
    assert cl_cfg["continual_learning"]["task_sequence"] == "libero_10"
    assert cl_cfg["continual_learning"]["steps_per_task"] == 321
    assert cl_cfg["continual_learning"]["algorithm"]["name"] == "mir"
    assert cl_cfg["continual_learning"]["algorithm"]["mir_refresh_interval"] == 7

    stdp = spec_for("qwen2_5_vl", "snn", "r_stdp")
    stdp["training"]["checkpoint"] = "/models/neuro"
    stdp["workflow"] = {"id": "r_stdp", "config": {"alpha": 0.6, "stdp_lr": 2e-5}}
    stdp_cfg = resolve_experiment(stdp, REPO_ROOT, environment={})["resolved_config"]
    assert stdp_cfg["stdp"]["alpha"] == 0.6
    assert stdp_cfg["stdp"]["stdp_lr"] == 2e-5

    world = spec_for("cosmos2", "flow_matching_dit", "imitation_learning")
    world["resources"]["num_gpus"] = 4
    world["workflow"] = {
        "id": "world_model",
        "config": {"use_video_loss": False, "feature_layer_id": 12, "sigma_max": 42.0},
    }
    world_cfg = resolve_experiment(world, REPO_ROOT, environment={})["resolved_config"]
    assert world_cfg["framework"]["world_model"]["use_video_loss"] is False
    assert world_cfg["framework"]["world_model"]["feature_layer_id"] == 12
    assert world_cfg["framework"]["world_model"]["sigma_max"] == 42.0


def test_rl_workflow_supports_encoder_reuse_and_relative_gpu_roles() -> None:
    spec = spec_for("qwen2_5_vl", "mlp_regression", "rlt_td3")
    spec["training"]["checkpoint"] = "/models/qwen-vla"
    spec["workflow"] = {
        "id": "rl_token",
        "config": {
            "encoder_mode": "reuse",
            "encoder_checkpoint": "/models/encoder.pt",
            "task_scope": "subset",
            "task_ids": "0,2",
            "rollout_gpu_count": 2,
            "train_gpu_role": 2,
            "episodes_per_task": 9,
            "envs_per_task": 3,
        },
    }
    resolved = resolve_experiment(spec, REPO_ROOT, environment={})
    stages = resolved["command_preview"]["stages"]
    assert [stage["id"] for stage in stages] == ["rl_offpolicy"]
    assert stages[0]["requested_gpu_count"] == 3
    assert stages[0]["environment"]["ROLLOUT_GPUS"] == "0,1"
    assert stages[0]["environment"]["TRAIN_GPU"] == "2"
    assert stages[0]["environment"]["TASK_SCOPE"] == "subset"
    assert stages[0]["environment"]["TASK_IDS"] == "0,2"


def test_neurovla_pipeline_and_repository_backed_experimental_trainers() -> None:
    neuro = spec_for("qwen2_5_vl", "snn")
    neuro["workflow"] = {
        "id": "neurovla_pretrain",
        "config": {"pipeline_mode": "pretrain_then_stdp", "stdp_alpha": 0.7},
    }
    resolved = resolve_experiment(neuro, REPO_ROOT, environment={})
    stages = resolved["command_preview"]["stages"]
    assert [stage["id"] for stage in stages] == ["pretrain", "stdp"]
    assert stages[1]["depends_on"] == ["pretrain"]
    assert stages[1]["resolved_config"]["stdp"]["alpha"] == 0.7
    assert stages[1]["resolved_config"]["trainer"]["pretrained_checkpoint"].endswith("ui-test/final_model")

    for method, entrypoint in (
        ("cotrain", "AlphaBrain/training/train_alphabrain_cotrain.py"),
        ("vlm_only", "AlphaBrain/training/train_alphabrain_vlm.py"),
    ):
        experimental = spec_for("qwen2_5_vl", "flow_matching_dit", method)
        experimental["experimental"] = {"enabled": True, "risk_acknowledged": True}
        experimental["workflow"] = {"id": method, "config": {}}
        snapshot = resolve_experiment(experimental, REPO_ROOT, environment={})
        assert entrypoint in snapshot["command_preview"]["argv"]
        assert snapshot["resolved_config"]["framework"]["qwenvl"]["base_vlm"].endswith(
            "Qwen2.5-VL-3B-Instruct"
        )


def test_static_preflight_checks_paths_and_output_collision(tmp_path: Path) -> None:
    pretrained = tmp_path / "pretrained"
    (pretrained / "paligemma-3b-pt-224").mkdir(parents=True)
    data = tmp_path / "libero"
    data.mkdir()
    output = tmp_path / "output"

    spec = spec_for("paligemma", "mlp_regression")
    spec["resources"]["num_gpus"] = 2
    spec["parameters"]["output_root_dir"] = str(output)
    environment = {"PRETRAINED_MODELS_DIR": str(pretrained), "LIBERO_DATA_ROOT": str(data)}
    resolved = resolve_experiment(spec, REPO_ROOT, environment=environment)
    issues = preflight_static(spec, REPO_ROOT, resolved=resolved, environment=environment)
    assert "static_preflight_passed" in codes(issues)
    assert not any(item["severity"] == "error" for item in issues)

    (output / "ui-test").mkdir(parents=True)
    issues = preflight_static(spec, REPO_ROOT, resolved=resolved, environment=environment)
    assert "output_exists" in codes(issues)


def test_full_preflight_handles_output_collision_with_a_regular_file(tmp_path: Path) -> None:
    class Monitor:
        available = True
        error = ""

        def snapshot(self, reservations=None):
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

    pretrained = tmp_path / "pretrained"
    (pretrained / "Qwen2.5-VL-3B-Instruct").mkdir(parents=True)
    data = tmp_path / "libero"
    data.mkdir()
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    (output_root / "ui-test").write_text("collision")
    spec = spec_for()
    spec["parameters"]["output_root_dir"] = str(output_root)
    _resolved, _compatibility, issues, _previews = run_preflight(
        repo_root=REPO_ROOT,
        state_dir=tmp_path / "state",
        name="ui-test",
        spec=spec,
        configured_environment={"PRETRAINED_MODELS_DIR": str(pretrained), "LIBERO_DATA_ROOT": str(data)},
        gpu_monitor=Monitor(),
        results_roots=[str(output_root)],
        disk_min_free_gib=0,
        disk_min_free_percent=0,
        include_experimental=False,
    )
    assert "output_exists" in codes(issues)
