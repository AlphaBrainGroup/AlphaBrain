from __future__ import annotations

import json
from pathlib import Path

import yaml

from alphabrain_ui.deployment_registry import (
    build_adapter_command,
    get_deployment_catalog,
    inspect_checkpoint,
    resolve_checkpoint_source,
    resolve_deployment,
)


def issue_codes(result: dict) -> set[str]:
    return {item["code"] for item in result["issues"]}


def make_vlm_dir(path: Path, model_type: str) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    return path


def make_generic_checkpoint(path: Path, framework: dict) -> Path:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"not-deserialized-by-preflight")
    (path / "framework_config.yaml").write_text(
        yaml.safe_dump({"framework": framework, "trainer": {}}, sort_keys=False),
        encoding="utf-8",
    )
    (path / "dataset_statistics.json").write_text(json.dumps({"libero": {"action": {}}}), encoding="utf-8")
    return path


def test_deployment_catalog_is_json_only_and_exposes_only_wired_models() -> None:
    catalog = get_deployment_catalog()
    assert json.loads(json.dumps(catalog)) == catalog
    assert {item["id"] for item in catalog["deployment_adapters"]} == {
        "base_framework_websocket",
        "lerobot_pi05_websocket",
        "cosmos_policy_websocket",
    }
    assert catalog["deployment_adapters"] == catalog["adapters"]
    assert catalog["deployment_combinations"] == catalog["combinations"]
    assert all(item["status"] == "verified" for item in catalog["combinations"])
    assert all(not item["id"].startswith("deploy_rl_") for item in catalog["combinations"])
    assert all(
        not any(str(source).startswith("rl_") for source in item.get("source_combinations", []))
        for item in catalog["combinations"]
    )
    assert all(item.get("adapter") != "none" for item in catalog["combinations"])
    assert catalog["filters"] == {"include_experimental": False, "unsupported_hidden": True}


def test_checkpoint_index_and_relative_local_paths_resolve_without_database_imports(tmp_path: Path) -> None:
    checkpoint = tmp_path / "models" / "checkpoint"
    checkpoint.mkdir(parents=True)
    index = [{"id": "checkpoint-id", "path": str(checkpoint)}]

    indexed = resolve_checkpoint_source({"checkpoint_id": "checkpoint-id"}, checkpoint_index=index)
    relative = resolve_checkpoint_source("models/checkpoint", repo_root=tmp_path)

    assert indexed["valid"] is True
    assert indexed["kind"] == "checkpoint_index"
    assert indexed["path"] == str(checkpoint.resolve())
    assert relative["valid"] is True
    assert relative["kind"] == "local_path"
    assert relative["path"] == str(checkpoint.resolve())


def test_self_contained_checkpoint_detects_qwen3_from_embedded_metadata(tmp_path: Path) -> None:
    checkpoint = make_generic_checkpoint(tmp_path / "checkpoint", {"name": "QwenOFT", "qwenvl": {}})
    make_vlm_dir(checkpoint / "vlm_pretrained", "qwen3_vl")

    result = inspect_checkpoint(checkpoint, repo_root=tmp_path)

    assert result["valid"] is True
    assert result["deployable"] is True
    assert result["checkpoint"]["format"] == "self_contained"
    assert result["detected"]["framework"] == "QwenOFT"
    assert result["detected"]["backbone"] == "qwen3_vl"
    assert result["detected"]["action_head"] == "mlp_regression"
    assert result["detected"]["combination_id"] == "deploy_qwen3_oft"
    assert result["detected"]["adapter_id"] == "base_framework_websocket"


def test_lerobot_pi05_format_is_explicit_and_uses_dedicated_adapter(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pi05"
    checkpoint.mkdir()
    tokenizer = tmp_path / "paligemma-tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "config.json").write_text(json.dumps({"type": "pi05"}), encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"not-read-by-static-inspection")
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": str(tokenizer)},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "policy_postprocessor.json").write_text(json.dumps({"steps": []}), encoding="utf-8")

    inspected = inspect_checkpoint(checkpoint, repo_root=tmp_path)
    resolved = resolve_deployment(
        checkpoint,
        repo_root=tmp_path,
        python_executable="/env/bin/python",
    )

    assert inspected["valid"] is True
    assert inspected["checkpoint"]["checkpoint_family"] == "pi05"
    assert inspected["checkpoint"]["checkpoint_format"] == "lerobot"
    assert inspected["checkpoint"]["checkpoint_format_label"]["zh-CN"] == "LeRobot Pi0.5"
    assert inspected["detected"]["combination_id"] == "deploy_lerobot_pi05"
    assert inspected["detected"]["adapter_id"] == "lerobot_pi05_websocket"
    assert resolved["command"] == [
        "/env/bin/python",
        str(tmp_path / "deployment/model_server/server_policy_lerobot_pi05.py"),
        "--ckpt_path",
        str(checkpoint.resolve()),
        "--port",
        "10093",
        "--idle_timeout",
        "1800",
    ]


def test_openpi_and_alphabrain_pi05_formats_are_not_mislabeled(tmp_path: Path) -> None:
    openpi = tmp_path / "openpi"
    (openpi / "params").mkdir(parents=True)
    (openpi / "assets").mkdir()
    (openpi / "config.json").write_text(
        json.dumps({"format": "openpi", "policy_type": "pi05"}),
        encoding="utf-8",
    )
    base_model = make_vlm_dir(tmp_path / "paligemma", "paligemma")
    alphabrain = make_generic_checkpoint(
        tmp_path / "alphabrain",
        {"name": "PaliGemmaPi05", "paligemma": {"base_vlm": str(base_model)}},
    )

    openpi_result = inspect_checkpoint(openpi, repo_root=tmp_path)
    alphabrain_result = inspect_checkpoint(alphabrain, repo_root=tmp_path)

    assert openpi_result["checkpoint"]["checkpoint_format"] == "openpi"
    assert "openpi_pi05_adapter_not_wired" in issue_codes(openpi_result)
    assert alphabrain_result["checkpoint"]["checkpoint_format"] == "alphabrain"
    assert alphabrain_result["detected"]["combination_id"] == "deploy_paligemma_pi05"


def test_ambiguous_framework_returns_only_compatible_candidates(tmp_path: Path) -> None:
    base_model = make_vlm_dir(tmp_path / "custom-base-model", "custom")
    checkpoint = make_generic_checkpoint(
        tmp_path / "checkpoint",
        {"name": "QwenOFT", "qwenvl": {"base_vlm": str(base_model)}},
    )

    ambiguous = inspect_checkpoint(checkpoint, repo_root=tmp_path)
    selected = inspect_checkpoint(checkpoint, repo_root=tmp_path, combination_id="deploy_qwen2_5_oft")

    assert ambiguous["valid"] is False
    assert issue_codes(ambiguous) == {"deployment_combination_ambiguous"}
    assert set(ambiguous["detected"]["candidate_combination_ids"]) == {
        "deploy_qwen2_5_oft",
        "deploy_qwen3_oft",
    }
    assert selected["valid"] is True
    assert selected["detected"]["combination_id"] == "deploy_qwen2_5_oft"


def test_legacy_weight_file_uses_run_config_and_never_reads_weight_bytes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    weights = run_dir / "checkpoints" / "steps_100_pytorch_model.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"this-is-not-a-pickle")
    base_model = make_vlm_dir(tmp_path / "Qwen2.5-VL-3B-Instruct", "qwen2_5_vl")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {"framework": {"name": "QwenOFT", "qwenvl": {"base_vlm": str(base_model)}}, "trainer": {}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "dataset_statistics.json").write_text("{}", encoding="utf-8")

    result = inspect_checkpoint(weights, repo_root=tmp_path)

    assert result["valid"] is True
    assert result["checkpoint"]["format"] == "legacy_file"
    assert result["checkpoint"]["weights_path"] == str(weights.resolve())
    assert result["detected"]["combination_id"] == "deploy_qwen2_5_oft"


def test_legacy_neurovla_resolves_project_relative_vlm_and_persists_override(tmp_path: Path) -> None:
    project = tmp_path / "NeuroVLA"
    base_model = make_vlm_dir(
        project / "playground" / "Pretrained_models" / "Qwen2.5-VL-3B-Instruct",
        "qwen2_5_vl",
    )
    run_dir = project / "playground" / "DeploymentBundles" / "offline" / "run"
    weights = run_dir / "checkpoints" / "steps_200000_pytorch_model.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"static-inspection-only")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "framework": {
                    "name": "NeuroVLA",
                    "qwenvl": {
                        "base_vlm": "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct",
                        "attn_implementation": "flash_attention_2",
                    },
                },
                "trainer": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "dataset_statistics.json").write_text("{}", encoding="utf-8")

    inspected = inspect_checkpoint(weights, repo_root=tmp_path / "ui")
    resolved = resolve_deployment(
        weights,
        repo_root=tmp_path / "ui",
        python_executable="/env/bin/python",
        parameters={"precision": "bf16", "attention_backend": "auto"},
    )

    assert inspected["valid"] is True
    assert inspected["checkpoint"]["runtime_dependency_path"] == str(base_model.resolve())
    assert resolved["resolved_parameters"]["base_vlm_path"] == str(base_model.resolve())
    assert resolved["command"][-4:] == [
        "--base-vlm-path",
        str(base_model.resolve()),
        "--attention-backend",
        "auto",
    ]


def test_world_model_checkpoint_uses_backend_and_local_dependency(tmp_path: Path) -> None:
    base_model = tmp_path / "Cosmos-Predict2.5-2B-diffusers"
    base_model.mkdir()
    checkpoint = make_generic_checkpoint(
        tmp_path / "checkpoint",
        {
            "name": "WorldModelVLA",
            "world_model": {
                "backend": "cosmos2-diffusers",
                "backbone": "cosmos2.5",
                "checkpoint_path": str(base_model),
            },
        },
    )

    result = inspect_checkpoint(checkpoint, repo_root=tmp_path)

    assert result["valid"] is True
    assert result["detected"]["backbone"] == "cosmos2_5"
    assert result["detected"]["combination_id"] == "deploy_world_model_cosmos2_5"
    assert result["checkpoint"]["runtime_dependency_path"] == str(base_model.resolve())


def test_cosmos_policy_contract_and_command_resolution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "cosmos-policy"
    checkpoint.mkdir()
    (checkpoint / "cosmos_dit.pt").write_bytes(b"not-loaded")
    (checkpoint / "libero_t5_embeddings.pkl").write_bytes(b"not-unpickled")
    (checkpoint / "libero_dataset_statistics.json").write_text(
        json.dumps({"actions_min": [0], "actions_max": [1]}), encoding="utf-8"
    )
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "cosmos-policy"}), encoding="utf-8")
    pretrained = tmp_path / "Cosmos-Predict2-2B-Video2World"
    pretrained.mkdir()

    inspected = inspect_checkpoint(checkpoint, repo_root=tmp_path)
    resolved = resolve_deployment(
        checkpoint,
        repo_root=tmp_path,
        python_executable="/env/bin/python",
        parameters={"pretrained_dir": str(pretrained), "port": 12000, "idle_timeout_seconds": -1},
    )

    assert inspected["valid"] is True
    assert inspected["checkpoint"]["format"] == "cosmos_policy"
    assert inspected["required_parameters"] == ["pretrained_dir"]
    assert inspected["detected"]["combination_id"] == "deploy_cosmos_policy"
    assert resolved["valid"] is True
    assert resolved["adapter_id"] == "cosmos_policy_websocket"
    assert resolved["environment"] == {}
    assert resolved["command"] == [
        "/env/bin/python",
        str(tmp_path / "deployment/model_server/server_policy_cosmos.py"),
        "--ckpt_dir",
        str(checkpoint.resolve()),
        "--pretrained_dir",
        str(pretrained.resolve()),
        "--port",
        "12000",
        "--idle_timeout",
        "-1",
    ]


def test_missing_cosmos_files_and_rl_checkpoints_are_blocked(tmp_path: Path) -> None:
    incomplete = tmp_path / "cosmos"
    incomplete.mkdir()
    (incomplete / "cosmos_dit.pt").write_bytes(b"weights")
    (incomplete / "config.json").write_text(json.dumps({"model_type": "cosmos-policy"}), encoding="utf-8")
    rl_checkpoint = tmp_path / "rl_iter_00010"
    rl_checkpoint.mkdir()
    (rl_checkpoint / "encoder.pt").write_bytes(b"encoder")
    (rl_checkpoint / "actor.pt").write_bytes(b"actor")

    incomplete_result = inspect_checkpoint(incomplete, repo_root=tmp_path)
    rl_result = inspect_checkpoint(rl_checkpoint, repo_root=tmp_path)

    assert incomplete_result["valid"] is False
    assert {"cosmos_t5_embeddings_missing", "cosmos_dataset_statistics_missing"} <= issue_codes(incomplete_result)
    assert rl_result["valid"] is False
    assert issue_codes(rl_result) == {"rl_checkpoint_not_deployable"}
    assert rl_result["candidates"] == []


def test_missing_static_metadata_and_unwired_framework_are_blocking(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unsupported"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "framework_config.yaml").write_text(
        yaml.safe_dump({"framework": {"name": "LlamaPi05"}}), encoding="utf-8"
    )

    result = inspect_checkpoint(checkpoint, repo_root=tmp_path)

    assert result["valid"] is False
    assert {"dataset_statistics_missing", "deployment_adapter_not_wired"} <= issue_codes(result)
    assert result["candidates"] == []


def test_command_builder_rejects_secrets_and_returns_persistable_metadata(tmp_path: Path) -> None:
    result = build_adapter_command(
        "base_framework_websocket",
        tmp_path / "checkpoint",
        combination_id="deploy_qwen2_5_oft",
        backbone_id="qwen2_5_vl",
        action_head_id="mlp_regression",
        python_executable="/env/bin/python",
        repo_root=tmp_path,
        parameters={"port": 10093, "idle_timeout_seconds": 30, "precision": "fp32"},
    )
    rejected = build_adapter_command(
        "base_framework_websocket",
        tmp_path / "checkpoint",
        repo_root=tmp_path,
        parameters={"api_key": "must-not-leak"},
    )

    assert result["valid"] is True
    assert result["command"] == [
        "/env/bin/python",
        str(tmp_path / "deployment/model_server/server_policy.py"),
        "--ckpt_path",
        str(tmp_path / "checkpoint"),
        "--port",
        "10093",
        "--idle_timeout",
        "30",
    ]
    assert result["startup_timeout_seconds"] == 900
    assert result["environment"] == {}
    assert rejected["valid"] is False
    assert rejected["command"] == []
    assert "must-not-leak" not in json.dumps(rejected)
    assert issue_codes(rejected) == {"sensitive_adapter_parameter_rejected"}
