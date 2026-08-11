from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from alphabrain_ui.capabilities import get_catalog
from alphabrain_ui.configuration import resolve_experiment
from alphabrain_ui.launchers import build_launch_plan

REPO = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_finetune_snapshot_is_immutable_and_does_not_edit_source(tmp_path: Path) -> None:
    source = REPO / "configs/finetune_config.yaml"
    before = digest(source)
    spec = {
        "run_id": "ui-smoke",
        "training": {"family": "imitation_learning", "mode": "qwen_oft"},
        "dataset": {"id": "libero_goal", "root": "/datasets/libero"},
        "resources": {"gpu_count": 2},
        "parameters": {"max_train_steps": 10, "batch_size": 1},
    }
    stages = build_launch_plan(REPO, tmp_path, "ui-smoke", spec, snapshot_key="exp", write_snapshots=True)
    assert len(stages) == 1
    snapshot = Path(stages[0].config_snapshot_path)
    loaded = yaml.safe_load(snapshot.read_text())
    assert loaded["modes"]["qwen_oft"]["run_id"] == "ui-smoke"
    assert digest(source) == before


def test_rl_full_pipeline_has_dependency_and_known_outputs(tmp_path: Path) -> None:
    spec = {
        "run_id": "rlt-smoke",
        "training": {"family": "rl_token", "track": "rlt_a", "algorithm": "ppo", "full_pipeline": True},
        "checkpoint": "/models/vla",
        "resources": {"gpu_count": 1},
    }
    stages = build_launch_plan(REPO, tmp_path, "rlt-smoke", spec, snapshot_key="rl", write_snapshots=True)
    assert [stage.phase for stage in stages] == ["rl_pretrain", "rl_ppo"]
    assert stages[1].dependency_position == 0
    assert stages[1].environment["ENCODER_PATH"].endswith("checkpoints/pretrain_best/encoder.pt")
    assert all(stage.environment["ALPHABRAIN_UI_LAUNCH"] == "1" for stage in stages)


def test_capability_snapshot_launches_exact_resolved_config(tmp_path: Path) -> None:
    pretrained = tmp_path / "pretrained"
    (pretrained / "Qwen2.5-VL-3B-Instruct").mkdir(parents=True)
    data = tmp_path / "libero"
    data.mkdir()
    spec = {
        "name": "capability-smoke",
        "backbone_id": "qwen2_5_vl",
        "action_head_id": "mlp_regression",
        "method_id": "imitation_learning",
        "dataset_id": "libero",
        "resources": {"strategy": "auto", "gpu_count": 1, "gpu_ids": []},
        "parameters": {
            "run_id": "capability-smoke",
            "output_root_dir": str(tmp_path / "outputs" / ".." / "outputs"),
        },
        "expert_overrides": {},
    }
    snapshot = resolve_experiment(
        spec,
        REPO,
        environment={"PRETRAINED_MODELS_DIR": str(pretrained), "LIBERO_DATA_ROOT": str(data)},
    )
    stages = build_launch_plan(
        REPO,
        tmp_path,
        "capability-smoke",
        snapshot,
        snapshot_key="canonical",
        write_snapshots=True,
    )
    assert len(stages) == 1
    assert stages[0].command[:3] == ["python", "-m", "accelerate.commands.launch"]
    config = yaml.safe_load(Path(stages[0].config_snapshot_path).read_text())
    assert config["run_id"] == "capability-smoke"
    assert "modes" not in config
    assert stages[0].output_dir == str(tmp_path / "outputs" / "capability-smoke")


def test_every_runnable_registry_combination_has_a_launch_plan(tmp_path: Path) -> None:
    environment = {
        "PRETRAINED_MODELS_DIR": str(tmp_path / "models"),
        "LIBERO_DATA_ROOT": str(tmp_path / "data"),
        "LEROBOT_LIBERO_DATA_DIR": str(tmp_path / "data"),
    }
    checkpoint_methods = {
        "r_stdp",
        "rlt_td3",
        "rlt_a_td3",
        "rlt_a_grpo",
        "vanilla_vla_ppo",
    }
    combinations = get_catalog(include_experimental=True)["combinations"]
    assert combinations
    for index, combination in enumerate(combinations):
        run_id = f"catalog-{index}"
        spec = {
            "architecture": {
                "backbone": combination["backbone"],
                "action_head": combination["action_head"],
            },
            "training": {"method": combination["method"]},
            "dataset": {"id": combination["datasets"][0]},
            "parameters": {
                "run_id": run_id,
                "output_root_dir": str(tmp_path / "outputs"),
            },
            "resources": {
                "allocation": "auto",
                "num_gpus": max(1, int(combination.get("min_gpus", 1))),
                "gpu_ids": [],
                "main_process_port": 31000 + index,
                "deepspeed_config": "configs/deepspeed/accelerate_zero2.yaml",
            },
            "expert_overrides": {},
        }
        if combination["status"] == "experimental":
            spec["experimental"] = {"enabled": True, "risk_acknowledged": True}
        if combination["method"] in checkpoint_methods:
            checkpoint = str(tmp_path / "checkpoint")
            spec["training"]["checkpoint"] = checkpoint
            spec["parameters"]["pretrained_checkpoint"] = checkpoint
        snapshot = resolve_experiment(spec, REPO, environment=environment)
        stages = build_launch_plan(
            REPO,
            tmp_path,
            run_id,
            snapshot,
            snapshot_key=run_id,
            write_snapshots=True,
        )
        assert stages, combination["id"]
        assert all(stage.command for stage in stages), combination["id"]
