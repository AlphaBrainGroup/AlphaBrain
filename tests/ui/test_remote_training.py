from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from alphabrain_ui.database import Database, Experiment, ExperimentStage, GPUReservation, Job, User
from alphabrain_ui.jobs import JobManager
from alphabrain_ui.preflight import run_preflight
from alphabrain_ui.remote_training import (
    RemoteTrainingConfig,
    build_remote_script,
    validate_remote_training_config,
)
from alphabrain_ui.runtime import RuntimeConfig
from alphabrain_ui.services import SettingsService


def test_remote_config_requires_host_repo_and_gpu_ids() -> None:
    with pytest.raises(ValueError, match="remote_training_host_required"):
        validate_remote_training_config(
            RemoteTrainingConfig(enabled=True, repo_root="/srv/AlphaBrain", gpu_ids=(0,))
        )
    with pytest.raises(ValueError, match="remote_training_gpu_ids_required"):
        validate_remote_training_config(
            RemoteTrainingConfig(enabled=True, host="trainer", repo_root="/srv/AlphaBrain")
        )


def test_remote_script_maps_repo_paths_and_keeps_secrets_out_of_argv(tmp_path: Path) -> None:
    repo = tmp_path / "AlphaBrain"
    state = repo / ".alphabrain-ui"
    snapshot = state / "configs" / "run-1" / "config.yaml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(f"output_root_dir: {repo / 'results' / 'training'}\n", encoding="utf-8")

    config = RemoteTrainingConfig(
        enabled=True,
        host="gpu.example.edu",
        user="researcher",
        port=2222,
        repo_root="/srv/AlphaBrain",
        identity_file=str(tmp_path / "id_ed25519"),
        gpu_ids=(0, 1),
        setup_command="source .venv/bin/activate",
    )
    ssh_command, script_bytes = build_remote_script(
        config=config,
        command=["bash", "scripts/run_finetune.sh", str(snapshot)],
        cwd=str(repo),
        environment={"CUDA_VISIBLE_DEVICES": "0,1", "WANDB_API_KEY": "top-secret"},
        local_repo_root=repo,
        local_state_dir=state,
    )
    script = script_bytes.decode("utf-8")

    assert ssh_command[-4:] == ["researcher@gpu.example.edu", "bash", "-s", "--"]
    assert "top-secret" not in " ".join(ssh_command)
    assert "export WANDB_API_KEY=top-secret" in script
    assert "cd /srv/AlphaBrain" in script
    assert "/srv/AlphaBrain/.alphabrain-ui/configs/run-1/config.yaml" in script
    assert str(repo) not in script


def test_remote_config_rejects_unsafe_target_and_duplicate_gpus() -> None:
    with pytest.raises(ValueError, match="remote_training_host_invalid"):
        validate_remote_training_config(
            RemoteTrainingConfig(enabled=True, host="-oProxyCommand=bad", repo_root="/srv/repo", gpu_ids=(0,))
        )
    with pytest.raises(ValueError, match="remote_training_gpu_ids_invalid"):
        validate_remote_training_config(
            RemoteTrainingConfig(enabled=True, host="trainer", repo_root="/srv/repo", gpu_ids=(0, 0))
        )


def test_remote_scheduler_uses_configured_gpus_without_local_nvml(tmp_path: Path) -> None:
    class UnexpectedLocalMonitor:
        def snapshot(self, reservations=None):  # type: ignore[no-untyped-def]
            raise AssertionError("remote training must not inspect local GPUs")

    async def scenario() -> None:
        database = Database(tmp_path / "remote.sqlite3")
        database.create_all()
        with database.session() as db:
            user = User(username="local", display_name="Local", role="administrator", is_local=True)
            db.add(user)
            db.flush()
            experiment = Experiment(
                owner_id=user.id,
                name="remote",
                family="test",
                spec={},
                resolved={},
                status="queued",
            )
            db.add(experiment)
            db.flush()
            stage = ExperimentStage(
                experiment_id=experiment.id,
                name="Train",
                phase="train",
                position=0,
                resolved={},
            )
            db.add(stage)
            db.flush()
            job = Job(
                experiment_id=experiment.id,
                stage_id=stage.id,
                owner_id=user.id,
                status="queued",
                requested_gpu_count=1,
                requested_gpu_ids=[],
                command=["bash", "train.sh"],
                environment={},
                cwd=str(tmp_path),
                output_dir=str(tmp_path / "output"),
                log_path=str(tmp_path / "remote.log"),
                metrics_path=str(tmp_path / "metrics.jsonl"),
            )
            db.add(job)
            db.flush()
            job_id = job.id
            settings = SettingsService(db)
            settings.set("remote_training_enabled", True)
            settings.set("remote_training_host", "gpu.example.edu")
            settings.set("remote_training_repo_root", "/srv/AlphaBrain")
            settings.set("remote_training_gpu_ids", [4, 6])

        manager = JobManager(
            RuntimeConfig(
                repo_root=tmp_path,
                state_dir=tmp_path,
                database_path=tmp_path / "remote.sqlite3",
                frontend_dist=tmp_path / "dist",
            ),
            database,
            UnexpectedLocalMonitor(),  # type: ignore[arg-type]
        )
        spawned: list[str] = []

        async def record_spawn(spawned_job_id: str) -> None:
            spawned.append(spawned_job_id)

        manager._spawn = record_spawn  # type: ignore[method-assign]
        await manager._start_queue_head()

        with database.session() as db:
            persisted = db.get(Job, job_id)
            assert persisted is not None
            assert persisted.status == "starting"
            assert persisted.assigned_gpu_ids == [4]
            assert db.get(GPUReservation, 4).job_id == job_id
        assert spawned == [job_id]

    asyncio.run(scenario())


def test_remote_preflight_does_not_require_local_nvml(tmp_path: Path) -> None:
    class UnexpectedLocalMonitor:
        available = False
        error = "NVML is unavailable"

        def snapshot(self, reservations=None):  # type: ignore[no-untyped-def]
            raise AssertionError("remote preflight must not inspect local GPUs")

    repo_root = Path(__file__).resolve().parents[2]
    pretrained = tmp_path / "pretrained"
    (pretrained / "Qwen2.5-VL-3B-Instruct").mkdir(parents=True)
    dataset = tmp_path / "libero"
    dataset.mkdir()
    spec = {
        "architecture": {"backbone": "qwen2_5_vl", "action_head": "mlp_regression"},
        "training": {"method": "imitation_learning"},
        "dataset": {"id": "libero", "mix": "libero_goal"},
        "resources": {"allocation": "auto", "num_gpus": 1},
        "parameters": {"run_id": "remote-preflight"},
        "expert_overrides": {},
    }
    _resolved, _compatibility, issues, _previews = run_preflight(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        name="remote-preflight",
        spec=spec,
        configured_environment={
            "PRETRAINED_MODELS_DIR": str(pretrained),
            "LIBERO_DATA_ROOT": str(dataset),
        },
        gpu_monitor=UnexpectedLocalMonitor(),  # type: ignore[arg-type]
        results_roots=[str(tmp_path / "results")],
        disk_min_free_gib=0,
        disk_min_free_percent=0,
        include_experimental=False,
        remote_gpu_ids=[4],
    )
    assert "nvml_unavailable" not in {item["code"] for item in issues}
