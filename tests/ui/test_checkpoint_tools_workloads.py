from __future__ import annotations

import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from alphabrain_ui.app import create_app
from alphabrain_ui.checkpoint_tools import discover_lora_bundle
from alphabrain_ui.database import (
    Checkpoint,
    Experiment,
    ExperimentStage,
    Job,
    UtilityRun,
)
from alphabrain_ui.runtime import RuntimeConfig
from alphabrain_ui.utility_worker import archive_path

REPO = Path(__file__).resolve().parents[2]


def make_app(tmp_path: Path):  # type: ignore[no-untyped-def]
    return create_app(
        RuntimeConfig(
            repo_root=REPO,
            state_dir=tmp_path / "state",
            database_path=tmp_path / "state/ui.sqlite3",
            frontend_dist=tmp_path / "missing-dist",
            scheduler_interval=60,
            stop_grace_seconds=0,
        )
    )


def setup(client: TestClient, results_root: Path) -> str:
    response = client.post(
        "/api/v1/setup",
        json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
    )
    assert response.status_code == 200
    assert client.patch(
        "/api/v1/settings", json={"results_roots": [str(results_root)]}
    ).status_code == 200
    return response.json()["user"]["id"]


def seed_lora_checkpoint(app, owner_id: str, results_root: Path) -> tuple[str, Path]:  # type: ignore[no-untyped-def]
    run_root = results_root / "run"
    checkpoint_root = run_root / "checkpoints"
    adapter = checkpoint_root / "task_4_steps_500_lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint_root / "task_4_steps_500_action_model.pt").write_bytes(b"action")
    with app.state.database.session() as db:
        experiment = Experiment(
            owner_id=owner_id,
            name="LoRA run",
            family="continual_learning",
            status="completed",
            spec={},
            resolved={},
        )
        db.add(experiment)
        db.flush()
        stage = ExperimentStage(
            experiment_id=experiment.id,
            name="train",
            phase="train",
            position=0,
            resolved={},
        )
        db.add(stage)
        db.flush()
        job = Job(
            experiment_id=experiment.id,
            stage_id=stage.id,
            owner_id=owner_id,
            status="completed",
            requested_gpu_count=1,
            requested_gpu_ids=[],
            assigned_gpu_ids=[],
            command=[],
            environment={},
            cwd=str(REPO),
            output_dir=str(run_root),
            log_path=str(run_root / "train.log"),
            metrics_path=str(run_root / "metrics.jsonl"),
        )
        db.add(job)
        db.flush()
        checkpoint = Checkpoint(
            experiment_id=experiment.id,
            job_id=job.id,
            path=str(adapter),
            name=adapter.name,
            is_complete=False,
            is_resumable=False,
        )
        db.add(checkpoint)
        db.flush()
        return checkpoint.id, adapter


def test_lora_bundle_discovery_and_managed_merge_queue(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = setup(client, results_root)
        checkpoint_id, adapter = seed_lora_checkpoint(app, owner_id, results_root)
        discovered = discover_lora_bundle(adapter)
        assert discovered["available"] is True
        assert discovered["action_model_path"].endswith("_action_model.pt")

        detail = client.get(f"/api/v1/checkpoints/{checkpoint_id}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["tools"]["merge_lora"]["available"] is True

        created = client.post(
            f"/api/v1/checkpoints/{checkpoint_id}/merge-lora",
            json={
                "model": "qwengr00t",
                "output_name": "task_4_steps_500_merged.pt",
                "resources": {"strategy": "auto", "gpu_count": 1, "gpu_ids": []},
            },
        )
        assert created.status_code == 200, created.text
        assert created.json()["queue_class"] == "gpu"
        assert created.json()["status"] == "queued"
        with app.state.database.session() as db:
            run = db.get(UtilityRun, created.json()["id"])
            assert run is not None
            assert run.kind == "merge_lora_checkpoint"
            assert run.command[1:4] == [
                "-m",
                "AlphaBrain.training.trainer_utils.peft.merge_lora_checkpoint",
                "--model",
            ]
            assert Path(run.output_path).parent == adapter.parent
            assert not Path(run.output_path).exists()

        duplicate = client.post(
            f"/api/v1/checkpoints/{checkpoint_id}/merge-lora",
            json={"model": "qwengr00t", "output_name": "task_4_steps_500_merged.pt"},
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["detail"] == "lora_output_already_queued"

        workloads = client.get("/api/v1/workloads").json()
        utility = next(item for item in workloads if item["id"] == created.json()["id"])
        assert utility["kind"] == "utility"
        assert utility["queue_scope"] == "gpu_global"
        assert utility["queue_position"] == 1


def test_completed_training_can_be_deleted_without_removing_output_files(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = setup(client, results_root)
        checkpoint_id, adapter = seed_lora_checkpoint(app, owner_id, results_root)
        training = next(
            item for item in client.get("/api/v1/workloads").json()
            if item["kind"] == "training"
        )
        assert training["can_delete"] is True

        wrong_confirmation = client.request(
            "DELETE",
            f"/api/v1/jobs/{training['id']}",
            json={"confirmation": "wrong"},
        )
        assert wrong_confirmation.status_code == 422

        deleted = client.request(
            "DELETE",
            f"/api/v1/jobs/{training['id']}",
            json={"confirmation": training["name"]},
        )
        assert deleted.status_code == 200
        assert deleted.json() == {"ok": True}
        assert all(
            item["id"] != training["id"]
            for item in client.get("/api/v1/workloads").json()
        )
        assert client.get(f"/api/v1/checkpoints/{checkpoint_id}").status_code == 404
        assert adapter.exists()


def test_active_training_cannot_be_deleted(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = setup(client, results_root)
        seed_lora_checkpoint(app, owner_id, results_root)
        with app.state.database.session() as db:
            job = db.execute(select(Job)).scalar_one()
            job.status = "running"
            job_id = job.id

        training = next(
            item for item in client.get("/api/v1/workloads").json()
            if item["id"] == job_id
        )
        assert training["can_delete"] is False
        response = client.request(
            "DELETE",
            f"/api/v1/jobs/{job_id}",
            json={"confirmation": training["name"]},
        )
        assert response.status_code == 409
        assert response.json()["detail"] == "training_must_be_terminal_before_delete"


def test_checkpoint_package_reports_progress_and_serves_zip(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = setup(client, results_root)
        checkpoint_id, adapter = seed_lora_checkpoint(app, owner_id, results_root)
        with app.state.database.session() as db:
            checkpoint = db.get(Checkpoint, checkpoint_id)
            assert checkpoint is not None
            checkpoint.is_complete = True

        created = client.post(f"/api/v1/checkpoints/{checkpoint_id}/package")
        assert created.status_code == 200, created.text
        payload = created.json()
        assert payload["kind"] == "checkpoint_package"
        assert payload["status"] == "queued"
        duplicate = client.post(f"/api/v1/checkpoints/{checkpoint_id}/package")
        assert duplicate.status_code == 409

        with app.state.database.session() as db:
            run = db.get(UtilityRun, payload["id"])
            assert run is not None
            progress_path = Path(run.parameters["_progress_path"])
            archive_path(adapter, Path(run.output_path), run.id, progress_path)
            run.status = "completed"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        assert progress["percent"] == 100
        with zipfile.ZipFile(payload["output_path"]) as archive:
            assert any(name.endswith("adapter_model.safetensors") for name in archive.namelist())

        refreshed = next(
            row for row in client.get("/api/v1/utilities").json()
            if row["id"] == payload["id"]
        )
        assert refreshed["progress"]["percent"] == 100
        downloaded = client.get(f"/api/v1/utilities/{payload['id']}/output")
        assert downloaded.status_code == 200
        assert downloaded.headers["content-type"] == "application/zip"


def test_completed_training_can_be_packaged_in_background(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = setup(client, results_root)
        seed_lora_checkpoint(app, owner_id, results_root)
        training = next(
            item for item in client.get("/api/v1/workloads").json()
            if item["kind"] == "training"
        )
        assert training["can_package"] is True

        created = client.post(f"/api/v1/jobs/{training['id']}/package")
        assert created.status_code == 200, created.text
        payload = created.json()
        assert payload["kind"] == "training_package"
        assert payload["parameters"]["job_id"] == training["id"]

        refreshed = next(
            item for item in client.get("/api/v1/workloads").json()
            if item["id"] == training["id"]
        )
        assert refreshed["package_run"]["id"] == payload["id"]


def test_model_publication_rejects_checkpoint_outside_results_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed-results"
    outside = tmp_path / "outside" / "checkpoint"
    outside.mkdir(parents=True)
    (outside / "model.safetensors").write_bytes(b"weights")
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = setup(client, allowed_root)
        with app.state.database.session() as db:
            experiment = Experiment(
                owner_id=owner_id,
                name="outside",
                family="imitation_learning",
                status="completed",
                spec={},
                resolved={},
            )
            db.add(experiment)
            db.flush()
            checkpoint = Checkpoint(
                experiment_id=experiment.id,
                path=str(outside),
                name="outside",
                is_complete=True,
                is_resumable=False,
            )
            db.add(checkpoint)
            db.flush()
            checkpoint_id = checkpoint.id
        assert client.put(
            "/api/v1/users/me/huggingface/publish-token", json={"token": "hf_publish_secret_123"}
        ).status_code == 200
        response = client.post(
            "/api/v1/model-publications",
            json={"checkpoint_id": checkpoint_id, "repo_id": "lab/model", "revision": "main"},
        )
        assert response.status_code == 403
        assert response.json()["detail"] == "checkpoint_path_outside_results_roots"
