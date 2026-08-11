from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from alphabrain_ui.app import create_app
from alphabrain_ui.database import (
    AuditEvent,
    Checkpoint,
    Experiment,
    ExperimentStage,
    Job,
    ModelDeployment,
    utcnow,
)
from alphabrain_ui.gpu import GPUInfo
from alphabrain_ui.runtime import RuntimeConfig


class BusyVisibleGPUMonitor:
    """Expose GPUs to preflight while keeping scheduler work queued."""

    available = True
    error = ""

    def snapshot(self, reservations=None):  # type: ignore[no-untyped-def]
        reservations = reservations or {}
        return [
            GPUInfo(
                index=index,
                uuid=f"GPU-{index}",
                name=f"Fake GPU {index}",
                memory_total_bytes=24 * 1024**3,
                memory_used_bytes=0,
                memory_free_bytes=24 * 1024**3,
                utilization_percent=0,
                temperature_c=30,
                processes=[],
                reserved_by_job_id=reservations.get(index),
                available=False,
            )
            for index in range(2)
        ]


def make_app(tmp_path: Path):
    return create_app(
        RuntimeConfig(
            repo_root=Path(__file__).resolve().parents[2],
            state_dir=tmp_path,
            database_path=tmp_path / "ui.sqlite3",
            frontend_dist=tmp_path / "missing-dist",
            scheduler_interval=0.01,
            stop_grace_seconds=0,
        )
    )


def make_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"preflight-never-deserializes-this")
    (path / "framework_config.yaml").write_text(
        yaml.safe_dump({"framework": {"name": "QwenOFT", "qwenvl": {}}, "trainer": {}}),
        encoding="utf-8",
    )
    (path / "dataset_statistics.json").write_text("{}", encoding="utf-8")
    embedded = path / "vlm_pretrained"
    embedded.mkdir()
    (embedded / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}), encoding="utf-8")
    (embedded / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    return path


def index_checkpoint(app, owner_id: str, checkpoint_path: Path) -> str:  # type: ignore[no-untyped-def]
    with app.state.database.session() as db:
        experiment = Experiment(
            owner_id=owner_id,
            name="trained-model",
            family="imitation_learning",
            status="completed",
            spec={},
            resolved={},
        )
        db.add(experiment)
        db.flush()
        checkpoint = Checkpoint(
            experiment_id=experiment.id,
            path=str(checkpoint_path.resolve()),
            name=checkpoint_path.name,
            is_complete=True,
            is_resumable=False,
        )
        db.add(checkpoint)
        db.flush()
        return checkpoint.id


def deployment_payload(checkpoint_id: str, *, port: int | None = None) -> dict:
    endpoint = {
        "scope": "local",
        "advertised_host": "127.0.0.1",
        "idle_timeout_seconds": 1800,
    }
    if port is not None:
        endpoint["port"] = port
    return {
        "name": "LIBERO inference",
        "checkpoint_source": {"kind": "indexed", "checkpoint_id": checkpoint_id},
        "combination_id": "deploy_qwen2_5_oft",
        "resources": {"strategy": "auto", "gpu_count": 1, "gpu_ids": []},
        "endpoint": endpoint,
        "parameters": {"precision": "fp32"},
    }


def seed_deployment(
    app,  # type: ignore[no-untyped-def]
    owner_id: str,
    *,
    status: str = "queued",
    port: int | None = None,
    log_path: Path | None = None,
) -> str:
    with app.state.database.session() as db:
        deployment = ModelDeployment(
            owner_id=owner_id,
            name="seeded",
            checkpoint_path=str(app.state.config.state_dir / "checkpoint"),
            combination_id="deploy_qwen2_5_oft",
            adapter_id="base_framework_websocket",
            backbone_id="qwen2_5_vl",
            action_head_id="mlp_regression",
            status=status,
            requested_gpu_count=1,
            requested_gpu_ids=[],
            assigned_gpu_ids=[],
            parameters={"precision": "fp32"},
            port=port,
            api_key_hash="a" * 64,
            api_key_prefix="ab_seed",
            log_path=str(log_path or app.state.config.state_dir / "seeded.log"),
        )
        db.add(deployment)
        db.flush()
        return deployment.id


def test_deployment_preflight_create_and_key_is_returned_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", BusyVisibleGPUMonitor)
    monkeypatch.setattr(
        "alphabrain_ui.deployment_preflight.probe_model_server_python",
        lambda *_args, **_kwargs: {"ok": True, "errors": {}},
    )
    app = make_app(tmp_path)
    checkpoint_path = make_checkpoint(tmp_path / "checkpoint")
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        checkpoint_id = index_checkpoint(app, setup.json()["user"]["id"], checkpoint_path)
        payload = deployment_payload(checkpoint_id)

        capabilities = client.get("/api/v1/deployment/capabilities")
        assert capabilities.status_code == 200
        assert {row["id"] for row in capabilities.json()["catalog"]["adapters"]} == {
            "base_framework_websocket",
            "lerobot_pi05_websocket",
            "cosmos_policy_websocket",
        }
        preflight = client.post("/api/v1/deployments/preflight", json=payload)
        assert preflight.status_code == 200, preflight.text
        assert preflight.json()["ok"] is True, preflight.text
        assert "api_key" not in json.dumps(preflight.json()).lower()

        created = client.post("/api/v1/deployments", json=payload)
        assert created.status_code == 200, created.text
        body = created.json()
        raw_key = body["api_key"]
        deployment_id = body["deployment"]["id"]
        assert raw_key.startswith("ab_")
        assert body["deployment"]["status"] == "queued"
        assert body["deployment"]["endpoint"]["scope"] == "local"
        assert body["deployment"]["parameters"] == {"precision": "fp32"}
        assert "api_key_hash" not in body["deployment"]

        detail = client.get(f"/api/v1/deployments/{deployment_id}").json()
        listed = client.get("/api/v1/deployments").json()
        assert raw_key not in json.dumps(detail)
        assert raw_key not in json.dumps(listed)
        assert "api_key_hash" not in detail
        assert "api_key_hash" not in listed[0]
        with app.state.database.session() as db:
            row = db.get(ModelDeployment, deployment_id)
            assert row.api_key_hash == hashlib.sha256(raw_key.encode()).hexdigest()
            assert raw_key not in json.dumps(row.command)
            assert raw_key not in json.dumps(row.parameters)
            audit = db.query(AuditEvent).filter(AuditEvent.target_id == deployment_id).all()
            assert raw_key not in json.dumps([item.detail for item in audit])


def test_deployment_preflight_blocks_reserved_port_and_missing_runtime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", BusyVisibleGPUMonitor)
    monkeypatch.setattr(
        "alphabrain_ui.deployment_preflight.probe_model_server_python",
        lambda *_args, **_kwargs: {"ok": False, "errors": {"torch": "ModuleNotFoundError"}},
    )
    app = make_app(tmp_path)
    checkpoint_path = make_checkpoint(tmp_path / "checkpoint")
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        owner_id = setup.json()["user"]["id"]
        checkpoint_id = index_checkpoint(app, owner_id, checkpoint_path)
        seed_deployment(app, owner_id, port=18443)
        response = client.post("/api/v1/deployments/preflight", json=deployment_payload(checkpoint_id, port=18443))
        assert response.status_code == 200
        assert response.json()["ok"] is False
        codes = {item["code"] for item in response.json()["issues"]}
        assert {"deployment_port_reserved", "model_server_dependencies_missing"} <= codes
        rejected = client.post("/api/v1/deployments", json=deployment_payload(checkpoint_id, port=18443))
        assert rejected.status_code == 422


def test_global_queue_positions_include_training_and_deployments(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", BusyVisibleGPUMonitor)
    app = make_app(tmp_path)
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        owner_id = setup.json()["user"]["id"]
        deployment_id = seed_deployment(app, owner_id)
        with app.state.database.session() as db:
            experiment = Experiment(
                owner_id=owner_id,
                name="queue",
                family="test",
                status="queued",
                spec={},
                resolved={},
            )
            db.add(experiment)
            db.flush()
            stage = ExperimentStage(experiment_id=experiment.id, name="train", phase="train", resolved={})
            db.add(stage)
            db.flush()
            job = Job(
                experiment_id=experiment.id,
                stage_id=stage.id,
                owner_id=owner_id,
                status="queued",
                cwd=str(tmp_path),
                output_dir=str(tmp_path / "output"),
                queued_at=utcnow() + timedelta(seconds=1),
            )
            db.add(job)
            db.flush()
            job_id = job.id

        deployment = client.get(f"/api/v1/deployments/{deployment_id}").json()
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert deployment["queue_position"] == 1
        assert job["queue_position"] == 2


def test_lab_users_can_view_all_but_only_mutate_owned_deployments(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", BusyVisibleGPUMonitor)
    app = make_app(tmp_path)
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "lab", "username": "admin", "display_name": "Admin", "password": "password123"},
        )
        admin_id = setup.json()["user"]["id"]
        client.headers["X-CSRF-Token"] = client.cookies["alphabrain_csrf"]
        researcher = client.post(
            "/api/v1/users",
            json={
                "username": "researcher",
                "display_name": "Researcher",
                "password": "password123",
                "role": "researcher",
            },
        ).json()
        admin_deployment = seed_deployment(app, admin_id)
        own_deployment = seed_deployment(app, researcher["id"])
        assert client.post("/api/v1/auth/logout").status_code == 200
        client.headers.pop("X-CSRF-Token")
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "researcher", "password": "password123"},
        )
        assert login.status_code == 200
        client.headers["X-CSRF-Token"] = client.cookies["alphabrain_csrf"]

        assert len(client.get("/api/v1/deployments").json()) == 2
        assert client.post(f"/api/v1/deployments/{admin_deployment}/cancel").status_code == 403
        assert client.post(f"/api/v1/deployments/{own_deployment}/cancel").status_code == 200
        assert client.post(f"/api/v1/deployments/{admin_deployment}/force-kill").status_code == 403


def test_deployment_events_and_active_deployment_guards(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", BusyVisibleGPUMonitor)
    app = make_app(tmp_path)
    log_path = tmp_path / "deployment.log"
    log_path.write_text("model ready\n", encoding="utf-8")
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        owner_id = setup.json()["user"]["id"]
        stopped_id = seed_deployment(app, owner_id, status="stopped", log_path=log_path)
        queued_id = seed_deployment(app, owner_id, status="queued")

        events = client.get(f"/api/v1/deployments/{stopped_id}/events")
        assert events.status_code == 200
        assert "model ready" in events.text
        assert '"type": "status"' in events.text
        assert "event: end" in events.text
        switch = client.patch(
            "/api/v1/settings",
            json={"deployment_mode": "lab", "admin_password": "password123"},
        )
        assert switch.status_code == 409
        assert client.post(f"/api/v1/deployments/{queued_id}/cancel").status_code == 200
