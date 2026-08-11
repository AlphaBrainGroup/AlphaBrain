from __future__ import annotations

import asyncio
import hashlib
import io
import json
import stat
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from alphabrain_ui.app import create_app
from alphabrain_ui.database import (
    AuditEvent,
    Checkpoint,
    Experiment,
    InferenceRun,
    ModelDeployment,
    ModelPublication,
    UtilityRun,
)
from alphabrain_ui.deployments import DeploymentManager
from alphabrain_ui.inference import InferenceTransportError
from alphabrain_ui.runtime import RuntimeConfig
from alphabrain_ui.utilities import UtilityManager


REPO = Path(__file__).resolve().parents[2]


def make_app(tmp_path: Path):
    return create_app(
        RuntimeConfig(
            repo_root=REPO,
            state_dir=tmp_path / "state",
            database_path=tmp_path / "state/ui.sqlite3",
            frontend_dist=tmp_path / "missing-dist",
            scheduler_interval=0.01,
            stop_grace_seconds=0,
        )
    )


def setup_personal(client: TestClient) -> str:
    response = client.post(
        "/api/v1/setup",
        json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
    )
    assert response.status_code == 200
    return response.json()["user"]["id"]


def seed_deployment(app, owner_id: str, *, status: str = "running", port: int = 19431) -> str:  # type: ignore[no-untyped-def]
    with app.state.database.session() as db:
        row = ModelDeployment(
            owner_id=owner_id,
            name="Playground model",
            checkpoint_path=str(app.state.config.state_dir / "checkpoint"),
            combination_id="deploy_qwen2_5_oft",
            adapter_id="base_framework_websocket",
            backbone_id="qwen2_5_vl",
            action_head_id="mlp_regression",
            status=status,
            requested_gpu_count=1,
            requested_gpu_ids=[],
            assigned_gpu_ids=[0] if status == "running" else [],
            parameters={"precision": "fp32"},
            bind_host="127.0.0.1",
            advertised_host="127.0.0.1",
            port=port,
            api_key_hash=hashlib.sha256(b"public-key").hexdigest(),
            api_key_prefix="ab_public",
            command=[],
            environment={},
            log_path=str(app.state.config.state_dir / "deployment.log"),
        )
        db.add(row)
        db.flush()
        return row.id


def png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (5, 4), color=(12, 34, 56)).save(stream, format="PNG")
    return stream.getvalue()


def test_controller_secret_command_redaction_and_delete_lifecycle(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = setup_personal(client)
        deployment_id = seed_deployment(app, owner_id, status="stopped")
        manager: DeploymentManager = app.state.deployment_manager
        manager.create_controller_key(deployment_id)
        raw = manager.controller_key(deployment_id)
        assert raw and raw.startswith("ab_ctl_")
        secret_path = tmp_path / "state/secrets/deployment-controller" / deployment_id
        assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600

        with app.state.database.session() as db:
            deployment = db.get(ModelDeployment, deployment_id)
            assert deployment is not None
            executable, persisted, _timeout = manager._runtime_command(deployment, "python")
            controller_digest = hashlib.sha256(raw.encode()).hexdigest()
            assert executable[executable.index("--controller-api-key-sha256") + 1] == controller_digest
            assert persisted[persisted.index("--controller-api-key-sha256") + 1] == "[REDACTED]"
            assert raw not in json.dumps(executable)
            assert raw not in json.dumps(persisted)
            assert controller_digest not in json.dumps(persisted)
            assert raw not in json.dumps(deployment.command)
            assert raw not in json.dumps(deployment.environment)

        detail = client.get(f"/api/v1/deployments/{deployment_id}")
        assert detail.status_code == 200
        assert detail.json()["managed_inference_available"] is True
        assert raw not in detail.text
        deleted = client.request(
            "DELETE",
            f"/api/v1/deployments/{deployment_id}",
            json={"confirmation": "Playground model"},
        )
        assert deleted.status_code == 200
        assert manager.controller_key(deployment_id) is None
        assert not secret_path.exists()


def test_multipart_playground_persists_summary_output_and_optional_inputs(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_infer(*, port: int, controller_key: str, request):  # type: ignore[no-untyped-def]
        captured.update({"port": port, "controller_key": controller_key, "request": request})
        return (
            {"ok": True, "request_id": request.request_id, "actions": [[0.1, 0.2], [0.3, 0.4]]},
            {"framework": "QwenOFT", "action_horizon": 2},
            12.5,
        )

    monkeypatch.setattr("alphabrain_ui.app.infer_via_websocket", fake_infer)
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = setup_personal(client)
        deployment_id = seed_deployment(app, owner_id)
        app.state.deployment_manager.create_controller_key(deployment_id)
        controller_key = app.state.deployment_manager.controller_key(deployment_id)

        response = client.post(
            f"/api/v1/deployments/{deployment_id}/infer",
            data={
                "instructions": json.dumps(["pick up the cup", "put down the cup"]),
                "states": json.dumps([[1, 2, 3], [4, 5, 6]]),
                "save_inputs": "true",
            },
            files=[("images", ("camera.png", png_bytes(), "image/png"))],
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "completed"
        assert body["batch_size"] == 2
        assert body["image_count"] == 1
        assert body["inputs_saved"] is True
        assert body["output"]["actions"] == [[0.1, 0.2], [0.3, 0.4]]
        assert body["deployment_metadata"] == {"framework": "QwenOFT", "action_horizon": 2}
        assert body["latency_ms"] == 12.5
        assert captured["controller_key"] == controller_key
        assert captured["port"] == 19431
        assert captured["request"].payload["states"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert len(captured["request"].payload["batch_images"]) == 2

        listed = client.get("/api/v1/inference-runs", params={"deployment_id": deployment_id})
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == body["id"]
        assert listed.json()[0]["output"] is None
        detail = client.get(f"/api/v1/inference-runs/{body['id']}")
        assert detail.json()["output"] == body["output"]
        assert controller_key not in response.text + listed.text + detail.text

        with app.state.database.session() as db:
            run = db.get(InferenceRun, body["id"])
            assert run is not None
            assert run.request_summary["image_mode"] == "broadcast"
            assert run.request_summary["images"][0]["sha256"]
            assert "data" not in run.request_summary["images"][0]
            assert run.output_json["actions"] == [[0.1, 0.2], [0.3, 0.4]]
            assert len(run.input_paths) == 1
            saved = Path(run.input_paths[0])
            assert saved.read_bytes() == png_bytes()
            assert stat.S_IMODE(saved.stat().st_mode) == 0o600
            audit = db.query(AuditEvent).filter(AuditEvent.action == "deployment.infer").one()
            assert controller_key not in json.dumps(audit.detail)

        # Deployment deletion removes only the controller credential. The
        # audited inference result remains readable through its SET NULL FK.
        with app.state.database.session() as db:
            deployment = db.get(ModelDeployment, deployment_id)
            assert deployment is not None
            deployment.status = "stopped"
        deleted = client.request(
            "DELETE",
            f"/api/v1/deployments/{deployment_id}",
            json={"confirmation": "Playground model"},
        )
        assert deleted.status_code == 200
        persisted = client.get(f"/api/v1/inference-runs/{body['id']}").json()
        assert persisted["deployment_id"] is None
        assert persisted["output"] == body["output"]


def test_playground_failure_is_persisted_with_stable_error_code(tmp_path: Path, monkeypatch) -> None:
    async def timeout(**_kwargs):  # type: ignore[no-untyped-def]
        raise InferenceTransportError("deployment_inference_timeout", "remote secret detail")

    monkeypatch.setattr("alphabrain_ui.app.infer_via_websocket", timeout)
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = setup_personal(client)
        deployment_id = seed_deployment(app, owner_id)

        # Existing deployments without the UI-only controller key do not gain
        # managed inference access implicitly and do not create run records.
        unavailable = client.post(
            f"/api/v1/deployments/{deployment_id}/infer", data={"instructions": "pick"}
        )
        assert unavailable.status_code == 409
        assert unavailable.json()["detail"] == "deployment_controller_unavailable"
        assert client.get("/api/v1/inference-runs").json() == []

        app.state.deployment_manager.create_controller_key(deployment_id)
        failed = client.post(
            f"/api/v1/deployments/{deployment_id}/infer", data={"instructions": "pick"}
        )
        assert failed.status_code == 504
        assert failed.json()["detail"] == "deployment_inference_timeout"
        assert "remote secret detail" not in failed.text
        runs = client.get("/api/v1/inference-runs").json()
        assert len(runs) == 1
        assert runs[0]["status"] == "timed_out"
        assert runs[0]["error"] == "deployment_inference_timeout"


def seed_checkpoint(app, owner_id: str, path: Path) -> str:  # type: ignore[no-untyped-def]
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"weights")
    with app.state.database.session() as db:
        experiment = Experiment(
            owner_id=owner_id,
            name="publish-source",
            family="imitation_learning",
            status="completed",
            spec={},
            resolved={},
        )
        db.add(experiment)
        db.flush()
        checkpoint = Checkpoint(
            experiment_id=experiment.id,
            path=str(path.resolve()),
            name=path.name,
            size_bytes=7,
            is_complete=True,
            is_resumable=False,
        )
        db.add(checkpoint)
        db.flush()
        return checkpoint.id


def test_model_publication_persists_without_token_in_api_argv_or_sqlite(tmp_path: Path, monkeypatch) -> None:
    async def no_schedule(_self):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(UtilityManager, "_schedule_once", no_schedule)
    app = make_app(tmp_path)
    token = "hf_personal_publish_secret_123"
    with TestClient(app) as client:
        owner_id = setup_personal(client)
        configured_results = tmp_path / "results"
        assert client.patch(
            "/api/v1/settings",
            json={"results_roots": [str(configured_results)]},
        ).status_code == 200
        checkpoint_id = seed_checkpoint(app, owner_id, tmp_path / "results/checkpoint-100")
        missing = client.post(
            "/api/v1/model-publications",
            json={"checkpoint_id": checkpoint_id, "repo_id": "lab/model", "revision": "main"},
        )
        assert missing.status_code == 409
        assert missing.json()["detail"] == "huggingface_publish_token_not_configured"
        assert client.put(
            "/api/v1/users/me/huggingface/publish-token", json={"token": token}
        ).json() == {"configured": True}

        created = client.post(
            "/api/v1/model-publications",
            json={
                "checkpoint_id": checkpoint_id,
                "repo_id": "lab/model",
                "revision": "experiment/v1",
                "private": True,
            },
        )
        assert created.status_code == 200, created.text
        publication = created.json()
        assert publication["status"] == "queued"
        assert publication["repo_id"] == "lab/model"
        assert token not in created.text

        listed = client.get("/api/v1/model-publications")
        detail = client.get(f"/api/v1/model-publications/{publication['id']}")
        assert token not in listed.text + detail.text
        with app.state.database.session() as db:
            stored = db.get(ModelPublication, publication["id"])
            assert stored is not None and stored.utility_run_id
            utility = db.get(UtilityRun, stored.utility_run_id)
            assert utility is not None
            assert utility.status == "queued"
            assert utility.parameters["hf_auth"] == "user_publish"
            assert utility.parameters["hf_user_id"] == owner_id
            persisted = json.dumps({
                "publication": {
                    "metadata": stored.metadata_json,
                    "source": stored.source_path,
                    "error": stored.error,
                },
                "utility": {
                    "command": utility.command,
                    "environment": utility.environment,
                    "parameters": utility.parameters,
                },
            })
            assert token not in persisted
            assert "HF_TOKEN" not in utility.environment
            assert token not in Path(utility.log_path).read_text(encoding="utf-8") if Path(utility.log_path).exists() else True


def test_hf_publish_token_is_injected_only_into_child_environment(tmp_path: Path, monkeypatch) -> None:
    app = make_app(tmp_path)
    token = "hf_child_only_secret_456"
    with app.state.database.session() as db:
        # Seed a minimal local user without starting the API scheduler.
        from alphabrain_ui.database import User

        user = User(username="publisher", display_name="Publisher", role="researcher")
        db.add(user)
        db.flush()
        owner_id = user.id
    app.state.hf_secret_store.set_user_publish_token(owner_id, token)
    run = app.state.utility_manager.create_run(
        owner_id=owner_id,
        kind="publish_huggingface",
        resource_id="lab/model",
        command=["python", "-m", "alphabrain_ui.utility_tasks", "publish-huggingface", "--source", str(tmp_path)],
        cwd=REPO,
        parameters={"hf_auth": "user_publish", "hf_user_id": owner_id},
    )
    with app.state.database.session() as db:
        stored = db.get(UtilityRun, run.id)
        assert stored is not None
        stored.status = "starting"

    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 43210
        returncode = None

        def __init__(self) -> None:
            self.release = asyncio.Event()

        async def wait(self) -> int:
            await self.release.wait()
            self.returncode = 0
            return 0

    fake = FakeProcess()

    async def fake_subprocess(*command, **kwargs):  # type: ignore[no-untyped-def]
        captured["command"] = list(command)
        captured["environment"] = dict(kwargs["env"])
        return fake

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)

    async def scenario() -> None:
        manager: UtilityManager = app.state.utility_manager
        await manager._spawn(run.id, assigned_gpu_ids=[])
        assert captured["environment"]["HF_TOKEN"] == token
        assert token not in json.dumps(captured["command"])
        fake.release.set()
        watchers = list(manager._watchers.values())
        if watchers:
            await asyncio.gather(*watchers)

    asyncio.run(scenario())
    with app.state.database.session() as db:
        stored = db.get(UtilityRun, run.id)
        assert stored is not None
        assert stored.status == "completed"
        assert token not in json.dumps(stored.command)
        assert token not in json.dumps(stored.environment)
        assert token not in json.dumps(stored.parameters)
        assert "HF_TOKEN" not in stored.environment
