from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from alphabrain_ui.app import _artifact_token, create_app
from alphabrain_ui.database import (
    Checkpoint,
    DeploymentGPUReservation,
    EvaluationRun,
    Experiment,
    ExperimentStage,
    Job,
    ModelDeployment,
    utcnow,
)
from alphabrain_ui.gpu import GPUInfo
from alphabrain_ui.runtime import RuntimeConfig
from alphabrain_ui.services import SettingsService


class BusyVisibleGPUMonitor:
    available = True
    error = ""

    def snapshot(self, reservations=None):  # type: ignore[no-untyped-def]
        reservations = reservations or {}
        return [
            GPUInfo(
                index=0,
                uuid="GPU-0",
                name="Fake GPU",
                memory_total_bytes=24 * 1024**3,
                memory_used_bytes=0,
                memory_free_bytes=24 * 1024**3,
                utilization_percent=0,
                temperature_c=30,
                processes=[],
                reserved_by_job_id=reservations.get(0),
                available=False,
            )
        ]


def make_app(tmp_path: Path):
    return create_app(
        RuntimeConfig(
            repo_root=Path(__file__).resolve().parents[2],
            state_dir=tmp_path / "state",
            database_path=tmp_path / "state/ui.sqlite3",
            frontend_dist=tmp_path / "missing-dist",
            scheduler_interval=0.01,
            stop_grace_seconds=0,
        )
    )


def make_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"static-only")
    (path / "framework_config.yaml").write_text(
        yaml.safe_dump({"framework": {"name": "QwenOFT", "qwenvl": {}}, "trainer": {}}),
        encoding="utf-8",
    )
    (path / "dataset_statistics.json").write_text("{}", encoding="utf-8")
    embedded = path / "vlm_pretrained"
    embedded.mkdir()
    (embedded / "config.json").write_text('{"model_type":"qwen2_5_vl"}', encoding="utf-8")
    (embedded / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    return path


def make_external_vlm_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"static-only")
    (path / "framework_config.yaml").write_text(
        yaml.safe_dump(
            {
                "framework": {
                    "name": "QwenOFT",
                    "qwenvl": {
                        "base_vlm": "${PRETRAINED_MODELS_DIR}/Qwen2.5-VL-7B-Instruct"
                    },
                },
                "trainer": {},
            }
        ),
        encoding="utf-8",
    )
    (path / "dataset_statistics.json").write_text("{}", encoding="utf-8")
    return path


def make_ambiguous_external_vlm_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"static-only")
    (path / "framework_config.yaml").write_text(
        yaml.safe_dump(
            {
                "framework": {
                    "name": "QwenOFT",
                    "qwenvl": {"base_vlm": "${PRETRAINED_MODELS_DIR}/vlm"},
                },
                "trainer": {},
            }
        ),
        encoding="utf-8",
    )
    (path / "dataset_statistics.json").write_text("{}", encoding="utf-8")
    return path


def index_checkpoint(app, owner_id: str, path: Path) -> str:  # type: ignore[no-untyped-def]
    with app.state.database.session() as db:
        experiment = Experiment(
            owner_id=owner_id,
            name="trained",
            family="imitation_learning",
            status="completed",
            spec={},
            resolved={},
        )
        db.add(experiment)
        db.flush()
        row = Checkpoint(
            experiment_id=experiment.id,
            path=str(path.resolve()),
            name=path.name,
            is_complete=True,
            is_resumable=False,
        )
        db.add(row)
        db.flush()
        return row.id


def evaluation_payload(checkpoint_id: str) -> dict:
    return {
        "name": "LIBERO quick",
        "checkpoint_source": {"kind": "indexed", "checkpoint_id": checkpoint_id},
        "combination_id": "deploy_qwen2_5_oft",
        "benchmark_id": "libero",
        "preset": "quick",
        "suite": "libero_goal",
        "parameters": {},
        "resources": {"strategy": "auto", "gpu_count": 1, "gpu_ids": []},
        "wandb": {
            "enabled": False,
            "mode": "online",
            "project": "AlphaBrain-Evaluation",
            "categories": ["summary", "tasks", "config"],
        },
    }


def configure_libero(app, path: Path) -> None:  # type: ignore[no-untyped-def]
    path.mkdir(parents=True)
    with app.state.database.session() as db:
        SettingsService(db).set(
            "environment",
            {"LIBERO_HOME": str(path), "LIBERO_PYTHON": str(Path(__import__("sys").executable))},
        )


def test_checkpoint_inspection_detects_and_filters_evaluation_combinations(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    checkpoint = make_checkpoint(tmp_path / "checkpoint")
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        checkpoint_id = index_checkpoint(app, setup.json()["user"]["id"], checkpoint)

        response = client.post(
            "/api/v1/evaluations/inspect-checkpoint",
            json={"kind": "indexed", "checkpoint_id": checkpoint_id},
        )

    assert response.status_code == 200
    payload = response.json()
    candidate_ids = {item["combination_id"] for item in payload["candidates"]}
    assert "deploy_qwen2_5_oft" in candidate_ids
    assert payload["detected"]["candidate_combination_ids"]
    assert set(payload["compatible_benchmark_ids"]) <= {"libero", "libero_plus"}


def test_checkpoint_inspection_surfaces_unset_model_directory_before_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("PRETRAINED_MODELS_DIR", raising=False)
    app = make_app(tmp_path)
    checkpoint = make_external_vlm_checkpoint(tmp_path / "external-checkpoint")
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        checkpoint_id = index_checkpoint(app, setup.json()["user"]["id"], checkpoint)

        response = client.post(
            "/api/v1/evaluations/inspect-checkpoint",
            json={"kind": "indexed", "checkpoint_id": checkpoint_id},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert "deploy_qwen2_5_oft" in {item["combination_id"] for item in payload["candidates"]}
    assert "pretrained_path_unresolved" in {item["code"] for item in payload["issues"]}


def test_checkpoint_inspection_rechecks_runtime_after_resolving_ambiguous_combination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("PRETRAINED_MODELS_DIR", raising=False)
    app = make_app(tmp_path)
    checkpoint = make_ambiguous_external_vlm_checkpoint(tmp_path / "ambiguous-checkpoint")
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        checkpoint_id = index_checkpoint(app, setup.json()["user"]["id"], checkpoint)
        source = {"kind": "indexed", "checkpoint_id": checkpoint_id}

        initial = client.post("/api/v1/evaluations/inspect-checkpoint", json=source)
        selected = client.post(
            "/api/v1/evaluations/inspect-checkpoint",
            json={**source, "combination_id": "deploy_qwen2_5_oft"},
        )

    assert initial.status_code == selected.status_code == 200
    initial_payload = initial.json()
    selected_payload = selected.json()
    assert len(initial_payload["candidates"]) > 1
    assert "deployment_combination_ambiguous" in {
        item["code"] for item in initial_payload["issues"]
    }
    assert "pretrained_path_unresolved" not in {
        item["code"] for item in initial_payload["issues"]
    }
    assert selected_payload["detected"]["combination_id"] == "deploy_qwen2_5_oft"
    assert "pretrained_path_unresolved" in {
        item["code"] for item in selected_payload["issues"]
    }


def test_evaluation_catalog_preflight_create_and_global_queue(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", BusyVisibleGPUMonitor)
    monkeypatch.setattr(
        "alphabrain_ui.evaluation_preflight.probe_model_server_python",
        lambda *_args, **_kwargs: {"ok": True, "errors": {}},
    )
    monkeypatch.setattr(
        "alphabrain_ui.evaluation_preflight.probe_benchmark_python",
        lambda *_args, **_kwargs: {"ok": True, "errors": {}},
    )
    app = make_app(tmp_path)
    checkpoint = make_checkpoint(tmp_path / "checkpoint")
    configure_libero(app, tmp_path / "LIBERO")
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        checkpoint_id = index_checkpoint(app, setup.json()["user"]["id"], checkpoint)

        catalog = client.get("/api/v1/evaluation-benchmarks")
        assert catalog.status_code == 200
        assert {row["id"] for row in catalog.json()["benchmarks"]} == {"libero", "robocasa365"}
        libero = next(row for row in catalog.json()["benchmarks"] if row["id"] == "libero")
        assert "deploy_qwen2_5_oft" in libero["compatibility"]["combination_ids"]

        payload = evaluation_payload(checkpoint_id)
        preflight = client.post("/api/v1/evaluations/preflight", json=payload)
        assert preflight.status_code == 200, preflight.text
        assert preflight.json()["ok"] is True, preflight.text
        assert preflight.json()["resolved"]["parameters"]["num_trials"] == 1

        created = client.post("/api/v1/evaluations", json=payload)
        assert created.status_code == 200, created.text
        evaluation = created.json()["evaluation"]
        assert evaluation["status"] == "queued"
        assert evaluation["queue_position"] == 1
        assert evaluation["requested_gpu_count"] == 1
        assert not Path(evaluation["output_dir"]).exists()


def test_managed_deployment_evaluation_reuses_endpoint_without_second_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", BusyVisibleGPUMonitor)
    monkeypatch.setattr(
        "alphabrain_ui.evaluation_preflight.probe_benchmark_python",
        lambda *_args, **_kwargs: {"ok": True, "errors": {}},
    )
    app = make_app(tmp_path)
    configure_libero(app, tmp_path / "LIBERO")
    controller_key = "controller-key-that-never-enters-api-or-database"
    with TestClient(app) as client:
        owner_id = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        ).json()["user"]["id"]
        with app.state.database.session() as db:
            deployment = ModelDeployment(
                owner_id=owner_id,
                name="Managed model",
                checkpoint_path=str(tmp_path / "checkpoint"),
                combination_id="deploy_qwen2_5_oft",
                adapter_id="base_framework_websocket",
                backbone_id="qwen2_5_vl",
                action_head_id="mlp_regression",
                status="running",
                assigned_gpu_ids=[0],
                port=12345,
                api_key_hash="a" * 64,
                api_key_prefix="ab_test",
            )
            db.add(deployment)
            db.flush()
            db.add(DeploymentGPUReservation(gpu_index=0, deployment_id=deployment.id))
            deployment_id = deployment.id
        app.state.deployment_manager.controller_secret_store.set(deployment_id, controller_key)

        # Keep this API test deterministic: the manager lifecycle itself is
        # covered separately, while this verifies the persisted queue contract.
        async def leave_queued(_evaluation_id: str) -> None:
            return None

        monkeypatch.setattr(
            app.state.evaluation_manager, "start_queued_unlocked", leave_queued
        )
        payload = {
            "name": "Managed LIBERO quick",
            "source_kind": "managed_deployment",
            "deployment_id": deployment_id,
            "benchmark_id": "libero",
            "preset": "quick",
            "suite": "libero_goal",
            "parameters": {},
            "resources": {"strategy": "auto", "gpu_count": 1, "gpu_ids": []},
            "wandb": {
                "enabled": False,
                "mode": "online",
                "project": "AlphaBrain-Evaluation",
                "categories": ["summary", "tasks", "config"],
            },
        }
        preflight = client.post("/api/v1/evaluations/preflight", json=payload)
        assert preflight.status_code == 200, preflight.text
        assert preflight.json()["ok"] is True, preflight.text
        resolved = preflight.json()["resolved"]
        assert resolved["reuse_server"] is True
        assert resolved["managed_server"] == {
            "host": "127.0.0.1",
            "port": 12345,
            "assigned_gpu_ids": [0],
        }
        assert resolved["resources"]["gpu_count"] == 0
        assert controller_key not in preflight.text

        created = client.post("/api/v1/evaluations", json=payload)
        assert created.status_code == 200, created.text
        evaluation = created.json()["evaluation"]
        assert evaluation["source_kind"] == "managed_deployment"
        assert evaluation["deployment_id"] == deployment_id
        assert evaluation["requested_gpu_count"] == 0
        assert controller_key not in created.text
        with app.state.database.session() as db:
            stored = db.get(EvaluationRun, evaluation["id"])
            assert stored.environment.get("ALPHABRAIN_POLICY_API_KEY") is None
            assert controller_key not in json.dumps(stored.command)
            assert controller_key not in json.dumps(stored.environment)

        app.state.deployment_manager.controller_secret_store.delete(deployment_id)
        missing_key = client.post("/api/v1/evaluations/preflight", json=payload)
        assert missing_key.status_code == 200
        assert missing_key.json()["ok"] is False
        assert "managed_deployment_controller_key_missing" in {
            item["code"] for item in missing_key.json()["issues"]
        }


def test_evaluation_result_artifacts_compare_and_path_confinement(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", BusyVisibleGPUMonitor)
    app = make_app(tmp_path)
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        owner_id = setup.json()["user"]["id"]
        ids = []
        for index, successes in enumerate((1, 0)):
            output = tmp_path / f"evaluation-{index}"
            video = output / "videos/rollout.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            result = {
                "schema_version": "evaluation-result-v1",
                "status": "completed",
                "benchmark": "libero",
                "checkpoint": str(tmp_path / "checkpoint"),
                "suite": {"name": "libero_goal"},
                "started_at": "2026-07-16T00:00:00Z",
                "finished_at": "2026-07-16T00:00:01Z",
                "duration_seconds": 1.0,
                "summary": {
                    "num_tasks": 1,
                    "num_episodes": 1,
                    "num_successes": successes,
                    "success_rate": float(successes),
                },
                "tasks": [{
                    "id": "0", "name": "pick", "num_episodes": 1,
                    "num_successes": successes, "success_rate": float(successes),
                }],
                "episodes": [{
                    "task_id": "0", "task_name": "pick", "episode_index": 0,
                    "success": bool(successes), "steps": 3, "duration_seconds": 1.0,
                    "video": "videos/rollout.mp4",
                }],
                "videos": [{
                    "path": "videos/rollout.mp4", "task_id": "0", "task_name": "pick",
                    "episode_index": 0, "success": bool(successes),
                }],
                "parameters": {},
                "metadata": {},
                "error": None,
            }
            result_path = output / "evaluation-result-v1.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with app.state.database.session() as db:
                row = EvaluationRun(
                    owner_id=owner_id,
                    name=f"evaluation-{index}",
                    checkpoint_path=str(tmp_path / "checkpoint"),
                    combination_id="deploy_qwen2_5_oft",
                    adapter_id="base_framework_websocket",
                    backbone_id="qwen2_5_vl",
                    action_head_id="mlp_regression",
                    benchmark_id="libero",
                    benchmark_status="verified",
                    preset="quick",
                    suite="libero_goal",
                    parameters={
                        "task_ids": " 0 " if index == 0 else "0",
                        "task_list": "",
                        "task_limit": 1,
                    },
                    model_parameters={},
                    wandb={},
                    status="completed",
                    cwd=str(tmp_path),
                    output_dir=str(output),
                    result_path=str(result_path),
                    result_summary=result["summary"],
                )
                db.add(row)
                db.flush()
                ids.append(row.id)

        detail = client.get(f"/api/v1/evaluations/{ids[0]}")
        assert detail.status_code == 200
        assert detail.json()["result"]["summary"]["success_rate"] == 1.0
        artifacts = client.get(f"/api/v1/evaluations/{ids[0]}/artifacts").json()
        video_artifact = next(item for item in artifacts if item["kind"] == "video")
        streamed = client.get(video_artifact["url"])
        assert streamed.status_code == 200
        assert streamed.content == b"video"

        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        traversal = _artifact_token("../outside.txt")
        assert client.get(f"/api/v1/evaluations/{ids[0]}/artifacts/{traversal}").status_code == 404

        compared = client.post("/api/v1/evaluations/compare", json={"evaluation_ids": ids})
        assert compared.status_code == 200
        assert compared.json()["compatible"] is True
        assert compared.json()["signature"].endswith(
            "task_ids=0 | task_list= | task_limit=1"
        )
        assert [item["result"]["summary"]["success_rate"] for item in compared.json()["items"]] == [1.0, 0.0]

        baseline = {"task_ids": "0", "task_list": "", "task_limit": 1}
        for field, value in (("task_ids", "1"), ("task_list", "pick,place"), ("task_limit", 2)):
            with app.state.database.session() as db:
                row = db.get(EvaluationRun, ids[1])
                row.parameters = {**baseline, field: value}
            mismatched = client.post(
                "/api/v1/evaluations/compare",
                json={"evaluation_ids": ids},
            )
            assert mismatched.status_code == 200
            assert mismatched.json()["compatible"] is False
            assert "signature_mismatch" in mismatched.json()["warnings"]

        with app.state.database.session() as db:
            row = db.get(EvaluationRun, ids[1])
            row.parameters = baseline
            row.status = "failed"
        rejected = client.post("/api/v1/evaluations/compare", json={"evaluation_ids": ids})
        assert rejected.status_code == 409
        assert rejected.json()["detail"] == "evaluation_not_completed"

        with app.state.database.session() as db:
            row = db.get(EvaluationRun, ids[1])
            row.status = "completed"
            result_path = Path(row.result_path)
        partial_result = json.loads(result_path.read_text(encoding="utf-8"))
        partial_result["status"] = "partial"
        partial_result["error"] = "incomplete"
        result_path.write_text(json.dumps(partial_result), encoding="utf-8")
        rejected_partial = client.post(
            "/api/v1/evaluations/compare",
            json={"evaluation_ids": ids},
        )
        assert rejected_partial.status_code == 409
        assert rejected_partial.json()["detail"] == "evaluation_not_completed"


def test_wandb_retry_is_enqueued_in_the_background(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", BusyVisibleGPUMonitor)
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        ).json()["user"]["id"]
        with app.state.database.session() as db:
            evaluation = EvaluationRun(
                owner_id=owner_id,
                name="W&B retry",
                checkpoint_path=str(tmp_path / "checkpoint"),
                combination_id="deploy_qwen2_5_oft",
                adapter_id="base_framework_websocket",
                backbone_id="qwen2_5_vl",
                action_head_id="mlp_regression",
                benchmark_id="libero",
                wandb={"enabled": True, "mode": "offline", "categories": ["summary"]},
                wandb_status="failed",
                status="completed",
                cwd=str(tmp_path),
                output_dir=str(tmp_path / "wandb-evaluation"),
                result_path=str(tmp_path / "wandb-evaluation/evaluation-result-v1.json"),
            )
            db.add(evaluation)
            db.flush()
            evaluation_id = evaluation.id

        task_names: list[str] = []

        async def background_only(_evaluation_id: str, _secret_store=None):  # type: ignore[no-untyped-def]
            task_name = __import__("asyncio").current_task().get_name()
            task_names.append(task_name)
            assert task_name == f"wandb-evaluation-{evaluation_id}"

        monkeypatch.setattr(app.state.evaluation_manager, "retry_upload", background_only)
        response = client.post(f"/api/v1/evaluations/{evaluation_id}/wandb/retry")

        assert response.status_code == 200
        assert response.json()["wandb_status"] == "pending"
        assert task_names == [f"wandb-evaluation-{evaluation_id}"]


def test_global_queue_positions_include_evaluations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", BusyVisibleGPUMonitor)
    app = make_app(tmp_path)
    with TestClient(app) as client:
        owner_id = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        ).json()["user"]["id"]
        with app.state.database.session() as db:
            evaluation = EvaluationRun(
                owner_id=owner_id,
                name="evaluation",
                checkpoint_path=str(tmp_path / "checkpoint"),
                combination_id="deploy_qwen2_5_oft",
                adapter_id="base_framework_websocket",
                backbone_id="qwen2_5_vl",
                action_head_id="mlp_regression",
                benchmark_id="libero",
                status="queued",
                cwd=str(tmp_path),
                output_dir=str(tmp_path / "evaluation"),
                queued_at=utcnow(),
            )
            db.add(evaluation)
            experiment = Experiment(
                owner_id=owner_id, name="train", family="test", status="queued", spec={}, resolved={}
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
                output_dir=str(tmp_path / "training"),
                queued_at=utcnow() + timedelta(seconds=1),
            )
            db.add(job)
            deployment = ModelDeployment(
                owner_id=owner_id,
                name="deployment",
                checkpoint_path=str(tmp_path / "checkpoint"),
                combination_id="deploy_qwen2_5_oft",
                adapter_id="base_framework_websocket",
                backbone_id="qwen2_5_vl",
                action_head_id="mlp_regression",
                status="queued",
                api_key_hash="a" * 64,
                api_key_prefix="ab_test",
                queued_at=utcnow() + timedelta(seconds=2),
            )
            db.add(deployment)
            db.flush()
            evaluation_id, job_id, deployment_id = evaluation.id, job.id, deployment.id

        assert client.get(f"/api/v1/evaluations/{evaluation_id}").json()["queue_position"] == 1
        assert client.get(f"/api/v1/jobs/{job_id}").json()["queue_position"] == 2
        assert client.get(f"/api/v1/deployments/{deployment_id}").json()["queue_position"] == 3
