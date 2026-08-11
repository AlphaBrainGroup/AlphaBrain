from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from alphabrain_ui.app import create_app
from alphabrain_ui.demo import DEMO_DATASET_ID, DEMO_TEMPLATE_ID, demo_template_spec
from alphabrain_ui.runtime import RuntimeConfig


def _wait_for(client: TestClient, path: str, statuses: set[str], timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        response = client.get(path)
        assert response.status_code == 200, response.text
        last = response.json()
        if last.get("status") in statuses:
            return last
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {path}; last payload: {last}")


def test_demo_mode_runs_training_deployment_inference_and_evaluation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    app = create_app(
        RuntimeConfig(
            repo_root=repo_root,
            state_dir=tmp_path,
            database_path=tmp_path / "ui.sqlite3",
            frontend_dist=tmp_path / "missing-dist",
            scheduler_interval=0.05,
            stop_grace_seconds=0,
            demo_mode=True,
        )
    )

    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["demo_mode"] is True
        assert client.get("/api/v1/setup/status").json()["initialized"] is True
        assert len(client.get("/api/v1/gpus").json()["items"]) == 2
        assert client.get("/api/v1/training/target").json() == {
            "mode": "local",
            "gpu_ids": [],
        }
        assert any(row["id"] == DEMO_DATASET_ID for row in client.get("/api/v1/datasets").json())
        assert any(row["id"] == DEMO_TEMPLATE_ID for row in client.get("/api/v1/templates").json())
        registry = client.get("/api/v1/registry").json()["view"]
        assert "toy" in {row["id"] for row in registry["components"]}
        assert "toy_libero_debug" in {row["id"] for row in registry["combinations"]}
        capabilities = client.get("/api/v1/capabilities").json()["catalog"]
        assert "toy" in {row["id"] for row in capabilities["backbones"]}
        assert "toy_libero_debug" in {row["id"] for row in capabilities["combinations"]}

        experiment_request = {"name": "Demo full flow", "spec": demo_template_spec()}
        preflight = client.post("/api/v1/experiments/preflight", json=experiment_request)
        assert preflight.status_code == 200, preflight.text
        assert preflight.json()["can_submit"] is True
        submitted = client.post("/api/v1/experiments/submit", json=experiment_request)
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["jobs"][0]["id"]
        job = _wait_for(client, f"/api/v1/jobs/{job_id}", {"completed", "failed"})
        assert job["status"] == "completed", job

        checkpoints = client.get("/api/v1/checkpoints").json()
        checkpoint = next(row for row in checkpoints if row["job_id"] == job_id)
        assert checkpoint["is_complete"] is True

        deployment_request = {
            "name": "Demo policy",
            "checkpoint_source": {"kind": "indexed", "checkpoint_id": checkpoint["id"]},
            "combination_id": "deploy_qwen2_5_oft",
            "resources": {"strategy": "auto", "gpu_count": 1, "gpu_ids": []},
            "endpoint": {"scope": "local", "idle_timeout_seconds": -1},
            "parameters": {},
        }
        deployment_preflight = client.post("/api/v1/deployments/preflight", json=deployment_request)
        assert deployment_preflight.status_code == 200, deployment_preflight.text
        assert deployment_preflight.json()["ok"] is True
        deployment_response = client.post("/api/v1/deployments", json=deployment_request)
        assert deployment_response.status_code == 200, deployment_response.text
        deployment_id = deployment_response.json()["deployment"]["id"]
        deployment = _wait_for(
            client,
            f"/api/v1/deployments/{deployment_id}",
            {"running", "failed"},
        )
        assert deployment["status"] == "running", deployment

        inference = client.post(
            f"/api/v1/deployments/{deployment_id}/infer",
            data={
                "instructions": json.dumps(["pick up the red block"]),
                "states": json.dumps([[0.0] * 7]),
                "save_inputs": "false",
            },
        )
        assert inference.status_code == 200, inference.text
        assert inference.json()["status"] == "completed"
        assert inference.json()["output"]["data"]["simulated"] is True

        evaluation_request = {
            "name": "Demo LIBERO quick eval",
            "kind": "standard",
            "source_kind": "temporary_checkpoint",
            "checkpoint_source": {"kind": "indexed", "checkpoint_id": checkpoint["id"]},
            "combination_id": "deploy_qwen2_5_oft",
            "benchmark_id": "libero",
            "preset": "quick",
            "suite": "libero_goal",
            "parameters": {},
            "model_parameters": {},
            "resources": {"strategy": "auto", "gpu_count": 1, "gpu_ids": []},
            "wandb": {"enabled": False, "mode": "disabled"},
        }
        evaluation_preflight = client.post("/api/v1/evaluations/preflight", json=evaluation_request)
        assert evaluation_preflight.status_code == 200, evaluation_preflight.text
        assert evaluation_preflight.json()["ok"] is True
        evaluation_response = client.post("/api/v1/evaluations", json=evaluation_request)
        assert evaluation_response.status_code == 200, evaluation_response.text
        evaluation_id = evaluation_response.json()["evaluation"]["id"]
        evaluation = _wait_for(
            client,
            f"/api/v1/evaluations/{evaluation_id}",
            {"completed", "failed"},
        )
        assert evaluation["status"] == "completed", evaluation
        result = client.get(f"/api/v1/evaluations/{evaluation_id}/result")
        assert result.status_code == 200, result.text
        assert result.json()["summary"]["success_rate"] == 0.75
        assert result.json()["metadata"]["simulated"] is True

        stopped = client.post(f"/api/v1/deployments/{deployment_id}/stop")
        assert stopped.status_code == 200, stopped.text
        deployment = _wait_for(
            client,
            f"/api/v1/deployments/{deployment_id}",
            {"stopped", "failed"},
        )
        assert deployment["status"] == "stopped", deployment
