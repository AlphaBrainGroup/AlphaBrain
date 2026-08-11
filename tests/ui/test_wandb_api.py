from __future__ import annotations

import json
import stat
from pathlib import Path

from fastapi.testclient import TestClient

from alphabrain_ui.app import create_app
from alphabrain_ui.database import AuditEvent
from alphabrain_ui.runtime import RuntimeConfig


def make_app(tmp_path: Path):
    return create_app(
        RuntimeConfig(
            repo_root=Path(__file__).resolve().parents[2],
            state_dir=tmp_path,
            database_path=tmp_path / "ui.sqlite3",
            frontend_dist=tmp_path / "missing-dist",
            scheduler_interval=0.02,
        )
    )


def setup_personal(client: TestClient) -> None:
    response = client.post(
        "/api/v1/setup",
        json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
    )
    assert response.status_code == 200


def wandb_spec() -> dict:
    return {
        "architecture": {"backbone": "qwen2_5_vl", "action_head": "mlp_regression"},
        "training": {"method": "imitation_learning"},
        "dataset": {"id": "libero", "mix": "libero_goal"},
        "resources": {"allocation": "auto", "num_gpus": 1},
        "parameters": {"run_id": "wandb-api-test"},
        "expert_overrides": {},
        "wandb": {
            "enabled": True,
            "mode": "online",
            "project": "AlphaBrain",
            "categories": ["metrics", "config", "system"],
        },
    }


def test_wandb_api_key_is_write_only_and_audit_safe(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    secret = "test-wandb-key-123456789"
    with TestClient(app) as client:
        setup_personal(client)
        assert client.get("/api/v1/settings/wandb").json() == {"configured": False}

        saved = client.put("/api/v1/settings/wandb/api-key", json={"api_key": secret})
        assert saved.status_code == 200
        assert saved.json() == {"configured": True}
        assert client.get("/api/v1/settings/wandb").json() == {"configured": True}
        assert secret not in json.dumps(client.get("/api/v1/settings").json())

        key_path = tmp_path / "secrets" / "wandb_api_key"
        assert stat.S_IMODE(key_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
        with app.state.database.session() as db:
            audit_payload = json.dumps(
                [row.detail for row in db.query(AuditEvent).order_by(AuditEvent.created_at).all()]
            )
        assert secret not in audit_payload
        assert secret.encode() not in (tmp_path / "ui.sqlite3").read_bytes()

        deleted = client.delete("/api/v1/settings/wandb/api-key")
        assert deleted.status_code == 200
        assert deleted.json() == {"configured": False}
        assert not key_path.exists()


def test_validation_errors_do_not_reflect_credentials(tmp_path: Path) -> None:
    secret = "xyZ$42"
    with TestClient(make_app(tmp_path)) as client:
        setup_personal(client)
        response = client.put("/api/v1/settings/wandb/api-key", json={"api_key": secret})
        assert response.status_code == 422
        assert secret not in response.text

        spec = wandb_spec()
        inline_secret = "must-never-be-reflected-123456"
        spec["wandb"]["api_key"] = inline_secret
        response = client.post(
            "/api/v1/experiments/preflight",
            json={"name": "unsafe", "spec": spec, "acknowledge_experimental": False},
        )
        assert response.status_code == 422
        assert inline_secret not in response.text


def test_online_wandb_preflight_requires_the_secret_store(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        setup_personal(client)
        payload = {"name": "wandb-api-test", "spec": wandb_spec(), "acknowledge_experimental": False}
        missing = client.post("/api/v1/experiments/preflight", json=payload)
        assert missing.status_code == 200
        assert "wandb_api_key_missing" in {item["code"] for item in missing.json()["issues"]}

        client.put("/api/v1/settings/wandb/api-key", json={"api_key": "configured-key-123456"})
        configured = client.post("/api/v1/experiments/preflight", json=payload)
        assert configured.status_code == 200
        assert "wandb_api_key_missing" not in {item["code"] for item in configured.json()["issues"]}


def test_capabilities_expose_per_combination_wandb_content_support(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        setup_personal(client)
        combinations = client.get("/api/v1/capabilities").json()["catalog"]["combinations"]
        imitation = next(item for item in combinations if item["id"] == "qwen_oft_libero")
        categories = {item["id"]: item for item in imitation["wandb_categories"]}
        assert categories["metrics"]["supported"] is True
        assert categories["config"]["supported"] is True
        assert categories["system"]["supported"] is True
        assert categories["videos"]["supported"] is False
        assert categories["checkpoints"]["supported"] is False
