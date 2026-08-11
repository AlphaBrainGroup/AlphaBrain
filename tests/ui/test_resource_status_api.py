from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from alphabrain_ui.app import create_app
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
    assert client.post(
        "/api/v1/setup",
        json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
    ).status_code == 200


def qwen_status(client: TestClient) -> dict:
    catalog = client.get("/api/v1/capabilities").json()["catalog"]
    qwen = next(item for item in catalog["backbones"] if item["id"] == "qwen2_5_vl")
    return qwen["pretrained_directory"]


def test_capability_route_uses_configured_pretrained_model_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PRETRAINED_MODELS_DIR", raising=False)
    pretrained = tmp_path / "pretrained"
    model = pretrained / "Qwen2.5-VL-3B-Instruct"
    model.mkdir(parents=True)
    with TestClient(make_app(tmp_path)) as client:
        setup_personal(client)
        assert client.patch(
            "/api/v1/settings",
            json={"environment": {"PRETRAINED_MODELS_DIR": ""}},
        ).status_code == 200
        assert qwen_status(client)["configured"] is False

        response = client.patch(
            "/api/v1/settings",
            json={"environment": {"PRETRAINED_MODELS_DIR": str(pretrained)}},
        )
        assert response.status_code == 200
        status = qwen_status(client)
        assert status == {
            "required": True,
            "configured": True,
            "path": str(model),
            "exists": True,
            "issue": None,
        }

        missing_root = tmp_path / "missing-pretrained"
        assert client.patch(
            "/api/v1/settings",
            json={"environment": {"PRETRAINED_MODELS_DIR": str(missing_root)}},
        ).status_code == 200
        status = qwen_status(client)
        assert status["configured"] is True
        assert status["exists"] is False
        assert status["issue"]["code"] == "pretrained_directory_not_found"


def test_dashboard_exposes_cpu_and_ram_snapshot(tmp_path: Path, monkeypatch) -> None:
    snapshot = {
        "available": True,
        "cpu_percent": 12.5,
        "load": {"one_minute": 1.0, "five_minutes": 0.5, "fifteen_minutes": 0.25},
        "memory": {
            "total_bytes": 1000,
            "used_bytes": 400,
            "available_bytes": 600,
            "percent": 40.0,
        },
    }
    monkeypatch.setattr("alphabrain_ui.app.collect_system_metrics", lambda: snapshot)
    with TestClient(make_app(tmp_path)) as client:
        setup_personal(client)
        response = client.get("/api/v1/dashboard")
        assert response.status_code == 200
        assert response.json()["system_metrics"] == snapshot
