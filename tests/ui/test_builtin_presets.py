from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from alphabrain_ui.app import create_app
from alphabrain_ui.builtin_presets import builtin_checkpoint_candidates, builtin_templates
from alphabrain_ui.capabilities import validate_compatibility
from alphabrain_ui.runtime import RuntimeConfig


def _make_local_resources(root: Path) -> Path:
    for name in ("Qwen2.5-VL-3B-Instruct", "Qwen3-VL-4B-Instruct"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")

    vjepa = root / "vjepa2"
    vjepa.mkdir()
    (vjepa / "vjepa2_1_vitG_384.pt").write_bytes(b"weights")
    wan = root / "Wan2.2-TI2V-5B"
    wan.mkdir()
    (wan / "config.json").write_text("{}", encoding="utf-8")

    cosmos = root / "Cosmos-Policy-LIBERO-Predict2-2B"
    cosmos.mkdir()
    (cosmos / "Cosmos-Policy-LIBERO-Predict2-2B.pt").write_bytes(b"weights")
    (cosmos / "config.json").write_text(json.dumps({"model_type": "cosmos-policy"}), encoding="utf-8")
    (cosmos / "libero_dataset_statistics.json").write_text("{}", encoding="utf-8")
    (cosmos / "libero_t5_embeddings.pkl").write_bytes(b"embeddings")

    pi05 = root / "pi05_base"
    pi05.mkdir()
    (pi05 / "model.safetensors").write_bytes(b"weights")
    (pi05 / "config.json").write_text(json.dumps({"type": "pi05"}), encoding="utf-8")
    tokenizer = root / "paligemma-tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (pi05 / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"registry_name": "tokenizer_processor", "config": {"tokenizer_name": str(tokenizer)}}]}),
        encoding="utf-8",
    )
    (pi05 / "policy_postprocessor.json").write_text(json.dumps({"steps": []}), encoding="utf-8")
    return root


def _make_app(tmp_path: Path):
    return create_app(
        RuntimeConfig(
            repo_root=Path(__file__).resolve().parents[2],
            state_dir=tmp_path / "state",
            database_path=tmp_path / "state" / "ui.sqlite3",
            frontend_dist=tmp_path / "missing-dist",
            scheduler_interval=0.02,
        )
    )


def test_builtin_templates_follow_local_resource_availability(tmp_path: Path) -> None:
    pretrained = _make_local_resources(tmp_path / "pretrained")

    rows = builtin_templates(pretrained_root=pretrained, environment={})

    assert len(rows) == 5
    assert all(row["builtin"] is True for row in rows)
    assert all(row["availability"] == "partial" for row in rows)
    assert all(validate_compatibility(row["spec"]) == [] for row in rows)
    assert {row["recommended_gpu_count"] for row in rows} == {1, 4}


def test_builtin_checkpoint_candidates_only_report_present_complete_packages(tmp_path: Path) -> None:
    pretrained = _make_local_resources(tmp_path / "pretrained")

    rows = builtin_checkpoint_candidates(pretrained)

    assert {row["id"] for row in rows} == {
        "builtin-checkpoint:cosmos-policy-libero-predict2-2b",
        "builtin-checkpoint:pi05-base",
    }
    assert all(row["complete"] is True for row in rows)
    assert all(row["can_delete"] is False for row in rows)


def test_builtin_template_and_checkpoint_api_are_read_only_and_inspected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("PRETRAINED_MODELS_DIR", raising=False)
    pretrained = _make_local_resources(tmp_path / "pretrained")

    with TestClient(_make_app(tmp_path)) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        assert setup.status_code == 200
        configured = client.patch("/api/v1/settings", json={"pretrained_root": str(pretrained)})
        assert configured.status_code == 200, configured.text

        templates = client.get("/api/v1/templates")
        assert templates.status_code == 200
        assert len([row for row in templates.json() if row.get("builtin")]) == 5
        template_id = "builtin-template:qwen25-oft-libero"
        assert client.get(f"/api/v1/templates/{template_id}").status_code == 200
        assert client.patch(f"/api/v1/templates/{template_id}", json={"name": "changed"}).status_code == 409
        assert client.delete(f"/api/v1/templates/{template_id}").status_code == 409

        response = client.get("/api/v1/checkpoints")
        assert response.status_code == 200, response.text
        checkpoints = {row["id"]: row for row in response.json() if row.get("builtin")}
        cosmos_id = "builtin-checkpoint:cosmos-policy-libero-predict2-2b"
        pi05_id = "builtin-checkpoint:pi05-base"
        assert checkpoints[cosmos_id]["deployable"] is True
        assert checkpoints[cosmos_id]["inspection_summary"]["format"] == "cosmos_policy"
        assert checkpoints[pi05_id]["deployable"] is True
        assert checkpoints[pi05_id]["checkpoint_format"] == "lerobot"
        assert checkpoints[pi05_id]["checkpoint_format_label_i18n"]["zh-CN"] == "LeRobot Pi0.5"
        assert client.get(f"/api/v1/checkpoints/{cosmos_id}").status_code == 200
        assert client.get(f"/api/v1/checkpoints/{pi05_id}").status_code == 200
