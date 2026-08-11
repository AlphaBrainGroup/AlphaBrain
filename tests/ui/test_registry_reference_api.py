from __future__ import annotations

import json
import stat
from copy import deepcopy
from pathlib import Path

from fastapi.testclient import TestClient

from alphabrain_ui.app import create_app
from alphabrain_ui.database import AuditEvent
from alphabrain_ui.registry_overlay import empty_overlay
from alphabrain_ui.runtime import RuntimeConfig


def make_app(tmp_path: Path):  # type: ignore[no-untyped-def]
    return create_app(
        RuntimeConfig(
            repo_root=Path(__file__).resolve().parents[2],
            state_dir=tmp_path,
            database_path=tmp_path / "ui.sqlite3",
            frontend_dist=tmp_path / "missing-dist",
            scheduler_interval=0.02,
        )
    )


def setup_personal(client: TestClient) -> str:
    response = client.post(
        "/api/v1/setup",
        json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
    )
    assert response.status_code == 200
    return response.json()["user"]["id"]


def valid_overlay() -> dict:
    document = empty_overlay("api-v1")
    document["additions"]["components"]["backbones"] = [
        {
            "id": "lab_qwen_variant",
            "based_on": "qwen2_5_vl",
            "label": {"zh-CN": "实验室 Qwen 变体", "en-US": "Lab Qwen variant"},
            "description": {
                "zh-CN": "继承内置且已接通的实现。",
                "en-US": "Derives from a wired built-in implementation.",
            },
        }
    ]
    document["disables"]["combinations"] = ["toy_libero_debug"]
    return document


def executable_overlay() -> dict:
    document = valid_overlay()
    document["overlay_version"] = "app-effective-v1"
    document["additions"]["combinations"] = [
        {
            "id": "lab_qwen_oft_libero",
            "based_on": "qwen_oft_libero",
            "label": {"zh-CN": "实验室 Qwen OFT", "en-US": "Lab Qwen OFT"},
            "backbone": "lab_qwen_variant",
            "action_head": "mlp_regression",
            "method": "imitation_learning",
            "datasets": ["libero"],
            "dataset_mixes": ["libero_goal"],
            "min_gpus": 1,
        }
    ]
    document["additions"]["deployment_combinations"] = [
        {
            "id": "lab_deploy_qwen_oft",
            "based_on": "deploy_qwen2_5_oft",
            "label": {"zh-CN": "实验室部署", "en-US": "Lab deployment"},
            "backbone": "lab_qwen_variant",
            "action_head": "mlp_regression",
            "adapter": "base_framework_websocket",
            "source_combinations": ["lab_qwen_oft_libero"],
            "benchmarks": ["libero"],
            "recommended_gpu_count": 1,
        }
    ]
    return document


def test_regular_platform_hides_and_rejects_demo_only_models(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        setup_personal(client)

        registry = client.get("/api/v1/registry")
        assert registry.status_code == 200
        assert "toy" not in {
            row["id"] for row in registry.json()["view"]["components"]
        }
        assert "toy_libero_debug" not in {
            row["id"] for row in registry.json()["view"]["combinations"]
        }

        capabilities = client.get("/api/v1/capabilities")
        assert capabilities.status_code == 200
        assert "toy" not in {
            row["id"] for row in capabilities.json()["catalog"]["backbones"]
        }
        assert "toy_libero_debug" not in {
            row["id"] for row in capabilities.json()["catalog"]["combinations"]
        }

        blocked = client.post(
            "/api/v1/experiments/submit",
            json={
                "name": "Hidden demo model",
                "spec": {
                    "architecture": {
                        "backbone": "toy",
                        "action_head": "mlp_regression",
                    },
                    "training": {"method": "imitation_learning"},
                    "dataset": {"id": "libero", "mix": "libero_goal"},
                    "resources": {"allocation": "auto", "num_gpus": 1},
                    "parameters": {"run_id": "hidden-demo-model"},
                    "expert_overrides": {},
                },
            },
        )
        assert blocked.status_code == 422
        assert "config_resolution_failed" in {
            issue["code"] for issue in blocked.json()["detail"]["issues"]
        }


def test_registry_overlay_api_previews_persists_and_builds_effective_catalog(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        setup_personal(client)

        initial = client.get("/api/v1/registry")
        assert initial.status_code == 200
        assert initial.json()["overlay_status"]["configured"] is False
        assert "path" not in initial.json()["overlay_status"]

        preview = client.post("/api/v1/registry/overlay/preview", json=valid_overlay())
        assert preview.status_code == 200
        assert preview.json()["overlay"]["overlay_version"] == "api-v1"
        assert preview.json()["view"]["summary"]["overlay_component_count"] == 1
        assert "effective_catalog" not in preview.json()
        assert client.get("/api/v1/registry/overlay").json()["overlay"] is None

        saved = client.put("/api/v1/registry/overlay", json=valid_overlay())
        assert saved.status_code == 200
        assert saved.json()["status"]["configured"] is True
        assert saved.json()["status"]["overlay_version"] == "api-v1"
        overlay_path = tmp_path / "registry" / "overlay.yaml"
        assert stat.S_IMODE(overlay_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(overlay_path.stat().st_mode) == 0o600

        effective = client.get("/api/v1/registry").json()
        added = next(
            row for row in effective["view"]["components"] if row["id"] == "lab_qwen_variant"
        )
        assert added["status"] == "experimental"
        assert not any(
            row["id"] == "toy_libero_debug" for row in effective["view"]["combinations"]
        )
        assert added["origin"] == "overlay"
        assert "catalog" not in effective

        deleted = client.delete("/api/v1/registry/overlay")
        assert deleted.status_code == 200
        assert deleted.json() == {
            "status": {
                "configured": False,
                "sha256": None,
                "overlay_version": None,
                "component_additions": 0,
                "combination_additions": 0,
                "deployment_combination_additions": 0,
                "disabled_entries": 0,
            },
            "overlay": None,
        }
        assert not overlay_path.exists()

    with app.state.database.session() as db:
        audits = db.query(AuditEvent).filter(AuditEvent.action.like("registry_overlay.%")).all()
        actions = {row.action for row in audits}
        serialized = json.dumps([row.detail for row in audits])
    assert actions == {"registry_overlay.update", "registry_overlay.delete"}
    assert "default_pretrained" not in serialized
    assert "description" not in serialized


def test_overlay_validation_is_bilingual_and_never_reflects_submitted_secret(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        setup_personal(client)
        assert client.put("/api/v1/registry/overlay", json=valid_overlay()).status_code == 200
        before = client.get("/api/v1/registry/overlay").json()["status"]["sha256"]
        document = valid_overlay()
        secret = "must-never-be-reflected-123456789"
        document["additions"]["components"]["backbones"][0]["api_key"] = secret
        response = client.put("/api/v1/registry/overlay", json=document)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["code"] == "overlay_credentials_forbidden"
        assert set(detail["message_i18n"]) == {"zh-CN", "en-US"}
        assert secret not in response.text
        assert client.get("/api/v1/registry/overlay").json()["status"]["sha256"] == before
    assert secret.encode() not in (tmp_path / "ui.sqlite3").read_bytes()


def test_overlay_conflict_and_corrupt_persisted_file_have_distinct_safe_errors(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        setup_personal(client)
        conflict = valid_overlay()
        conflict["additions"]["components"]["backbones"][0]["id"] = "qwen2_5_vl"
        response = client.put("/api/v1/registry/overlay", json=conflict)
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "overlay_id_conflict"

        oversized = empty_overlay("oversized-v1")
        oversized["additions"]["components"]["backbones"] = [
            {
                "id": f"lab_variant_{index:03d}",
                "based_on": "qwen2_5_vl",
                "label": {"zh-CN": "中" * 2000, "en-US": "x" * 2000},
            }
            for index in range(70)
        ]
        response = client.put("/api/v1/registry/overlay", json=oversized)
        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "overlay_too_large"

        overlay_path = tmp_path / "registry" / "overlay.yaml"
        overlay_path.parent.mkdir(mode=0o700)
        overlay_path.write_text("schema: [invalid", encoding="utf-8")
        overlay_path.chmod(0o600)
        response = client.get("/api/v1/registry")
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "overlay_invalid_yaml"
        assert str(tmp_path) not in response.text
        recovered = client.delete("/api/v1/registry/overlay")
        assert recovered.status_code == 200
        assert recovered.json()["status"]["configured"] is False
        assert client.get("/api/v1/registry").status_code == 200


def test_noop_overlay_put_and_delete_do_not_add_audit_events(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        setup_personal(client)
        assert client.delete("/api/v1/registry/overlay").status_code == 200
        assert client.put("/api/v1/registry/overlay", json=valid_overlay()).status_code == 200
        assert client.put("/api/v1/registry/overlay", json=valid_overlay()).status_code == 200
        assert client.delete("/api/v1/registry/overlay").status_code == 200
        assert client.delete("/api/v1/registry/overlay").status_code == 200

    with app.state.database.session() as db:
        actions = [
            row.action
            for row in db.query(AuditEvent)
            .filter(AuditEvent.action.like("registry_overlay.%"))
            .order_by(AuditEvent.created_at)
            .all()
        ]
    assert actions == ["registry_overlay.update", "registry_overlay.delete"]


def test_registry_overlay_management_requires_an_administrator_in_lab_mode(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        setup = client.post(
            "/api/v1/setup",
            json={
                "mode": "lab",
                "username": "admin",
                "display_name": "Admin",
                "password": "password123",
            },
        )
        assert setup.status_code == 200
        client.headers["X-CSRF-Token"] = client.cookies["alphabrain_csrf"]
        created = client.post(
            "/api/v1/users",
            json={
                "username": "researcher",
                "display_name": "Researcher",
                "password": "password123",
                "role": "researcher",
            },
        )
        assert created.status_code == 200
        assert client.post("/api/v1/auth/logout").status_code == 200
        client.headers.pop("X-CSRF-Token")
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "researcher", "password": "password123"},
        )
        assert login.status_code == 200
        client.headers["X-CSRF-Token"] = client.cookies["alphabrain_csrf"]

        assert client.get("/api/v1/registry").status_code == 200
        assert client.get("/api/v1/reference-results").status_code == 200
        assert client.get("/api/v1/registry/overlay").status_code == 403
        assert client.post("/api/v1/registry/overlay/preview", json=valid_overlay()).status_code == 403
        assert client.put("/api/v1/registry/overlay", json=valid_overlay()).status_code == 403
        assert client.delete("/api/v1/registry/overlay").status_code == 403


def test_reference_results_api_matches_only_the_exact_versioned_signature(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        setup_personal(client)
        snapshot = client.get("/api/v1/reference-results")
        assert snapshot.status_code == 200
        assert snapshot.json()["version"] == "2026.07.16.1"
        assert snapshot.json()["source_url"] == "https://www.alphabrain-platform.com/#features"

        matched = client.post(
            "/api/v1/reference-results/match",
            json={
                "benchmark_id": "libero",
                "signature": {
                    "suite": "libero_all",
                    "protocol": "website_backbone_table",
                    "local_run_id": "extra-fields-are-allowed",
                },
            },
        )
        assert matched.status_code == 200
        assert [item["id"] for item in matched.json()["items"]] == ["libero_backbone_reference"]

        mismatch = client.post(
            "/api/v1/reference-results/match",
            json={
                "benchmark_id": "libero",
                "signature": {"suite": "libero_goal", "protocol": "website_backbone_table"},
            },
        )
        assert mismatch.status_code == 200
        assert mismatch.json()["items"] == []

        invalid = client.post(
            "/api/v1/reference-results/match",
            json={"benchmark_id": "libero", "signature": {"suite": {"nested": "forbidden"}}},
        )
        assert invalid.status_code == 422
        detail = invalid.json()["detail"]
        assert detail["code"] == "reference_invalid_signature"
        assert set(detail["message_i18n"]) == {"zh-CN", "en-US"}

        missing = client.post(
            "/api/v1/reference-results/match",
            json={"signature": {"suite": "libero_all"}},
        )
        assert missing.status_code == 422
        assert missing.json()["detail"]["code"] == "reference_invalid_id"

        unknown = client.post(
            "/api/v1/reference-results/match",
            json={"benchmark_id": "libero", "signature": {}, "endpoint": "not-accepted"},
        )
        assert unknown.status_code == 422
        assert unknown.json()["detail"]["code"] == "reference_unknown_field"

        non_finite = client.post(
            "/api/v1/reference-results/match",
            content='{"benchmark_id":"libero","signature":{"temperature":NaN}}',
            headers={"Content-Type": "application/json"},
        )
        assert non_finite.status_code == 422
        assert non_finite.json()["detail"]["code"] == "reference_invalid_signature"

    with app.state.database.session() as db:
        audits = db.query(AuditEvent).filter(AuditEvent.action == "reference_results.match").all()
    assert audits == []


def test_overlay_preview_does_not_mutate_input_or_the_builtin_catalog(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        setup_personal(client)
        document = valid_overlay()
        original = deepcopy(document)
        assert client.post("/api/v1/registry/overlay/preview", json=document).status_code == 200
        assert document == original
        builtin = client.get("/api/v1/registry").json()
        assert builtin["overlay_status"]["configured"] is False
        assert not any(
            row["id"] == "lab_qwen_variant" for row in builtin["view"]["components"]
        )


def test_overlay_is_consumed_by_app_capabilities_resolve_deployment_and_evaluation_catalogs(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    pretrained = first_root / "pretrained/Qwen2.5-VL-3B-Instruct"
    libero = first_root / "libero"
    results = first_root / "results"
    pretrained.mkdir(parents=True)
    libero.mkdir(parents=True)
    results.mkdir(parents=True)
    first = make_app(first_root)
    second = make_app(second_root)
    with TestClient(first) as client:
        setup_personal(client)
        assert client.patch(
            "/api/v1/settings",
            json={
                "experimental_globally_enabled": True,
                "environment": {
                    "PRETRAINED_MODELS_DIR": str(pretrained.parent),
                    "LIBERO_DATA_ROOT": str(libero),
                },
                "results_roots": [str(results)],
                "disk_min_free_gib": 0,
                "disk_min_free_percent": 0,
            },
        ).status_code == 200
        assert client.patch(
            "/api/v1/users/me/preferences", json={"experimental_enabled": True}
        ).status_code == 200
        assert client.put(
            "/api/v1/registry/overlay", json=executable_overlay()
        ).status_code == 200

        capabilities = client.get(
            "/api/v1/capabilities", params={"include_experimental": True}
        ).json()["catalog"]
        assert "lab_qwen_oft_libero" in {
            row["id"] for row in capabilities["combinations"]
        }
        deployment = client.get(
            "/api/v1/deployment/capabilities", params={"include_experimental": True}
        ).json()["catalog"]
        assert "lab_deploy_qwen_oft" in {
            row["id"] for row in deployment["deployment_combinations"]
        }
        evaluation = client.get(
            "/api/v1/evaluation-benchmarks", params={"include_experimental": True}
        ).json()["catalog"]
        assert "lab_deploy_qwen_oft" in {
            row["id"] for row in evaluation["combinations"]
        }

        spec = {
            "architecture": {
                "backbone": "lab_qwen_variant",
                "action_head": "mlp_regression",
            },
            "training": {"method": "imitation_learning"},
            "dataset": {"id": "libero", "mix": "libero_goal"},
            "resources": {"allocation": "auto", "num_gpus": 1},
            "parameters": {
                "run_id": "overlay-app-resolve",
                "output_root_dir": str(results),
            },
            "experimental": {"enabled": True, "risk_acknowledged": True},
            "expert_overrides": {},
        }
        resolved = client.post(
            "/api/v1/experiments/resolve",
            json={"name": "overlay-app-resolve", "spec": spec, "acknowledge_experimental": True},
        )
        assert resolved.status_code == 200, resolved.text
        assert resolved.json()["resolved"]["combination"]["id"] == "lab_qwen_oft_libero"
        assert not {"unknown_backbone", "incompatible_combination"} & {
            row["code"] for row in resolved.json()["issues"]
        }

    # Effective catalogs are scoped to the app state directory; no global
    # loader or monkeypatch may leak the first app's overlay into this app.
    with TestClient(second) as client:
        setup_personal(client)
        capabilities = client.get(
            "/api/v1/capabilities", params={"include_experimental": True}
        ).json()["catalog"]
        assert "lab_qwen_variant" not in {
            row["id"] for row in capabilities["components"]["backbones"]
        }
