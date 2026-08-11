from __future__ import annotations

import json
import stat
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from alphabrain_ui.app import create_app
from alphabrain_ui.database import UtilityRun
from alphabrain_ui.runtime import RuntimeConfig
from alphabrain_ui.secrets import HuggingFaceSecretStore, SecureSecretStore
from alphabrain_ui.utility_worker import _safe_hf_etag


def make_app(tmp_path: Path):
    return create_app(
        RuntimeConfig(
            repo_root=Path(__file__).resolve().parents[2],
            state_dir=tmp_path / "state",
            database_path=tmp_path / "state/ui.sqlite3",
            frontend_dist=tmp_path / "missing-dist",
            scheduler_interval=0.02,
        )
    )


def setup(client: TestClient) -> None:
    assert client.post(
        "/api/v1/setup",
        json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
    ).status_code == 200


def make_lerobot_dataset(root: Path) -> Path:
    dataset = root / "tiny_lerobot"
    meta = dataset / "meta"
    data = dataset / "data/chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({
        "features": {"observation.images.front": {"dtype": "image"}},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
        "total_episodes": 1,
    }))
    (meta / "modality.json").write_text(json.dumps({
        "state": {"joints": {"start": 0, "end": 7}},
        "action": {"joints": {"start": 0, "end": 7}},
        "video": {"front": {"original_key": "observation.images.front"}},
    }))
    (meta / "tasks.jsonl").write_text('{"task_index": 0, "task": "pick"}\n')
    (meta / "episodes.jsonl").write_text('{"episode_index": 0, "length": 2}\n')
    (meta / "stats_gr00t.json").write_text("{}")
    (data / "episode_000000.parquet").write_bytes(b"PAR1dataPAR1")
    return dataset


def make_cosmos_dataset(root: Path) -> Path:
    dataset = root / "cosmos_libero"
    success = dataset / "success_only"
    success.mkdir(parents=True)
    (success / "dataset_statistics.json").write_text("{}")
    (success / "pick_demo.hdf5").write_bytes(b"HDF5")
    return dataset


def make_vlm_dataset(root: Path) -> Path:
    dataset = root / "vlm_json"
    (dataset / "images").mkdir(parents=True)
    (dataset / "annotations.jsonl").write_text('{"id": 1, "image": "images/one.png"}\n')
    return dataset


def experiment_spec(*, registration_id: str | None = None, mixture_id: str | None = None) -> dict:
    dataset = {"id": "libero"}
    if registration_id:
        dataset["registration_id"] = registration_id
    if mixture_id:
        dataset["mixture_id"] = mixture_id
    return {
        "architecture": {"backbone": "qwen2_5_vl", "action_head": "mlp_regression"},
        "training": {"method": "imitation_learning"},
        "dataset": dataset,
        "resources": {"allocation": "auto", "num_gpus": 1, "gpu_ids": []},
        "parameters": {"run_id": "dataset-center-reference"},
        "expert_overrides": {},
    }


def test_generic_secret_store_is_atomic_owner_only_and_never_returns_values(tmp_path: Path) -> None:
    store = SecureSecretStore(tmp_path, "test")
    store.set("subject", "abcdefgh")
    path = tmp_path / "secrets/test/subject"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert store.read("subject") == "abcdefgh"

    hf = HuggingFaceSecretStore(tmp_path)
    assert hf.set_global_download_token("hf_download_123") == {"configured": True}
    assert hf.global_status() == {"configured": True}
    assert hf.set_user_publish_token("user-id", "hf_publish_123") == {"configured": True}
    assert hf.user_status("user-id") == {"configured": True}
    assert "hf_download_123" not in json.dumps(hf.global_status())


def test_resource_inventory_hf_status_and_world_model_registration(tmp_path: Path) -> None:
    world_model = tmp_path / "cosmos"
    world_model.mkdir()
    (world_model / "config.json").write_text("{}")
    with TestClient(make_app(tmp_path)) as client:
        setup(client)
        inventory = client.get("/api/v1/resources")
        assert inventory.status_code == 200
        items = {item["id"]: item for item in inventory.json()["items"]}
        ids = set(items)
        assert "pretrained.Qwen2.5-VL-3B-Instruct" in ids
        assert "dataset.libero" in ids
        assert "world_model.cosmos_predict2" in ids
        assert items["pretrained.paligemma-3b-pt-224"]["requires_hf_token"] is True
        gated = client.post(
            "/api/v1/resources/pretrained.paligemma-3b-pt-224/install",
            json={"target_root": str(tmp_path / "models")},
        )
        assert gated.status_code == 409
        assert gated.json()["detail"] == "hf_download_token_required"

        token = client.put(
            "/api/v1/settings/huggingface/download-token", json={"token": "hf_download_123"}
        )
        assert token.json() == {"configured": True}
        assert "hf_download_123" not in client.get("/api/v1/resources").text

        registered = client.put(
            "/api/v1/resources/world_model.cosmos_predict2/path", json={"path": str(world_model)}
        )
        assert registered.status_code == 200
        assert registered.json()["status"] == "installed"


def test_huggingface_etag_is_safe_for_windows_cache_paths() -> None:
    assert _safe_hf_etag('"abc:*?<>|/\\def"') == "_abc________def_"


def test_resource_install_creates_background_run_and_cancel_cleans_partial_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_install_command(resource_id: str, *, repo_root: Path, target_root: Path):
        del repo_root
        name = resource_id.removeprefix("pretrained.")
        output = target_root / name
        return [sys.executable, "-c", "import time; time.sleep(30)"], {}, str(output)

    monkeypatch.setattr("alphabrain_ui.app.install_command", fake_install_command)
    app = make_app(tmp_path)
    with TestClient(app) as client:
        setup(client)
        target_root = tmp_path / "models"
        created = client.post(
            "/api/v1/resources/pretrained.Qwen2.5-VL-3B-Instruct/install",
            json={"target_root": str(target_root)},
        )
        assert created.status_code == 200, created.text
        run_id = created.json()["id"]
        with app.state.database.session() as db:
            run = db.get(UtilityRun, run_id)
            assert run is not None
            partial = Path(run.parameters["_cleanup_paths"][0])
        partial.mkdir(parents=True)
        (partial / "partial.bin").write_bytes(b"incomplete")

        cancelled = client.post(f"/api/v1/utilities/{run_id}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] in {"cancelled", "stopped"}
        assert not partial.exists()
        resource = next(
            item for item in client.get("/api/v1/resources").json()["items"]
            if item["id"] == "pretrained.Qwen2.5-VL-3B-Instruct"
        )
        assert resource["install_root"] == str(target_root)
        assert resource["target_path"] == str(target_root / "Qwen2.5-VL-3B-Instruct")
        cleared = client.patch(
            "/api/v1/settings",
            json={"pretrained_root": "", "environment": {}},
        )
        assert cleared.status_code == 200
        assert cleared.json()["pretrained_root"] == ""
        resource = next(
            item for item in client.get("/api/v1/resources").json()["items"]
            if item["id"] == "pretrained.Qwen2.5-VL-3B-Instruct"
        )
        assert resource["status"] == "unconfigured"


def test_dataset_registration_preview_mixture_and_stats(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    dataset = make_lerobot_dataset(datasets_root)
    outside = make_lerobot_dataset(tmp_path / "outside")
    with TestClient(make_app(tmp_path)) as client:
        setup(client)
        assert client.patch(
            "/api/v1/settings", json={"dataset_roots": [str(datasets_root)]}
        ).status_code == 200
        forbidden = client.post("/api/v1/datasets", json={"name": "outside", "path": str(outside)})
        assert forbidden.status_code == 403

        created = client.post("/api/v1/datasets", json={
            "name": "Tiny", "path": str(dataset), "storage_mode": "reference", "visibility": "shared"
        })
        assert created.status_code == 200, created.text
        row = created.json()
        assert row["status"] == "ready"
        assert row["episode_count"] == 1
        assert row["validation"]["format_family"] == "lerobot"
        assert row["validation"]["builder_support"] == "direct"
        preview = client.get(f"/api/v1/datasets/{row['id']}/preview")
        assert preview.status_code == 200
        assert preview.json()["episodes"][0]["episode_index"] == 0

        mixture = client.post("/api/v1/datasets/mixtures", json={
            "name": "Tiny mix", "visibility": "shared",
            "members": [{"registration_id": row["id"], "weight": 1.0, "robot_type": "libero_franka"}],
        })
        assert mixture.status_code == 200, mixture.text
        assert mixture.json()["mixture_spec"][0]["path"] == str(dataset)
        assert mixture.json()["owner_name"] == "Admin"
        assert mixture.json()["resolved_members"][0]["format_family"] == "lerobot"

        stats = client.post(f"/api/v1/datasets/{row['id']}/stats")
        assert stats.status_code == 200
        run_id = stats.json()["id"]
        terminal = None
        for _ in range(40):
            runs = client.get("/api/v1/utilities").json()
            terminal = next(item for item in runs if item["id"] == run_id)
            if terminal["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)
        assert terminal and terminal["status"] == "completed", terminal
        refreshed = client.get(f"/api/v1/datasets/{row['id']}").json()
        assert refreshed["stats_status"] == "ready"
        assert refreshed["metadata"]["filesystem_stats"]["file_count"] >= 6

        removed = client.delete(f"/api/v1/datasets/{row['id']}")
        assert removed.status_code == 200
        assert removed.json()["data_deleted"] is False
        assert dataset.is_dir()


def test_dataset_inspection_reports_supported_formats_and_enforces_roots(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    lerobot = make_lerobot_dataset(datasets_root / "single")
    collection_root = datasets_root / "collection"
    make_lerobot_dataset(collection_root)
    cosmos = make_cosmos_dataset(datasets_root)
    vlm = make_vlm_dataset(datasets_root)
    unknown = datasets_root / "unknown"
    unknown.mkdir(parents=True)
    outside = make_lerobot_dataset(tmp_path / "outside")

    with TestClient(make_app(tmp_path)) as client:
        setup(client)
        assert client.patch(
            "/api/v1/settings", json={"dataset_roots": [str(datasets_root)]}
        ).status_code == 200

        cases = [
            (lerobot, "lerobot", "direct", True),
            (collection_root, "lerobot_collection", "mixture_only", False),
            (cosmos, "cosmos", "inventory_only", False),
            (vlm, "vlm_json", "inventory_only", False),
        ]
        for path, family, support, builder_ready in cases:
            response = client.post("/api/v1/datasets/inspect", json={"path": str(path)})
            assert response.status_code == 200, response.text
            report = response.json()
            assert report["valid"] is True
            assert report["format_family"] == family
            assert report["builder_support"] == support
            assert report["builder_ready"] is builder_ready

        unrecognized = client.post("/api/v1/datasets/inspect", json={"path": str(unknown)})
        assert unrecognized.status_code == 200
        assert unrecognized.json()["valid"] is False
        assert unrecognized.json()["builder_support"] == "unsupported"

        forbidden = client.post("/api/v1/datasets/inspect", json={"path": str(outside)})
        assert forbidden.status_code == 403
        assert forbidden.json()["detail"]["code"] == "dataset_path_outside_configured_roots"


def test_experiment_resolves_registered_dataset_and_versioned_mixture(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    dataset = make_lerobot_dataset(datasets_root)
    with TestClient(make_app(tmp_path)) as client:
        setup(client)
        assert client.patch(
            "/api/v1/settings",
            json={
                "dataset_roots": [str(datasets_root)],
                "disk_min_free_gib": 0,
                "disk_min_free_percent": 0,
            },
        ).status_code == 200
        registration = client.post(
            "/api/v1/datasets",
            json={"name": "Tiny", "path": str(dataset), "visibility": "shared"},
        ).json()
        mixture = client.post(
            "/api/v1/datasets/mixtures",
            json={
                "name": "Tiny mix",
                "visibility": "shared",
                "members": [
                    {
                        "registration_id": registration["id"],
                        "pattern": "",
                        "weight": 0.75,
                        "robot_type": "libero_franka",
                    }
                ],
            },
        ).json()

        registered = client.post(
            "/api/v1/experiments/resolve",
            json={
                "name": "registered",
                "spec": experiment_spec(registration_id=registration["id"]),
            },
        )
        assert registered.status_code == 200, registered.text
        registered_body = registered.json()
        registered_dataset = registered_body["resolved"]["spec"]["dataset"]
        assert registered_dataset["root"] == str(dataset)
        assert registered_dataset["source"] == {
            "kind": "registration",
            "id": registration["id"],
            "fingerprint": registration["fingerprint"],
        }
        assert registered_body["resolved"]["resolved_config"]["datasets"]["vla_data"][
            "mixture_spec"
        ][0]["path"] == str(dataset)
        assert "dataset_mixture_not_found" not in {
            item["code"] for item in registered_body["issues"]
        }

        mixed = client.post(
            "/api/v1/experiments/resolve",
            json={"name": "mixture", "spec": experiment_spec(mixture_id=mixture["id"])},
        )
        assert mixed.status_code == 200, mixed.text
        mixed_body = mixed.json()["resolved"]
        assert mixed_body["spec"]["dataset"]["source"] == {
            "kind": "mixture",
            "id": mixture["id"],
            "version": mixture["version"],
        }
        mixture_spec = mixed_body["resolved_config"]["datasets"]["vla_data"]["mixture_spec"]
        assert mixture_spec == [
            {
                "path": str(dataset),
                "pattern": "",
                "weight": 0.75,
                "robot_type": "libero_franka",
                "trajectory_limit": None,
            }
        ]

        arbitrary = experiment_spec()
        arbitrary["dataset"]["mixture_spec"] = [{"path": str(tmp_path), "weight": 1.0}]
        rejected = client.post(
            "/api/v1/experiments/preflight",
            json={"name": "untrusted", "spec": arbitrary},
        )
        assert rejected.status_code == 200
        assert rejected.json()["can_submit"] is False
        assert "untrusted_dataset_mixture_spec" in {
            item["code"] for item in rejected.json()["issues"]
        }


def test_private_dataset_references_are_hidden_from_other_researchers(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    private_dataset = make_lerobot_dataset(datasets_root / "private")
    shared_dataset = make_lerobot_dataset(datasets_root / "shared")
    with TestClient(make_app(tmp_path)) as client:
        setup_response = client.post(
            "/api/v1/setup",
            json={
                "mode": "lab",
                "username": "admin",
                "display_name": "Admin",
                "password": "password123",
            },
        )
        assert setup_response.status_code == 200
        client.headers["X-CSRF-Token"] = client.cookies["alphabrain_csrf"]
        assert client.patch(
            "/api/v1/settings", json={"dataset_roots": [str(datasets_root)]}
        ).status_code == 200
        assert client.post(
            "/api/v1/users",
            json={
                "username": "researcher",
                "display_name": "Researcher",
                "password": "password123",
                "role": "researcher",
            },
        ).status_code == 200
        private_registration = client.post(
            "/api/v1/datasets",
            json={"name": "Private", "path": str(private_dataset), "visibility": "private"},
        ).json()
        shared_registration = client.post(
            "/api/v1/datasets",
            json={"name": "Shared", "path": str(shared_dataset), "visibility": "shared"},
        ).json()
        private_mixture = client.post(
            "/api/v1/datasets/mixtures",
            json={
                "name": "Private mix",
                "visibility": "private",
                "members": [{"registration_id": shared_registration["id"], "weight": 1.0}],
            },
        ).json()
        assert client.post("/api/v1/auth/logout").status_code == 200
        client.headers.pop("X-CSRF-Token")
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "researcher", "password": "password123"},
        )
        assert login.status_code == 200
        client.headers["X-CSRF-Token"] = client.cookies["alphabrain_csrf"]

        private_registration_result = client.post(
            "/api/v1/experiments/preflight",
            json={
                "name": "private-registration",
                "spec": experiment_spec(registration_id=private_registration["id"]),
            },
        ).json()
        assert "dataset_registration_not_found" in {
            item["code"] for item in private_registration_result["issues"]
        }
        private_mixture_result = client.post(
            "/api/v1/experiments/preflight",
            json={
                "name": "private-mixture",
                "spec": experiment_spec(mixture_id=private_mixture["id"]),
            },
        ).json()
        assert "dataset_mixture_not_found" in {
            item["code"] for item in private_mixture_result["issues"]
        }
