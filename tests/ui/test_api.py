from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from alphabrain_ui.app import create_app
from alphabrain_ui.database import Experiment, ExperimentStage, Job
from alphabrain_ui.gpu import GPUInfo
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


class GateGPUMonitor:
    """Visible to preflight while keeping scheduler-owned jobs queued."""

    available = True
    error = ""

    def snapshot(self, reservations=None):
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
                available=reservations is None,
            )
        ]


def test_personal_setup_templates_and_preferences(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        initial_status = client.get("/api/v1/setup/status").json()
        assert initial_status["initialized"] is False
        assert initial_status["deployment_mode"] == "personal"
        response = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        assert response.status_code == 200
        me = client.get("/api/v1/auth/me").json()
        assert me["deployment_mode"] == "personal"
        assert me["user"]["gpu_refresh_interval_seconds"] == 5

        created = client.post(
            "/api/v1/templates",
            json={"name": "Shared baseline", "visibility": "shared", "spec": {"training": {"method": "imitation"}}},
        )
        assert created.status_code == 200
        assert any(
            row["name"] == "Shared baseline" and row.get("builtin") is not True
            for row in client.get("/api/v1/templates").json()
        )

        updated = client.patch(
            "/api/v1/users/me/preferences",
            json={"language": "en-US", "theme": "dark", "gpu_refresh_interval_seconds": 30},
        )
        assert updated.status_code == 200
        assert updated.json()["language"] == "en-US"
        assert updated.json()["theme"] == "dark"
        assert updated.json()["gpu_refresh_interval_seconds"] == 30
        assert client.get("/api/v1/auth/me").json()["user"]["gpu_refresh_interval_seconds"] == 30
        assert client.patch(
            "/api/v1/users/me/preferences",
            json={"gpu_refresh_interval_seconds": -1},
        ).status_code == 422
        assert client.patch(
            "/api/v1/users/me/preferences",
            json={"gpu_refresh_interval_seconds": 1},
        ).status_code == 422
        manual = client.patch(
            "/api/v1/users/me/preferences",
            json={"gpu_refresh_interval_seconds": 0},
        )
        assert manual.status_code == 200
        assert manual.json()["gpu_refresh_interval_seconds"] == 0


def test_template_duplicate_detection_override_update_and_audit(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        spec = {
            "architecture": {"backbone": "pi0_5", "action_head": "flow_matching"},
            "training": {"method": "sft", "checkpoint": "/models/pi05"},
            "dataset": {"id": "libero"},
            "parameters": {"run_id": "run-one", "learning_rate": 0.0001},
            "resources": {"allocation": "fixed", "num_gpus": 1, "gpu_ids": [0]},
            "metadata": {"description": "first"},
        }
        first = client.post(
            "/api/v1/templates",
            json={"name": "PI05 baseline", "visibility": "private", "spec": spec},
        )
        assert first.status_code == 200
        first_id = first.json()["id"]

        same_spec = {
            **spec,
            "parameters": {**spec["parameters"], "run_id": "run-two"},
            "resources": {**spec["resources"], "gpu_ids": [3]},
            "metadata": {"description": "second"},
        }
        duplicate = client.post(
            "/api/v1/templates",
            json={"name": "Another name", "visibility": "private", "spec": same_spec},
        )
        assert duplicate.status_code == 409
        detail = duplicate.json()["detail"]
        assert detail["code"] == "duplicate_template"
        assert detail["matches"] == [
            {
                "id": first_id,
                "name": "PI05 baseline",
                "owner_id": first.json()["owner_id"],
                "owner_name": "Admin",
                "visibility": "personal",
                "builtin": False,
                "same_name": False,
                "same_spec": True,
            }
        ]

        same_name = client.post(
            "/api/v1/templates",
            json={"name": "  ｐｉ０５ BASELINE ", "visibility": "private", "spec": {"dataset": {"id": "bridge"}}},
        )
        assert same_name.status_code == 409
        assert same_name.json()["detail"]["matches"][0]["same_name"] is True

        # Self is excluded when the identity changes and no other match exists.
        assert client.patch(
            f"/api/v1/templates/{first_id}",
            json={"name": "PI05 renamed", "spec": spec},
        ).status_code == 200

        override = client.post(
            "/api/v1/templates",
            json={
                "name": "Another name",
                "visibility": "private",
                "spec": same_spec,
                "allow_duplicate": True,
            },
        )
        assert override.status_code == 200

        # Presentation-only edits do not re-trigger an already acknowledged duplicate.
        assert client.patch(
            f"/api/v1/templates/{first_id}",
            json={"description": "Documentation only", "visibility": "shared"},
        ).status_code == 200

        audit = client.get("/api/v1/audit").json()
        override_event = next(row for row in audit if row["target_id"] == override.json()["id"])
        assert override_event["action"] == "template.create"
        assert override_event["detail"] == {
            "duplicate_override": True,
            "duplicate_match_ids": [first_id],
        }


def test_template_duplicate_visibility_and_builtin_matching(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        client.post(
            "/api/v1/setup",
            json={"mode": "lab", "username": "admin", "display_name": "Admin", "password": "password123"},
        )
        client.headers["X-CSRF-Token"] = client.cookies["alphabrain_csrf"]
        shared = client.post(
            "/api/v1/templates",
            json={"name": "Admin shared", "visibility": "shared", "spec": {"dataset": {"id": "shared-data"}}},
        )
        private = client.post(
            "/api/v1/templates",
            json={"name": "Admin private", "visibility": "private", "spec": {"dataset": {"id": "private-data"}}},
        )
        assert shared.status_code == private.status_code == 200
        assert client.post(
            "/api/v1/users",
            json={"username": "alice", "display_name": "Alice", "password": "password123", "role": "researcher"},
        ).status_code == 200
        assert client.post("/api/v1/auth/logout").status_code == 200

        client.headers.pop("X-CSRF-Token")
        login = client.post("/api/v1/auth/login", json={"username": "alice", "password": "password123"})
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]

        shared_match = client.post(
            "/api/v1/templates",
            json={"name": "Alice shared copy", "visibility": "private", "spec": {"dataset": {"id": "shared-data"}}},
        )
        assert shared_match.status_code == 409
        assert [item["id"] for item in shared_match.json()["detail"]["matches"]] == [shared.json()["id"]]

        # Another owner's private template neither blocks the save nor appears in the response.
        private_copy = client.post(
            "/api/v1/templates",
            json={"name": "Alice private copy", "visibility": "private", "spec": {"dataset": {"id": "private-data"}}},
        )
        assert private_copy.status_code == 200

        builtin = next(item for item in client.get("/api/v1/templates").json() if item.get("builtin"))
        builtin_match = client.post(
            "/api/v1/templates",
            json={"name": "Builtin copy", "visibility": "private", "spec": builtin["spec"]},
        )
        assert builtin_match.status_code == 409
        assert any(
            item["id"] == builtin["id"] and item["builtin"] and item["same_spec"]
            for item in builtin_match.json()["detail"]["matches"]
        )


def _make_lerobot_v2_dataset(root: Path) -> Path:
    dataset = root / "libero_goal_no_noops_1.0.0_lerobot"
    meta = dataset / "meta"
    data = dataset / "data/chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.front": {
                        "dtype": "image",
                        "shape": [3, 224, 224],
                        "names": ["channel", "height", "width"],
                    }
                },
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "chunks_size": 1000,
                "total_episodes": 1,
            }
        )
    )
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "state": {"joints": {"start": 0, "end": 7, "original_key": "observation.state"}},
                "action": {"joints": {"start": 0, "end": 7, "original_key": "action"}},
                "video": {"image_0": {"original_key": "observation.images.front"}},
            }
        )
    )
    (meta / "tasks.jsonl").write_text('{"task_index": 0, "task": "pick"}\n')
    (meta / "episodes.jsonl").write_text('{"episode_index": 0, "length": 2, "tasks": ["pick"]}\n')
    (meta / "stats_gr00t.json").write_text("{}")
    (data / "episode_000000.parquet").write_bytes(b"PAR1dataPAR1")
    return dataset


def test_local_dataset_browser_and_validation(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets"
    dataset = _make_lerobot_v2_dataset(dataset_root)
    with TestClient(make_app(tmp_path)) as client:
        client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )

        browsed = client.get("/api/v1/datasets/directories", params={"path": str(dataset_root)})
        assert browsed.status_code == 200
        assert browsed.json()["directories"] == [{"name": dataset.name, "path": str(dataset)}]

        validated = client.post(
            "/api/v1/datasets/validate",
            json={"path": str(dataset_root), "dataset_id": "libero"},
        )
        assert validated.status_code == 200
        assert validated.json()["valid"] is True, validated.text
        assert validated.json()["normalized_root"] == str(dataset_root)
        assert validated.json()["format"] == "LeRobot v2.0"
        assert validated.json()["episode_count"] == 1

        # Selecting the leaf directory is normalized to the root expected by the mixture recipe.
        leaf = client.post(
            "/api/v1/datasets/validate",
            json={"path": str(dataset), "dataset_id": "libero"},
        ).json()
        assert leaf["valid"] is True
        assert leaf["normalized_root"] == str(dataset_root)

        invalid = client.post(
            "/api/v1/datasets/validate",
            json={"path": str(tmp_path / "missing"), "dataset_id": "libero"},
        ).json()
        assert invalid["valid"] is False
        assert {item["code"] for item in invalid["issues"]} == {"dataset_directory_not_found"}


def test_startup_mode_only_suggests_lab_before_initialization(tmp_path: Path) -> None:
    config = RuntimeConfig(
        repo_root=Path(__file__).resolve().parents[2],
        state_dir=tmp_path,
        database_path=tmp_path / "ui.sqlite3",
        frontend_dist=tmp_path / "missing-dist",
        initial_mode="lab",
        scheduler_interval=0.02,
    )
    with TestClient(create_app(config)) as client:
        status = client.get("/api/v1/setup/status").json()
        assert status == {"initialized": False, "deployment_mode": "lab", "default_language": "zh-CN"}
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        assert setup.status_code == 200

    # A persisted choice wins over a different startup suggestion.
    with TestClient(create_app(config)) as client:
        status = client.get("/api/v1/setup/status").json()
        assert status["initialized"] is True
        assert status["deployment_mode"] == "personal"


def test_lab_login_and_csrf(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "lab", "username": "admin", "display_name": "Admin", "password": "password123"},
        )
        assert setup.status_code == 200
        payload = {"name": "Private", "visibility": "private", "spec": {}}
        assert client.post("/api/v1/templates", json=payload).status_code == 403

        client.headers["X-CSRF-Token"] = client.cookies["alphabrain_csrf"]
        assert client.post("/api/v1/templates", json=payload).status_code == 200
        assert client.post("/api/v1/auth/logout").status_code == 200

        client.headers.pop("X-CSRF-Token")
        login = client.post("/api/v1/auth/login", json={"username": "admin", "password": "password123"})
        assert login.status_code == 200
        assert "csrf_token" in login.json()


def test_settings_reject_secrets_and_experimental_is_double_gated(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        rejected = client.patch("/api/v1/settings", json={"environment": {"HF_TOKEN": "secret"}})
        assert rejected.status_code == 422
        assert client.patch("/api/v1/users/me/preferences", json={"experimental_enabled": True}).status_code == 409
        assert client.patch("/api/v1/settings", json={"experimental_globally_enabled": True}).status_code == 200
        enabled = client.patch("/api/v1/users/me/preferences", json={"experimental_enabled": True})
        assert enabled.status_code == 200
        assert enabled.json()["experimental_enabled"] is True

        invalid_root = tmp_path / "not-a-directory"
        invalid_root.write_text("file")
        bad_storage = client.patch("/api/v1/settings", json={"results_roots": [str(invalid_root)]})
        assert bad_storage.status_code == 422
        primary_results = tmp_path / "results-primary"
        secondary_results = tmp_path / "results-secondary"
        primary_results.mkdir()
        secondary_results.mkdir()
        multiple_roots = client.patch(
            "/api/v1/settings",
            json={"results_roots": [str(primary_results), str(secondary_results)]},
        )
        assert multiple_roots.status_code == 200
        assert multiple_roots.json()["results_roots"] == [
            str(primary_results),
            str(secondary_results),
        ]
        dashboard = client.get("/api/v1/dashboard")
        assert dashboard.status_code == 200
        assert dashboard.json()["remote_training"] == {"enabled": False, "target": ""}
        assert client.get("/api/v1/remote-training/metrics").json() == {"enabled": False}
        assert client.get("/api/v1/training/target").json() == {
            "mode": "local",
            "gpu_ids": [],
        }


def test_ssh_settings_save_ignores_unchanged_missing_default_dataset_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.shutil.which", lambda name: f"/usr/bin/{name}")
    with TestClient(make_app(tmp_path)) as client:
        client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )

        response = client.patch(
            "/api/v1/settings",
            json={
                "dataset_roots": ["data"],
                "remote_training_enabled": True,
                "remote_training_host": "180.76.111.187",
                "remote_training_user": "root",
                "remote_training_port": 5918,
                "remote_training_repo_root": "/share/chenhonghan/AlphaBrain",
                "remote_training_gpu_ids": [0, 1, 2, 3],
            },
        )

        assert response.status_code == 200
        assert response.json()["dataset_roots"] == ["data"]
        assert response.json()["remote_training_host"] == "180.76.111.187"
        assert response.json()["remote_training_port"] == 5918
        assert client.get("/api/v1/training/target").json() == {
            "mode": "remote",
            "gpu_ids": [0, 1, 2, 3],
        }

        changed_missing_root = client.patch(
            "/api/v1/settings",
            json={"dataset_roots": [str(tmp_path / "missing-dataset")]},
        )
        assert changed_missing_root.status_code == 422
        assert changed_missing_root.json()["detail"]["code"] == "dataset_root_unavailable"


def test_dashboard_storage_path_can_be_configured_independently(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        first_results = tmp_path / "results-primary"
        monitored = tmp_path / "monitored-volume" / "nested"
        first_results.mkdir()
        monitored.mkdir(parents=True)

        defaults = client.patch(
            "/api/v1/settings",
            json={"results_roots": [str(first_results)]},
        )
        assert defaults.status_code == 200
        assert client.get("/api/v1/storage").json()[0]["path"] == str(first_results)

        configured = client.patch(
            "/api/v1/settings",
            json={"storage_monitor_path": str(monitored)},
        )
        assert configured.status_code == 200
        assert configured.json()["storage_monitor_path"] == str(monitored)
        storage = client.get("/api/v1/storage").json()
        assert len(storage) == 1
        assert storage[0]["path"] == str(monitored)
        assert storage[0]["mount_point"]
        assert storage[0]["total_bytes"] > 0
        assert client.get("/api/v1/dashboard").json()["storage"][0]["path"] == str(monitored)

        missing = client.patch(
            "/api/v1/settings",
            json={"storage_monitor_path": str(tmp_path / "missing")},
        )
        assert missing.status_code == 422
        assert missing.json()["detail"]["code"] == "storage_monitor_path_not_found"

        file_path = tmp_path / "not-a-directory"
        file_path.write_text("file")
        not_directory = client.patch(
            "/api/v1/settings",
            json={"storage_monitor_path": str(file_path)},
        )
        assert not_directory.status_code == 422
        assert not_directory.json()["detail"]["code"] == "storage_monitor_path_not_directory"

        cleared = client.patch("/api/v1/settings", json={"storage_monitor_path": ""})
        assert cleared.status_code == 200
        assert cleared.json()["storage_monitor_path"] == ""
        assert client.get("/api/v1/storage").json()[0]["path"] == str(first_results)


def test_current_administrator_cannot_lock_out_own_account(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        user_id = setup.json()["user"]["id"]

        assert client.patch(f"/api/v1/users/{user_id}", json={"role": "researcher"}).status_code == 409
        assert client.patch(f"/api/v1/users/{user_id}", json={"is_active": False}).status_code == 409
        assert client.get("/api/v1/settings").status_code == 200


def test_force_kill_missing_job_returns_404(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        assert client.post("/api/v1/jobs/missing/force-kill").status_code == 404


def test_usernames_are_normalized_and_validated(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        created = client.post(
            "/api/v1/users",
            json={"username": " alice ", "display_name": "Alice", "password": "password123", "role": "researcher"},
        )
        assert created.status_code == 200
        assert created.json()["username"] == "alice"
        duplicate = client.post(
            "/api/v1/users",
            json={"username": "alice", "display_name": "Alice", "password": "password123", "role": "researcher"},
        )
        assert duplicate.status_code == 409
        invalid = client.post(
            "/api/v1/users",
            json={"username": "bad name", "display_name": "Bad", "password": "password123", "role": "researcher"},
        )
        assert invalid.status_code == 422


def test_secure_cookie_environment_is_a_security_floor(tmp_path: Path) -> None:
    app = create_app(
        RuntimeConfig(
            repo_root=Path(__file__).resolve().parents[2],
            state_dir=tmp_path,
            database_path=tmp_path / "secure.sqlite3",
            frontend_dist=tmp_path / "missing-dist",
            secure_cookies=True,
        )
    )
    with TestClient(app, base_url="https://testserver") as client:
        response = client.post(
            "/api/v1/setup",
            json={"mode": "lab", "username": "admin", "display_name": "Admin", "password": "password123"},
        )
        cookies = response.headers.get_list("set-cookie")
        assert len(cookies) == 2
        assert all("Secure" in value for value in cookies)
        client.headers["X-CSRF-Token"] = client.cookies["alphabrain_csrf"]
        settings = client.get("/api/v1/settings")
        assert settings.status_code == 200
        assert settings.json()["secure_cookies"] is True
        assert settings.json()["secure_cookies_locked"] is True
        attempted_disable = client.patch("/api/v1/settings", json={"secure_cookies": False})
        assert attempted_disable.status_code == 200
        assert attempted_disable.json()["secure_cookies"] is True
        assert attempted_disable.json()["secure_cookies_locked"] is True


def test_sqlite_datetimes_are_returned_as_utc(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        created_at = setup.json()["user"]["created_at"]
        listed_at = client.get("/api/v1/users").json()[0]["created_at"]
        assert created_at.endswith(("Z", "+00:00"))
        assert listed_at.endswith(("Z", "+00:00"))


def test_checkpoint_list_and_guarded_delete_survive_symlink_replacement(tmp_path: Path) -> None:
    results = tmp_path / "results"
    output = results / "run"
    checkpoint = output / "checkpoints" / "steps_10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    app = make_app(tmp_path)
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        user_id = setup.json()["user"]["id"]
        assert client.patch("/api/v1/settings", json={"results_roots": [str(results)]}).status_code == 200
        with app.state.database.session() as db:
            experiment = Experiment(
                owner_id=user_id,
                name="delete-me",
                family="test",
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
            db.add(
                Job(
                    experiment_id=experiment.id,
                    stage_id=stage.id,
                    owner_id=user_id,
                    status="completed",
                    cwd=str(tmp_path),
                    output_dir=str(output),
                )
            )

        listed = client.get("/api/v1/checkpoints")
        assert listed.status_code == 200
        row = next(item for item in listed.json() if item["experiment_name"] == "delete-me")
        assert row["experiment_name"] == "delete-me"
        assert row["owner_name"] == "Admin"
        assert row["can_delete"] is True
        assert row["step"] == 10

        with app.state.database.session() as db:
            dependent = Experiment(
                owner_id=user_id,
                name="dependent",
                family="test",
                status="queued",
                spec={},
                resolved={},
            )
            db.add(dependent)
            db.flush()
            dependent_stage = ExperimentStage(
                experiment_id=dependent.id,
                name="train",
                phase="train",
                position=0,
                resolved={},
            )
            db.add(dependent_stage)
            db.flush()
            dependent_job = Job(
                experiment_id=dependent.id,
                stage_id=dependent_stage.id,
                owner_id=user_id,
                status="queued",
                cwd=str(tmp_path),
                output_dir=str(results / "dependent"),
                input_checkpoint_path=row["path"],
            )
            db.add(dependent_job)
            db.flush()
            dependent_job_id = dependent_job.id

        refreshed = next(
            item for item in client.get("/api/v1/checkpoints").json()
            if item["id"] == row["id"]
        )
        assert refreshed["can_delete"] is False
        referenced = client.request(
            "DELETE",
            f"/api/v1/checkpoints/{row['id']}",
            json={"confirmation": "delete-me"},
        )
        assert referenced.status_code == 409
        with app.state.database.session() as db:
            db.get(Job, dependent_job_id).status = "cancelled"

        target = results / "other-experiment"
        target.mkdir()
        sentinel = target / "keep.txt"
        sentinel.write_text("keep")
        for child in checkpoint.iterdir():
            child.unlink()
        checkpoint.rmdir()
        checkpoint.symlink_to(target, target_is_directory=True)

        wrong = client.request(
            "DELETE",
            f"/api/v1/checkpoints/{row['id']}",
            json={"confirmation": "wrong"},
        )
        assert wrong.status_code == 422
        deleted = client.request(
            "DELETE",
            f"/api/v1/checkpoints/{row['id']}",
            json={"confirmation": "delete-me"},
        )
        assert deleted.status_code == 200
        assert sentinel.read_text() == "keep"
        assert not checkpoint.exists()


def test_duplicate_output_directory_is_blocked_while_first_job_is_queued(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", GateGPUMonitor)
    pretrained = tmp_path / "pretrained"
    (pretrained / "Qwen2.5-VL-3B-Instruct").mkdir(parents=True)
    data = tmp_path / "libero"
    data.mkdir()
    output = tmp_path / "outputs"
    spec = {
        "architecture": {"backbone": "qwen2_5_vl", "action_head": "mlp_regression"},
        "training": {"method": "imitation_learning"},
        "dataset": {"id": "libero", "mix": "libero_goal"},
        "resources": {"allocation": "auto", "num_gpus": 1, "gpu_ids": []},
        "parameters": {"run_id": "duplicate", "output_root_dir": str(output)},
        "expert_overrides": {},
    }
    with TestClient(make_app(tmp_path)) as client:
        client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        client.patch(
            "/api/v1/settings",
            json={
                "environment": {
                    "PRETRAINED_MODELS_DIR": str(pretrained),
                    "LIBERO_DATA_ROOT": str(data),
                },
                "results_roots": [str(output)],
                "disk_min_free_gib": 0,
                "disk_min_free_percent": 0,
            },
        )
        payload = {"name": "duplicate", "spec": spec, "acknowledge_experimental": False}
        first = client.post("/api/v1/experiments/submit", json=payload)
        assert first.status_code == 200, first.text
        second_preflight = client.post("/api/v1/experiments/preflight", json=payload)
        assert second_preflight.status_code == 200
        assert second_preflight.json()["can_submit"] is False
        assert "output_directory_reserved" in {item["code"] for item in second_preflight.json()["issues"]}
        second = client.post("/api/v1/experiments/submit", json=payload)
        assert second.status_code == 422


def test_job_api_reports_fifo_position_progress_and_latest_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.app.GPUMonitor", GateGPUMonitor)
    app = make_app(tmp_path)
    metrics_path = tmp_path / "first-metrics.jsonl"
    metrics_path.write_text('{"phase":"train","step":25,"metrics":{"loss":0.5}}\n')
    with TestClient(app) as client:
        setup = client.post(
            "/api/v1/setup",
            json={"mode": "personal", "username": "admin", "display_name": "Admin", "password": ""},
        )
        user_id = setup.json()["user"]["id"]
        with app.state.database.session() as db:
            experiment = Experiment(
                owner_id=user_id,
                name="queue",
                family="test",
                status="queued",
                spec={"parameters": {"max_train_steps": 100}},
                resolved={},
            )
            db.add(experiment)
            db.flush()
            job_ids = []
            for index in range(2):
                stage = ExperimentStage(
                    experiment_id=experiment.id,
                    name=f"stage-{index}",
                    phase="train",
                    position=index,
                    resolved={},
                )
                db.add(stage)
                db.flush()
                job = Job(
                    experiment_id=experiment.id,
                    stage_id=stage.id,
                    owner_id=user_id,
                    status="queued",
                    cwd=str(tmp_path),
                    output_dir=str(tmp_path / f"queue-{index}"),
                    metrics_path=str(metrics_path if index == 0 else tmp_path / "missing.jsonl"),
                )
                db.add(job)
                db.flush()
                job_ids.append(job.id)

        rows = {row["id"]: row for row in client.get("/api/v1/jobs").json()}
        assert rows[job_ids[0]]["queue_position"] == 1
        assert rows[job_ids[1]]["queue_position"] == 2
        detail = client.get(f"/api/v1/jobs/{job_ids[0]}").json()
        assert detail["latest_metrics"] == {"loss": 0.5}
        assert detail["progress"] == 0.25
