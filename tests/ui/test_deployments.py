from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

import psutil

from alphabrain_ui.database import Database, DeploymentGPUReservation, ModelDeployment, User
from alphabrain_ui.deployments import DeploymentManager
from alphabrain_ui.runtime import RuntimeConfig


def _runtime(tmp_path: Path, database_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        repo_root=tmp_path,
        state_dir=tmp_path,
        database_path=database_path,
        frontend_dist=tmp_path / "dist",
        scheduler_interval=0.01,
    )


def _seed_deployment(
    database: Database,
    tmp_path: Path,
    *,
    status: str = "starting",
    pid: int | None = None,
    pgid: int | None = None,
    created_at: float | None = None,
    parameters: dict | None = None,
) -> tuple[str, str]:
    digest = hashlib.sha256(b"deployment-test-key").hexdigest()
    with database.session() as db:
        user = User(username="local", display_name="Local", role="administrator", is_local=True)
        db.add(user)
        db.flush()
        deployment = ModelDeployment(
            owner_id=user.id,
            name="Test deployment",
            checkpoint_path=str(tmp_path / "checkpoint"),
            combination_id="deploy_qwen2_5_oft",
            adapter_id="base_framework_websocket",
            backbone_id="qwen2_5_vl",
            action_head_id="mlp_regression",
            status=status,
            requested_gpu_count=1,
            requested_gpu_ids=[0],
            assigned_gpu_ids=[0],
            parameters=parameters
            or {
                "precision": "bf16",
                "_startup_timeout_seconds": 37,
            },
            bind_host="127.0.0.1",
            advertised_host="127.0.0.1",
            port=12093,
            idle_timeout_seconds=77,
            api_key_hash=digest,
            api_key_prefix="ab_test",
            command=[],
            environment={},
            log_path=str(tmp_path / "deployment.log"),
            pid=pid,
            pgid=pgid,
            process_created_at=created_at,
        )
        db.add(deployment)
        db.flush()
        db.add(DeploymentGPUReservation(gpu_index=0, deployment_id=deployment.id))
        return deployment.id, digest


async def _start_process(tmp_path: Path, *, ignore_term: bool) -> asyncio.subprocess.Process:
    ready = tmp_path / f"ready-{ignore_term}"
    handler = "signal.signal(signal.SIGTERM, lambda *_: None);" if ignore_term else ""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        f"import pathlib,signal,time;{handler}pathlib.Path({str(ready)!r}).write_text('ready');time.sleep(30)",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = asyncio.get_running_loop().time() + 2
    while not ready.exists() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    assert ready.exists()
    return process


def test_runtime_command_uses_registry_and_only_persists_redacted_digest(tmp_path: Path) -> None:
    database_path = tmp_path / "command.sqlite3"
    database = Database(database_path)
    database.create_all()
    deployment_id, digest = _seed_deployment(
        database,
        tmp_path,
        parameters={
            "precision": "bf16",
            "port": 19999,
            "idle_timeout_seconds": 1,
            "_startup_timeout_seconds": 37,
            "_restart_after_stop": "rotate",
        },
    )
    manager = DeploymentManager(_runtime(tmp_path, database_path), database, object())

    with database.session() as db:
        deployment = db.get(ModelDeployment, deployment_id)
        command, persisted, startup_timeout = manager._runtime_command(deployment, sys.executable)

    assert command[command.index("--port") + 1] == "12093"
    assert command[command.index("--idle_timeout") + 1] == "77"
    assert command[command.index("--api-key-sha256") + 1] == digest
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--deployment-id") + 1] == deployment_id
    assert persisted[persisted.index("--api-key-sha256") + 1] == "[REDACTED]"
    assert digest not in json.dumps(persisted)
    assert startup_timeout == 37


def test_spawn_failure_cannot_persist_digest_from_exception(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "spawn-failure.sqlite3"
        database = Database(database_path)
        database.create_all()
        deployment_id, digest = _seed_deployment(database, tmp_path)
        manager = DeploymentManager(_runtime(tmp_path, database_path), database, object())

        async def fail_spawn(*args, **kwargs):
            raise RuntimeError(f"synthetic process error containing {digest}")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_spawn)
        await manager._spawn(deployment_id)

        with database.session() as db:
            deployment = db.get(ModelDeployment, deployment_id)
            persisted = json.dumps(
                {
                    "command": deployment.command,
                    "error": deployment.error,
                    "parameters": deployment.parameters,
                    "environment": deployment.environment,
                }
            )
            assert deployment.status == "failed"
            assert deployment.error == "Failed to start model server."
            assert digest not in persisted
            assert db.query(DeploymentGPUReservation).count() == 0
        assert digest not in (tmp_path / "deployment.log").read_text()

    asyncio.run(scenario())


def test_graceful_stop_escalates_to_sigkill_and_detached_watcher_releases_gpu(tmp_path: Path) -> None:
    async def scenario() -> None:
        process = await _start_process(tmp_path, ignore_term=True)
        try:
            database_path = tmp_path / "stop.sqlite3"
            database = Database(database_path)
            database.create_all()
            created_at = psutil.Process(process.pid).create_time()
            deployment_id, _ = _seed_deployment(
                database,
                tmp_path,
                status="running",
                pid=process.pid,
                pgid=process.pid,
                created_at=created_at,
            )
            manager = DeploymentManager(
                _runtime(tmp_path, database_path),
                database,
                object(),
                graceful_stop_timeout=0.05,
            )
            watcher = asyncio.create_task(
                manager._watch_detached(deployment_id, process.pid, process.pid, created_at)
            )
            manager._waiters[deployment_id] = watcher

            result = await manager.stop(deployment_id)
            assert result.status == "stopping"
            assert await asyncio.wait_for(process.wait(), timeout=2) == -signal.SIGKILL
            await asyncio.wait_for(watcher, timeout=2)

            with database.session() as db:
                deployment = db.get(ModelDeployment, deployment_id)
                assert deployment.status == "stopped"
                assert deployment.error == ""
                assert deployment.pid is None
                assert db.query(DeploymentGPUReservation).count() == 0
            assert deployment_id not in manager._waiters
            assert deployment_id not in manager._escalations
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

    asyncio.run(scenario())


def test_key_rotation_restarts_from_detached_watcher_without_releasing_gpu(tmp_path: Path) -> None:
    async def scenario() -> None:
        process = await _start_process(tmp_path, ignore_term=False)
        try:
            database_path = tmp_path / "rotate.sqlite3"
            database = Database(database_path)
            database.create_all()
            created_at = psutil.Process(process.pid).create_time()
            deployment_id, old_digest = _seed_deployment(
                database,
                tmp_path,
                status="running",
                pid=process.pid,
                pgid=process.pid,
                created_at=created_at,
            )
            manager = DeploymentManager(
                _runtime(tmp_path, database_path),
                database,
                object(),
                graceful_stop_timeout=0.5,
            )
            spawned = asyncio.Event()

            async def fake_spawn(requested_id: str) -> None:
                assert requested_id == deployment_id
                spawned.set()

            manager._spawn = fake_spawn
            watcher = asyncio.create_task(
                manager._watch_detached(deployment_id, process.pid, process.pid, created_at)
            )
            manager._waiters[deployment_id] = watcher

            result, raw_key = await manager.rotate_key(deployment_id)
            assert result.status == "stopping"
            await asyncio.wait_for(process.wait(), timeout=2)
            await asyncio.wait_for(spawned.wait(), timeout=2)
            await asyncio.wait_for(watcher, timeout=2)

            new_digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
            with database.session() as db:
                deployment = db.get(ModelDeployment, deployment_id)
                persisted = json.dumps(
                    {
                        "command": deployment.command,
                        "error": deployment.error,
                        "parameters": deployment.parameters,
                        "environment": deployment.environment,
                    }
                )
                assert deployment.status == "starting"
                assert deployment.api_key_hash == new_digest
                assert deployment.api_key_prefix == raw_key[:10]
                assert deployment.pid is None
                assert "_restart_after_stop" not in deployment.parameters
                assert db.query(DeploymentGPUReservation).count() == 1
                assert raw_key not in persisted
                assert old_digest not in persisted
                assert new_digest not in persisted
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

    asyncio.run(scenario())


def test_running_restart_uses_detached_watcher_and_preserves_key_and_gpu(tmp_path: Path) -> None:
    async def scenario() -> None:
        process = await _start_process(tmp_path, ignore_term=False)
        try:
            database_path = tmp_path / "restart.sqlite3"
            database = Database(database_path)
            database.create_all()
            created_at = psutil.Process(process.pid).create_time()
            deployment_id, original_digest = _seed_deployment(
                database,
                tmp_path,
                status="running",
                pid=process.pid,
                pgid=process.pid,
                created_at=created_at,
            )
            manager = DeploymentManager(
                _runtime(tmp_path, database_path),
                database,
                object(),
                graceful_stop_timeout=0.5,
            )
            spawned = asyncio.Event()

            async def fake_spawn(requested_id: str) -> None:
                assert requested_id == deployment_id
                spawned.set()

            manager._spawn = fake_spawn
            watcher = asyncio.create_task(
                manager._watch_detached(deployment_id, process.pid, process.pid, created_at)
            )
            manager._waiters[deployment_id] = watcher

            result = await manager.restart(deployment_id)
            assert result.status == "stopping"
            await asyncio.wait_for(process.wait(), timeout=2)
            await asyncio.wait_for(spawned.wait(), timeout=2)
            await asyncio.wait_for(watcher, timeout=2)

            with database.session() as db:
                deployment = db.get(ModelDeployment, deployment_id)
                assert deployment.status == "starting"
                assert deployment.api_key_hash == original_digest
                assert deployment.pid is None
                assert "_restart_after_stop" not in deployment.parameters
                assert db.query(DeploymentGPUReservation).count() == 1
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

    asyncio.run(scenario())


def test_fast_exit_rotation_persists_new_key_and_intent_before_signal(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "fast-rotation.sqlite3"
        database = Database(database_path)
        database.create_all()
        deployment_id, old_digest = _seed_deployment(
            database,
            tmp_path,
            status="running",
            pid=424_242,
            pgid=424_242,
            created_at=1.0,
        )
        manager = DeploymentManager(_runtime(tmp_path, database_path), database, object())
        manager._matches_process = lambda _deployment: True
        observed: dict[str, object] = {}
        spawned: list[str] = []

        def exit_during_signal(pgid: int, sig: int) -> None:
            assert (pgid, sig) == (424_242, signal.SIGTERM)
            with database.session() as db:
                deployment = db.get(ModelDeployment, deployment_id)
                observed.update(
                    status=deployment.status,
                    intent=deployment.parameters.get("_restart_after_stop"),
                    digest=deployment.api_key_hash,
                )
            raise ProcessLookupError

        async def fake_spawn(requested_id: str) -> None:
            spawned.append(requested_id)

        monkeypatch.setattr(os, "killpg", exit_during_signal)
        manager._spawn = fake_spawn
        result, raw_key = await manager.rotate_key(deployment_id)

        new_digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        assert observed == {"status": "stopping", "intent": "rotate", "digest": new_digest}
        assert observed["digest"] != old_digest
        assert spawned == [deployment_id]
        assert result.status == "starting"
        with database.session() as db:
            deployment = db.get(ModelDeployment, deployment_id)
            assert deployment.status == "starting"
            assert deployment.api_key_hash == new_digest
            assert "_restart_after_stop" not in deployment.parameters
            assert db.query(DeploymentGPUReservation).count() == 1

    asyncio.run(scenario())


def test_reconcile_completes_pending_in_place_restart_after_ui_was_offline(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "reconcile.sqlite3"
        database = Database(database_path)
        database.create_all()
        deployment_id, _ = _seed_deployment(
            database,
            tmp_path,
            status="stopping",
            pid=999_999_999,
            pgid=999_999_999,
            created_at=1.0,
            parameters={
                "precision": "bf16",
                "_startup_timeout_seconds": 37,
                "_restart_after_stop": "rotate",
            },
        )
        manager = DeploymentManager(_runtime(tmp_path, database_path), database, object())
        spawned: list[str] = []

        async def fake_spawn(requested_id: str) -> None:
            spawned.append(requested_id)

        manager._spawn = fake_spawn
        await manager._reconcile_existing()

        assert spawned == [deployment_id]
        with database.session() as db:
            deployment = db.get(ModelDeployment, deployment_id)
            assert deployment.status == "starting"
            assert deployment.pid is None
            assert deployment.stop_requested_at is None
            assert "_restart_after_stop" not in deployment.parameters
            assert db.query(DeploymentGPUReservation).count() == 1

    asyncio.run(scenario())
