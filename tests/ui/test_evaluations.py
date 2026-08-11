from __future__ import annotations

import asyncio
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from alphabrain_ui.database import (
    Database,
    DeploymentGPUReservation,
    EvaluationGPUReservation,
    EvaluationRun,
    ModelDeployment,
    User,
)
from alphabrain_ui.evaluations import EvaluationManager
from alphabrain_ui.runtime import RuntimeConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def _runtime(tmp_path: Path, database_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        repo_root=REPO_ROOT,
        state_dir=tmp_path / "state",
        database_path=database_path,
        frontend_dist=tmp_path / "dist",
        scheduler_interval=0.01,
    )


def _result(path: Path, *, status: str = "partial", successes: int = 1) -> dict:
    value = {
        "schema_version": "evaluation-result-v1",
        "status": status,
        "benchmark": "libero",
        "checkpoint": str(path.parent / "checkpoint"),
        "suite": {"name": "libero_goal"},
        "started_at": "2026-07-16T00:00:00Z",
        "finished_at": "2026-07-16T00:00:01Z",
        "duration_seconds": 1.0,
        "summary": {
            "num_tasks": 1,
            "num_episodes": 2,
            "num_successes": successes,
            "success_rate": successes / 2,
        },
        "tasks": [],
        "episodes": [],
        "videos": [],
        "parameters": {},
        "metadata": {},
        "error": "benchmark stopped before all episodes completed" if status != "completed" else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def _seed_evaluation(
    database: Database,
    tmp_path: Path,
    *,
    owner_id: str,
    name: str,
    status: str,
    wandb_status: str = "disabled",
    wandb: dict | None = None,
    pid: int | None = None,
    result_status: str | None = None,
) -> str:
    output_dir = tmp_path / name
    result_path = output_dir / "evaluation-result-v1.json"
    if result_status is not None:
        _result(result_path, status=result_status)
    with database.session() as db:
        evaluation = EvaluationRun(
            owner_id=owner_id,
            name=name,
            checkpoint_path=str(tmp_path / "checkpoint"),
            combination_id="deploy_qwen2_5_oft",
            adapter_id="base_framework_websocket",
            backbone_id="qwen2_5_vl",
            action_head_id="mlp_regression",
            benchmark_id="libero",
            suite="libero_goal",
            wandb=wandb or {},
            wandb_status=wandb_status,
            status=status,
            requested_gpu_ids=[0],
            assigned_gpu_ids=[0],
            cwd=str(tmp_path),
            output_dir=str(output_dir),
            result_path=str(result_path),
            pid=pid,
            pgid=pid,
        )
        db.add(evaluation)
        db.flush()
        if pid is not None:
            db.add(EvaluationGPUReservation(gpu_index=0, evaluation_id=evaluation.id))
        return evaluation.id


def _database_with_owner(tmp_path: Path) -> tuple[Database, str, Path]:
    database_path = tmp_path / "evaluation.sqlite3"
    database = Database(database_path)
    database.create_all()
    with database.session() as db:
        owner = User(
            username="local",
            display_name="Local",
            role="administrator",
            is_local=True,
        )
        db.add(owner)
        db.flush()
        owner_id = owner.id
    return database, owner_id, database_path


def test_reconcile_resumes_wandb_uploads_and_caches_partial_summary(tmp_path: Path) -> None:
    async def scenario() -> None:
        database, owner_id, database_path = _database_with_owner(tmp_path)
        wandb = {"enabled": True, "mode": "offline", "categories": ["summary"]}
        pending_id = _seed_evaluation(
            database,
            tmp_path,
            owner_id=owner_id,
            name="pending-upload",
            status="completed",
            wandb_status="pending",
            wandb=wandb,
        )
        uploading_id = _seed_evaluation(
            database,
            tmp_path,
            owner_id=owner_id,
            name="uploading-upload",
            status="completed",
            wandb_status="uploading",
            wandb=wandb,
        )
        partial_id = _seed_evaluation(
            database,
            tmp_path,
            owner_id=owner_id,
            name="partial-result",
            status="running",
            pid=999_999_991,
            result_status="partial",
        )
        manager = EvaluationManager(_runtime(tmp_path, database_path), database, object())
        scheduled: list[str] = []
        manager._schedule_upload = (  # type: ignore[method-assign]
            lambda evaluation_id, _secret_store=None: scheduled.append(evaluation_id)
        )

        await manager._reconcile_existing()

        assert scheduled == sorted((pending_id, uploading_id))
        with database.session() as db:
            assert db.get(EvaluationRun, pending_id).wandb_status == "pending"
            assert db.get(EvaluationRun, uploading_id).wandb_status == "pending"
            partial = db.get(EvaluationRun, partial_id)
            assert partial.status == "failed"
            assert partial.result_summary == {
                "num_tasks": 1,
                "num_episodes": 2,
                "num_successes": 1,
                "success_rate": 0.5,
            }
            assert db.query(EvaluationGPUReservation).count() == 0

    asyncio.run(scenario())


def test_managed_evaluation_injects_controller_key_only_into_child_environment(
    tmp_path: Path, monkeypatch
) -> None:
    async def scenario() -> None:
        database, owner_id, database_path = _database_with_owner(tmp_path)
        controller_key = "managed-controller-key-with-more-than-32-characters"
        with database.session() as db:
            deployment = ModelDeployment(
                owner_id=owner_id,
                name="managed",
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
            deployment_id = deployment.id
            db.add(DeploymentGPUReservation(gpu_index=0, deployment_id=deployment.id))
            evaluation = EvaluationRun(
                owner_id=owner_id,
                name="managed-evaluation",
                source_kind="managed_deployment",
                deployment_id=deployment.id,
                checkpoint_path=deployment.checkpoint_path,
                combination_id=deployment.combination_id,
                adapter_id=deployment.adapter_id,
                backbone_id=deployment.backbone_id,
                action_head_id=deployment.action_head_id,
                benchmark_id="libero",
                suite="libero_goal",
                parameters={"num_trials": 1},
                status="queued",
                requested_gpu_count=1,
                requested_gpu_ids=[],
                assigned_gpu_ids=[],
                environment={},
                command=[],
                cwd=str(REPO_ROOT),
                output_dir=str(tmp_path / "managed-evaluation"),
            )
            db.add(evaluation)
            db.flush()
            evaluation_id = evaluation.id

        class NoSnapshotMonitor:
            def snapshot(self, _reservations=None):  # type: ignore[no-untyped-def]
                raise AssertionError("managed deployment reuse must not request a second GPU")

        manager = EvaluationManager(
            _runtime(tmp_path, database_path), database, NoSnapshotMonitor()
        )
        manager.controller_secret_store.set(deployment_id, controller_key)
        captured: dict[str, object] = {}

        async def fake_subprocess(*command, **kwargs):  # type: ignore[no-untyped-def]
            captured["command"] = list(command)
            captured["env"] = dict(kwargs["env"])
            return SimpleNamespace(pid=999_999_987)

        async def hold_monitor(_evaluation_id, _process, log_handle):  # type: ignore[no-untyped-def]
            try:
                await asyncio.Event().wait()
            finally:
                log_handle.close()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
        manager._monitor_process = hold_monitor  # type: ignore[method-assign]
        await manager.start_queued_unlocked(evaluation_id)

        child_env = captured["env"]
        assert isinstance(child_env, dict)
        assert child_env["ALPHABRAIN_POLICY_API_KEY"] == controller_key
        assert controller_key not in " ".join(captured["command"])
        config_path = tmp_path / "managed-evaluation/evaluation-config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        mode = config["modes"]["ui_evaluation"]
        assert mode["reuse_server"] is True
        assert mode["host"] == "127.0.0.1"
        assert mode["port"] == 12345
        assert mode["server_entrypoint"] == ""
        assert mode["server_args"] == []
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o444

        with database.session() as db:
            stored = db.get(EvaluationRun, evaluation_id)
            assert stored.status == "running"
            assert stored.requested_gpu_count == 0
            assert stored.assigned_gpu_ids == [0]
            assert "ALPHABRAIN_POLICY_API_KEY" not in stored.environment
            assert controller_key not in json.dumps(stored.command)
            assert db.query(EvaluationGPUReservation).count() == 0
            reservation = db.get(DeploymentGPUReservation, 0)
            assert reservation.deployment_id == deployment_id

        await manager.shutdown()

    asyncio.run(scenario())


def test_detached_watcher_caches_failed_result_summary(tmp_path: Path) -> None:
    async def scenario() -> None:
        database, owner_id, database_path = _database_with_owner(tmp_path)
        evaluation_id = _seed_evaluation(
            database,
            tmp_path,
            owner_id=owner_id,
            name="detached-failed",
            status="running",
            pid=999_999_992,
            result_status="failed",
        )
        manager = EvaluationManager(_runtime(tmp_path, database_path), database, object())

        await manager._watch_detached(
            evaluation_id,
            999_999_992,
            999_999_992,
            None,
        )

        with database.session() as db:
            evaluation = db.get(EvaluationRun, evaluation_id)
            assert evaluation.status == "failed"
            assert evaluation.result_summary["num_episodes"] == 2
            assert evaluation.result_summary["success_rate"] == 0.5
            assert db.query(EvaluationGPUReservation).count() == 0

    asyncio.run(scenario())


def test_evaluation_runtime_servers_are_forced_to_loopback(tmp_path: Path) -> None:
    database_path = tmp_path / "unused.sqlite3"
    manager = EvaluationManager(
        _runtime(tmp_path, database_path),
        object(),
        object(),
    )
    pretrained = tmp_path / "cosmos-pretrained"
    pretrained.mkdir()
    cases = (
        (
            "base_framework_websocket",
            "deploy_qwen2_5_oft",
            "qwen2_5_vl",
            "mlp_regression",
            {"precision": "bf16"},
        ),
        (
            "cosmos_policy_websocket",
            "deploy_cosmos_policy",
            "cosmos2",
            "cosmos_policy_dit",
            {"pretrained_dir": str(pretrained)},
        ),
    )
    for adapter_id, combination_id, backbone_id, action_head_id, model_parameters in cases:
        evaluation = SimpleNamespace(
            adapter_id=adapter_id,
            checkpoint_path=str(tmp_path / f"{combination_id}-checkpoint"),
            combination_id=combination_id,
            backbone_id=backbone_id,
            action_head_id=action_head_id,
            model_parameters=model_parameters,
            parameters={},
            server_port=12093,
            suite="libero_goal",
            task_set="",
            split="",
            benchmark_id="libero",
            assigned_gpu_ids=[0],
            output_dir=str(tmp_path / f"{combination_id}-output"),
        )

        runtime_config, command = manager._runtime_config(
            evaluation,
            model_python=sys.executable,
        )
        server_args = runtime_config["modes"]["ui_evaluation"]["server_args"]

        assert command.count("--host") == 1
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert server_args.count("--host") == 1
        assert server_args[server_args.index("--host") + 1] == "127.0.0.1"


@pytest.mark.parametrize(
    "adapter_command, expected_host_argument",
    [
        ([sys.executable, "server.py", "--host", "0.0.0.0"], "--host"),
        ([sys.executable, "server.py", "--host=0.0.0.0"], "--host=127.0.0.1"),
    ],
)
def test_evaluation_runtime_rewrites_adapter_provided_hosts(
    tmp_path: Path,
    monkeypatch,
    adapter_command: list[str],
    expected_host_argument: str,
) -> None:
    server_path = REPO_ROOT / "server.py"
    command = [*adapter_command]
    command[1] = str(server_path)
    monkeypatch.setattr(
        "alphabrain_ui.evaluations.build_adapter_command",
        lambda *_args, **_kwargs: {"valid": True, "command": command, "issues": []},
    )
    manager = EvaluationManager(
        _runtime(tmp_path, tmp_path / "unused.sqlite3"),
        object(),
        object(),
    )
    evaluation = SimpleNamespace(
        adapter_id="fixture",
        checkpoint_path=str(tmp_path / "checkpoint"),
        combination_id="fixture",
        backbone_id="fixture",
        action_head_id="fixture",
        model_parameters={},
        parameters={},
        server_port=12093,
        suite="libero_goal",
        task_set="",
        split="",
        benchmark_id="libero",
        assigned_gpu_ids=[0],
        output_dir=str(tmp_path / "output"),
    )

    runtime_config, resolved = manager._runtime_config(evaluation, model_python=sys.executable)
    server_args = runtime_config["modes"]["ui_evaluation"]["server_args"]

    assert "0.0.0.0" not in " ".join(resolved)
    assert expected_host_argument in server_args
    if expected_host_argument == "--host":
        assert server_args[server_args.index("--host") + 1] == "127.0.0.1"
