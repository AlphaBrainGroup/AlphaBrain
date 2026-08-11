from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from alphabrain_ui.database import (
    Database,
    DeploymentGPUReservation,
    EvaluationGPUReservation,
    EvaluationRun,
    ModelDeployment,
    User,
)
from alphabrain_ui.evaluation_preflight import validate_evaluation_request
from alphabrain_ui.evaluation_groups import expand_evaluation_request
from alphabrain_ui.evaluations import EvaluationManager, POLICY_API_KEY_ENV
from alphabrain_ui.runtime import RuntimeConfig
from alphabrain_ui.schemas import EvaluationRequest
from alphabrain_ui.secrets import SecureSecretStore
from alphabrain_ui.services import SettingsService
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


class NoGPUProbe:
    available = False
    error = "NVML must not be consulted"

    def snapshot(self, _reservations=None):  # type: ignore[no-untyped-def]
        raise AssertionError("managed deployment reuse must not allocate another GPU")


def _runtime(tmp_path: Path, database_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        repo_root=REPO_ROOT,
        state_dir=tmp_path / "state",
        database_path=database_path,
        frontend_dist=tmp_path / "dist",
        scheduler_interval=0.01,
    )


def _seed(tmp_path: Path, *, status: str = "running") -> tuple[Database, RuntimeConfig, str, str]:
    database_path = tmp_path / "state/ui.sqlite3"
    database = Database(database_path)
    database.create_all()
    runtime = _runtime(tmp_path, database_path)
    libero_home = tmp_path / "LIBERO"
    libero_home.mkdir(parents=True)
    with database.session() as db:
        owner = User(
            username="local",
            display_name="Local",
            role="administrator",
            is_local=True,
        )
        db.add(owner)
        db.flush()
        deployment = ModelDeployment(
            owner_id=owner.id,
            name="managed",
            checkpoint_path=str(tmp_path / "checkpoint"),
            combination_id="deploy_qwen2_5_oft",
            adapter_id="base_framework_websocket",
            backbone_id="qwen2_5_vl",
            action_head_id="mlp_regression",
            status=status,
            assigned_gpu_ids=[3],
            port=18765,
            api_key_hash="a" * 64,
            api_key_prefix="ab_test",
            parameters={"precision": "bf16"},
        )
        db.add(deployment)
        db.flush()
        db.add(DeploymentGPUReservation(gpu_index=3, deployment_id=deployment.id))
        SettingsService(db).set(
            "environment",
            {"LIBERO_HOME": str(libero_home), "LIBERO_PYTHON": sys.executable},
        )
        SettingsService(db).set("results_roots", [str(tmp_path / "results")])
        return database, runtime, owner.id, deployment.id


def _request(deployment_id: str) -> EvaluationRequest:
    return EvaluationRequest.model_validate(
        {
            "name": "reuse deployment",
            "source_kind": "managed_deployment",
            "deployment_id": deployment_id,
            "benchmark_id": "libero",
            "suite": "libero_goal",
        }
    )


def _seed_evaluation(
    database: Database,
    tmp_path: Path,
    *,
    owner_id: str,
    deployment_id: str,
    status: str,
) -> str:
    with database.session() as db:
        deployment = db.get(ModelDeployment, deployment_id)
        evaluation = EvaluationRun(
            owner_id=owner_id,
            source_kind="managed_deployment",
            deployment_id=deployment_id,
            name="managed evaluation",
            checkpoint_path=deployment.checkpoint_path,
            combination_id=deployment.combination_id,
            adapter_id=deployment.adapter_id,
            backbone_id=deployment.backbone_id,
            action_head_id=deployment.action_head_id,
            benchmark_id="libero",
            suite="libero_goal",
            parameters={"num_trials": 1},
            model_parameters={},
            status=status,
            requested_gpu_count=0,
            assigned_gpu_ids=[] if status == "queued" else [3],
            server_port=None if status == "queued" else 18765,
            environment={POLICY_API_KEY_ENV: "must-not-be-persisted"},
            cwd=str(REPO_ROOT),
            output_dir=str(tmp_path / "managed-evaluation"),
        )
        db.add(evaluation)
        db.flush()
        return evaluation.id


def test_managed_preflight_skips_model_server_and_gpu_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, runtime, _owner_id, deployment_id = _seed(tmp_path)
    store = SecureSecretStore(runtime.state_dir, "deployment-controller")
    store.set(deployment_id, "ab_ctl_" + "x" * 40, min_length=32)
    monkeypatch.setattr(
        "alphabrain_ui.evaluation_preflight.probe_model_server_python",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model probe must be skipped")),
    )
    monkeypatch.setattr(
        "alphabrain_ui.evaluation_preflight.probe_benchmark_python",
        lambda *_args, **_kwargs: {"ok": True, "errors": {}},
    )

    with database.session() as db:
        result = validate_evaluation_request(
            _request(deployment_id),
            db=db,
            runtime=runtime,
            gpu_monitor=NoGPUProbe(),
            include_experimental=False,
            controller_secret_store=store,
        )

    assert result["ok"] is True, result["issues"]
    assert result["resolved"]["checkpoint_path"].endswith("checkpoint")
    assert result["resolved"]["combination_id"] == "deploy_qwen2_5_oft"
    assert result["resolved"]["resources"] == {
        "strategy": "managed_deployment",
        "gpu_count": 0,
        "gpu_ids": [],
        "assigned_gpu_ids": [3],
        "single_gpu_mode": True,
        "visible_gpu_ids": [3],
        "reuse_deployment": True,
    }
    assert POLICY_API_KEY_ENV not in json.dumps(result)


@pytest.mark.parametrize(
    ("deployment_status", "configure_key", "expected_code"),
    [
        ("stopped", True, "managed_deployment_not_running"),
        ("running", False, "managed_deployment_controller_key_missing"),
    ],
)
def test_managed_preflight_blocks_unsafe_deployments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deployment_status: str,
    configure_key: bool,
    expected_code: str,
) -> None:
    database, runtime, _owner_id, deployment_id = _seed(tmp_path, status=deployment_status)
    store = SecureSecretStore(runtime.state_dir, "deployment-controller")
    if configure_key:
        store.set(deployment_id, "ab_ctl_" + "x" * 40, min_length=32)
    monkeypatch.setattr(
        "alphabrain_ui.evaluation_preflight.probe_benchmark_python",
        lambda *_args, **_kwargs: {"ok": True, "errors": {}},
    )

    with database.session() as db:
        result = validate_evaluation_request(
            _request(deployment_id),
            db=db,
            runtime=runtime,
            gpu_monitor=NoGPUProbe(),
            include_experimental=False,
            controller_secret_store=store,
        )

    assert result["ok"] is False
    assert expected_code in {item["code"] for item in result["issues"]}


def test_managed_scheduler_reuses_deployment_gpu_and_port_without_reservation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database, runtime, owner_id, deployment_id = _seed(tmp_path)
        store = SecureSecretStore(runtime.state_dir, "deployment-controller")
        store.set(deployment_id, "ab_ctl_" + "x" * 40, min_length=32)
        evaluation_id = _seed_evaluation(
            database,
            tmp_path,
            owner_id=owner_id,
            deployment_id=deployment_id,
            status="queued",
        )
        manager = EvaluationManager(
            runtime,
            database,
            NoGPUProbe(),
            controller_secret_store=store,
        )
        spawned: list[str] = []

        async def record_spawn(value: str) -> None:
            spawned.append(value)

        manager._spawn = record_spawn  # type: ignore[method-assign]
        await manager.start_queued_unlocked(evaluation_id)

        assert spawned == [evaluation_id]
        with database.session() as db:
            evaluation = db.get(EvaluationRun, evaluation_id)
            assert evaluation.status == "starting"
            assert evaluation.server_port == 18765
            assert evaluation.assigned_gpu_ids == [3]
            assert evaluation.requested_gpu_count == 0
            assert db.query(EvaluationGPUReservation).count() == 0

    asyncio.run(scenario())


def test_managed_spawn_injects_controller_key_only_into_child_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        database, runtime, owner_id, deployment_id = _seed(tmp_path)
        secret = "ab_ctl_" + "s" * 40
        store = SecureSecretStore(runtime.state_dir, "deployment-controller")
        store.set(deployment_id, secret, min_length=32)
        evaluation_id = _seed_evaluation(
            database,
            tmp_path,
            owner_id=owner_id,
            deployment_id=deployment_id,
            status="starting",
        )
        manager = EvaluationManager(
            runtime,
            database,
            NoGPUProbe(),
            controller_secret_store=store,
        )
        waiting = asyncio.Event()
        captured: dict[str, object] = {}

        class FakeProcess:
            pid = 987654321

            async def wait(self) -> int:
                await waiting.wait()
                return 0

        async def fake_subprocess(*command: str, **kwargs):  # type: ignore[no-untyped-def]
            captured["command"] = list(command)
            captured["env"] = dict(kwargs["env"])
            return FakeProcess()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
        await manager._spawn(evaluation_id)
        task = manager._waiters[evaluation_id]
        try:
            child_env = captured["env"]
            command = captured["command"]
            assert child_env[POLICY_API_KEY_ENV] == secret
            assert secret not in json.dumps(command)
            with database.session() as db:
                evaluation = db.get(EvaluationRun, evaluation_id)
                assert evaluation.environment == {}
                assert secret not in json.dumps(evaluation.command)
                config_text = Path(evaluation.config_path).read_text(encoding="utf-8")
                log_text = Path(evaluation.log_path).read_text(encoding="utf-8")
                assert secret not in config_text
                assert secret not in log_text
                assert "reuse_server: true" in config_text
                assert "server_entrypoint: ''" in config_text
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_managed_source_schema_does_not_require_checkpoint() -> None:
    payload = _request("00000000-0000-0000-0000-000000000000")
    assert payload.checkpoint_source is None
    with pytest.raises(ValueError):
        EvaluationRequest.model_validate(
            {
                "name": "unsupported",
                "kind": "cl_matrix",
                "source_kind": "managed_deployment",
                "deployment_id": "00000000-0000-0000-0000-000000000000",
                "benchmark_id": "libero",
            }
        )

    batch = EvaluationRequest.model_validate(
        {
            "name": "batch",
            "kind": "batch",
            "source_kind": "managed_deployment",
            "deployment_id": "00000000-0000-0000-0000-000000000000",
            "benchmark_id": "libero",
            "suites": ["libero_goal", "libero_object"],
        }
    )
    children = expand_evaluation_request(batch)
    assert [child.request.suite for child in children] == ["libero_goal", "libero_object"]
    assert all(child.request.checkpoint_source is None for child in children)


def test_run_eval_reuse_mode_never_starts_or_stops_policy_server(tmp_path: Path) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.1)
    port = listener.getsockname()[1]
    stop = threading.Event()

    def accept_connections() -> None:
        while not stop.is_set():
            try:
                connection, _address = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            connection.close()

    thread = threading.Thread(target=accept_connections, daemon=True)
    thread.start()
    fake_python = tmp_path / "fake-libero-python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "path = sys.argv[sys.argv.index('--args.result-out-path') + 1]\n"
        "value = {\n"
        " 'schema_version': 'evaluation-result-v1', 'status': 'completed',\n"
        " 'benchmark': 'libero', 'checkpoint': '/managed/checkpoint',\n"
        " 'suite': {'name': 'libero_goal'},\n"
        " 'started_at': '2026-07-16T00:00:00Z',\n"
        " 'finished_at': '2026-07-16T00:00:01Z', 'duration_seconds': 1.0,\n"
        " 'summary': {'num_tasks': 0, 'num_episodes': 0, 'num_successes': 0, 'success_rate': 0.0},\n"
        " 'tasks': [], 'episodes': [], 'videos': [], 'parameters': {},\n"
        " 'metadata': {'controller_key_present': bool(os.environ.get('ALPHABRAIN_POLICY_API_KEY'))},\n"
        " 'error': None}\n"
        "with open(path, 'w', encoding='utf-8') as stream: json.dump(value, stream)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output_dir = tmp_path / "runner-output"
    config_path = tmp_path / "reuse.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "modes": {
                    "ui_evaluation": {
                        "type": "eval",
                        "reuse_server": True,
                        "checkpoint": "/managed/checkpoint",
                        "benchmark": "libero",
                        "task_suite": "libero_goal",
                        "num_trials": 1,
                        "host": "127.0.0.1",
                        "port": port,
                        "gpu_id": 3,
                        "server_python": sys.executable,
                        "server_entrypoint": "",
                        "server_args": [],
                        "client_entrypoint": "benchmarks/LIBERO/eval/eval_libero.py",
                        "task_limit": 1,
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    secret = "ab_ctl_" + "z" * 40
    environment = os.environ.copy()
    environment.update(
        {
            "ALPHABRAIN_UI_MANAGED": "1",
            "ALPHABRAIN_POLICY_API_KEY": secret,
            "EVAL_OUTPUT_DIR": str(output_dir),
            "LIBERO_PYTHON": str(fake_python),
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    try:
        completed = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts/run_eval.sh"), "ui_evaluation", str(config_path)],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "Reusing managed policy server" in completed.stdout
        assert "Starting policy server" not in completed.stdout
        assert not (output_dir / "server.log").exists()
        assert listener.fileno() >= 0
        result = json.loads((output_dir / "evaluation-result-v1.json").read_text(encoding="utf-8"))
        assert result["metadata"]["controller_key_present"] is True
        assert secret not in completed.stdout
        assert secret not in completed.stderr
        for path in output_dir.rglob("*"):
            if path.is_file():
                assert secret.encode() not in path.read_bytes()
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=1)
