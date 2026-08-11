from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from alphabrain_ui.database import (
    Checkpoint,
    Database,
    Experiment,
    ExperimentStage,
    GPUReservation,
    Job,
    ModelDeployment,
    User,
    new_id,
    utcnow,
)
from alphabrain_ui.gpu import GPUInfo
from alphabrain_ui.jobs import JobManager
from alphabrain_ui.runtime import RuntimeConfig
from alphabrain_ui.wandb import WandbSecretStore


class FakeGPUMonitor:
    available = True
    error = ""

    def snapshot(self, reservations=None):
        reservations = reservations or {}
        return [
            GPUInfo(
                index=0,
                uuid="GPU-fake",
                name="Fake GPU",
                memory_total_bytes=24 * 1024**3,
                memory_used_bytes=0,
                memory_free_bytes=24 * 1024**3,
                utilization_percent=0,
                temperature_c=30,
                processes=[],
                reserved_by_job_id=reservations.get(0),
                available=0 not in reservations,
            )
        ]


class FakeTwoGPUMonitor:
    available = True
    error = ""

    def snapshot(self, reservations=None):
        reservations = reservations or {}
        return [
            GPUInfo(
                index=index,
                uuid=f"GPU-{index}",
                name=f"Fake GPU {index}",
                memory_total_bytes=24 * 1024**3,
                memory_used_bytes=0,
                memory_free_bytes=24 * 1024**3,
                utilization_percent=0,
                temperature_c=30,
                processes=[],
                reserved_by_job_id=reservations.get(index),
                available=index not in reservations,
            )
            for index in range(2)
        ]


class SplitAvailabilityGPUMonitor:
    available = True
    error = ""

    def snapshot(self, reservations=None):  # type: ignore[no-untyped-def]
        reservations = reservations or {}
        return [
            GPUInfo(
                index=index,
                uuid=f"GPU-{index}",
                name=f"Fake GPU {index}",
                memory_total_bytes=24 * 1024**3,
                memory_used_bytes=0,
                memory_free_bytes=24 * 1024**3,
                utilization_percent=0,
                temperature_c=30,
                processes=[],
                reserved_by_job_id=reservations.get(index),
                available=index == 1 and index not in reservations,
            )
            for index in range(2)
        ]


class RecordingDeploymentManager:
    def __init__(self):
        self.started: list[str] = []

    async def start_queued_unlocked(self, deployment_id: str) -> None:
        self.started.append(deployment_id)

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


async def wait_for_status(database: Database, job_id: str, expected: set[str], timeout: float = 5) -> str:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        with database.session() as db:
            status = db.get(Job, job_id).status
        if status in expected:
            return status
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {expected}")


def seed_job(database: Database, root: Path, name: str, sleep: float) -> str:
    with database.session() as db:
        user = db.query(User).first()
        if user is None:
            user = User(username="local", display_name="Local", role="administrator", is_local=True)
            db.add(user)
            db.flush()
        experiment = Experiment(
            owner_id=user.id,
            name=name,
            family="test",
            spec={},
            resolved={},
            status="queued",
        )
        db.add(experiment)
        db.flush()
        stage = ExperimentStage(experiment_id=experiment.id, name="Test", phase="test", position=0, resolved={})
        db.add(stage)
        db.flush()
        job_id = new_id()
        output = root / name
        job = Job(
            id=job_id,
            experiment_id=experiment.id,
            stage_id=stage.id,
            owner_id=user.id,
            status="queued",
            requested_gpu_count=1,
            requested_gpu_ids=[],
            command=[
                sys.executable,
                "-c",
                (
                    "import pathlib,time; print('started',flush=True); "
                    f"time.sleep({sleep}); pathlib.Path(r'{output}').mkdir("
                    "parents=True,exist_ok=True)"
                ),
            ],
            environment={},
            cwd=str(root),
            output_dir=str(output),
            log_path=str(root / f"{job_id}.log"),
            metrics_path=str(output / "metrics.jsonl"),
        )
        db.add(job)
        return job_id


def test_strict_fifo_runs_one_job_at_a_time(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = Database(tmp_path / "jobs.sqlite3")
        database.create_all()
        cfg = RuntimeConfig(
            repo_root=tmp_path,
            state_dir=tmp_path,
            database_path=tmp_path / "jobs.sqlite3",
            frontend_dist=tmp_path / "dist",
            scheduler_interval=0.01,
        )
        first = seed_job(database, tmp_path, "first", 0.15)
        second = seed_job(database, tmp_path, "second", 0.01)
        manager = JobManager(cfg, database, FakeGPUMonitor())
        await manager.start()
        try:
            assert await wait_for_status(database, first, {"running", "completed"}) in {"running", "completed"}
            with database.session() as db:
                assert db.get(Job, second).status == "queued"
            assert await wait_for_status(database, first, {"completed"}) == "completed"
            assert await wait_for_status(database, second, {"completed"}) == "completed"
            assert (tmp_path / "first").is_dir()
            assert (tmp_path / "second").is_dir()
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_safe_stop_uses_process_group_and_releases_gpu(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = Database(tmp_path / "stop.sqlite3")
        database.create_all()
        cfg = RuntimeConfig(
            repo_root=tmp_path,
            state_dir=tmp_path,
            database_path=tmp_path / "stop.sqlite3",
            frontend_dist=tmp_path / "dist",
            scheduler_interval=0.01,
        )
        job_id = seed_job(database, tmp_path, "long", 30)
        manager = JobManager(cfg, database, FakeGPUMonitor())
        await manager.start()
        try:
            assert await wait_for_status(database, job_id, {"running"}) == "running"
            await manager.request_stop(job_id)
            assert await wait_for_status(database, job_id, {"stopped"}) == "stopped"
            with database.session() as db:
                assert db.query(GPUReservation).count() == 0
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_fast_processes_do_not_leave_starting_jobs_or_gpu_reservations(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = Database(tmp_path / "fast.sqlite3")
        database.create_all()
        cfg = RuntimeConfig(
            repo_root=tmp_path,
            state_dir=tmp_path,
            database_path=tmp_path / "fast.sqlite3",
            frontend_dist=tmp_path / "dist",
            scheduler_interval=0.001,
        )
        job_ids = []
        for index in range(30):
            job_id = seed_job(database, tmp_path, f"fast-{index}", 0)
            with database.session() as db:
                db.get(Job, job_id).command = ["/bin/true"]
            job_ids.append(job_id)

        manager = JobManager(cfg, database, FakeGPUMonitor())
        await manager.start()
        try:
            for job_id in job_ids:
                assert await wait_for_status(database, job_id, {"completed"}) == "completed"
            with database.session() as db:
                assert db.query(GPUReservation).count() == 0
                assert db.query(Job).filter(Job.status == "starting").count() == 0
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_mixed_completed_and_cancelled_stages_mark_experiment_stopped(tmp_path: Path) -> None:
    database = Database(tmp_path / "mixed.sqlite3")
    database.create_all()
    with database.session() as db:
        user = User(username="local", role="administrator", is_local=True)
        db.add(user)
        db.flush()
        experiment = Experiment(owner_id=user.id, name="mixed", family="test", spec={}, resolved={}, status="queued")
        db.add(experiment)
        db.flush()
        for position, status in enumerate(("completed", "cancelled")):
            stage = ExperimentStage(
                experiment_id=experiment.id,
                name=f"stage-{position}",
                phase="test",
                position=position,
                resolved={},
            )
            db.add(stage)
            db.flush()
            db.add(
                Job(
                    experiment_id=experiment.id,
                    stage_id=stage.id,
                    owner_id=user.id,
                    status=status,
                    cwd=str(tmp_path),
                    output_dir=str(tmp_path / f"mixed-{position}"),
                )
            )
        JobManager._refresh_experiment_statuses(db)
        assert experiment.status == "stopped"


def test_parallel_jobs_receive_distinct_main_process_ports(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = Database(tmp_path / "ports.sqlite3")
        database.create_all()
        cfg = RuntimeConfig(
            repo_root=tmp_path,
            state_dir=tmp_path,
            database_path=tmp_path / "ports.sqlite3",
            frontend_dist=tmp_path / "dist",
            scheduler_interval=0.005,
        )
        job_ids = [seed_job(database, tmp_path, f"port-{index}", 0.25) for index in range(2)]
        for index, job_id in enumerate(job_ids):
            port_file = tmp_path / f"port-{index}.txt"
            with database.session() as db:
                job = db.get(Job, job_id)
                job.environment = {
                    "MASTER_PORT": "{main_process_port}",
                    "ALPHABRAIN_UI_PREFERRED_PORT": "29500",
                }
                job.command = [
                    sys.executable,
                    "-c",
                    (
                        "import os,pathlib,time; "
                        f"pathlib.Path(r'{port_file}').write_text(os.environ['MASTER_PORT']); "
                        "time.sleep(0.25)"
                    ),
                ]
        manager = JobManager(cfg, database, FakeTwoGPUMonitor())
        await manager.start()
        try:
            for job_id in job_ids:
                assert await wait_for_status(database, job_id, {"running", "completed"}) in {"running", "completed"}
            for job_id in job_ids:
                assert await wait_for_status(database, job_id, {"completed"}) == "completed"
            ports = {int((tmp_path / f"port-{index}.txt").read_text()) for index in range(2)}
            assert len(ports) == 2
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_checkpoint_index_failure_cannot_leak_gpu_reservation(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = Database(tmp_path / "index-failure.sqlite3")
        database.create_all()
        cfg = RuntimeConfig(
            repo_root=tmp_path,
            state_dir=tmp_path,
            database_path=tmp_path / "index-failure.sqlite3",
            frontend_dist=tmp_path / "dist",
            scheduler_interval=0.005,
        )
        job_id = seed_job(database, tmp_path, "index-failure", 0)
        manager = JobManager(cfg, database, FakeGPUMonitor())

        def fail_index(_db, _job):
            raise PermissionError("unreadable checkpoint")

        manager._index_checkpoints = fail_index
        await manager.start()
        try:
            assert await wait_for_status(database, job_id, {"completed"}) == "completed"
            await asyncio.sleep(0.02)
            with database.session() as db:
                assert db.query(GPUReservation).count() == 0
            assert job_id not in manager._waiters
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_checkpoint_index_understands_current_names_and_skips_symlinks(tmp_path: Path) -> None:
    database = Database(tmp_path / "checkpoints.sqlite3")
    database.create_all()
    job_id = seed_job(database, tmp_path, "checkpoint-run", 0)
    output = tmp_path / "checkpoint-run"
    root = output / "checkpoints"
    root.mkdir(parents=True)
    (root / "steps_10000_model.safetensors").write_bytes(b"weights")
    vla = root / "checkpoint-200" / "vla"
    vla.mkdir(parents=True)
    (vla / "model.safetensors").write_bytes(b"weights")
    os.symlink(tmp_path, root / "unsafe-link")
    with database.session() as db:
        db.get(Job, job_id).status = "completed"
    cfg = RuntimeConfig(
        repo_root=tmp_path,
        state_dir=tmp_path,
        database_path=tmp_path / "checkpoints.sqlite3",
        frontend_dist=tmp_path / "dist",
    )
    JobManager(cfg, database, FakeGPUMonitor()).refresh_checkpoints()
    with database.session() as db:
        rows = {row.name: row for row in db.query(Checkpoint).all()}
        assert set(rows) == {"steps_10000_model.safetensors", "checkpoint-200"}
        assert rows["steps_10000_model.safetensors"].step == 10000
        assert rows["steps_10000_model.safetensors"].is_complete is True
        assert rows["checkpoint-200"].step == 200
        assert rows["checkpoint-200"].is_complete is True


def test_launch_environment_precedence_is_dotenv_then_process_then_ui(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        (tmp_path / ".env").write_text("WANDB_BASE_URL=dotenv\nWANDB_MODE=dotenv\n")
        monkeypatch.setenv("WANDB_BASE_URL", "process")
        monkeypatch.setenv("WANDB_MODE", "process")
        database = Database(tmp_path / "environment.sqlite3")
        database.create_all()
        job_id = seed_job(database, tmp_path, "environment", 0)
        output = tmp_path / "environment.json"
        with database.session() as db:
            job = db.get(Job, job_id)
            job.environment = {"WANDB_MODE": "ui"}
            job.command = [
                sys.executable,
                "-c",
                (
                    "import json,os,pathlib; "
                    f"pathlib.Path(r'{output}').write_text(json.dumps({{"
                    "'WANDB_BASE_URL': os.environ['WANDB_BASE_URL'], "
                    "'WANDB_MODE': os.environ['WANDB_MODE']}))"
                ),
            ]
        cfg = RuntimeConfig(
            repo_root=tmp_path,
            state_dir=tmp_path,
            database_path=tmp_path / "environment.sqlite3",
            frontend_dist=tmp_path / "dist",
            scheduler_interval=0.005,
        )
        manager = JobManager(cfg, database, FakeGPUMonitor())
        await manager.start()
        try:
            assert await wait_for_status(database, job_id, {"completed"}) == "completed"
            assert json.loads(output.read_text()) == {"WANDB_BASE_URL": "process", "WANDB_MODE": "ui"}
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_wandb_key_is_injected_only_into_child_environment(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        secret = "secure-test-key-123456789"
        monkeypatch.setenv("WANDB_API_KEY", "inherited-key-must-not-win")
        database_path = tmp_path / "wandb-environment.sqlite3"
        database = Database(database_path)
        database.create_all()
        job_id = seed_job(database, tmp_path, "wandb-environment", 0)
        output = tmp_path / "wandb-child-result.txt"
        with database.session() as db:
            job = db.get(Job, job_id)
            job.environment = {
                "WANDB_API_KEY": "persisted-attempt-must-be-removed",
                "WANDB_MODE": "online",
                "ALPHABRAIN_WANDB_CATEGORIES": "metrics",
            }
            job.command = [
                sys.executable,
                "-c",
                (
                    "import os,pathlib; value=os.environ.get('WANDB_API_KEY',''); "
                    f"pathlib.Path(r'{output}').write_text('secure' if value.startswith('secure-') else 'wrong')"
                ),
            ]
        store = WandbSecretStore(tmp_path)
        store.set_api_key(secret)
        cfg = RuntimeConfig(
            repo_root=tmp_path,
            state_dir=tmp_path,
            database_path=database_path,
            frontend_dist=tmp_path / "dist",
            scheduler_interval=0.005,
        )
        manager = JobManager(cfg, database, FakeGPUMonitor(), wandb_secret_store=store)
        await manager.start()
        try:
            assert await wait_for_status(database, job_id, {"completed"}) == "completed"
            assert output.read_text() == "secure"
            with database.session() as db:
                job = db.get(Job, job_id)
                assert "WANDB_API_KEY" not in job.environment
                assert secret not in json.dumps(job.command)
            assert secret.encode() not in database_path.read_bytes()
            assert secret not in Path(tmp_path / f"{job_id}.log").read_text()
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_wandb_key_is_not_exposed_to_jobs_without_ui_wandb_config(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setenv("WANDB_API_KEY", "inherited-key-must-be-removed")
        database_path = tmp_path / "non-wandb.sqlite3"
        database = Database(database_path)
        database.create_all()
        job_id = seed_job(database, tmp_path, "non-wandb", 0)
        output = tmp_path / "non-wandb-child-result.txt"
        with database.session() as db:
            job = db.get(Job, job_id)
            job.environment = {"WANDB_MODE": "online"}
            job.command = [
                sys.executable,
                "-c",
                (
                    "import os,pathlib; "
                    f"pathlib.Path(r'{output}').write_text('present' if os.environ.get('WANDB_API_KEY') else 'absent')"
                ),
            ]
        store = WandbSecretStore(tmp_path)
        store.set_api_key("secure-test-key-123456789")
        manager = JobManager(
            RuntimeConfig(
                repo_root=tmp_path,
                state_dir=tmp_path,
                database_path=database_path,
                frontend_dist=tmp_path / "dist",
                scheduler_interval=0.005,
            ),
            database,
            FakeGPUMonitor(),
            wandb_secret_store=store,
        )
        await manager.start()
        try:
            assert await wait_for_status(database, job_id, {"completed"}) == "completed"
            assert output.read_text() == "absent"
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def _seed_queued_deployment(
    database: Database,
    root: Path,
    owner_id: str,
    *,
    queued_at,  # type: ignore[no-untyped-def]
    gpu_id: int,
) -> str:
    with database.session() as db:
        deployment = ModelDeployment(
            owner_id=owner_id,
            name="queued-deployment",
            checkpoint_path=str(root / "checkpoint"),
            combination_id="deploy_qwen2_5_oft",
            adapter_id="base_framework_websocket",
            backbone_id="qwen2_5_vl",
            action_head_id="mlp_regression",
            status="queued",
            requested_gpu_count=1,
            requested_gpu_ids=[gpu_id],
            assigned_gpu_ids=[],
            parameters={"precision": "fp32"},
            port=None,
            api_key_hash="a" * 64,
            api_key_prefix="ab_queue",
            log_path=str(root / "deployment.log"),
            queued_at=queued_at,
        )
        db.add(deployment)
        db.flush()
        return deployment.id


def test_global_fifo_blocks_later_deployment_when_training_head_waits_for_gpu(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "training-head.sqlite3"
        database = Database(database_path)
        database.create_all()
        job_id = seed_job(database, tmp_path, "training-head", 0)
        with database.session() as db:
            job = db.get(Job, job_id)
            job.queued_at = utcnow()
            job.requested_gpu_ids = [0]
            owner_id = job.owner_id
            deployment_time = job.queued_at + timedelta(seconds=1)
        deployment_id = _seed_queued_deployment(
            database,
            tmp_path,
            owner_id,
            queued_at=deployment_time,
            gpu_id=1,
        )
        deployment_manager = RecordingDeploymentManager()
        manager = JobManager(
            RuntimeConfig(
                repo_root=tmp_path,
                state_dir=tmp_path,
                database_path=database_path,
                frontend_dist=tmp_path / "dist",
            ),
            database,
            SplitAvailabilityGPUMonitor(),
            deployment_manager=deployment_manager,
        )

        await manager.tick()

        assert deployment_manager.started == []
        with database.session() as db:
            assert db.get(Job, job_id).status == "queued"
            assert db.get(ModelDeployment, deployment_id).status == "queued"

    asyncio.run(scenario())


def test_global_fifo_blocks_later_training_when_deployment_head_waits_for_gpu(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "deployment-head.sqlite3"
        database = Database(database_path)
        database.create_all()
        job_id = seed_job(database, tmp_path, "later-training", 0)
        with database.session() as db:
            job = db.get(Job, job_id)
            owner_id = job.owner_id
            deployment_time = utcnow()
            job.queued_at = deployment_time + timedelta(seconds=1)
            job.requested_gpu_ids = [1]
        deployment_id = _seed_queued_deployment(
            database,
            tmp_path,
            owner_id,
            queued_at=deployment_time,
            gpu_id=0,
        )
        deployment_manager = RecordingDeploymentManager()
        manager = JobManager(
            RuntimeConfig(
                repo_root=tmp_path,
                state_dir=tmp_path,
                database_path=database_path,
                frontend_dist=tmp_path / "dist",
            ),
            database,
            SplitAvailabilityGPUMonitor(),
            deployment_manager=deployment_manager,
        )

        await manager.tick()

        assert deployment_manager.started == [deployment_id]
        with database.session() as db:
            assert db.get(Job, job_id).status == "queued"
            assert db.get(ModelDeployment, deployment_id).status == "queued"

    asyncio.run(scenario())
