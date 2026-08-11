from __future__ import annotations

import asyncio
from pathlib import Path

from alphabrain_ui.database import Database, UtilityGPUReservation, UtilityRun, User
from alphabrain_ui.gpu import GPUInfo
from alphabrain_ui.jobs import JobManager
from alphabrain_ui.runtime import RuntimeConfig
from alphabrain_ui.secrets import HuggingFaceSecretStore
from alphabrain_ui.utilities import UtilityManager


REPO_ROOT = Path(__file__).resolve().parents[2]


class OneAvailableGPU:
    available = True
    error = ""

    def snapshot(self, reservations=None):  # type: ignore[no-untyped-def]
        reservations = reservations or {}
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
                reserved_by_job_id=reservations.get(0),
                available=0 not in reservations,
            )
        ]


def test_gpu_utility_reserves_shared_gpu_and_releases_it_after_exit(
    tmp_path: Path, monkeypatch
) -> None:
    async def scenario() -> None:
        runtime = RuntimeConfig(
            repo_root=REPO_ROOT,
            state_dir=tmp_path / "state",
            database_path=tmp_path / "state/ui.sqlite3",
            frontend_dist=tmp_path / "dist",
            scheduler_interval=0.01,
        )
        database = Database(runtime.database_path)
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

        utility_manager = UtilityManager(
            runtime,
            database,
            HuggingFaceSecretStore(runtime.state_dir),
            interval=0.01,
        )
        run = utility_manager.create_run(
            owner_id=owner_id,
            kind="lora_merge",
            command=["python", "fake-merge.py"],
            cwd=REPO_ROOT,
            output_path=str(tmp_path / "merged"),
            queue_class="gpu",
            requested_gpu_count=1,
        )
        finished = asyncio.Event()
        captured: dict[str, object] = {}

        class FakeProcess:
            pid = 999_999_981
            returncode = None

            async def wait(self) -> int:
                await finished.wait()
                self.returncode = 0
                return 0

        async def fake_subprocess(*command, **kwargs):  # type: ignore[no-untyped-def]
            captured["command"] = list(command)
            captured["env"] = dict(kwargs["env"])
            return FakeProcess()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
        scheduler = JobManager(
            runtime,
            database,
            OneAvailableGPU(),
            utility_manager=utility_manager,
        )
        await scheduler._start_gpu_utility(run.id)

        with database.session() as db:
            active = db.get(UtilityRun, run.id)
            assert active.status == "running"
            assert active.assigned_gpu_ids == [0]
            reservation = db.get(UtilityGPUReservation, 0)
            assert reservation.utility_run_id == run.id
        assert captured["command"] == ["python", "fake-merge.py"]
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"

        watcher = utility_manager._watchers[run.id]
        finished.set()
        await watcher
        with database.session() as db:
            completed = db.get(UtilityRun, run.id)
            assert completed.status == "completed"
            assert completed.exit_code == 0
            assert db.query(UtilityGPUReservation).count() == 0

    asyncio.run(scenario())
