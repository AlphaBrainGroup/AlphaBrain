from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
import socket
from pathlib import Path

import psutil
from sqlalchemy import delete, select

from .database import (
    Checkpoint,
    DeploymentGPUReservation,
    EvaluationGPUReservation,
    EvaluationRun,
    Experiment,
    GPUReservation,
    Job,
    ModelDeployment,
    UtilityGPUReservation,
    UtilityRun,
    utcnow,
)
from .gpu import GPUMonitor
from .process_control import FORCE_KILL_SIGNAL, new_process_group_kwargs, process_group_matches, signal_process_group
from .remote_training import RemoteTrainingConfig, build_remote_script
from .runtime import RuntimeConfig
from .services import SettingsService
from .wandb import WANDB_API_KEY_ENV, WANDB_CATEGORIES_ENV, WandbSecretStore

ACTIVE_STATUSES = {"starting", "running", "stopping"}
TERMINAL_STATUSES = {"completed", "failed", "stopped", "cancelled", "dependency_failed", "interrupted"}


def _replace_tokens(value: str, gpu_ids: list[int], main_process_port: int) -> str:
    mapping = {
        "{gpu_ids}": ",".join(str(x) for x in gpu_ids),
        "{first_gpu}": str(gpu_ids[0]) if gpu_ids else "0",
        "{gpu_count}": str(len(gpu_ids)),
        "{main_process_port}": str(main_process_port),
    }
    for token, replacement in mapping.items():
        value = value.replace(token, replacement)
    return value


def _port_is_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def _allocate_port(preferred: int, excluded: set[int]) -> int:
    if 1024 <= preferred <= 65535 and preferred not in excluded and _port_is_available(preferred):
        return preferred
    for _attempt in range(32):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in excluded:
            return port
    raise RuntimeError("Unable to allocate a free main-process port")


def _path_size(path: Path) -> int:
    if path.is_file():
        with contextlib.suppress(OSError):
            return path.stat().st_size
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            with contextlib.suppress(OSError):
                total += child.stat().st_size
    return total


class JobManager:
    def __init__(
        self,
        config: RuntimeConfig,
        database,
        gpu_monitor: GPUMonitor,
        *,
        lock: asyncio.Lock | None = None,
        deployment_manager=None,
        evaluation_manager=None,
        utility_manager=None,
        wandb_secret_store: WandbSecretStore | None = None,
    ):
        self.config = config
        self.database = database
        self.gpu_monitor = gpu_monitor
        self._stop = asyncio.Event()
        self._scheduler_task: asyncio.Task | None = None
        self._waiters: dict[str, asyncio.Task] = {}
        self._lock = lock or asyncio.Lock()
        self.deployment_manager = deployment_manager
        self.evaluation_manager = evaluation_manager
        self.utility_manager = utility_manager
        self.wandb_secret_store = wandb_secret_store or WandbSecretStore(config.state_dir)

    async def start(self) -> None:
        self._stop.clear()
        await self._reconcile_existing_jobs()
        if self.deployment_manager is not None:
            await self.deployment_manager.start()
        if self.evaluation_manager is not None:
            await self.evaluation_manager.start()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop(), name="alphabrain-ui-scheduler")

    async def shutdown(self) -> None:
        # Deliberately do not terminate training processes. They run in their
        # own process groups and are reconciled when the UI comes back.
        self._stop.set()
        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
        waiters = list(self._waiters.values())
        for task in waiters:
            task.cancel()
        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)
        if self.deployment_manager is not None:
            await self.deployment_manager.shutdown()
        if self.evaluation_manager is not None:
            await self.evaluation_manager.shutdown()

    async def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Scheduler failures must not crash the API. The next tick can
                # recover; endpoint health exposes the queue state.
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.scheduler_interval)
            except asyncio.TimeoutError:
                pass

    async def tick(self) -> None:
        async with self._lock:
            self._advance_dependencies()
            await self._start_queue_head()

    def _advance_dependencies(self) -> None:
        with self.database.session() as db:
            blocked = db.execute(select(Job).where(Job.status == "blocked").order_by(Job.queued_at)).scalars().all()
            for job in blocked:
                dependency_id = job.stage.dependency_stage_id
                if not dependency_id:
                    job.status = "queued"
                    continue
                dependency_jobs = (
                    db.execute(select(Job).where(Job.stage_id == dependency_id).order_by(Job.created_at.desc()))
                    .scalars()
                    .all()
                )
                if not dependency_jobs:
                    continue
                dep_status = dependency_jobs[0].status
                if dep_status == "completed":
                    job.status = "queued"
                    job.queued_at = utcnow()
                elif dep_status in TERMINAL_STATUSES:
                    job.status = "dependency_failed"
                    job.error = f"Dependency finished with status {dep_status}"
                    job.finished_at = utcnow()
            self._refresh_experiment_statuses(db)

    async def _start_queue_head(self) -> None:
        with self.database.session() as db:
            job_head = (
                db.execute(select(Job).where(Job.status == "queued").order_by(Job.queued_at, Job.created_at))
                .scalars()
                .first()
            )
            deployment_head = (
                db.execute(
                    select(ModelDeployment)
                    .where(ModelDeployment.status == "queued")
                    .order_by(ModelDeployment.queued_at, ModelDeployment.created_at)
                )
                .scalars()
                .first()
            )
            evaluation_head = (
                db.execute(
                    select(EvaluationRun)
                    .where(EvaluationRun.status == "queued")
                    .order_by(EvaluationRun.queued_at, EvaluationRun.created_at)
                )
                .scalars()
                .first()
            )
            utility_head = (
                db.execute(
                    select(UtilityRun)
                    .where(UtilityRun.status == "queued", UtilityRun.queue_class == "gpu")
                    .order_by(UtilityRun.queued_at, UtilityRun.created_at)
                )
                .scalars()
                .first()
            )
            candidates = []
            if job_head is not None:
                candidates.append((job_head.queued_at, job_head.created_at, "job", job_head.id))
            if deployment_head is not None:
                candidates.append(
                    (deployment_head.queued_at, deployment_head.created_at, "deployment", deployment_head.id)
                )
            if evaluation_head is not None:
                candidates.append(
                    (evaluation_head.queued_at, evaluation_head.created_at, "evaluation", evaluation_head.id)
                )
            if utility_head is not None:
                candidates.append(
                    (utility_head.queued_at, utility_head.created_at, "utility", utility_head.id)
                )
            queue_head = min(candidates, default=None, key=lambda item: (item[0], item[1], item[2]))

        if queue_head is None:
            return
        if queue_head[2] == "deployment" and self.deployment_manager is not None:
            await self.deployment_manager.start_queued_unlocked(queue_head[3])
            return
        if queue_head[2] == "evaluation" and self.evaluation_manager is not None:
            await self.evaluation_manager.start_queued_unlocked(queue_head[3])
            return
        if queue_head[2] == "utility" and self.utility_manager is not None:
            await self._start_gpu_utility(queue_head[3])
            return

        with self.database.session() as db:
            job = (
                db.execute(select(Job).where(Job.status == "queued").order_by(Job.queued_at, Job.created_at))
                .scalars()
                .first()
            )
            if job is None:
                return
            remote_config = RemoteTrainingConfig.from_settings(SettingsService(db).all())
            if remote_config.enabled:
                reservations = {
                    row.gpu_index: row.job_id
                    for row in db.execute(select(GPUReservation)).scalars().all()
                }
                configured = list(remote_config.gpu_ids)
                if job.requested_gpu_ids:
                    chosen = [int(value) for value in job.requested_gpu_ids]
                    if configured and any(index not in configured for index in chosen):
                        job.status = "failed"
                        job.error = f"Requested GPU IDs are not configured on the remote server: {chosen}"
                        job.finished_at = utcnow()
                        self._refresh_experiment_statuses(db)
                        return
                    if any(index in reservations for index in chosen):
                        return
                else:
                    available = [index for index in configured if index not in reservations]
                    if len(available) < job.requested_gpu_count:
                        if job.requested_gpu_count > len(configured):
                            job.status = "failed"
                            job.error = (
                                f"Requested {job.requested_gpu_count} GPUs, but only "
                                f"{len(configured)} remote GPU IDs are configured."
                            )
                            job.finished_at = utcnow()
                            self._refresh_experiment_statuses(db)
                        return
                    chosen = available[: job.requested_gpu_count]
                for index in chosen:
                    db.add(GPUReservation(gpu_index=index, job_id=job.id))
                job.assigned_gpu_ids = chosen
                job.status = "starting"
                job.started_at = utcnow()
                job_id = job.id
                self._refresh_experiment_statuses(db)
            else:
                reservations = {row.gpu_index: row.job_id for row in db.execute(select(GPUReservation)).scalars().all()}
                reservations.update(
                    {
                        row.gpu_index: f"deployment:{row.deployment_id}"
                        for row in db.execute(select(DeploymentGPUReservation)).scalars().all()
                    }
                )
                reservations.update(
                    {
                        row.gpu_index: f"utility:{row.utility_run_id}"
                        for row in db.execute(select(UtilityGPUReservation)).scalars().all()
                    }
                )
                reservations.update(
                    {
                        row.gpu_index: f"evaluation:{row.evaluation_id}"
                        for row in db.execute(select(EvaluationGPUReservation)).scalars().all()
                    }
                )
                snapshot = self.gpu_monitor.snapshot(reservations)
                if not snapshot:
                    return
                by_index = {gpu.index: gpu for gpu in snapshot}
                if job.requested_gpu_ids:
                    chosen = [int(x) for x in job.requested_gpu_ids]
                    missing = [idx for idx in chosen if idx not in by_index]
                    if missing:
                        job.status = "failed"
                        job.error = f"Requested GPU IDs are no longer visible: {missing}"
                        job.finished_at = utcnow()
                        self._refresh_experiment_statuses(db)
                        return
                    failed_inspection = [idx for idx in chosen if by_index[idx].error]
                    if failed_inspection:
                        job.status = "failed"
                        job.error = f"Unable to inspect requested GPU IDs: {failed_inspection}"
                        job.finished_at = utcnow()
                        self._refresh_experiment_statuses(db)
                        return
                    if any(not by_index[idx].available for idx in chosen):
                        return
                else:
                    healthy = [gpu for gpu in snapshot if not gpu.error]
                    if job.requested_gpu_count > len(healthy):
                        job.status = "failed"
                        job.error = (
                            f"Requested {job.requested_gpu_count} GPUs, but only "
                            f"{len(healthy)} can be inspected."
                        )
                        job.finished_at = utcnow()
                        self._refresh_experiment_statuses(db)
                        return
                    available = [gpu.index for gpu in snapshot if gpu.available]
                    if len(available) < job.requested_gpu_count:
                        return
                    chosen = available[: job.requested_gpu_count]
                for index in chosen:
                    db.add(GPUReservation(gpu_index=index, job_id=job.id))
                job.assigned_gpu_ids = chosen
                job.status = "starting"
                job.started_at = utcnow()
                job_id = job.id
                self._refresh_experiment_statuses(db)
        await self._spawn(job_id)

    async def _start_gpu_utility(self, run_id: str) -> None:
        chosen: list[int] = []
        with self.database.session() as db:
            run = db.get(UtilityRun, run_id)
            if run is None or run.status != "queued" or run.queue_class != "gpu":
                return
            reservations = {
                row.gpu_index: row.job_id
                for row in db.execute(select(GPUReservation)).scalars().all()
            }
            reservations.update(
                {
                    row.gpu_index: f"deployment:{row.deployment_id}"
                    for row in db.execute(select(DeploymentGPUReservation)).scalars().all()
                }
            )
            reservations.update(
                {
                    row.gpu_index: f"evaluation:{row.evaluation_id}"
                    for row in db.execute(select(EvaluationGPUReservation)).scalars().all()
                }
            )
            reservations.update(
                {
                    row.gpu_index: f"utility:{row.utility_run_id}"
                    for row in db.execute(select(UtilityGPUReservation)).scalars().all()
                }
            )
            snapshot = self.gpu_monitor.snapshot(reservations)
            if not snapshot:
                return
            by_index = {gpu.index: gpu for gpu in snapshot}
            if run.requested_gpu_ids:
                chosen = [int(value) for value in run.requested_gpu_ids]
                if any(index not in by_index for index in chosen):
                    run.status = "failed"
                    run.error = "requested_gpu_not_visible"
                    run.finished_at = utcnow()
                    return
                if any(by_index[index].error for index in chosen):
                    run.status = "failed"
                    run.error = "requested_gpu_inspection_failed"
                    run.finished_at = utcnow()
                    return
                if any(not by_index[index].available for index in chosen):
                    return
            else:
                healthy = [gpu for gpu in snapshot if not gpu.error]
                if run.requested_gpu_count > len(healthy):
                    run.status = "failed"
                    run.error = "requested_gpu_count_unavailable"
                    run.finished_at = utcnow()
                    return
                available = [gpu.index for gpu in snapshot if gpu.available]
                if len(available) < run.requested_gpu_count:
                    return
                chosen = available[: run.requested_gpu_count]
            for index in chosen:
                db.add(UtilityGPUReservation(gpu_index=index, utility_run_id=run.id))
            run.assigned_gpu_ids = chosen
        try:
            await self.utility_manager.activate_gpu_run(run_id, chosen)
        except Exception:
            with self.database.session() as db:
                run = db.get(UtilityRun, run_id)
                if run and run.status == "queued":
                    run.status = "failed"
                    run.error = "utility_gpu_activation_failed"
                    run.finished_at = utcnow()
                db.execute(
                    delete(UtilityGPUReservation).where(
                        UtilityGPUReservation.utility_run_id == run_id
                    )
                )

    async def _spawn(self, job_id: str) -> None:
        with self.database.session() as db:
            job = db.get(Job, job_id)
            if job is None or job.status != "starting":
                return
            gpu_ids = [int(x) for x in job.assigned_gpu_ids]
            log_handle = None
            try:
                active_jobs = db.execute(select(Job).where(Job.status.in_(ACTIVE_STATUSES))).scalars().all()
                used_ports = {
                    int(row.environment["MASTER_PORT"])
                    for row in active_jobs
                    if row.id != job.id and str(row.environment.get("MASTER_PORT", "")).isdigit()
                }
                preferred_port = int(job.environment.get("ALPHABRAIN_UI_PREFERRED_PORT", 0))
                main_process_port = _allocate_port(preferred_port, used_ports)
                command = [_replace_tokens(str(arg), gpu_ids, main_process_port) for arg in job.command]
                resolved_environment = {
                    str(key): _replace_tokens(str(value), gpu_ids, main_process_port)
                    for key, value in job.environment.items()
                    if str(key).upper() != WANDB_API_KEY_ENV
                }
                resolved_environment["MASTER_PORT"] = str(main_process_port)
                job.command = command
                job.environment = resolved_environment
                # Read repository .env only at launch time. This gives direct
                # Python launchers the same environment as the existing shell
                # wrappers without persisting credentials in SQLite.
                from .preflight import read_dotenv

                env = read_dotenv(self.config.repo_root / ".env")
                env.update(os.environ)
                env.update(resolved_environment)
                # Never trust or persist a W&B key supplied through generic
                # environment settings. The owner-only secret file is the
                # sole source, and its value exists only in this child env.
                env.pop(WANDB_API_KEY_ENV, None)
                wandb_managed = WANDB_CATEGORIES_ENV in env
                selected_wandb_categories = {
                    item.strip()
                    for item in env.get(WANDB_CATEGORIES_ENV, "").split(",")
                    if item.strip()
                }
                if wandb_managed and "config" not in selected_wandb_categories:
                    env.pop("WANDB_CONFIG_PATHS", None)
                wandb_mode = env.get("WANDB_MODE", "online").strip().lower()
                if wandb_managed and wandb_mode not in {"disabled", "offline"}:
                    wandb_api_key = self.wandb_secret_store.read_api_key()
                    if wandb_api_key is not None:
                        env[WANDB_API_KEY_ENV] = wandb_api_key
                env.pop("ALPHABRAIN_UI_PREFERRED_PORT", None)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
                log_path = Path(job.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("ab", buffering=0)
                remote_config = RemoteTrainingConfig.from_settings(SettingsService(db).all())
                if remote_config.enabled:
                    remote_environment = {
                        key: value
                        for key, value in env.items()
                        if key in resolved_environment
                        or key in {WANDB_API_KEY_ENV, "CUDA_VISIBLE_DEVICES", "MASTER_PORT"}
                    }
                    ssh_command, remote_script = build_remote_script(
                        config=remote_config,
                        command=command,
                        cwd=job.cwd,
                        environment=remote_environment,
                        local_repo_root=self.config.repo_root,
                        local_state_dir=self.config.state_dir,
                    )
                    process = await asyncio.create_subprocess_exec(
                        *ssh_command,
                        cwd=self.config.repo_root,
                        env=os.environ.copy(),
                        stdin=asyncio.subprocess.PIPE,
                        stdout=log_handle,
                        stderr=asyncio.subprocess.STDOUT,
                        **new_process_group_kwargs(),
                    )
                    assert process.stdin is not None
                    process.stdin.write(remote_script)
                    await process.stdin.drain()
                    process.stdin.close()
                else:
                    # Scripts invoking `python` should resolve to the same
                    # environment that runs the UI backend.
                    import sys

                    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=job.cwd,
                        env=env,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=asyncio.subprocess.STDOUT,
                        **new_process_group_kwargs(),
                    )
            except Exception as exc:
                if log_handle is not None:
                    log_handle.close()
                job.status = "failed"
                job.error = f"Failed to start process: {exc}"
                job.finished_at = utcnow()
                db.execute(delete(GPUReservation).where(GPUReservation.job_id == job.id))
                self._refresh_experiment_statuses(db)
                return
            job.pid = process.pid
            # start_new_session=True makes the child the leader of a new
            # process group, so its PGID is its PID. Avoid os.getpgid() here:
            # a very short command may already have exited by this point.
            job.pgid = process.pid
            with contextlib.suppress(psutil.Error):
                job.process_created_at = psutil.Process(process.pid).create_time()
            job.status = "running"
            self._refresh_experiment_statuses(db)
        waiter = asyncio.create_task(self._wait_for_process(job_id, process, log_handle), name=f"job-{job_id}")
        self._waiters[job_id] = waiter

    async def _wait_for_process(self, job_id: str, process: asyncio.subprocess.Process, log_handle) -> None:
        try:
            try:
                exit_code = await process.wait()
            except asyncio.CancelledError:
                # UI shutdown leaves the detached process untouched.
                raise
            finally:
                log_handle.close()
            # Commit lifecycle state and the GPU release before best-effort
            # artifact discovery. A malformed or unreadable checkpoint must
            # never leave an exited process marked as running.
            with self.database.session() as db:
                job = db.get(Job, job_id)
                if job is None:
                    return
                job.exit_code = exit_code
                if job.stop_requested_at is not None:
                    job.status = "stopped"
                else:
                    job.status = "completed" if exit_code == 0 else "failed"
                if exit_code != 0 and not job.error:
                    job.error = f"Process exited with code {exit_code}"
                job.finished_at = utcnow()
                db.execute(delete(GPUReservation).where(GPUReservation.job_id == job.id))
                self._refresh_experiment_statuses(db)
            try:
                with self.database.session() as db:
                    job = db.get(Job, job_id)
                    if job is not None:
                        self._index_checkpoints(db, job)
            except OSError:
                pass
        finally:
            self._waiters.pop(job_id, None)

    async def _reconcile_existing_jobs(self) -> None:
        with self.database.session() as db:
            jobs = db.execute(select(Job).where(Job.status.in_(ACTIVE_STATUSES))).scalars().all()
            for job in jobs:
                if not self._matches_process(job):
                    job.status = "interrupted"
                    job.error = "The UI restarted and the recorded training process is no longer running."
                    job.finished_at = utcnow()
                    db.execute(delete(GPUReservation).where(GPUReservation.job_id == job.id))
                    continue
                # A detached process cannot provide an exit code, but its PID
                # and outputs remain observable.
                job.status = "running"
                for index in job.assigned_gpu_ids:
                    if db.get(GPUReservation, int(index)) is None:
                        db.add(GPUReservation(gpu_index=int(index), job_id=job.id))
                self._waiters[job.id] = asyncio.create_task(
                    self._watch_detached(job.id, int(job.pid), job.process_created_at),
                    name=f"detached-job-{job.id}",
                )
            self._refresh_experiment_statuses(db)

    def _matches_process(self, job: Job) -> bool:
        if not job.pid:
            return False
        try:
            process = psutil.Process(job.pid)
            if job.process_created_at is not None and abs(process.create_time() - job.process_created_at) > 1:
                return False
            if job.pgid is not None and not process_group_matches(job.pid, job.pgid):
                return False
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (OSError, psutil.Error):
            return False

    async def _watch_detached(self, job_id: str, pid: int, expected_created_at: float | None) -> None:
        try:
            while not self._stop.is_set():
                try:
                    process = psutil.Process(pid)
                    same_process = expected_created_at is None or abs(process.create_time() - expected_created_at) <= 1
                    alive = same_process and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                except psutil.Error:
                    alive = False
                if not alive:
                    with self.database.session() as db:
                        job = db.get(Job, job_id)
                        if job and job.status in ACTIVE_STATUSES:
                            job.status = "stopped" if job.stop_requested_at else "interrupted"
                            job.error = job.error or (
                                "Detached process exited while the UI was offline; exit code is unavailable."
                            )
                            job.finished_at = utcnow()
                            db.execute(delete(GPUReservation).where(GPUReservation.job_id == job.id))
                            self._refresh_experiment_statuses(db)
                    try:
                        with self.database.session() as db:
                            job = db.get(Job, job_id)
                            if job is not None:
                                self._index_checkpoints(db, job)
                    except OSError:
                        pass
                    return
                await asyncio.sleep(self.config.scheduler_interval)
        finally:
            self._waiters.pop(job_id, None)

    async def cancel(self, job_id: str) -> Job:
        async with self._lock:
            with self.database.session() as db:
                job = db.get(Job, job_id)
                if job is None:
                    raise KeyError(job_id)
                if job.status not in {"queued", "blocked"}:
                    raise ValueError("Only queued or blocked jobs can be cancelled")
                job.status = "cancelled"
                job.finished_at = utcnow()
                self._refresh_experiment_statuses(db)
                return job

    async def signal(self, job_id: str, sig: int) -> Job:
        async with self._lock:
            with self.database.session() as db:
                job = db.get(Job, job_id)
                if job is None:
                    raise KeyError(job_id)
                if job.status not in ACTIVE_STATUSES or not job.pgid:
                    raise ValueError("Job is not running")
                if not self._matches_process(job):
                    job.status = "interrupted"
                    job.error = job.error or "The recorded process identity no longer matches."
                    job.finished_at = utcnow()
                    db.execute(delete(GPUReservation).where(GPUReservation.job_id == job.id))
                    self._refresh_experiment_statuses(db)
                    return job
                try:
                    signal_process_group(job.pid, job.pgid, sig)
                except ProcessLookupError:
                    job.status = "interrupted"
                    job.error = job.error or "The recorded process no longer exists."
                    job.finished_at = utcnow()
                    db.execute(delete(GPUReservation).where(GPUReservation.job_id == job.id))
                    self._refresh_experiment_statuses(db)
                    return job
                job.stop_requested_at = utcnow()
                job.status = "stopping"
                return job

    async def request_stop(self, job_id: str) -> Job:
        return await self.signal(job_id, signal.SIGINT)

    async def terminate(self, job_id: str) -> Job:
        return await self.signal(job_id, signal.SIGTERM)

    async def force_kill(self, job_id: str) -> Job:
        return await self.signal(job_id, FORCE_KILL_SIGNAL)

    def refresh_checkpoints(self, experiment_id: str | None = None) -> None:
        """Discover checkpoints without waiting for the training job to end."""
        with self.database.session() as db:
            stmt = select(Job).where(Job.status.in_(ACTIVE_STATUSES | TERMINAL_STATUSES))
            if experiment_id:
                stmt = stmt.where(Job.experiment_id == experiment_id)
            for job in db.execute(stmt).scalars().all():
                with contextlib.suppress(OSError):
                    self._index_checkpoints(db, job)

    @staticmethod
    def _refresh_experiment_statuses(db) -> None:
        experiments = db.execute(select(Experiment)).scalars().all()
        for experiment in experiments:
            jobs = db.execute(select(Job).where(Job.experiment_id == experiment.id)).scalars().all()
            if not jobs:
                continue
            statuses = {job.status for job in jobs}
            if "running" in statuses or "starting" in statuses or "stopping" in statuses:
                experiment.status = "running"
            elif "queued" in statuses or "blocked" in statuses:
                experiment.status = "queued"
            elif statuses == {"completed"}:
                experiment.status = "completed"
            elif statuses <= {"completed", "cancelled", "stopped"}:
                experiment.status = "stopped"
            elif statuses & {"failed", "dependency_failed", "interrupted"}:
                experiment.status = "failed"

    @staticmethod
    def _index_checkpoints(db, job: Job) -> None:
        output = Path(job.output_dir)
        if not output.exists():
            return
        output_root = output.resolve()
        candidates: list[Path] = []
        checkpoint_root = output / "checkpoints"
        if checkpoint_root.is_dir():
            candidates.extend(child for child in checkpoint_root.iterdir() if child.is_file() or child.is_dir())
        for name in ("final_model", "final"):
            candidate = output / name
            if candidate.exists():
                candidates.append(candidate)
        for path in candidates:
            if path.is_symlink():
                continue
            resolved = str(path.resolve())
            resolved_path = Path(resolved)
            if resolved_path == output_root or not resolved_path.is_relative_to(output_root):
                continue
            existing = db.execute(select(Checkpoint).where(Checkpoint.path == resolved)).scalar_one_or_none()
            step_match = re.search(r"(?:steps?_?|iter_|checkpoint[-_]?)(\d+)", path.name)
            step = int(step_match.group(1)) if step_match else None
            weight_names = ("model.safetensors", "pytorch_model.pt", "encoder.pt", "vla_state_dict.pt")
            if path.is_dir():
                complete = any(
                    (path / filename).exists() or (path / "vla" / filename).exists() for filename in weight_names
                )
            else:
                complete = path.name in weight_names or path.name.endswith(
                    (
                        "_model.safetensors",
                        "_pytorch_model.pt",
                        "_encoder.pt",
                        "_vla_state_dict.pt",
                        "_merged.pt",
                    )
                )
            resumable = path.is_dir() and ((path / "training_state").exists() or (path / "resume_meta.json").exists())
            if existing is None:
                db.add(
                    Checkpoint(
                        experiment_id=job.experiment_id,
                        job_id=job.id,
                        path=resolved,
                        name=path.name,
                        step=step,
                        size_bytes=_path_size(path),
                        is_complete=complete,
                        is_resumable=resumable,
                        metadata_json={"phase": job.stage.phase},
                    )
                )
            else:
                existing.size_bytes = _path_size(path)
                existing.is_complete = complete
                existing.is_resumable = resumable
