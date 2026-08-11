from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import secrets
import signal
import socket
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import psutil
from sqlalchemy import delete, select

from .database import (
    DeploymentGPUReservation,
    EvaluationGPUReservation,
    GPUReservation,
    ModelDeployment,
    UtilityGPUReservation,
    utcnow,
)
from .deployment_registry import build_adapter_command
from .gpu import GPUMonitor
from .preflight import read_dotenv
from .process_control import (
    FORCE_KILL_SIGNAL,
    new_process_group_kwargs,
    process_group_matches,
    signal_process_group,
)
from .runtime import RuntimeConfig
from .secrets import SecureSecretStore
from .services import SettingsService

DEPLOYMENT_ACTIVE_STATUSES = {"starting", "running", "stopping"}
DEPLOYMENT_TERMINAL_STATUSES = {"stopped", "failed", "cancelled"}
_RESTART_PARAMETER = "_restart_after_stop"
_LEGACY_ROTATE_PARAMETER = "_rotate_restart"


class DeploymentCommandError(RuntimeError):
    """A deliberately non-sensitive adapter command construction error."""


def generate_api_key() -> tuple[str, str, str]:
    raw = "ab_" + secrets.token_urlsafe(32)
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest(), raw[:10]


def model_server_python(settings: SettingsService) -> str:
    configured = str(settings.get("model_server_python", "") or "").strip()
    return configured or os.environ.get("ALPHABRAIN_PYTHON", "").strip() or sys.executable


def _merged_reservations(db) -> dict[int, str]:  # type: ignore[no-untyped-def]
    result = {row.gpu_index: row.job_id for row in db.execute(select(GPUReservation)).scalars().all()}
    result.update(
        {
            row.gpu_index: f"deployment:{row.deployment_id}"
            for row in db.execute(select(DeploymentGPUReservation)).scalars().all()
        }
    )
    result.update(
        {
            row.gpu_index: f"evaluation:{row.evaluation_id}"
            for row in db.execute(select(EvaluationGPUReservation)).scalars().all()
        }
    )
    result.update(
        {
            row.gpu_index: f"utility:{row.utility_run_id}"
            for row in db.execute(select(UtilityGPUReservation)).scalars().all()
        }
    )
    return result


async def _health_snapshot(port: int, timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=timeout)
        writer.write(
            b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nAccept: application/json\r\n\r\n"
        )
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(256 * 1024), timeout=timeout)
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        head, _, body = raw.partition(b"\r\n\r\n")
        if b" 200 " not in head.split(b"\r\n", 1)[0]:
            return None
        value = json.loads(body.decode("utf-8"))
        return value if isinstance(value, dict) and value.get("status") == "ok" else None
    except (OSError, asyncio.TimeoutError, UnicodeDecodeError, json.JSONDecodeError):
        return None


class DeploymentManager:
    def __init__(
        self,
        config: RuntimeConfig,
        database,
        gpu_monitor: GPUMonitor,
        *,
        lock: asyncio.Lock | None = None,
        graceful_stop_timeout: float | None = None,
        controller_secret_store: SecureSecretStore | None = None,
        source_catalog_provider: Callable[[], Mapping[str, Any]] | None = None,
    ):
        self.config = config
        self.database = database
        self.gpu_monitor = gpu_monitor
        self.lock = lock or asyncio.Lock()
        self._stop = asyncio.Event()
        self._waiters: dict[str, asyncio.Task] = {}
        self._escalations: dict[str, asyncio.Task] = {}
        configured_stop_timeout = config.stop_grace_seconds if graceful_stop_timeout is None else graceful_stop_timeout
        self._graceful_stop_timeout = max(0.0, float(configured_stop_timeout))
        self.controller_secret_store = controller_secret_store or SecureSecretStore(
            config.state_dir,
            "deployment-controller",
        )
        self.source_catalog_provider = source_catalog_provider

    def create_controller_key(self, deployment_id: str) -> None:
        """Create the UI-only credential used by managed inference clients.

        The plaintext is deliberately never returned to an API caller or
        stored in SQLite.  Existing deployments aren't silently backfilled:
        if the secret is missing, their public endpoint remains usable while
        managed Playground/evaluation access stays disabled.
        """

        raw = "ab_ctl_" + secrets.token_urlsafe(40)
        self.controller_secret_store.set(deployment_id, raw, min_length=32)

    def controller_key(self, deployment_id: str) -> str | None:
        return self.controller_secret_store.read(deployment_id)

    def controller_key_configured(self, deployment_id: str) -> bool:
        return self.controller_secret_store.configured(deployment_id)

    def delete_controller_key(self, deployment_id: str) -> None:
        self.controller_secret_store.delete(deployment_id)

    async def start(self) -> None:
        self._stop.clear()
        await self._reconcile_existing()

    async def shutdown(self) -> None:
        self._stop.set()
        waiters = list(self._waiters.values())
        escalations = list(self._escalations.values())
        for task in waiters:
            task.cancel()
        for task in escalations:
            task.cancel()
        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)
        if escalations:
            await asyncio.gather(*escalations, return_exceptions=True)

    async def start_queued_unlocked(self, deployment_id: str) -> None:
        with self.database.session() as db:
            deployment = db.get(ModelDeployment, deployment_id)
            if deployment is None or deployment.status != "queued":
                return
            snapshot = self.gpu_monitor.snapshot(_merged_reservations(db))
            if not snapshot:
                return
            by_index = {gpu.index: gpu for gpu in snapshot}
            if deployment.requested_gpu_ids:
                chosen = [int(value) for value in deployment.requested_gpu_ids]
                missing = [index for index in chosen if index not in by_index]
                if missing:
                    deployment.status = "failed"
                    deployment.error = f"Requested GPU IDs are no longer visible: {missing}"
                    deployment.finished_at = utcnow()
                    return
                failed = [index for index in chosen if by_index[index].error]
                if failed:
                    deployment.status = "failed"
                    deployment.error = f"Unable to inspect requested GPU IDs: {failed}"
                    deployment.finished_at = utcnow()
                    return
                if any(not by_index[index].available for index in chosen):
                    return
            else:
                healthy = [gpu for gpu in snapshot if not gpu.error]
                if deployment.requested_gpu_count > len(healthy):
                    deployment.status = "failed"
                    deployment.error = (
                        f"Requested {deployment.requested_gpu_count} GPUs, "
                        f"but only {len(healthy)} can be inspected."
                    )
                    deployment.finished_at = utcnow()
                    return
                available = [gpu.index for gpu in snapshot if gpu.available]
                if len(available) < deployment.requested_gpu_count:
                    return
                chosen = available[: deployment.requested_gpu_count]

            active_ports = {
                int(value)
                for value in db.execute(
                    select(ModelDeployment.port).where(
                        ModelDeployment.status.in_(DEPLOYMENT_ACTIVE_STATUSES),
                        ModelDeployment.id != deployment.id,
                        ModelDeployment.port.is_not(None),
                    )
                ).scalars()
                if value is not None
            }
            requested_port = int(deployment.port or 0)
            if requested_port and (
                requested_port in active_ports
                or not self._port_is_available(requested_port, deployment.bind_host)
            ):
                deployment.status = "failed"
                deployment.error = f"Requested model-server port is unavailable: {requested_port}"
                deployment.finished_at = utcnow()
                return
            deployment.port = requested_port or self._allocate_port(deployment.bind_host, active_ports)
            for index in chosen:
                db.add(DeploymentGPUReservation(gpu_index=index, deployment_id=deployment.id))
            deployment.assigned_gpu_ids = chosen
            deployment.status = "starting"
            deployment.started_at = utcnow()
            deployment.finished_at = None
            deployment.stop_requested_at = None
            deployment.error = ""
        await self._spawn(deployment_id)

    @staticmethod
    def _port_is_available(port: int, bind_host: str = "127.0.0.1") -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((bind_host, port))
            return True
        except OSError:
            return False

    @staticmethod
    def _allocate_port(bind_host: str, excluded: set[int]) -> int:
        for _attempt in range(32):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((bind_host, 0))
                port = int(sock.getsockname()[1])
            if port not in excluded:
                return port
        raise RuntimeError("Unable to allocate a free model-server port")

    def _runtime_command(
        self,
        deployment: ModelDeployment,
        python: str,
    ) -> tuple[list[str], list[str], int]:
        """Resolve an adapter command and append ephemeral authentication data.

        ``build_adapter_command`` owns adapter-specific arguments. The returned
        persisted command is a separate list where the digest is redacted; the
        executable list must never be assigned to an ORM field or included in
        an exception.
        """
        if len(deployment.api_key_hash) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in deployment.api_key_hash
        ):
            raise DeploymentCommandError("Deployment authentication is not configured.")
        if self.config.demo_mode:
            command = [
                python,
                "-m",
                "alphabrain_ui.demo_worker",
                "serve",
                "--host",
                deployment.bind_host,
                "--port",
                str(int(deployment.port or 0)),
                "--deployment-id",
                deployment.id,
                "--checkpoint",
                deployment.checkpoint_path,
                "--idle-timeout",
                str(deployment.idle_timeout_seconds),
                "--api-key-sha256",
                deployment.api_key_hash,
            ]
            persisted = [*command[:-1], "[REDACTED]"]
            controller_key = self.controller_key(deployment.id)
            if controller_key:
                digest = hashlib.sha256(controller_key.encode("utf-8")).hexdigest()
                command.extend(["--controller-api-key-sha256", digest])
                persisted.extend(["--controller-api-key-sha256", "[REDACTED]"])
            return command, persisted, 15

        parameters = {
            str(key): value
            for key, value in dict(deployment.parameters or {}).items()
            if not str(key).startswith("_")
        }
        parameters["port"] = int(deployment.port or 0)
        parameters["idle_timeout_seconds"] = int(deployment.idle_timeout_seconds)
        try:
            resolved = build_adapter_command(
                deployment.adapter_id,
                deployment.checkpoint_path,
                combination_id=deployment.combination_id,
                backbone_id=deployment.backbone_id,
                action_head_id=deployment.action_head_id,
                python_executable=python,
                repo_root=self.config.repo_root,
                parameters=parameters,
                # Experimental permission is enforced when the immutable
                # deployment record is created. Existing accepted records must
                # remain restartable if the UI setting later changes.
                include_experimental=True,
                source_catalog=(
                    self.source_catalog_provider()
                    if self.source_catalog_provider is not None
                    else None
                ),
            )
        except Exception:
            raise DeploymentCommandError("Deployment adapter command construction failed.") from None
        if not resolved.get("valid") or not resolved.get("command"):
            issue_codes = sorted(
                {
                    str(issue.get("code", "invalid_adapter_command"))
                    for issue in resolved.get("issues", [])
                    if isinstance(issue, dict)
                }
            )
            suffix = f" ({', '.join(issue_codes)})" if issue_codes else ""
            raise DeploymentCommandError(f"Deployment adapter command is invalid{suffix}.")

        base_command = [str(value) for value in resolved["command"]]
        public_arguments = [
            "--host",
            deployment.bind_host,
            "--deployment-id",
            deployment.id,
        ]
        command = [
            *base_command,
            *public_arguments,
            "--api-key-sha256",
            deployment.api_key_hash,
        ]
        persisted = [*base_command, *public_arguments, "--api-key-sha256", "[REDACTED]"]
        controller_key = self.controller_key(deployment.id)
        controller_digest = None
        if controller_key:
            controller_digest = hashlib.sha256(controller_key.encode("utf-8")).hexdigest()
            command.extend(["--controller-api-key-sha256", controller_digest])
            persisted.extend(["--controller-api-key-sha256", "[REDACTED]"])
        # Defense in depth in case a future adapter accidentally echoes the
        # digest into another argument.
        persisted = [value.replace(deployment.api_key_hash, "[REDACTED]") for value in persisted]
        if controller_digest:
            persisted = [value.replace(controller_digest, "[REDACTED]") for value in persisted]
        startup_timeout = int(
            dict(deployment.parameters or {}).get(
                "_startup_timeout_seconds",
                resolved.get("startup_timeout_seconds", 900),
            )
        )
        return command, persisted, max(1, startup_timeout)

    async def _spawn(self, deployment_id: str) -> None:
        with self.database.session() as db:
            deployment = db.get(ModelDeployment, deployment_id)
            if deployment is None or deployment.status != "starting":
                return
            settings = SettingsService(db)
            python = model_server_python(settings)
            log_handle = None
            try:
                command, persisted_command, startup_timeout = self._runtime_command(deployment, python)
                deployment.command = persisted_command
                env = read_dotenv(self.config.repo_root / ".env")
                env.update(os.environ)
                env.update({str(key): str(value) for key, value in settings.get("environment", {}).items()})
                env.update({str(key): str(value) for key, value in deployment.environment.items()})
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in deployment.assigned_gpu_ids)
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONPATH"] = str(self.config.repo_root) + os.pathsep + env.get("PYTHONPATH", "")
                python_path = Path(python).expanduser()
                if python_path.parent != Path("."):
                    env["PATH"] = str(python_path.parent) + os.pathsep + env.get("PATH", "")
                log_path = Path(deployment.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("ab", buffering=0)
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(self.config.repo_root),
                    env=env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    **new_process_group_kwargs(),
                )
            except Exception:
                if log_handle is not None:
                    log_handle.close()
                deployment.status = "failed"
                # Never persist exception text from process construction: a
                # mocked or platform-specific exception may contain argv.
                deployment.error = "Failed to start model server."
                deployment.finished_at = utcnow()
                db.execute(
                    delete(DeploymentGPUReservation).where(
                        DeploymentGPUReservation.deployment_id == deployment.id
                    )
                )
                return
            deployment.pid = process.pid
            deployment.pgid = process.pid
            with contextlib.suppress(psutil.Error):
                deployment.process_created_at = psutil.Process(process.pid).create_time()
            port = int(deployment.port or 0)
        waiter = asyncio.create_task(
            self._monitor_process(deployment_id, process, log_handle, port, startup_timeout),
            name=f"model-deployment-{deployment_id}",
        )
        self._waiters[deployment_id] = waiter

    @staticmethod
    def _set_restart_intent(deployment: ModelDeployment, intent: str) -> None:
        parameters = dict(deployment.parameters or {})
        parameters.pop(_LEGACY_ROTATE_PARAMETER, None)
        parameters[_RESTART_PARAMETER] = intent
        deployment.parameters = parameters

    @staticmethod
    def _clear_restart_intent(deployment: ModelDeployment) -> str | None:
        parameters = dict(deployment.parameters or {})
        intent = parameters.pop(_RESTART_PARAMETER, None)
        if parameters.pop(_LEGACY_ROTATE_PARAMETER, False) and not intent:
            intent = "rotate"
        deployment.parameters = parameters
        return str(intent) if intent else None

    @staticmethod
    def _process_matches_identity(pid: int, pgid: int, created_at: float | None) -> bool:
        try:
            process = psutil.Process(pid)
            if created_at is not None and abs(process.create_time() - created_at) > 1:
                return False
            if not process_group_matches(pid, pgid):
                return False
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (OSError, psutil.Error):
            return False

    def _cancel_escalation(self, deployment_id: str) -> None:
        task = self._escalations.pop(deployment_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    def _schedule_escalation(
        self,
        deployment_id: str,
        *,
        pid: int,
        pgid: int,
        created_at: float | None,
        delay: float | None = None,
    ) -> bool:
        current = self._escalations.get(deployment_id)
        if current is not None and not current.done():
            return False
        task = asyncio.create_task(
            self._escalate_after_timeout(
                deployment_id,
                pid=pid,
                pgid=pgid,
                created_at=created_at,
                delay=self._graceful_stop_timeout if delay is None else max(0.0, delay),
            ),
            name=f"kill-model-deployment-{deployment_id}",
        )
        self._escalations[deployment_id] = task

        def cleanup(done: asyncio.Task) -> None:
            if self._escalations.get(deployment_id) is done:
                self._escalations.pop(deployment_id, None)

        task.add_done_callback(cleanup)
        return True

    async def _escalate_after_timeout(
        self,
        deployment_id: str,
        *,
        pid: int,
        pgid: int,
        created_at: float | None,
        delay: float,
    ) -> None:
        await asyncio.sleep(delay)
        with self.database.session() as db:
            deployment = db.get(ModelDeployment, deployment_id)
            if (
                deployment is None
                or deployment.status not in DEPLOYMENT_ACTIVE_STATUSES
                or deployment.pid != pid
                or deployment.pgid != pgid
            ):
                return
            if not self._process_matches_identity(pid, pgid, created_at):
                return
            try:
                signal_process_group(pid, pgid, FORCE_KILL_SIGNAL)
            except (ProcessLookupError, PermissionError):
                return

    def _remaining_grace_period(self, deployment: ModelDeployment) -> float:
        if deployment.stop_requested_at is None:
            return self._graceful_stop_timeout
        elapsed = max(0.0, (utcnow() - deployment.stop_requested_at).total_seconds())
        return max(0.0, self._graceful_stop_timeout - elapsed)

    @staticmethod
    def _prepare_in_place_restart(deployment: ModelDeployment) -> None:
        deployment.status = "starting"
        deployment.pid = None
        deployment.pgid = None
        deployment.process_created_at = None
        deployment.exit_code = None
        deployment.started_at = utcnow()
        deployment.ready_at = None
        deployment.finished_at = None
        deployment.stop_requested_at = None
        deployment.error = ""

    @staticmethod
    def _release_reservations(db, deployment_id: str) -> None:  # type: ignore[no-untyped-def]
        db.execute(
            delete(DeploymentGPUReservation).where(DeploymentGPUReservation.deployment_id == deployment_id)
        )

    async def _monitor_process(
        self,
        deployment_id: str,
        process: asyncio.subprocess.Process,
        log_handle,
        port: int,
        startup_timeout: int,
    ) -> None:
        startup_failed = False
        restart_in_place = False
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + max(1, startup_timeout)
            while process.returncode is None and loop.time() < deadline:
                health = await _health_snapshot(port)
                if health is not None and health.get("deployment_id") == deployment_id:
                    with self.database.session() as db:
                        deployment = db.get(ModelDeployment, deployment_id)
                        if deployment and deployment.status == "starting":
                            deployment.status = "running"
                            deployment.ready_at = utcnow()
                    break
                await asyncio.sleep(1)
            else:
                if process.returncode is None:
                    startup_failed = True
                    with self.database.session() as db:
                        deployment = db.get(ModelDeployment, deployment_id)
                        if deployment:
                            deployment.error = f"Model server did not become healthy within {startup_timeout}s"
                    with contextlib.suppress(ProcessLookupError):
                        signal_process_group(process.pid, process.pid, signal.SIGTERM)
                    with self.database.session() as db:
                        deployment = db.get(ModelDeployment, deployment_id)
                        created_at = deployment.process_created_at if deployment else None
                    self._schedule_escalation(
                        deployment_id,
                        pid=process.pid,
                        pgid=process.pid,
                        created_at=created_at,
                    )

            try:
                exit_code = await process.wait()
            except asyncio.CancelledError:
                raise
            finally:
                log_handle.close()
            self._cancel_escalation(deployment_id)

            with self.database.session() as db:
                deployment = db.get(ModelDeployment, deployment_id)
                if deployment is None:
                    return
                deployment.exit_code = exit_code
                restart_in_place = self._clear_restart_intent(deployment) is not None
                if restart_in_place:
                    self._prepare_in_place_restart(deployment)
                else:
                    deployment.pid = None
                    deployment.pgid = None
                    deployment.process_created_at = None
                    if deployment.stop_requested_at is not None:
                        deployment.status = "stopped"
                    elif startup_failed or exit_code != 0:
                        deployment.status = "failed"
                    else:
                        deployment.status = "stopped"
                    if deployment.status == "failed" and exit_code != 0 and not deployment.error:
                        deployment.error = f"Model server exited with code {exit_code}"
                    deployment.finished_at = utcnow()
                    self._release_reservations(db, deployment.id)
            if restart_in_place:
                await self._spawn(deployment_id)
        finally:
            current = self._waiters.get(deployment_id)
            if current is asyncio.current_task():
                self._waiters.pop(deployment_id, None)

    async def _reconcile_existing(self) -> None:
        restart_ids: list[str] = []
        watch_specs: list[tuple[str, int, int, float | None]] = []
        escalation_specs: list[tuple[str, int, int, float | None, float]] = []
        with self.database.session() as db:
            rows = db.execute(
                select(ModelDeployment).where(ModelDeployment.status.in_(DEPLOYMENT_ACTIVE_STATUSES))
            ).scalars().all()
            for deployment in rows:
                if not self._matches_process(deployment):
                    restart_in_place = self._clear_restart_intent(deployment) is not None
                    if restart_in_place:
                        self._prepare_in_place_restart(deployment)
                        for index in deployment.assigned_gpu_ids:
                            if db.get(DeploymentGPUReservation, int(index)) is None:
                                db.add(
                                    DeploymentGPUReservation(gpu_index=int(index), deployment_id=deployment.id)
                                )
                        restart_ids.append(deployment.id)
                    else:
                        normal_stop = deployment.stop_requested_at is not None
                        deployment.status = "stopped" if normal_stop else "failed"
                        if not normal_stop:
                            deployment.error = (
                                deployment.error or "The recorded model-server process is no longer running."
                            )
                        deployment.pid = None
                        deployment.pgid = None
                        deployment.process_created_at = None
                        deployment.finished_at = utcnow()
                        self._release_reservations(db, deployment.id)
                    continue
                for index in deployment.assigned_gpu_ids:
                    if db.get(DeploymentGPUReservation, int(index)) is None:
                        db.add(DeploymentGPUReservation(gpu_index=int(index), deployment_id=deployment.id))
                pid = int(deployment.pid)
                pgid = int(deployment.pgid or deployment.pid)
                watch_specs.append((deployment.id, pid, pgid, deployment.process_created_at))
                if deployment.status == "stopping":
                    escalation_specs.append(
                        (
                            deployment.id,
                            pid,
                            pgid,
                            deployment.process_created_at,
                            self._remaining_grace_period(deployment),
                        )
                    )
        for deployment_id, pid, pgid, created_at in watch_specs:
            self._waiters[deployment_id] = asyncio.create_task(
                self._watch_detached(deployment_id, pid, pgid, created_at),
                name=f"detached-deployment-{deployment_id}",
            )
        for deployment_id, pid, pgid, created_at, delay in escalation_specs:
            self._schedule_escalation(
                deployment_id,
                pid=pid,
                pgid=pgid,
                created_at=created_at,
                delay=delay,
            )
        for deployment_id in restart_ids:
            await self._spawn(deployment_id)

    @staticmethod
    def _matches_process(deployment: ModelDeployment) -> bool:
        if not deployment.pid:
            return False
        return DeploymentManager._process_matches_identity(
            int(deployment.pid),
            int(deployment.pgid or deployment.pid),
            deployment.process_created_at,
        )

    async def _watch_detached(
        self,
        deployment_id: str,
        pid: int,
        pgid: int,
        created_at: float | None,
    ) -> None:
        try:
            while not self._stop.is_set():
                alive = self._process_matches_identity(pid, pgid, created_at)
                if not alive:
                    self._cancel_escalation(deployment_id)
                    restart_in_place = False
                    with self.database.session() as db:
                        deployment = db.get(ModelDeployment, deployment_id)
                        if (
                            deployment
                            and deployment.status in DEPLOYMENT_ACTIVE_STATUSES
                            and deployment.pid == pid
                        ):
                            restart_in_place = self._clear_restart_intent(deployment) is not None
                            if restart_in_place:
                                self._prepare_in_place_restart(deployment)
                            else:
                                normal_stop = deployment.stop_requested_at is not None
                                deployment.status = "stopped" if normal_stop else "failed"
                                if not normal_stop:
                                    deployment.error = deployment.error or (
                                        "Detached model server exited while the UI was offline."
                                    )
                                deployment.pid = None
                                deployment.pgid = None
                                deployment.process_created_at = None
                                deployment.finished_at = utcnow()
                                self._release_reservations(db, deployment.id)
                    if restart_in_place:
                        await self._spawn(deployment_id)
                    return

                port = 0
                should_check_health = False
                with self.database.session() as db:
                    deployment = db.get(ModelDeployment, deployment_id)
                    if (
                        deployment is None
                        or deployment.status not in DEPLOYMENT_ACTIVE_STATUSES
                        or deployment.pid != pid
                    ):
                        return
                    should_check_health = deployment.status == "starting"
                    port = int(deployment.port or 0)
                if should_check_health and port:
                    health = await _health_snapshot(port)
                    if health is not None and health.get("deployment_id") == deployment_id:
                        with self.database.session() as db:
                            deployment = db.get(ModelDeployment, deployment_id)
                            if deployment and deployment.status == "starting" and deployment.pid == pid:
                                deployment.status = "running"
                                deployment.ready_at = utcnow()
                await asyncio.sleep(self.config.scheduler_interval)
        finally:
            current = self._waiters.get(deployment_id)
            if current is asyncio.current_task():
                self._waiters.pop(deployment_id, None)

    async def cancel(self, deployment_id: str) -> ModelDeployment:
        async with self.lock:
            with self.database.session() as db:
                deployment = self._get(db, deployment_id)
                if deployment.status != "queued":
                    raise ValueError("Only queued deployments can be cancelled")
                deployment.status = "cancelled"
                deployment.finished_at = utcnow()
                return deployment

    async def stop(self, deployment_id: str, *, force: bool = False) -> ModelDeployment:
        async with self.lock:
            signal_spec: tuple[int, int, float | None] | None = None
            previous_state: tuple[str, Any, dict[str, Any]] | None = None
            with self.database.session() as db:
                deployment = self._get(db, deployment_id)
                if deployment.status not in DEPLOYMENT_ACTIVE_STATUSES:
                    raise ValueError("Deployment is not running")
                if not deployment.pid or not deployment.pgid:
                    self._clear_restart_intent(deployment)
                    deployment.status = "stopped"
                    deployment.finished_at = utcnow()
                    deployment.stop_requested_at = utcnow()
                    self._release_reservations(db, deployment.id)
                    return deployment
                if not self._matches_process(deployment):
                    deployment.status = "failed"
                    deployment.error = "The recorded model-server process identity no longer matches."
                    deployment.finished_at = utcnow()
                    deployment.pid = None
                    deployment.pgid = None
                    deployment.process_created_at = None
                    self._release_reservations(db, deployment.id)
                    return deployment
                previous_state = (
                    deployment.status,
                    deployment.stop_requested_at,
                    dict(deployment.parameters or {}),
                )
                # Persist the stop before signalling. A fast child exit can
                # then never be misclassified as an unexpected failure.
                self._clear_restart_intent(deployment)
                deployment.stop_requested_at = utcnow()
                deployment.status = "stopping"
                if self.config.demo_mode:
                    # The demo child has no external resources to drain. Mark
                    # it stopped before signalling so a Windows process-exit
                    # callback cannot race the API response for the SQLite
                    # write lock.
                    deployment.status = "stopped"
                    deployment.finished_at = utcnow()
                    self._release_reservations(db, deployment.id)
                signal_spec = (
                    int(deployment.pid),
                    int(deployment.pgid),
                    deployment.process_created_at,
                )
                result = deployment
            assert signal_spec is not None and previous_state is not None
            pid, pgid, created_at = signal_spec
            escalation_created = False
            if not force and not self.config.demo_mode:
                escalation_created = self._schedule_escalation(
                    deployment_id,
                    pid=pid,
                    pgid=pgid,
                    created_at=created_at,
                )
            try:
                signal_process_group(pid, pgid, FORCE_KILL_SIGNAL if force else signal.SIGTERM)
            except ProcessLookupError:
                self._cancel_escalation(deployment_id)
                with self.database.session() as db:
                    current = self._get(db, deployment_id)
                    if current.pid == pid and current.pgid == pgid:
                        current.status = "stopped"
                        current.finished_at = utcnow()
                        current.pid = None
                        current.pgid = None
                        current.process_created_at = None
                        self._release_reservations(db, current.id)
                    result = current
            except PermissionError:
                if escalation_created:
                    self._cancel_escalation(deployment_id)
                restored = False
                with self.database.session() as db:
                    current = self._get(db, deployment_id)
                    if current.pid == pid and current.pgid == pgid:
                        previous_status, previous_stop_requested_at, previous_parameters = previous_state
                        current.status = previous_status
                        current.stop_requested_at = previous_stop_requested_at
                        current.parameters = previous_parameters
                        restored = True
                    result = current
                if restored:
                    raise ValueError("The model-server process could not be signalled.") from None
            else:
                if force:
                    self._cancel_escalation(deployment_id)
            return result

    async def restart(self, deployment_id: str) -> ModelDeployment:
        async with self.lock:
            signal_spec: tuple[int, int, float | None] | None = None
            previous_state: tuple[str, Any, dict[str, Any]] | None = None
            spawn_now = False
            with self.database.session() as db:
                deployment = self._get(db, deployment_id)
                if deployment.status in DEPLOYMENT_TERMINAL_STATUSES:
                    self._clear_restart_intent(deployment)
                    deployment.status = "queued"
                    deployment.queued_at = utcnow()
                    deployment.started_at = None
                    deployment.ready_at = None
                    deployment.finished_at = None
                    deployment.stop_requested_at = None
                    deployment.exit_code = None
                    deployment.error = ""
                    deployment.assigned_gpu_ids = []
                    self._release_reservations(db, deployment.id)
                    return deployment
                if deployment.status not in DEPLOYMENT_ACTIVE_STATUSES:
                    raise ValueError("Deployment cannot be restarted from its current state")
                if not deployment.pid or not deployment.pgid or not self._matches_process(deployment):
                    self._clear_restart_intent(deployment)
                    self._prepare_in_place_restart(deployment)
                    spawn_now = True
                else:
                    previous_state = (
                        deployment.status,
                        deployment.stop_requested_at,
                        dict(deployment.parameters or {}),
                    )
                    self._set_restart_intent(deployment, "restart")
                    deployment.stop_requested_at = utcnow()
                    deployment.status = "stopping"
                    signal_spec = (
                        int(deployment.pid),
                        int(deployment.pgid),
                        deployment.process_created_at,
                    )
                result = deployment
            if spawn_now:
                self._cancel_escalation(deployment_id)
                await self._spawn(deployment_id)
            elif signal_spec is not None and previous_state is not None:
                pid, pgid, created_at = signal_spec
                escalation_created = self._schedule_escalation(
                    deployment_id,
                    pid=pid,
                    pgid=pgid,
                    created_at=created_at,
                )
                try:
                    signal_process_group(pid, pgid, signal.SIGTERM)
                except ProcessLookupError:
                    self._cancel_escalation(deployment_id)
                    should_spawn = False
                    with self.database.session() as db:
                        current = self._get(db, deployment_id)
                        if current.pid == pid and current.pgid == pgid:
                            self._clear_restart_intent(current)
                            self._prepare_in_place_restart(current)
                            should_spawn = True
                        result = current
                    if should_spawn:
                        await self._spawn(deployment_id)
                except PermissionError:
                    if escalation_created:
                        self._cancel_escalation(deployment_id)
                    restored = False
                    with self.database.session() as db:
                        current = self._get(db, deployment_id)
                        if current.pid == pid and current.pgid == pgid:
                            previous_status, previous_stop_requested_at, previous_parameters = previous_state
                            current.status = previous_status
                            current.stop_requested_at = previous_stop_requested_at
                            current.parameters = previous_parameters
                            restored = True
                        result = current
                    if restored:
                        raise ValueError("The model-server process could not be signalled.") from None
            return result

    async def rotate_key(self, deployment_id: str) -> tuple[ModelDeployment, str]:
        async with self.lock:
            signal_spec: tuple[int, int, float | None] | None = None
            previous_state: tuple[str, Any, dict[str, Any], str, str] | None = None
            spawn_now = False
            with self.database.session() as db:
                deployment = self._get(db, deployment_id)
                raw, digest, prefix = generate_api_key()
                previous_state = (
                    deployment.status,
                    deployment.stop_requested_at,
                    dict(deployment.parameters or {}),
                    deployment.api_key_hash,
                    deployment.api_key_prefix,
                )
                # Persist the new digest and restart intent before signalling;
                # even an immediately exiting child is then restarted with the
                # new key by either monitor implementation.
                deployment.api_key_hash = digest
                deployment.api_key_prefix = prefix
                if deployment.status in DEPLOYMENT_ACTIVE_STATUSES and deployment.pid and deployment.pgid:
                    if not self._matches_process(deployment):
                        self._clear_restart_intent(deployment)
                        self._prepare_in_place_restart(deployment)
                        spawn_now = True
                    else:
                        self._set_restart_intent(deployment, "rotate")
                        deployment.stop_requested_at = utcnow()
                        deployment.status = "stopping"
                        signal_spec = (
                            int(deployment.pid),
                            int(deployment.pgid),
                            deployment.process_created_at,
                        )
                if deployment.status in DEPLOYMENT_ACTIVE_STATUSES and not deployment.pid:
                    # A starting record without a child will pick up the new
                    # digest when its pending spawn runs.
                    if deployment.status == "stopping":
                        self._clear_restart_intent(deployment)
                        self._prepare_in_place_restart(deployment)
                        spawn_now = True
                result = deployment
            if spawn_now:
                self._cancel_escalation(deployment_id)
                await self._spawn(deployment_id)
            elif signal_spec is not None and previous_state is not None:
                pid, pgid, created_at = signal_spec
                escalation_created = self._schedule_escalation(
                    deployment_id,
                    pid=pid,
                    pgid=pgid,
                    created_at=created_at,
                )
                try:
                    signal_process_group(pid, pgid, signal.SIGTERM)
                except ProcessLookupError:
                    self._cancel_escalation(deployment_id)
                    should_spawn = False
                    with self.database.session() as db:
                        current = self._get(db, deployment_id)
                        if current.pid == pid and current.pgid == pgid:
                            self._clear_restart_intent(current)
                            self._prepare_in_place_restart(current)
                            should_spawn = True
                        result = current
                    if should_spawn:
                        await self._spawn(deployment_id)
                except PermissionError:
                    if escalation_created:
                        self._cancel_escalation(deployment_id)
                    restored = False
                    with self.database.session() as db:
                        current = self._get(db, deployment_id)
                        if current.pid == pid and current.pgid == pgid:
                            (
                                previous_status,
                                previous_stop_requested_at,
                                previous_parameters,
                                previous_hash,
                                previous_prefix,
                            ) = previous_state
                            current.status = previous_status
                            current.stop_requested_at = previous_stop_requested_at
                            current.parameters = previous_parameters
                            current.api_key_hash = previous_hash
                            current.api_key_prefix = previous_prefix
                            restored = True
                        result = current
                    if restored:
                        raise ValueError("The model-server process could not be signalled.") from None
            return result, raw

    @staticmethod
    def _get(db, deployment_id: str) -> ModelDeployment:  # type: ignore[no-untyped-def]
        deployment = db.get(ModelDeployment, deployment_id)
        if deployment is None:
            raise KeyError(deployment_id)
        return deployment
