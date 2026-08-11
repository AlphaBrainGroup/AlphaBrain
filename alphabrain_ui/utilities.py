"""Managed CPU/network utility queue.

GPU utility rows use the same lifecycle but are deliberately not claimed by
the CPU loop.  The shared GPU scheduler can call ``activate_gpu_run`` after it
atomically reserves the requested devices.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .database import (
    Database,
    DatasetRegistration,
    ModelPublication,
    UtilityGPUReservation,
    UtilityRun,
    utcnow,
)
from .dataset_registry import inspect_dataset
from .process_control import FORCE_KILL_SIGNAL, new_process_group_kwargs, signal_process_group
from .runtime import RuntimeConfig
from .secrets import HuggingFaceSecretStore

UTILITY_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "stopped", "interrupted"}
UTILITY_ACTIVE_STATUSES = {"starting", "running", "stopping"}


class UtilityManager:
    def __init__(
        self,
        runtime: RuntimeConfig,
        database: Database,
        hf_secrets: HuggingFaceSecretStore,
        *,
        cpu_concurrency: int = 2,
        interval: float = 1.0,
    ):
        self.runtime = runtime
        self.database = database
        self.hf_secrets = hf_secrets
        self.cpu_concurrency = max(1, min(int(cpu_concurrency), 16))
        self.interval = interval
        self._scheduler: asyncio.Task[None] | None = None
        self._watchers: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._closing = False

    async def start(self) -> None:
        interrupted: list[UtilityRun] = []
        with self.database.session() as db:
            rows = db.execute(select(UtilityRun).where(UtilityRun.status.in_(UTILITY_ACTIVE_STATUSES))).scalars()
            for row in rows:
                row.status = "interrupted"
                row.error = "ui_restarted_while_utility_was_active"
                row.finished_at = utcnow()
                row.pid = None
                row.pgid = None
                interrupted.append(row)
            db.execute(delete(UtilityGPUReservation))
        for row in interrupted:
            self._cleanup_run_paths(row)
        self._closing = False
        self._scheduler = asyncio.create_task(self._schedule_loop(), name="alphabrain-utility-scheduler")

    async def shutdown(self) -> None:
        self._closing = True
        if self._scheduler:
            self._scheduler.cancel()
            await asyncio.gather(self._scheduler, return_exceptions=True)
            self._scheduler = None
        # Utility processes are intentionally terminated on UI shutdown so a
        # detached download/copy can never outlive its auditable DB lifecycle.
        for run_id in list(self._processes):
            await self.stop(run_id)
        if self._watchers:
            await asyncio.gather(*self._watchers.values(), return_exceptions=True)

    def create_run(
        self,
        *,
        owner_id: str,
        kind: str,
        command: list[str],
        cwd: Path,
        output_path: str = "",
        resource_id: str = "",
        parameters: dict[str, Any] | None = None,
        environment: dict[str, str] | None = None,
        queue_class: str = "cpu",
        requested_gpu_count: int = 0,
        requested_gpu_ids: list[int] | None = None,
        db_session: Session | None = None,
    ) -> UtilityRun:
        if queue_class not in {"cpu", "gpu"}:
            raise ValueError("invalid_utility_queue_class")
        run = UtilityRun(
            owner_id=owner_id,
            kind=kind,
            resource_id=resource_id,
            status="queued",
            queue_class=queue_class,
            requested_gpu_count=requested_gpu_count if queue_class == "gpu" else 0,
            requested_gpu_ids=list(requested_gpu_ids or []),
            parameters=dict(parameters or {}),
            command=[str(item) for item in command],
            environment={str(key): str(value) for key, value in (environment or {}).items()},
            cwd=str(cwd.resolve(strict=False)),
            output_path=output_path,
            log_path=str(self.runtime.state_dir / "logs" / "utilities" / "pending.log"),
        )
        if db_session is not None:
            db_session.add(run)
            db_session.flush()
            self._prepare_run(run)
            run.log_path = str(self.runtime.state_dir / "logs" / "utilities" / f"{run.id}.log")
            db_session.flush()
        else:
            with self.database.session() as db:
                db.add(run)
                db.flush()
                self._prepare_run(run)
                run.log_path = str(self.runtime.state_dir / "logs" / "utilities" / f"{run.id}.log")
                db.flush()
                db.expunge(run)
        return run

    def _prepare_run(self, run: UtilityRun) -> None:
        progress_path = self.runtime.state_dir / "progress" / "utilities" / f"{run.id}.json"
        replacements = {
            "{run_id}": run.id,
            "{progress_path}": str(progress_path),
        }

        def expand(value: Any) -> Any:
            if isinstance(value, str):
                for token, replacement in replacements.items():
                    value = value.replace(token, replacement)
                return value
            if isinstance(value, list):
                return [expand(item) for item in value]
            if isinstance(value, dict):
                return {str(key): expand(item) for key, item in value.items()}
            return value

        run.command = [str(expand(item)) for item in run.command]
        run.output_path = str(expand(run.output_path))
        run.parameters = {
            **expand(dict(run.parameters or {})),
            "_progress_path": str(progress_path),
        }

    @staticmethod
    def _cleanup_path(path: Path, run_id: str) -> None:
        if not path.name.endswith(f".partial-{run_id}"):
            return
        try:
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError:
            pass

    def _cleanup_run_paths(self, run: UtilityRun) -> None:
        values = (run.parameters or {}).get("_cleanup_paths", [])
        for value in values if isinstance(values, list) else []:
            self._cleanup_path(Path(str(value)), run.id)

    async def _schedule_loop(self) -> None:
        while not self._closing:
            try:
                await self._schedule_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A bad utility row must not kill all future scheduling.
                pass
            await asyncio.sleep(self.interval)

    async def _schedule_once(self) -> None:
        with self.database.session() as db:
            active = db.execute(
                select(func.count(UtilityRun.id)).where(
                    UtilityRun.queue_class == "cpu", UtilityRun.status.in_(UTILITY_ACTIVE_STATUSES)
                )
            ).scalar_one()
            capacity = max(0, self.cpu_concurrency - int(active))
            if not capacity:
                return
            ids = db.execute(
                select(UtilityRun.id)
                .where(UtilityRun.queue_class == "cpu", UtilityRun.status == "queued")
                .order_by(UtilityRun.queued_at, UtilityRun.created_at)
                .limit(capacity)
            ).scalars().all()
            for run_id in ids:
                row = db.get(UtilityRun, run_id)
                if row and row.status == "queued":
                    row.status = "starting"
                    row.started_at = utcnow()
        for run_id in ids:
            await self._spawn(run_id, assigned_gpu_ids=[])

    async def activate_gpu_run(self, run_id: str, assigned_gpu_ids: list[int]) -> None:
        """GPU-scheduler interface; caller owns reservation/release semantics."""
        with self.database.session() as db:
            row = db.get(UtilityRun, run_id)
            if row is None or row.status != "queued" or row.queue_class != "gpu":
                raise ValueError("utility_not_queued_for_gpu")
            if len(assigned_gpu_ids) < row.requested_gpu_count:
                raise ValueError("insufficient_assigned_gpus")
            row.status = "starting"
            row.started_at = utcnow()
            row.assigned_gpu_ids = list(assigned_gpu_ids)
        await self._spawn(run_id, assigned_gpu_ids=assigned_gpu_ids)

    async def _spawn(self, run_id: str, *, assigned_gpu_ids: list[int]) -> None:
        with self.database.session() as db:
            row = db.get(UtilityRun, run_id)
            if row is None or row.status != "starting":
                return
            command = list(row.command)
            environment = dict(row.environment or {})
            parameters = dict(row.parameters or {})
            cwd = row.cwd
            log_path = Path(row.log_path)
        environment = {**os.environ, **environment}
        if assigned_gpu_ids:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpu_ids))
        if parameters.get("hf_auth") == "global":
            token = self.hf_secrets.read_global_download_token()
            if token:
                environment["HF_TOKEN"] = token
        elif parameters.get("hf_auth") == "user_publish":
            user_id = str(parameters.get("hf_user_id") or "")
            token = self.hf_secrets.read_user_publish_token(user_id) if user_id else None
            if token:
                environment["HF_TOKEN"] = token
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            log_stream = log_path.open("ab", buffering=0)
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=environment,
                stdout=log_stream,
                stderr=asyncio.subprocess.STDOUT,
                **new_process_group_kwargs(),
            )
        except Exception as exc:
            with self.database.session() as db:
                row = db.get(UtilityRun, run_id)
                if row:
                    row.status = "failed"
                    row.error = f"spawn_failed: {type(exc).__name__}: {exc}"
                    row.finished_at = utcnow()
                db.execute(
                    delete(UtilityGPUReservation).where(
                        UtilityGPUReservation.utility_run_id == run_id
                    )
                )
            self.sync_publication_status(run_id)
            return
        finally:
            if "log_stream" in locals() and "process" not in locals():
                log_stream.close()
        self._processes[run_id] = process
        with self.database.session() as db:
            row = db.get(UtilityRun, run_id)
            if row:
                row.status = "running"
                row.pid = process.pid
                row.pgid = process.pid
                publication_id = str((row.parameters or {}).get("publication_id") or "")
                publication = db.get(ModelPublication, publication_id) if publication_id else None
                if publication is not None:
                    publication.status = "running"
                    publication.started_at = row.started_at or utcnow()
        watcher = asyncio.create_task(self._watch(run_id, process, log_stream), name=f"utility-{run_id}")
        self._watchers[run_id] = watcher

    async def _watch(self, run_id: str, process: asyncio.subprocess.Process, log_stream) -> None:  # type: ignore[no-untyped-def]
        try:
            exit_code = await process.wait()
        finally:
            log_stream.close()
        cleanup_run: UtilityRun | None = None
        with self.database.session() as db:
            row = db.get(UtilityRun, run_id)
            if row is None:
                return
            row.exit_code = exit_code
            row.pid = None
            row.pgid = None
            row.finished_at = utcnow()
            if row.status == "stopping":
                row.status = "stopped"
                cleanup_run = row
            elif exit_code == 0:
                row.status = "completed"
            else:
                row.status = "failed"
                row.error = f"utility exited with code {exit_code}"
                cleanup_run = row
            db.execute(
                delete(UtilityGPUReservation).where(
                    UtilityGPUReservation.utility_run_id == run_id
                )
            )
        if cleanup_run is not None:
            self._cleanup_run_paths(cleanup_run)
        if exit_code == 0:
            self._finalize_domain(run_id)
        else:
            self.sync_publication_status(run_id)
        self._processes.pop(run_id, None)
        self._watchers.pop(run_id, None)

    def _finalize_domain(self, run_id: str) -> None:
        with self.database.session() as db:
            run = db.get(UtilityRun, run_id)
            if run is None:
                return
            publication_id = str((run.parameters or {}).get("publication_id") or "")
            publication = db.get(ModelPublication, publication_id) if publication_id else None
            if publication is not None:
                publication.status = "completed"
                publication.started_at = publication.started_at or run.started_at
                publication.finished_at = run.finished_at or utcnow()
                publication.result_url = (
                    f"https://huggingface.co/{publication.repo_id}/tree/{publication.revision}"
                )
            registration_id = str((run.parameters or {}).get("registration_id") or "")
            registration = db.get(DatasetRegistration, registration_id) if registration_id else None
            if registration is None:
                return
            if run.kind == "dataset_copy":
                report = inspect_dataset(Path(registration.path))
                registration.validation = report
                registration.status = "ready" if report["valid"] else "invalid"
                registration.format = str(report["format"])
                registration.episode_count = int(report.get("episode_count", 0))
                registration.step_count = int(report.get("step_count", 0))
                registration.size_bytes = int(report.get("size_bytes", 0))
                registration.fingerprint = str(report.get("fingerprint", ""))
            elif run.kind == "dataset_stats":
                try:
                    result = json.loads(Path(run.output_path).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    registration.stats_status = "failed"
                    return
                registration.size_bytes = int(result.get("size_bytes", registration.size_bytes))
                registration.fingerprint = str(result.get("fingerprint", registration.fingerprint))
                registration.metadata_json = {**(registration.metadata_json or {}), "filesystem_stats": result}
                registration.stats_status = "ready"

    def sync_publication_status(self, run_id: str) -> None:
        """Propagate a utility failure/cancellation to its publication row."""

        with self.database.session() as db:
            run = db.get(UtilityRun, run_id)
            if run is None:
                return
            publication_id = str((run.parameters or {}).get("publication_id") or "")
            publication = db.get(ModelPublication, publication_id) if publication_id else None
            if publication is None:
                return
            publication.started_at = publication.started_at or run.started_at
            if run.status in UTILITY_TERMINAL_STATUSES:
                publication.status = run.status
                publication.error = run.error
                publication.finished_at = run.finished_at or utcnow()

    async def stop(self, run_id: str) -> None:
        process = self._processes.get(run_id)
        watcher = self._watchers.get(run_id)
        cancelled_while_queued = False
        cleanup_run: UtilityRun | None = None
        with self.database.session() as db:
            row = db.get(UtilityRun, run_id)
            if row is None:
                raise ValueError("utility_not_found")
            if row.status == "queued":
                row.status = "cancelled"
                row.finished_at = utcnow()
                cancelled_while_queued = True
                cleanup_run = row
            elif row.status not in UTILITY_ACTIVE_STATUSES:
                return
            else:
                row.status = "stopping"
                row.stop_requested_at = utcnow()
        if cancelled_while_queued:
            if cleanup_run is not None:
                self._cleanup_run_paths(cleanup_run)
            self.sync_publication_status(run_id)
            return
        if process and process.returncode is None:
            try:
                signal_process_group(process.pid, process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    asyncio.shield(process.wait()),
                    timeout=max(1, min(int(self.runtime.stop_grace_seconds), 10)),
                )
            except TimeoutError:
                try:
                    signal_process_group(process.pid, process.pid, FORCE_KILL_SIGNAL)
                except ProcessLookupError:
                    pass
                await process.wait()
        if watcher is not None:
            await asyncio.shield(watcher)


def serialize_utility(run: UtilityRun, *, queue_position: int | None = None) -> dict[str, Any]:
    progress = None
    raw_progress_path = str((run.parameters or {}).get("_progress_path") or "")
    progress_path = Path(raw_progress_path) if raw_progress_path else None
    if progress_path is not None and progress_path.name == f"{run.id}.json":
        try:
            candidate = json.loads(progress_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                progress = candidate
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    return {
        "id": run.id, "owner_id": run.owner_id, "kind": run.kind, "resource_id": run.resource_id,
        "status": run.status, "queue_class": run.queue_class, "requested_gpu_count": run.requested_gpu_count,
        "requested_gpu_ids": run.requested_gpu_ids, "assigned_gpu_ids": run.assigned_gpu_ids,
        "parameters": {key: value for key, value in (run.parameters or {}).items() if not str(key).startswith("_")},
        "output_path": run.output_path, "log_path": run.log_path, "error": run.error,
        "exit_code": run.exit_code, "queue_position": queue_position,
        "progress": progress,
        "queued_at": run.queued_at, "started_at": run.started_at, "finished_at": run.finished_at,
        "created_at": run.created_at, "updated_at": run.updated_at,
    }
