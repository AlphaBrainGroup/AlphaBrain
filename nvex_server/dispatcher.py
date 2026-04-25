from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .schemas import IterationArtifacts, IterationJob, IterationResultSummary, IterationStartRequest, PatchPlan


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobPaths:
    job_dir: Path
    log_path: Path
    metadata_path: Path
    exit_code_path: Path


class JobDispatcher:
    def __init__(self, repo_root: Path, jobs_root: Path) -> None:
        self.repo_root = repo_root
        self.jobs_root = jobs_root
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def start(self, patch_plan: PatchPlan, request: IterationStartRequest, before_success: float) -> IterationJob:
        iteration_id = f"iter_{uuid4().hex[:10]}"
        output_checkpoint = request.config.get("output_checkpoint") or f"{request.checkpoint}_patched"
        paths = self._paths(iteration_id)
        paths.job_dir.mkdir(parents=True, exist_ok=True)

        command = self._build_command(patch_plan, request)
        simulate = bool(request.config.get("simulate", False) or command is None)
        success_after = min(1.0, round(before_success + patch_plan.expected_uplift, 3))
        status = "completed" if simulate else "running"
        job = IterationJob(
            iteration_id=iteration_id,
            project_id=patch_plan.project_id,
            plan_id=patch_plan.plan_id,
            based_on_checkpoint=request.checkpoint,
            status=status,
            execution_backend=request.execution_backend or patch_plan.execution_backend,
            config=request.config,
            command=command,
            log_path=str(paths.log_path),
            output_checkpoint=output_checkpoint,
            result_summary=IterationResultSummary(success_before=before_success, success_after=success_after),
            artifacts=IterationArtifacts(
                logs=[str(paths.log_path)],
                videos=[],
                eval_runs=[patch_plan.based_on_eval_run],
                metadata_path=str(paths.metadata_path),
            ),
            created_at=utc_now(),
            updated_at=utc_now(),
        )

        if simulate:
            job.exit_code = 0
            paths.log_path.write_text(
                "Simulated iteration job. Provide config.simulate=false plus backend-specific config to run real commands.\n",
                encoding="utf-8",
            )
            self._write_metadata(job, paths)
            paths.exit_code_path.write_text("0", encoding="utf-8")
            return job

        wrapped_command = (
            f"{{ {command}; }} > {self._quote(paths.log_path)} 2>&1; "
            f"status=$?; echo $status > {self._quote(paths.exit_code_path)}; exit 0"
        )
        process = subprocess.Popen(
            ["bash", "-lc", wrapped_command],
            cwd=self.repo_root,
            start_new_session=True,
        )
        job.pid = process.pid
        self._write_metadata(job, paths)
        return job

    def refresh(self, job: IterationJob) -> IterationJob:
        paths = self._paths(job.iteration_id)
        exit_code = self._read_exit_code(paths)
        is_running = job.pid is not None and self._is_pid_running(job.pid)

        if exit_code is None and is_running:
            job.status = "running"
        elif exit_code is None and job.status == "running":
            job.status = "failed"
            job.exit_code = -1
        elif exit_code == 0:
            job.status = "completed"
            job.exit_code = 0
        elif exit_code is not None:
            job.status = "failed"
            job.exit_code = exit_code

        job.updated_at = utc_now()
        self._write_metadata(job, paths)
        return job

    def _build_command(self, patch_plan: PatchPlan, request: IterationStartRequest) -> str | None:
        config = request.config
        if "command" in config:
            return str(config["command"])

        backend = request.execution_backend or patch_plan.execution_backend
        if backend == "alphabrain_cl":
            yaml = config.get("yaml") or "configs/continual_learning/qwengr00t_cl_lora_libero.yaml"
            parts = ["bash scripts/run_continual_learning_scripts/run_cl_train.sh", f"--yaml {self._quote(yaml)}"]
            if run_id := config.get("run_id"):
                parts.append(f"--run-id {self._quote(run_id)}")
            if gpus := config.get("gpus"):
                parts.append(f"--gpus {self._quote(str(gpus))}")
            return " ".join(parts)

        if backend in {"alphabrain_finetune", "alphabrain_vlm_cotrain"}:
            mode = config.get("mode")
            if not mode:
                return None
            config_file = config.get("config_file")
            return " ".join(
                part
                for part in [
                    "bash scripts/run_finetune.sh",
                    self._quote(mode),
                    self._quote(config_file) if config_file else None,
                ]
                if part
            )

        if backend == "alphabrain_eval":
            mode = config.get("mode")
            if not mode:
                return None
            config_file = config.get("config_file")
            return " ".join(
                part
                for part in [
                    "bash scripts/run_eval.sh",
                    self._quote(mode),
                    self._quote(config_file) if config_file else None,
                ]
                if part
            )

        if backend == "alphabrain_world_model":
            command = config.get("world_model_command")
            return str(command) if command else None

        return None

    def _paths(self, iteration_id: str) -> JobPaths:
        job_dir = self.jobs_root / iteration_id
        return JobPaths(
            job_dir=job_dir,
            log_path=job_dir / "command.log",
            metadata_path=job_dir / "job.json",
            exit_code_path=job_dir / "exit_code.txt",
        )

    def _write_metadata(self, job: IterationJob, paths: JobPaths) -> None:
        paths.metadata_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def _read_exit_code(self, paths: JobPaths) -> int | None:
        if not paths.exit_code_path.exists():
            return None
        try:
            return int(paths.exit_code_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return -1

    def _is_pid_running(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _quote(self, value: str | Path) -> str:
        text = str(value)
        return subprocess.list2cmdline([text])