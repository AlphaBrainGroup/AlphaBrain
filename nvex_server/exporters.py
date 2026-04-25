from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from .schemas import ArtifactBundle, ArtifactType, EvalRun, FailureCluster, TaskBreakdownEntry

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class EvalArtifactExporter:
    def export(
        self,
        artifact_path: str,
        artifact_type: ArtifactType = "auto",
        *,
        project_id: str | None = None,
        benchmark_suite: str | None = None,
        checkpoint: str | None = None,
        run_id: str | None = None,
    ) -> EvalRun:
        path = Path(artifact_path)
        resolved_type = self._detect_type(path, artifact_type)
        parser = {
            "generic_json": self._from_generic_json,
            "libero_eval_json": self._from_libero_eval_json,
            "robocasa365_aggregate": self._from_robocasa365_aggregate,
            "robocasa_tabletop_stats": self._from_robocasa_tabletop_stats,
            "libero_log": self._from_libero_log,
        }[resolved_type]

        eval_run = parser(path, project_id=project_id, benchmark_suite=benchmark_suite, checkpoint=checkpoint, run_id=run_id)
        eval_run.artifacts.source_path = str(path)
        if eval_run.artifacts.metrics_json is None and path.suffix == ".json":
            eval_run.artifacts.metrics_json = str(path)
        return eval_run

    def _detect_type(self, path: Path, artifact_type: ArtifactType) -> ArtifactType:
        if artifact_type != "auto":
            return artifact_type

        if path.name == "aggregate_stats.json":
            return "robocasa365_aggregate"
        if path.name == "eval_results.json":
            return "libero_eval_json"
        if path.suffix == ".log":
            return "libero_log"
        return "generic_json"

    def _from_generic_json(
        self,
        path: Path,
        *,
        project_id: str | None,
        benchmark_suite: str | None,
        checkpoint: str | None,
        run_id: str | None,
    ) -> EvalRun:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        return self._build_eval_run(
            payload,
            project_id=project_id,
            benchmark_suite=benchmark_suite,
            checkpoint=checkpoint,
            run_id=run_id,
            path=path,
        )

    def _from_libero_eval_json(self, path: Path, **kwargs: str | None) -> EvalRun:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("benchmark_suite", payload.get("task_suite", kwargs.get("benchmark_suite") or "LIBERO_goal"))
        return self._build_eval_run(payload, path=path, **kwargs)

    def _from_robocasa365_aggregate(self, path: Path, **kwargs: str | None) -> EvalRun:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        task_breakdown = []
        for task_name, task_data in payload.get("tasks", {}).items():
            attempts = int(task_data.get("num_episodes", 0))
            success_rate = float(task_data.get("success_rate", 0.0))
            task_breakdown.append(
                {
                    "task_id": task_name,
                    "task_name": task_name,
                    "attempts": attempts,
                    "successes": round(attempts * success_rate),
                    "success_rate": success_rate,
                }
            )

        generic_payload = {
            "run_id": kwargs.get("run_id") or f"eval_{uuid4().hex[:10]}",
            "project_id": kwargs.get("project_id") or "proj_robocasa365",
            "benchmark_suite": kwargs.get("benchmark_suite") or "robocasa365",
            "checkpoint": kwargs.get("checkpoint"),
            "success_rate": float(payload.get("mean_success_rate", 0.0)),
            "task_breakdown": task_breakdown,
        }
        return self._build_eval_run(generic_payload, path=path, **kwargs)

    def _from_robocasa_tabletop_stats(self, path: Path, **kwargs: str | None) -> EvalRun:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        generic_payload = {
            "run_id": kwargs.get("run_id") or f"eval_{uuid4().hex[:10]}",
            "project_id": kwargs.get("project_id") or "proj_robocasa_tabletop",
            "benchmark_suite": kwargs.get("benchmark_suite") or "robocasa_tabletop",
            "checkpoint": kwargs.get("checkpoint"),
            "success_rate": float(payload.get("success_rate", 0.0)),
            "task_breakdown": [
                {
                    "task_id": path.parent.name,
                    "task_name": path.parent.name,
                    "attempts": int(payload.get("num_episodes", 0)),
                    "successes": round(int(payload.get("num_episodes", 0)) * float(payload.get("success_rate", 0.0))),
                    "success_rate": float(payload.get("success_rate", 0.0)),
                }
            ],
        }
        return self._build_eval_run(generic_payload, path=path, **kwargs)

    def _from_libero_log(self, path: Path, **kwargs: str | None) -> EvalRun:
        text = ANSI_ESCAPE_RE.sub("", path.read_text(encoding="utf-8"))
        task_rates = [float(match.group(1)) for match in re.finditer(r"Current task success rate:\s*([0-9.]+)", text)]
        total_match = re.search(r"Total success rate:\s*([0-9.]+)", text)
        overall_success = float(total_match.group(1)) if total_match else 0.0
        task_breakdown = [
            {
                "task_id": f"task_{index + 1}",
                "task_name": f"Task {index + 1}",
                "success_rate": task_rate,
            }
            for index, task_rate in enumerate(task_rates)
        ]
        generic_payload = {
            "run_id": kwargs.get("run_id") or f"eval_{uuid4().hex[:10]}",
            "project_id": kwargs.get("project_id") or "proj_libero",
            "benchmark_suite": kwargs.get("benchmark_suite") or "LIBERO_goal",
            "checkpoint": kwargs.get("checkpoint"),
            "success_rate": overall_success,
            "task_breakdown": task_breakdown,
        }
        return self._build_eval_run(generic_payload, path=path, **kwargs)

    def _build_eval_run(
        self,
        payload: dict,
        *,
        path: Path,
        project_id: str | None,
        benchmark_suite: str | None,
        checkpoint: str | None,
        run_id: str | None,
    ) -> EvalRun:
        task_breakdown_payload = payload.get("task_breakdown") or []
        task_breakdown = [
            TaskBreakdownEntry(
                task_id=item.get("task_id") or item.get("name") or f"task_{index + 1}",
                task_name=item.get("task_name") or item.get("name") or item.get("task_id") or f"Task {index + 1}",
                success_rate=float(item.get("success_rate", item.get("pct", 0.0))),
                attempts=item.get("attempts") or item.get("num_episodes"),
                successes=item.get("successes") or item.get("success_count"),
            )
            for index, item in enumerate(task_breakdown_payload)
        ]
        failure_clusters_payload = payload.get("failure_clusters") or self._infer_failure_clusters(task_breakdown)
        failure_clusters = [
            FailureCluster(
                cluster_id=item.get("cluster_id") or item.get("id") or chr(65 + index),
                label=item.get("label") or item.get("task_name") or f"Failure cluster {index + 1}",
                failure_pattern=item.get("failure_pattern") or item.get("label") or "task_reliability_gap",
                affected_tasks=item.get("affected_tasks") or [],
                share_of_failures=float(item.get("share_of_failures", item.get("pct", 0.0))),
                failure_count=int(item.get("failure_count", item.get("count", 0))),
                severity=item.get("severity") or item.get("sev") or "medium",
            )
            for index, item in enumerate(failure_clusters_payload)
        ]
        overall_success = float(payload.get("overall_success", payload.get("success_rate", 0.0)))

        return EvalRun(
            run_id=run_id or payload.get("run_id") or f"eval_{uuid4().hex[:10]}",
            project_id=project_id or payload.get("project_id") or "proj_001",
            benchmark_suite=benchmark_suite or payload.get("benchmark_suite") or "LIBERO_goal",
            checkpoint=checkpoint or payload.get("checkpoint"),
            overall_success=overall_success,
            task_breakdown=task_breakdown,
            failure_clusters=failure_clusters,
            artifacts=ArtifactBundle(
                videos=payload.get("videos", []),
                logs=payload.get("logs", []),
                metrics_json=str(path) if path.suffix == ".json" else None,
                source_path=str(path),
            ),
        )

    def _infer_failure_clusters(self, task_breakdown: list[TaskBreakdownEntry]) -> list[dict[str, object]]:
        if not task_breakdown:
            return []

        sorted_tasks = sorted(task_breakdown, key=lambda task: task.success_rate)
        weakest = sorted_tasks[: min(3, len(sorted_tasks))]
        total_failure_mass = sum(max(0.0, 1.0 - task.success_rate) for task in weakest) or 1.0
        severity_by_rank = ["critical", "high", "medium"]
        fallback_patterns = ["occlusion", "recovery", "generalization"]

        return [
            {
                "cluster_id": chr(65 + index),
                "label": f"{task.task_name} underperformance",
                "failure_pattern": fallback_patterns[index] if index < len(fallback_patterns) else "task_reliability_gap",
                "affected_tasks": [task.task_name],
                "share_of_failures": round(max(0.0, 1.0 - task.success_rate) / total_failure_mass, 3),
                "failure_count": max(1, task.attempts - (task.successes or 0)) if task.attempts is not None else 1,
                "severity": severity_by_rank[index] if index < len(severity_by_rank) else "medium",
            }
            for index, task in enumerate(weakest)
        ]