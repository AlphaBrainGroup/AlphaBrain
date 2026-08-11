"""Helpers for the benchmark-neutral ``evaluation-result-v1.json`` artifact.

This module intentionally depends only on the Python standard library.  The
benchmark clients are commonly executed from dedicated conda environments, so
importing the result writer must not pull in the AlphaBrain UI server stack.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


SCHEMA_VERSION = "evaluation-result-v1"
RESULT_FILENAME = f"{SCHEMA_VERSION}.json"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_result_path(result_out_path: str, video_out_path: str) -> Path:
    """Resolve an explicit result path or a backwards-compatible default.

    ``run_eval.sh`` always supplies an explicit path.  Direct invocations made
    before the UI integration only supplied ``video_out_path``; for those, put
    the result beside a conventional ``videos/`` directory, or inside any
    other video directory.
    """

    if result_out_path:
        return Path(result_out_path).expanduser().resolve()
    video_dir = Path(video_out_path).expanduser().resolve()
    result_dir = video_dir.parent if video_dir.name == "videos" else video_dir
    return result_dir / RESULT_FILENAME


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _safe_rate(successes: int, episodes: int) -> float:
    return float(successes) / float(episodes) if episodes else 0.0


def _artifact_path(path: Optional[str | Path], result_path: Path) -> Optional[str]:
    if path is None:
        return None
    artifact = Path(path).expanduser().resolve()
    try:
        return artifact.relative_to(result_path.parent).as_posix()
    except ValueError:
        return str(artifact)


class EvaluationResultWriter:
    """Incrementally and atomically materialize one evaluation result."""

    def __init__(
        self,
        result_path: str | Path,
        *,
        benchmark: str,
        checkpoint: str,
        suite: Mapping[str, Any],
        parameters: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.path = Path(result_path).expanduser().resolve()
        self._started_monotonic = time.monotonic()
        self.payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "benchmark": benchmark,
            "checkpoint": str(Path(checkpoint).expanduser()) if checkpoint else "",
            "suite": dict(suite),
            "started_at": _utc_now(),
            "finished_at": None,
            "duration_seconds": None,
            "summary": {
                "num_tasks": 0,
                "num_episodes": 0,
                "num_successes": 0,
                "success_rate": 0.0,
            },
            "tasks": [],
            "episodes": [],
            "videos": [],
            "parameters": dict(parameters or {}),
            "metadata": dict(metadata or {}),
            "error": None,
        }
        self.write()

    def _task(self, task_id: str, task_name: str) -> dict[str, Any]:
        for task in self.payload["tasks"]:
            if task["id"] == task_id:
                return task
        task = {
            "id": task_id,
            "name": task_name,
            "num_episodes": 0,
            "num_successes": 0,
            "success_rate": 0.0,
        }
        self.payload["tasks"].append(task)
        return task

    def record_episode(
        self,
        *,
        task_id: str,
        task_name: str,
        episode_index: int,
        success: bool,
        steps: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        video_path: Optional[str | Path] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        relative_video = _artifact_path(video_path, self.path)
        episode: dict[str, Any] = {
            "task_id": task_id,
            "task_name": task_name,
            "episode_index": int(episode_index),
            "success": bool(success),
            "steps": int(steps) if steps is not None else None,
            "duration_seconds": (
                round(float(duration_seconds), 6)
                if duration_seconds is not None
                else None
            ),
            "video": relative_video,
        }
        if metadata:
            episode["metadata"] = dict(metadata)
        self.payload["episodes"].append(episode)

        task = self._task(task_id, task_name)
        task["num_episodes"] += 1
        task["num_successes"] += int(bool(success))
        task["success_rate"] = _safe_rate(
            task["num_successes"], task["num_episodes"]
        )
        if relative_video:
            self.payload["videos"].append(
                {
                    "path": relative_video,
                    "task_id": task_id,
                    "task_name": task_name,
                    "episode_index": int(episode_index),
                    "success": bool(success),
                }
            )
        self._refresh_summary()
        self.write()

    def record_task_summary(
        self,
        *,
        task_id: str,
        task_name: str,
        num_episodes: int,
        num_successes: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record aggregate-only data, primarily for resumed legacy results."""

        task = self._task(task_id, task_name)
        task["num_episodes"] = int(num_episodes)
        task["num_successes"] = int(num_successes)
        task["success_rate"] = _safe_rate(num_successes, num_episodes)
        if metadata:
            task["metadata"] = dict(metadata)
        self._refresh_summary()
        self.write()

    def _refresh_summary(self) -> None:
        tasks = self.payload["tasks"]
        episodes = sum(int(task["num_episodes"]) for task in tasks)
        successes = sum(int(task["num_successes"]) for task in tasks)
        self.payload["summary"] = {
            "num_tasks": len(tasks),
            "num_episodes": episodes,
            "num_successes": successes,
            "success_rate": _safe_rate(successes, episodes),
        }

    def finish(self, metadata: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        if metadata:
            self.payload["metadata"].update(metadata)
        self.payload["status"] = "completed"
        self.payload["finished_at"] = _utc_now()
        self.payload["duration_seconds"] = round(
            time.monotonic() - self._started_monotonic, 6
        )
        self._refresh_summary()
        self.write()
        return copy.deepcopy(self.payload)

    def fail(self, error: str) -> dict[str, Any]:
        self.payload["status"] = "failed"
        self.payload["error"] = str(error)
        self.payload["finished_at"] = _utc_now()
        self.payload["duration_seconds"] = round(
            time.monotonic() - self._started_monotonic, 6
        )
        self._refresh_summary()
        self.write()
        return copy.deepcopy(self.payload)

    def write(self) -> None:
        _atomic_write_json(self.path, self.payload)


def mark_result_failed(
    result_path: str | Path,
    *,
    benchmark: str,
    checkpoint: str,
    error: str,
    suite: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Mark an existing partial result failed, or create a minimal failure."""

    path = Path(result_path).expanduser().resolve()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            payload = {}
    else:
        payload = {}
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("benchmark", benchmark)
    payload.setdefault("checkpoint", checkpoint)
    payload.setdefault("suite", dict(suite or {}))
    payload.setdefault("started_at", _utc_now())
    payload.setdefault("duration_seconds", None)
    payload.setdefault(
        "summary",
        {"num_tasks": 0, "num_episodes": 0, "num_successes": 0, "success_rate": 0.0},
    )
    payload.setdefault("tasks", [])
    payload.setdefault("episodes", [])
    payload.setdefault("videos", [])
    payload.setdefault("parameters", {})
    payload.setdefault("metadata", {})
    existing_error = payload.get("error")
    payload["status"] = "failed"
    payload["finished_at"] = _utc_now()
    payload["error"] = existing_error or error
    _atomic_write_json(path, payload)
    return payload


def aggregate_results(
    output_path: str | Path,
    input_paths: Iterable[str | Path],
    *,
    suite_name: str = "all",
) -> dict[str, Any]:
    """Combine suite result files and rebase their relative artifact paths."""

    output = Path(output_path).expanduser().resolve()
    inputs: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in input_paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported result schema in {path}")
        inputs.append((path, payload))
    if not inputs:
        raise ValueError("At least one result file is required")

    first = inputs[0][1]
    for path, payload in inputs[1:]:
        if payload.get("benchmark") != first.get("benchmark"):
            raise ValueError(f"Benchmark mismatch in {path}")
        if payload.get("checkpoint") != first.get("checkpoint"):
            raise ValueError(f"Checkpoint mismatch in {path}")
    tasks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    statuses: list[str] = []
    suite_entries: list[dict[str, Any]] = []
    errors: list[str] = []

    def rebase(value: Optional[str], source: Path) -> Optional[str]:
        if not value:
            return value
        artifact = Path(value)
        if not artifact.is_absolute():
            artifact = source.parent / artifact
        return _artifact_path(artifact, output)

    for source, payload in inputs:
        statuses.append(str(payload.get("status", "failed")))
        if payload.get("error"):
            errors.append(str(payload["error"]))
        suite_entries.append(dict(payload.get("suite") or {}))
        tasks.extend(copy.deepcopy(payload.get("tasks") or []))
        for episode in payload.get("episodes") or []:
            item = copy.deepcopy(episode)
            item["video"] = rebase(item.get("video"), source)
            episodes.append(item)
        for video in payload.get("videos") or []:
            item = copy.deepcopy(video)
            item["path"] = rebase(item.get("path"), source)
            videos.append(item)

    num_episodes = sum(int(task.get("num_episodes", 0)) for task in tasks)
    num_successes = sum(int(task.get("num_successes", 0)) for task in tasks)
    if all(status == "completed" for status in statuses):
        status = "completed"
    elif all(status == "failed" for status in statuses):
        status = "failed"
    else:
        status = "partial"

    started = [payload.get("started_at") for _, payload in inputs if payload.get("started_at")]
    finished = [payload.get("finished_at") for _, payload in inputs if payload.get("finished_at")]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "benchmark": first.get("benchmark", ""),
        "checkpoint": first.get("checkpoint", ""),
        "suite": {"name": suite_name, "suites": suite_entries},
        "started_at": min(started) if started else None,
        "finished_at": max(finished) if finished else None,
        "duration_seconds": round(
            sum(float(payload.get("duration_seconds") or 0.0) for _, payload in inputs), 6
        ),
        "summary": {
            "num_tasks": len(tasks),
            "num_episodes": num_episodes,
            "num_successes": num_successes,
            "success_rate": _safe_rate(num_successes, num_episodes),
        },
        "tasks": tasks,
        "episodes": episodes,
        "videos": videos,
        "parameters": {},
        "metadata": {"source_results": [str(path) for path, _ in inputs]},
        "error": "; ".join(errors) if errors else None,
    }
    _atomic_write_json(output, result)
    return result


def append_progress(
    path: str | Path,
    *,
    stage: str,
    status: str,
    message: str = "",
) -> None:
    progress_path = Path(path).expanduser().resolve()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": _utc_now(),
        "stage": stage,
        "status": status,
        "message": message,
    }
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaBrain evaluation result utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--suite-name", default="all")
    aggregate.add_argument("inputs", nargs="+")

    progress = subparsers.add_parser("progress")
    progress.add_argument("--path", required=True)
    progress.add_argument("--stage", required=True)
    progress.add_argument("--status", required=True)
    progress.add_argument("--message", default="")

    failure = subparsers.add_parser("fail")
    failure.add_argument("--output", required=True)
    failure.add_argument("--benchmark", required=True)
    failure.add_argument("--checkpoint", default="")
    failure.add_argument("--suite-name", default="")
    failure.add_argument("--error", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "aggregate":
        aggregate_results(args.output, args.inputs, suite_name=args.suite_name)
    elif args.command == "progress":
        append_progress(
            args.path, stage=args.stage, status=args.status, message=args.message
        )
    elif args.command == "fail":
        mark_result_failed(
            args.output,
            benchmark=args.benchmark,
            checkpoint=args.checkpoint,
            suite={"name": args.suite_name},
            error=args.error,
        )


if __name__ == "__main__":
    main()
