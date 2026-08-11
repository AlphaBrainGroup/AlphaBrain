"""Managed runners and parsers for non-standard AlphaBrain evaluations.

The heavy benchmark programs remain the source of truth.  This module gives
the UI one small, deterministic process to launch them and converts their
native outputs into ``evaluation-result-v2.json``.  Parser functions are kept
side-effect free so result fixtures can be tested without CUDA or simulators.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

RESULT_V2_FILENAME = "evaluation-result-v2.json"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_result_v2(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a result assembled by the API or manager."""

    _atomic_json(Path(path).expanduser().resolve(), payload)


def _rate(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result > 1.0 and result <= 100.0:
        result /= 100.0
    return max(0.0, min(1.0, result))


def _base_result(
    *,
    kind: str,
    benchmark: str,
    checkpoint: str,
    suite: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": "evaluation-result-v2",
        "kind": kind,
        "status": "completed",
        "benchmark": benchmark,
        "checkpoint": checkpoint,
        "suite": {"name": suite},
        "started_at": None,
        "finished_at": now,
        "duration_seconds": None,
        "summary": dict(summary),
        "tasks": [],
        "episodes": [],
        "videos": [],
        "artifacts": [],
        "matrices": [],
        "series": [],
        "comparisons": [],
        "children": [],
        "parameters": {},
        "metadata": {},
        "error": None,
    }


def parse_cl_matrix(matrix_dir: str | Path, *, checkpoint: str = "") -> dict[str, Any]:
    """Parse the T x T CL logs and calculate ASR, BWT, and forgetting."""

    root = Path(matrix_dir).expanduser().resolve()
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("task_")),
        key=lambda path: int(re.search(r"task_(\d+)", path.name).group(1))
        if re.search(r"task_(\d+)", path.name)
        else sys.maxsize,
    )
    if not directories:
        raise ValueError(f"No task_* checkpoint results found under {root}")
    matrix: list[list[float]] = []
    for directory in directories:
        log = directory / "eval.log"
        text = log.read_text(encoding="utf-8", errors="ignore")
        clean = re.sub(r"\x1b\[[^m]*m", "", text)
        row = [float(value) for value in re.findall(r"Current task success rate:\s*([0-9.]+)", clean)]
        if not row:
            raise ValueError(f"Could not parse per-task success rates from {log}")
        matrix.append(row)
    size = len(matrix)
    if any(len(row) < size for row in matrix):
        raise ValueError("Continual-learning result matrix is incomplete")
    matrix = [row[:size] for row in matrix]
    permutation = [max(range(size), key=lambda column: matrix[0][column])]
    for index in range(1, size):
        permutation.append(
            max(range(size), key=lambda column: matrix[index][column] - matrix[index - 1][column])
        )
    asr = sum(matrix[-1]) / size
    if size > 1:
        bwt_terms = [matrix[-1][permutation[index]] - matrix[index][permutation[index]] for index in range(size - 1)]
        forgetting_terms = [
            max(matrix[row][permutation[index]] for row in range(index, size - 1))
            - matrix[-1][permutation[index]]
            for index in range(size - 1)
        ]
        bwt = sum(bwt_terms) / len(bwt_terms)
        forgetting = sum(forgetting_terms) / len(forgetting_terms)
    else:
        bwt = forgetting = 0.0
    result = _base_result(
        kind="cl_matrix",
        benchmark="continual_learning",
        checkpoint=checkpoint,
        suite=root.name,
        summary={
            "success_rate": asr,
            "asr": asr,
            "bwt": bwt,
            "forgetting": forgetting,
            "num_checkpoints": size,
            "num_tasks": size,
            "num_episodes": 0,
            "num_successes": 0,
        },
    )
    result["matrices"] = [
        {
            "id": "continual_success_rate",
            "name": "Checkpoint x task success rate",
            "row_labels": [path.name for path in directories],
            "column_labels": [f"Task {index}" for index in range(size)],
            "values": matrix,
            "value_format": "rate",
            "metadata": {"train_to_eval_permutation": permutation},
        }
    ]
    result["artifacts"] = [
        {"kind": "log", "path": f"{path.name}/eval.log", "name": f"{path.name} eval log"}
        for path in directories
    ]
    return result


def parse_rl_iterations(summary_path: str | Path, *, checkpoint: str = "", suite: str = "") -> dict[str, Any]:
    """Parse ``all_iters_summary.json`` produced by ``run_eval_rlt.sh``."""

    path = Path(summary_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, list) or not raw:
        raise ValueError("RL iteration summary must be a non-empty list")
    overall_points: list[dict[str, Any]] = []
    per_task: dict[str, list[dict[str, Any]]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("iter", index))
        match = re.search(r"(\d+)$", label)
        x_value: int | str = int(match.group(1)) if match else label
        overall = _rate(item.get("overall_sr"))
        overall_points.append({"x": x_value, "y": overall, "metadata": {"label": label}})
        task_rates = item.get("per_task_sr")
        if isinstance(task_rates, Mapping):
            entries: Iterable[tuple[Any, Any]] = task_rates.items()
        elif isinstance(task_rates, list):
            entries = enumerate(task_rates)
        else:
            entries = []
        for task, value in entries:
            per_task.setdefault(str(task), []).append({"x": x_value, "y": _rate(value)})
    valid_overall = [point for point in overall_points if point["y"] is not None]
    if not valid_overall:
        raise ValueError("RL iteration summary contains no valid overall success rate")
    best = max(valid_overall, key=lambda point: float(point["y"]))
    final = valid_overall[-1]
    result = _base_result(
        kind="rl_iterations",
        benchmark="libero",
        checkpoint=checkpoint,
        suite=suite,
        summary={
            "success_rate": final["y"],
            "final_success_rate": final["y"],
            "best_success_rate": best["y"],
            "best_iteration": best["x"],
            "num_iterations": len(overall_points),
            "num_tasks": len(per_task),
            "num_episodes": 0,
            "num_successes": 0,
        },
    )
    result["series"] = [
        {
            "id": "overall_success_rate",
            "name": "Overall success rate",
            "x_label": "RL iteration",
            "y_label": "Success rate",
            "points": overall_points,
        },
        *[
            {
                "id": f"task_{task}_success_rate",
                "name": f"Task {task}",
                "x_label": "RL iteration",
                "y_label": "Success rate",
                "points": points,
            }
            for task, points in sorted(per_task.items())
        ],
    ]
    result["artifacts"] = [{"kind": "json", "path": path.name, "name": path.name}]
    return result


def parse_online_stdp(
    output_dir: str | Path,
    *,
    checkpoint: str,
    suite: str,
    adapted: bool,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    result_files = sorted(root.glob("*/eval_results.json")) or sorted(root.glob("**/eval_results.json"))
    if not result_files:
        raise ValueError(f"No eval_results.json found under {root}")
    task_rows: list[dict[str, Any]] = []
    rates: list[float] = []
    drift: float | None = None
    for path in result_files:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, Mapping):
            continue
        rate = _rate(data.get("success_rate"))
        if rate is not None:
            rates.append(rate)
        if data.get("final_weight_drift") is not None:
            drift = float(data["final_weight_drift"])
        task_rows.append(
            {
                "id": path.parent.name,
                "name": path.parent.name,
                "num_episodes": int(data.get("total_episodes", data.get("num_episodes", 0)) or 0),
                "num_successes": int(data.get("total_successes", data.get("num_successes", 0)) or 0),
                "success_rate": rate or 0.0,
                "metadata": {key: value for key, value in data.items() if key not in {"success_rate"}},
            }
        )
    success_rate = sum(rates) / len(rates) if rates else 0.0
    result = _base_result(
        kind="online_stdp",
        benchmark="libero",
        checkpoint=checkpoint,
        suite=suite,
        summary={
            "success_rate": success_rate,
            "num_tasks": len(task_rows),
            "num_episodes": sum(row["num_episodes"] for row in task_rows),
            "num_successes": sum(row["num_successes"] for row in task_rows),
            "final_weight_drift": drift,
            "adapted": adapted,
        },
    )
    result["tasks"] = task_rows
    result["metadata"] = {"online_stdp": adapted}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".mp4", ".webm", ".log"}:
            kind = "video" if path.suffix.lower() in {".mp4", ".webm"} else path.suffix.lstrip(".")
            result["artifacts"].append(
                {"kind": kind, "path": path.relative_to(root).as_posix(), "name": path.name}
            )
    return result


def upgrade_v1_result(
    result: Mapping[str, Any],
    *,
    kind: str,
    artifacts: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Wrap a valid v1 result without losing any existing fields."""

    if result.get("schema_version") not in {"evaluation-result-v1", "evaluation-result-v2"}:
        raise ValueError("Unsupported evaluation result schema")
    if result.get("schema_version") == "evaluation-result-v2":
        upgraded = dict(result)
        upgraded.setdefault("kind", kind)
        return upgraded
    upgraded = dict(result)
    upgraded.update(
        {
            "schema_version": "evaluation-result-v2",
            "kind": kind,
            "artifacts": [dict(item) for item in artifacts],
            "matrices": [],
            "series": [],
            "comparisons": [],
            "children": [],
        }
    )
    return upgraded


def _run(command: list[str], *, cwd: Path, environment: Mapping[str, str]) -> None:
    completed = subprocess.run(command, cwd=cwd, env=dict(environment), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"specialized evaluator exited with code {completed.returncode}")


def run_specialized(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(args.parameters_json or "{}")
    if not isinstance(parameters, dict):
        raise ValueError("parameters-json must contain an object")
    gpu_ids = [value for value in args.gpu_ids.split(",") if value]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root) + os.pathsep + environment.get("PYTHONPATH", "")
    started = _utc_now()

    if args.kind == "cl_matrix":
        run_id = str(parameters.get("run_id") or Path(args.checkpoint).name)
        command = [
            "bash",
            "scripts/run_continual_learning_scripts/run_cl_eval.sh",
            "--run-id",
            run_id,
            "--benchmark",
            str(parameters.get("benchmark", "libero")),
            "--gpus",
            ",".join(gpu_ids or ["0"]),
            "--output-base",
            str(output_dir),
        ]
        if parameters.get("model"):
            command.extend(["--model", str(parameters["model"])])
        if args.suite:
            command.extend(["--suite", args.suite])
        if parameters.get("trials"):
            command.extend(["--trials", str(parameters["trials"])])
        if parameters.get("last_only"):
            command.append("--last-only")
        _run(command, cwd=repo_root, environment=environment)
        result = parse_cl_matrix(output_dir, checkpoint=args.checkpoint)
    elif args.kind == "rl_iterations":
        run_dir = str(parameters.get("run_dir") or "")
        if not run_dir:
            raise ValueError("RL iteration evaluation requires parameters.run_dir")
        environment.update(
            {
                "RUN_DIR": run_dir,
                "VLA_CKPT": args.checkpoint,
                "GPUS": " ".join(gpu_ids or ["0"]),
                "SUITE": args.suite or "libero_goal",
                "TASK_IDS": str(parameters.get("task_ids", "0")),
                "N_EPS": str(parameters.get("n_eps", parameters.get("num_trials", 50))),
                "NUM_WORKERS": str(parameters.get("num_workers", 4)),
            }
        )
        for key in (
            "bottleneck_dim",
            "encoder_layers",
            "encoder_heads",
            "actor_hidden_dim",
            "ref_dropout",
            "fixed_std",
            "iter",
        ):
            if parameters.get(key) is not None:
                environment[key.upper()] = str(parameters[key])
        _run(["bash", "scripts/run_rl_scripts/run_eval_rlt.sh"], cwd=repo_root, environment=environment)
        run_root = Path(run_dir).expanduser()
        if not run_root.is_absolute():
            run_root = repo_root / run_root
        summary = run_root / "eval_all_iters_rlt" / "all_iters_summary.json"
        result = parse_rl_iterations(summary, checkpoint=args.checkpoint, suite=args.suite)
        # Keep the managed result self-contained even though native per-iter
        # outputs remain in the RL training directory.
        (output_dir / summary.name).write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
    elif args.kind in {"online_stdp_baseline", "online_stdp_adapted"}:
        adapted = args.kind.endswith("adapted")
        command = [
            "bash",
            "scripts/run_brain_inspired_scripts/run_eval_libero.sh",
            "--pretrained",
            args.checkpoint,
            "--suite",
            args.suite or "libero_goal",
            "--trials",
            str(parameters.get("num_trials", 10)),
            "--seed",
            str(parameters.get("seed", 7)),
            "--gpu",
            (gpu_ids or ["0"])[0],
            "--video-out",
            str(output_dir),
        ]
        if adapted:
            command.append("--online-stdp")
            for parameter, environment_key in (
                ("stdp_lr", "STDP_LR"),
                ("stdp_warmup", "STDP_WARMUP"),
                ("stdp_max_deviation", "STDP_MAX_DEV"),
                ("stdp_rollback_shrink", "STDP_ROLLBACK"),
            ):
                if parameters.get(parameter) is not None:
                    environment[environment_key] = str(parameters[parameter])
        _run(command, cwd=repo_root, environment=environment)
        result = parse_online_stdp(
            output_dir,
            checkpoint=args.checkpoint,
            suite=args.suite,
            adapted=adapted,
        )
    else:
        raise ValueError(f"Unsupported specialized evaluation kind: {args.kind}")
    result["started_at"] = started
    result["finished_at"] = _utc_now()
    result["parameters"] = parameters
    _atomic_json(Path(args.result_path), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--suite", default="")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--parameters-json", default="{}")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        run_specialized(args)
    except Exception as error:
        failure = _base_result(
            kind=args.kind.removesuffix("_baseline").removesuffix("_adapted"),
            benchmark="",
            checkpoint=args.checkpoint,
            suite=args.suite,
            summary={"success_rate": 0.0, "num_tasks": 0, "num_episodes": 0, "num_successes": 0},
        )
        failure["status"] = "failed"
        failure["error"] = f"{type(error).__name__}: {error}"
        _atomic_json(Path(args.result_path), failure)
        print(failure["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RESULT_V2_FILENAME",
    "parse_cl_matrix",
    "parse_online_stdp",
    "parse_rl_iterations",
    "run_specialized",
    "upgrade_v1_result",
    "write_result_v2",
]
