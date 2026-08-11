"""Small deterministic workers used by ``python -m alphabrain_ui --demo``."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train(output_dir: Path, metrics_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    print("[demo] Starting simulated VLA fine-tuning", flush=True)
    print("[demo] Dataset: LIBERO Mini (2 episodes, 12 frames)", flush=True)
    with metrics_path.open("a", encoding="utf-8") as metrics:
        for step in range(1, 11):
            loss = round(1.25 / (step + 1) + 0.025, 4)
            learning_rate = round(0.001 * (1 - step / 12), 7)
            record = {
                "step": step,
                "loss": loss,
                "learning_rate": learning_rate,
                "throughput": round(18.0 + step * 0.7, 2),
                "demo": True,
            }
            metrics.write(json.dumps(record, separators=(",", ":")) + "\n")
            metrics.flush()
            print(f"[demo] step={step:02d}/10 loss={loss:.4f} lr={learning_rate:.7f}", flush=True)
            time.sleep(0.12)
    checkpoint = output_dir / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "model.safetensors").write_bytes(b"ALPHABRAIN-DEMO-WEIGHTS\n")
    _write_json(
        checkpoint / "framework_config.yaml.json",
        {"framework": {"name": "DemoBaseFramework"}, "demo": True},
    )
    (checkpoint / "framework_config.yaml").write_text(
        "framework:\n  name: DemoBaseFramework\nbackbone: qwen2_5_vl\naction_head: mlp_regression\ndemo: true\n",
        encoding="utf-8",
    )
    _write_json(
        checkpoint / "dataset_statistics.json",
        {"action": {"mean": [0.0] * 7, "std": [1.0] * 7}, "demo": True},
    )
    _write_json(checkpoint / "resume_meta.json", {"step": 10, "demo": True})
    vlm = checkpoint / "vlm_pretrained"
    _write_json(vlm / "config.json", {"model_type": "qwen2_5_vl", "demo": True})
    _write_json(vlm / "preprocessor_config.json", {"processor_class": "DemoProcessor"})
    _write_json(
        output_dir / "run-summary.json",
        {"status": "completed", "steps": 10, "final_loss": 0.1386, "checkpoint": str(checkpoint), "demo": True},
    )
    print(f"[demo] Checkpoint saved: {checkpoint}", flush=True)
    print("[demo] Simulated training completed successfully", flush=True)


class DemoPolicy:
    def predict_action(self, **payload: Any) -> dict[str, Any]:
        instructions = payload.get("prompt") or payload.get("instructions") or payload.get("instruction") or [""]
        if isinstance(instructions, str):
            instructions = [instructions]
        batch_size = max(1, len(instructions) if isinstance(instructions, list) else 1)
        actions = []
        for batch_index in range(batch_size):
            actions.append(
                [
                    [round(0.05 * (batch_index + 1), 3), -0.1, 0.15, 0.0, 0.08, -0.04, 1.0],
                    [round(0.08 * (batch_index + 1), 3), -0.06, 0.1, 0.02, 0.04, -0.02, 0.0],
                ]
            )
        return {
            "actions": actions,
            "action_horizon": 2,
            "action_dimension": 7,
            "policy": "deterministic-demo-policy",
            "simulated": True,
        }


def serve(args: argparse.Namespace) -> None:
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    print(f"[demo] Starting deterministic policy server on {args.host}:{args.port}", flush=True)
    server = WebsocketPolicyServer(
        DemoPolicy(),
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={
            "model": "AlphaBrain Demo Policy",
            "checkpoint": args.checkpoint,
            "action_horizon": 2,
            "action_dimension": 7,
            "simulated": True,
        },
        api_key_sha256=args.api_key_sha256,
        controller_api_key_sha256=args.controller_api_key_sha256,
        deployment_id=args.deployment_id,
    )
    server.serve_forever()


def evaluate(args: argparse.Namespace) -> None:
    result_path = Path(args.result_path)
    progress_path = Path(args.progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    print(f"[demo] Starting {args.benchmark} evaluation for {args.suite}", flush=True)
    episodes = []
    task_specs = [
        ("pick-red-block", "Pick up the red block"),
        ("place-in-tray", "Place the block in the tray"),
    ]
    with progress_path.open("a", encoding="utf-8") as progress:
        for task_index, (task_id, task_name) in enumerate(task_specs):
            for episode_index in range(2):
                success = not (task_index == 1 and episode_index == 1)
                row = {
                    "type": "episode",
                    "task_id": task_id,
                    "task_name": task_name,
                    "episode_index": episode_index,
                    "success": success,
                    "steps": 31 + task_index * 6 + episode_index,
                    "timestamp": _now(),
                    "demo": True,
                }
                progress.write(json.dumps(row, separators=(",", ":")) + "\n")
                progress.flush()
                episodes.append(
                    {
                        "task_id": task_id,
                        "task_name": task_name,
                        "episode_index": episode_index,
                        "success": success,
                        "steps": row["steps"],
                        "duration_seconds": 0.18,
                        "metadata": {"demo": True},
                    }
                )
                print(f"[demo] {task_id} episode={episode_index} success={success}", flush=True)
                time.sleep(0.12)
    tasks = []
    for task_id, task_name in task_specs:
        task_episodes = [row for row in episodes if row["task_id"] == task_id]
        successes = sum(1 for row in task_episodes if row["success"])
        tasks.append(
            {
                "id": task_id,
                "name": task_name,
                "num_episodes": len(task_episodes),
                "num_successes": successes,
                "success_rate": successes / len(task_episodes),
                "metadata": {"demo": True},
            }
        )
    summary = {
        "num_tasks": len(tasks),
        "num_episodes": len(episodes),
        "num_successes": sum(1 for row in episodes if row["success"]),
        "success_rate": sum(1 for row in episodes if row["success"]) / len(episodes),
    }
    common = {
        "status": "completed",
        "benchmark": args.benchmark,
        "checkpoint": args.checkpoint,
        "suite": {"name": args.suite},
        "started_at": started,
        "finished_at": _now(),
        "duration_seconds": 0.5,
        "summary": summary,
        "tasks": tasks,
        "episodes": episodes,
        "videos": [],
        "parameters": {"demo": True},
        "metadata": {"demo": True, "simulated": True},
        "error": None,
    }
    if args.schema_version == "evaluation-result-v2":
        result = {
            "schema_version": "evaluation-result-v2",
            "kind": args.kind if args.kind in {"standard", "batch", "cl_matrix", "rl_iterations", "online_stdp", "world_model_video"} else "standard",
            **common,
            "artifacts": [],
            "matrices": [],
            "series": [],
            "comparisons": [],
            "children": [],
        }
    else:
        result = {"schema_version": "evaluation-result-v1", **common}
    _write_json(result_path, result)
    print(f"[demo] Evaluation result saved: {result_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaBrain demo workers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--metrics-path", type=Path, required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, required=True)
    serve_parser.add_argument("--deployment-id", required=True)
    serve_parser.add_argument("--checkpoint", default="")
    serve_parser.add_argument("--idle-timeout", type=int, default=-1)
    serve_parser.add_argument("--api-key-sha256", required=True)
    serve_parser.add_argument("--controller-api-key-sha256", default=None)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--result-path", required=True)
    eval_parser.add_argument("--progress-path", required=True)
    eval_parser.add_argument("--schema-version", default="evaluation-result-v1")
    eval_parser.add_argument("--kind", default="standard")
    eval_parser.add_argument("--benchmark", default="libero")
    eval_parser.add_argument("--checkpoint", default="")
    eval_parser.add_argument("--suite", default="libero_goal")

    args = parser.parse_args()
    if args.command == "train":
        train(args.output_dir, args.metrics_path)
    elif args.command == "serve":
        serve(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
