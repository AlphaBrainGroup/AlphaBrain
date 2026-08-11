"""
eval_libero_cosmos.py

LIBERO evaluation client for CosmosPolicy.

Key differences from eval_libero.py:
- Uses cosmos-format action unnormalization (actions_min/actions_max from
  libero_dataset_statistics.json), not VLA-Engine q01/q99 format.
- Sends wrist image as second element in images list (cosmos expects [primary, wrist]).
- No action ensemble by default (cosmos uses open-loop chunk execution).
- Proprio is passed as states (last timestep used by server).
- Proprio format: [gripper_qpos(2), eef_pos(3), eef_quat(4)] = 9 dims
  (matches original cosmos-policy, NOT axis-angle).

Usage:
    python benchmarks/LIBERO/eval/eval_libero_cosmos.py \
        --ckpt_dir data/pretrained_models/Cosmos-Policy-LIBERO-Predict2-2B \
        --host 127.0.0.1 --port 10093 \
        --task_suite_name libero_goal \
        --num_trials_per_task 2
"""

import dataclasses
import datetime as dt
import json
import logging
import os
import pathlib
import sys
import time
from typing import Optional

import imageio
import numpy as np
import tqdm
import tyro

# Allow running from the libero conda env
# Override via VLA_EXTRA_SYSPATH env var (colon-separated), e.g. "/path/to/LIBERO:/path/to/AlphaBrain"
import os as _os_syspath
for _p in [p for p in _os_syspath.environ.get("VLA_EXTRA_SYSPATH", "").split(":") if p]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from alphabrain_ui.evaluation_result import EvaluationResultWriter, resolve_result_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Server connection
    host: str = "127.0.0.1"
    port: int = 10093

    # Checkpoint dir (for dataset stats)
    ckpt_dir: str = "data/pretrained_models/Cosmos-Policy-LIBERO-Predict2-2B"

    # LIBERO task suite
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 2
    task_ids: str = ""  # Optional comma-separated suite task IDs; empty = all
    task_limit: int = 0  # Evaluate at most this many selected tasks; 0 = all

    # Output
    video_out_path: str = "experiments/libero/cosmos_logs"
    result_out_path: str = ""  # Explicit evaluation-result-v1.json path (optional)
    job_name: str = "cosmos_eval"
    seed: int = 0  # Must match original cosmos-policy (env.seed(0) affects object positions)


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _get_proprio_from_obs(obs):
    """
    Extract proprio from LIBERO observation matching original cosmos-policy format.
    Order: [gripper_qpos(2), eef_pos(3), eef_quat(4)] = 9 dims.
    """
    return np.concatenate([
        obs["robot0_gripper_qpos"],    # 2 dims
        obs["robot0_eef_pos"],         # 3 dims
        obs["robot0_eef_quat"],        # 4 dims (quaternion, NOT axis-angle)
    ])


def _unnormalize_actions_cosmos(normalized_actions: np.ndarray, dataset_stats: dict) -> np.ndarray:
    """
    Unnormalize actions using cosmos format (actions_min/actions_max).
    Formula: action = 0.5 * (norm + 1) * (max - min) + min
    """
    actions_min = dataset_stats["actions_min"]
    actions_max = dataset_stats["actions_max"]
    orig_shape = normalized_actions.shape
    actions = normalized_actions.reshape(-1, actions_min.shape[0])
    actions = np.clip(actions, -1.0, 1.0)
    actions = 0.5 * (actions + 1.0) * (actions_max - actions_min) + actions_min
    return actions.reshape(orig_shape)


def _binarize_gripper(open_val: float) -> float:
    """Convert continuous gripper value to binary {-1, 1}."""
    return 1.0 - 2.0 * (float(open_val) > 0.5)


def eval_libero_cosmos(args: Args) -> None:
    result_path = resolve_result_path(args.result_out_path, args.video_out_path)
    result_writer = EvaluationResultWriter(
        result_path,
        benchmark="libero",
        checkpoint=args.ckpt_dir,
        suite={"name": args.task_suite_name, "task_ids": args.task_ids},
        parameters={
            "num_trials_per_task": args.num_trials_per_task,
            "num_steps_wait": args.num_steps_wait,
            "task_limit": args.task_limit,
            "seed": args.seed,
            "num_views": 2,
        },
        metadata={"client_adapter": "cosmos_policy"},
    )
    try:
        _eval_libero_cosmos(args, result_writer)
    except BaseException as exc:
        result_writer.fail(f"{type(exc).__name__}: {exc}")
        raise


def _eval_libero_cosmos(args: Args, result_writer: EvaluationResultWriter) -> None:
    logging.info(f"CosmosPolicy LIBERO eval: {dataclasses.asdict(args)}")
    if args.num_trials_per_task <= 0:
        raise ValueError("num_trials_per_task must be positive")

    # Load dataset statistics
    stats_path = os.path.join(args.ckpt_dir, "libero_dataset_statistics.json")
    assert os.path.exists(stats_path), f"Dataset stats not found: {stats_path}"
    with open(stats_path) as f:
        dataset_stats = json.load(f)
    for k, v in dataset_stats.items():
        dataset_stats[k] = np.array(v)
    logging.info(f"Loaded dataset stats: keys={list(dataset_stats.keys())}")

    # Connect to policy server
    client = WebsocketClientPolicy(args.host, args.port)

    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}, {num_tasks} tasks")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_map.get(args.task_suite_name, 300)

    total_episodes, total_successes = 0, 0
    chunk_size = 16  # cosmos uses 16-step action chunks

    filter_ids = (
        {int(value) for value in args.task_ids.split(",") if value.strip()}
        if args.task_ids
        else None
    )
    if args.task_limit < 0:
        raise ValueError("task_limit must be non-negative")
    evaluated_tasks = 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        if filter_ids is not None and task_id not in filter_ids:
            continue
        if args.task_limit and evaluated_tasks >= args.task_limit:
            break
        evaluated_tasks += 1
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            episode_started_at = time.monotonic()
            logging.info(f"\nTask: {task_description}")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # Initialize proprio history (n=16 steps)
            # Cosmos proprio: [gripper_qpos(2), eef_pos(3), eef_quat(4)] = 9 dims
            n_proprio = 16
            state = _get_proprio_from_obs(obs)
            states = np.tile(state, (n_proprio, 1))  # (n, 9)

            t = 0
            done = False
            replay_images = []
            cached_actions = None
            step_in_chunk = 0

            while t < max_steps + args.num_steps_wait:
                # Wait for objects to stabilize
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # LIBERO images need vertical flip to match training data
                primary_img = np.ascontiguousarray(obs["agentview_image"][::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
                replay_images.append(primary_img)

                # Update proprio history
                state = _get_proprio_from_obs(obs)
                states = np.vstack([states[1:], state])  # rolling window

                # Query model every chunk_size steps
                if cached_actions is None or step_in_chunk >= chunk_size:
                    vla_input = {
                        "batch_images": [[primary_img, wrist_img]],
                        "instructions": [task_description],
                        "states": np.expand_dims(states, axis=0).astype(np.float32),
                    }
                    response = client.infer(vla_input)

                    if response.get("status") == "error" or "data" not in response:
                        err = response.get("error", {}).get("message", "unknown")
                        raise RuntimeError(f"Server error: {err}")

                    normalized_actions = np.array(response["data"]["normalized_actions"])
                    if normalized_actions.ndim == 3:
                        normalized_actions = normalized_actions[0]  # (chunk_size, action_dim)

                    # Unnormalize using cosmos stats
                    cached_actions = _unnormalize_actions_cosmos(normalized_actions, dataset_stats)
                    step_in_chunk = 0

                # Execute current action from chunk
                # Pass raw unnormalized action directly to env.step()
                # (matches original cosmos-policy — no gripper binarization)
                raw_action = cached_actions[step_in_chunk]
                step_in_chunk += 1

                obs, reward, done, info = env.step(raw_action.tolist())

                if done:
                    task_successes += 1
                    total_successes += 1
                    break

                t += 1

            task_episodes += 1
            total_episodes += 1

            # Save replay video
            suffix = "success" if done else "failure"
            task_seg = task_description.replace(" ", "_")[:40]
            video_path = (
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_seg}_ep{episode_idx}_{suffix}.mp4"
            )
            imageio.mimwrite(str(video_path), [np.asarray(x) for x in replay_images], fps=10)

            result_writer.record_episode(
                task_id=f"{args.task_suite_name}:{task_id}",
                task_name=str(task_description),
                episode_index=episode_idx,
                success=bool(done),
                steps=max(0, t - args.num_steps_wait),
                duration_seconds=time.monotonic() - episode_started_at,
                video_path=video_path,
                metadata={"suite_task_id": task_id},
            )

            logging.info(f"Episode {episode_idx}: {'SUCCESS' if done else 'FAILURE'}")
            logging.info(
                f"Total: {total_successes}/{total_episodes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(
            f"Task {task_id} ({task_description[:40]}): "
            f"{task_successes}/{task_episodes} = "
            f"{task_successes / max(task_episodes, 1) * 100:.1f}%"
        )
        env.close()

    if total_episodes == 0:
        raise ValueError("No LIBERO tasks were selected")
    logging.info(
        f"\n=== FINAL RESULTS ===\n"
        f"Task suite: {args.task_suite_name}\n"
        f"Total success rate: {total_successes}/{total_episodes} "
        f"= {total_successes / max(total_episodes, 1) * 100:.1f}%"
    )
    result_writer.finish(metadata={"client_adapter": "cosmos_policy"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    eval_libero_cosmos(args)
