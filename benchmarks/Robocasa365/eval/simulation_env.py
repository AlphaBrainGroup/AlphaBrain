import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import tyro

import robocasa  # noqa: F401
import robosuite  # noqa: F401

from benchmarks.Robocasa365.eval.model2robocasa365_interface import PolicyWarper
from benchmarks.Robocasa365.eval.wrappers.multistep_wrapper import MultiStepWrapper
from benchmarks.Robocasa365.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)


@dataclass
class VideoConfig:
    video_dir: Optional[str] = None
    steps_per_render: int = 2
    fps: int = 10
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    env_name: str
    split: str = "target"
    n_episodes: int = 50
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


class SimulationInferenceEnv:
    def __init__(self, model=None):
        self.model = model
        self.env = None

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.model.step(observations)

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        return gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    def run_simulation(self, config: SimulationConfig) -> Tuple[str, List[bool]]:
        start_time = time.time()
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} "
            f"(split={config.split}) with {config.n_envs} environments"
        )
        self.env = self.setup_environment(config)

        completed_episodes = 0
        current_successes = [False] * config.n_envs
        current_lengths = [0] * config.n_envs
        episode_successes = []

        obs, _ = self.env.reset()
        while completed_episodes < config.n_episodes:
            actions = self.get_action(obs)["actions"]
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_lengths[env_idx] += 1
                if terminations[env_idx] or truncations[env_idx]:
                    completed_episodes += 1
                    episode_successes.append(current_successes[env_idx])
                    cumulative_sr = float(np.mean(episode_successes))
                    print(
                        f"EP {len(episode_successes)} success: {current_successes[env_idx]}; "
                        f"length={current_lengths[env_idx]}; cumulative={cumulative_sr:.4f}"
                    )
                    current_successes[env_idx] = False
                    current_lengths[env_idx] = 0
            obs = next_obs

        self.env.reset()
        self.env.close()
        self.env = None
        print(f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds")
        return config.env_name, episode_successes


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    env = gym.make(config.env_name, split=config.split, enable_render=True)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5678
    resize_size: list[int] = dataclasses.field(default_factory=lambda: [224, 224])
    task_set: str = "target50"
    task_list: str = ""              # comma-separated env names, takes precedence over task_set if set
    sort_tasks: bool = True          # sort task names alphabetically before iterating
    split: str = "target"
    n_episodes: int = 50
    n_envs: int = 1
    n_action_steps: int = 16
    max_episode_steps: int = 1440
    video_out_path: str = "results/evaluation/robocasa365"
    seed: int = 7
    pretrained_path: str = ""


def eval_robocasa365(args: Args) -> None:
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    os.makedirs(args.video_out_path, exist_ok=True)
    np.random.seed(args.seed)

    # `--task-list` (free-form env names) takes precedence over `--task-set`
    # (registered preset).  This lets CL workflows evaluate on an arbitrary
    # subset of atomic / composite tasks (e.g. our `robocasa365_cl_atomic10`
    # 10-task stream) without having to register a new preset upstream.
    if args.task_list.strip():
        all_env_names = [name.strip() for name in args.task_list.split(",") if name.strip()]
        task_sets = ["custom_task_list"]
    else:
        if "," in args.task_set:
            task_sets = [task.strip() for task in args.task_set.split(",") if task.strip()]
        else:
            task_sets = [args.task_set]

        all_env_names = []
        for task_set in task_sets:
            if task_set not in TASK_SET_REGISTRY:
                raise ValueError(f"Unknown task_set `{task_set}`. Available keys include: {list(TASK_SET_REGISTRY.keys())[:8]}")
            all_env_names.extend(TASK_SET_REGISTRY[task_set])

    if args.sort_tasks:
        all_env_names = sorted(set(all_env_names))
    else:
        # Preserve user-supplied order, but de-duplicate.
        seen = set()
        all_env_names = [n for n in all_env_names if not (n in seen or seen.add(n))]

    model = PolicyWarper(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
    )

    aggregate_results = {}
    for env_name in all_env_names:
        task_video_dir = os.path.join(args.video_out_path, args.split, env_name)
        os.makedirs(task_video_dir, exist_ok=True)
        stats_path = os.path.join(task_video_dir, "stats.json")
        if os.path.exists(stats_path):
            print(f"Skipping {env_name}: stats.json already exists")
            with open(stats_path) as f:
                aggregate_results[env_name] = json.load(f)
            continue

        horizon = get_task_horizon(env_name)
        if not horizon:
            horizon = args.max_episode_steps

        config = SimulationConfig(
            env_name=f"robocasa/{env_name}",
            split=args.split,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            video=VideoConfig(video_dir=task_video_dir),
            multistep=MultiStepConfig(
                n_action_steps=args.n_action_steps,
                max_episode_steps=horizon,
            ),
        )

        print(f"Running simulation for {env_name}...")
        env_name_out, episode_successes = SimulationInferenceEnv(model=model).run_simulation(config)
        success_rate = float(np.mean(episode_successes))
        aggregate_results[env_name] = {
            "num_episodes": len(episode_successes),
            "success_rate": success_rate,
            "horizon": horizon,
        }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_results[env_name], f, indent=2)
        print(f"Results for {env_name_out}: success_rate={success_rate:.4f}")

    aggregate_path = os.path.join(args.video_out_path, args.split, "aggregate_stats.json")
    os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_set": task_sets,
                "split": args.split,
                "mean_success_rate": float(np.mean([item["success_rate"] for item in aggregate_results.values()])),
                "num_tasks": len(aggregate_results),
                "tasks": aggregate_results,
            },
            f,
            indent=2,
        )
    print(f"Saved aggregate stats to {aggregate_path}")


if __name__ == "__main__":
    tyro.cli(eval_robocasa365)
