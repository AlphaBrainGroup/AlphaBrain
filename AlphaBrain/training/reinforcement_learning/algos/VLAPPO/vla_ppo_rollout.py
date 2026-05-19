"""On-policy rollout collector for vanilla VLA-PPO.

Per-step pattern (chunked):
  1. Batched VLA forward over all active envs → action_mean (one VLA call)
  2. Sample mean + Gaussian noise → action; record old_log_prob
  3. Value head forward on action_queries → V(s) for GAE
  4. Execute action chunk in env (chunk_len env.step calls)
  5. Record per-step inputs (images, instruction, prop) so PPO update
     can re-forward VLA on the SAME inputs with grad enabled.

Episodes are grouped by initial state (reuse RLT_a's group_size convention
so GRPO-style baselines could be added later).
"""
from dataclasses import dataclass, field
from typing import List, Optional
import os
import time

import numpy as np
import torch

from AlphaBrain.training.reinforcement_learning.envs.libero_env import LiberoEnv
from AlphaBrain.training.reinforcement_learning.common.rollout import (
    _unnormalize, _postprocess_action, DUMMY_ACTION,
)
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_policy import (
    VLAPolicy, VLAValueHead, _gaussian_log_prob,
)


@dataclass
class VLAPPOStepRecord:
    """One step (one VLA forward = chunk_len env steps)."""
    images: list                    # [primary_img, wrist_img] uint8 numpy
    instruction: str
    prop_state: torch.Tensor        # (prop_dim,) float
    action_taken: torch.Tensor      # (chunk_len, action_dim) float — pre-unnorm
    action_mean: torch.Tensor       # (chunk_len, action_dim) float — VLA output at rollout
    old_log_prob: float
    value: float                    # V(action_queries) at rollout


@dataclass
class VLAPPOEpisode:
    step_records: List[VLAPPOStepRecord] = field(default_factory=list)
    reward: float = 0.0
    task_id: int = 0
    success: bool = False
    finish_step: int = 0             # number of step_records (each = chunk_len env steps)
    env_steps: int = 0               # cumulative env.step calls
    done_cache_idx: int = -1
    state_idx: int = -1


@torch.no_grad()
def vla_ppo_collect(
    *,
    policy: VLAPolicy,
    value_head: VLAValueHead,
    suite_name: str,
    task_id: int,
    n_initial_states: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    G: int = 8,
    libero_python: Optional[str] = None,
    seed: int = 42,
    num_steps_wait: int = 10,
    device: str = "cuda",
    num_envs: int = 4,
    group_idx: int = 0,
    group_size: int = 1,
    reward_coef: float = 5.0,
) -> List[VLAPPOEpisode]:
    """Collect G episodes on a single task with the VLA-policy.

    Mirrors action_token_collect_group's free-run rollout pattern but
    records (images, instruction, prop) per step so the PPO update can
    re-forward the VLA with grad enabled.
    """
    policy.vla.eval()  # no_grad context anyway; ensure dropout off
    value_head.eval()

    # Assign initial states with group_size repetition (same as RLT_a)
    num_unique = max(1, G // group_size)
    _rng = np.random.RandomState(seed + group_idx)
    unique_states = _rng.randint(0, n_initial_states, size=num_unique)
    state_ids = np.repeat(unique_states, group_size)[:G]

    episodes: List[VLAPPOEpisode] = []

    # Run G episodes in waves of num_envs envs.
    n_waves = (G + num_envs - 1) // num_envs
    for wave in range(n_waves):
        ep_start = wave * num_envs
        ep_end = min(ep_start + num_envs, G)
        active = ep_end - ep_start

        envs = [LiberoEnv(libero_python=libero_python) for _ in range(active)]
        obs_list = []
        try:
            task_description = None
            for i in range(active):
                obs = envs[i].reset(
                    suite_name=suite_name, task_id=task_id,
                    initial_state_idx=int(state_ids[ep_start + i]),
                    seed=seed + ep_start + i,
                )
                obs_list.append(obs)
                if task_description is None:
                    task_description = envs[i].task_description

            # Warmup: dummy steps
            for _ in range(num_steps_wait):
                for i in range(active):
                    obs_list[i], _, _ = envs[i].step(DUMMY_ACTION)

            local_eps = [VLAPPOEpisode(task_id=task_id,
                                       state_idx=int(state_ids[ep_start + i]))
                         for i in range(active)]
            alive = [True] * active
            env_steps_done = [0] * active

            max_chunks = max_steps // chunk_len + 1
            for chunk_idx in range(max_chunks):
                if not any(alive):
                    break
                idxs = [i for i in range(active) if alive[i]]
                batch_images = [[obs_list[i]["primary_image"], obs_list[i]["wrist_image"]] for i in idxs]
                batch_instrs = [task_description] * len(idxs)
                batch_props = [np.array(obs_list[i]["state"], dtype=np.float32) for i in idxs]

                # 1) Batched VLA forward (no grad) → action_mean + features
                mean, features = policy.forward_mean_and_features(batch_images, batch_instrs)
                # 2) Sample
                sampled, old_lp = policy.sample(mean)
                # 3) Value
                value = value_head(features)

                mean_cpu = mean.cpu()
                sampled_cpu = sampled.cpu()
                old_lp_cpu = old_lp.cpu()
                value_cpu = value.cpu()

                # Record per-env BEFORE stepping so we keep the obs that
                # produced this action.
                for j, i in enumerate(idxs):
                    rec = VLAPPOStepRecord(
                        images=[obs_list[i]["primary_image"].copy(),
                                obs_list[i]["wrist_image"].copy()],
                        instruction=task_description,
                        prop_state=torch.tensor(np.array(obs_list[i]["state"], dtype=np.float32)),
                        action_taken=sampled_cpu[j].clone(),
                        action_mean=mean_cpu[j].clone(),
                        old_log_prob=float(old_lp_cpu[j].item()),
                        value=float(value_cpu[j].item()),
                    )
                    local_eps[i].step_records.append(rec)

                # 4) Execute chunk in env (per-env loop; chunk_len env.step)
                for j, i in enumerate(idxs):
                    chunk_actions_unnorm = _unnormalize(sampled_cpu[j].numpy(), action_norm_stats)
                    last_obs, last_done, last_reward = None, False, 0.0
                    steps_executed = 0
                    for step_in_chunk in range(chunk_len):
                        env_a = _postprocess_action(chunk_actions_unnorm[step_in_chunk])
                        new_obs, r, done = envs[i].step(env_a)
                        steps_executed += 1
                        env_steps_done[i] += 1
                        last_obs, last_reward, last_done = new_obs, r, done
                        if done or env_steps_done[i] >= max_steps:
                            break
                    obs_list[i] = last_obs
                    local_eps[i].env_steps = env_steps_done[i]
                    if last_done or env_steps_done[i] >= max_steps:
                        alive[i] = False
                        local_eps[i].success = bool(last_done and last_reward > 0.5)
                        local_eps[i].reward = reward_coef if local_eps[i].success else 0.0
                        local_eps[i].finish_step = len(local_eps[i].step_records)
                        local_eps[i].done_cache_idx = steps_executed

            # Make sure finish_step is set on episodes that hit chunk loop end
            for i in range(active):
                if local_eps[i].finish_step == 0:
                    local_eps[i].finish_step = len(local_eps[i].step_records)

            episodes.extend(local_eps)
        finally:
            for e in envs:
                try: e.close()
                except Exception: pass

    return episodes
