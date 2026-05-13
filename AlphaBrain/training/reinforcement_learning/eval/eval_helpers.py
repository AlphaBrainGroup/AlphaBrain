"""Deterministic eval helpers shared by training loops and standalone eval."""
import logging
import os
from collections import defaultdict

import numpy as np
import torch

from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import ActionTokenActor
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import ActionTokenEncoderDecoder

logger = logging.getLogger(__name__)


@torch.no_grad()
def _eval_deterministic_local(
    frozen_vla,
    encoder: ActionTokenEncoderDecoder,
    actor: ActionTokenActor,
    suite_name: str,
    task_id: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    episode_indices: list,
    num_steps_wait: int,
    seed: int,
    device: str,
    rank: int = 0,
    video_dir=None,
) -> list:
    """Run eval episodes assigned to this rank. Returns list of (ep_idx, state_idx, success)."""
    from AlphaBrain.training.reinforcement_learning.envs.libero_env import LiberoEnv
    from AlphaBrain.training.reinforcement_learning.common.rollout import (
        DUMMY_ACTION,
        _postprocess_action,
        _save_video,
        _unnormalize,
    )

    frozen_vla.eval()
    encoder.eval()
    actor.eval()

    n_eps_total = len(episode_indices)
    # Print a progress line every ~20% of the chunk (min 1 ep).
    report_every = max(1, n_eps_total // 5)

    print(f"  [eval] task {task_id} rank {rank}: start — {n_eps_total} eps",
          flush=True)

    results = []
    n_success = 0
    for local_i, ep_idx in enumerate(episode_indices):
        state_idx = ep_idx % 50
        env = LiberoEnv(libero_python=os.environ.get("LIBERO_PYTHON"))
        try:
            obs = env.reset(
                suite_name=suite_name,
                task_id=task_id,
                initial_state_idx=state_idx,
                seed=seed,  # fixed seed for deterministic eval
            )
            task_desc = env.task_description
            frames = [] if video_dir else None
            env_step = 0
            action_cache = None
            cache_idx = 0
            success = False

            while env_step < max_steps + num_steps_wait:
                if env_step < num_steps_wait:
                    obs, _, done = env.step(DUMMY_ACTION)
                    env_step += 1
                    continue

                if action_cache is None or cache_idx >= chunk_len:
                    images = [[obs["primary_image"], obs["wrist_image"]]]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        action_queries, vla_actions = frozen_vla.get_vla_action(
                            batch_images=images, instructions=[task_desc])
                    rl_token = encoder.encode(action_queries)
                    # Slice VLA actions to match actor's chunk_len (VLA may output longer chunks)
                    if vla_actions.size(1) > chunk_len:
                        vla_actions = vla_actions[:, :chunk_len, :]
                    prop_state = torch.tensor(
                        np.array(obs["state"], dtype=np.float32)
                    ).unsqueeze(0).to(device)
                    action_t, _ = actor(rl_token, vla_actions, prop_state, deterministic=True)
                    action_np = action_t[0].cpu().numpy()
                    action_cache = _unnormalize(action_np, action_norm_stats)
                    cache_idx = 0

                env_action = _postprocess_action(action_cache[cache_idx])
                cache_idx += 1
                obs, reward, done = env.step(env_action)
                env_step += 1
                if frames is not None:
                    frames.append(obs["primary_image"].copy())
                if done:
                    success = bool(reward > 0.5)
                    break

            results.append((ep_idx, state_idx, success))
            if success:
                n_success += 1

            if frames and video_dir:
                os.makedirs(video_dir, exist_ok=True)
                status = "success" if success else "fail"
                vpath = os.path.join(video_dir, f"eval_s{state_idx:02d}_ep{ep_idx:02d}_{status}.mp4")
                _save_video(frames, vpath)

            # Compact progress: every ~20% of chunk, plus final ep.
            done_i = local_i + 1
            if done_i % report_every == 0 or done_i == n_eps_total:
                running_sr = n_success / done_i
                print(f"  [eval] task {task_id} rank {rank}: {done_i}/{n_eps_total}"
                      f"  running SR={running_sr:.2%}  (last ep {ep_idx} "
                      f"{'SUCCESS' if success else 'fail'})",
                      flush=True)
        finally:
            env.close()

    return results


def _eval_distributed(
    accelerator,
    frozen_vla,
    encoder: ActionTokenEncoderDecoder,
    actor: ActionTokenActor,
    suite_name: str,
    task_id: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    n_episodes: int,
    num_steps_wait: int,
    seed: int,
    device: str,
    video_dir=None,
) -> dict:
    """Distributed eval: split episodes across all ranks, gather results.

    E.g. 10 episodes across 6 GPUs → ranks get [2, 2, 2, 2, 1, 1] episodes.
    """
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    # Distribute episodes across ranks
    all_ep_indices = list(range(n_episodes))
    # Round-robin assignment for balance
    my_indices = [i for i in all_ep_indices if i % world_size == rank]

    if is_main:
        logger.info(f"[eval] Distributing {n_episodes} eval episodes across {world_size} GPUs "
                     f"({len(my_indices)} per rank)")

    with torch.no_grad():
        local_results = _eval_deterministic_local(
            frozen_vla=frozen_vla,
            encoder=encoder,
            actor=actor,
            suite_name=suite_name,
            task_id=task_id,
            action_norm_stats=action_norm_stats,
            max_steps=max_steps,
            chunk_len=chunk_len,
            episode_indices=my_indices,
            num_steps_wait=num_steps_wait,
            seed=seed,
            device=device,
            rank=rank,
            video_dir=video_dir,
        )

    # Gather results: encode as tensor (ep_idx, state_idx, success) padded to same length
    max_local = (n_episodes + world_size - 1) // world_size
    result_tensor = torch.zeros(max_local, 3, device=device, dtype=torch.float32)
    for i, (ep_idx, state_idx, success) in enumerate(local_results):
        result_tensor[i] = torch.tensor([ep_idx, state_idx, float(success)])

    # Mark valid entries
    valid_mask = torch.zeros(max_local, device=device)
    valid_mask[:len(local_results)] = 1.0

    # Gather from all ranks
    all_results = accelerator.gather(result_tensor)  # (world_size * max_local, 3)
    all_masks = accelerator.gather(valid_mask)       # (world_size * max_local,)

    eval_result = {}
    if is_main:
        per_state = defaultdict(list)
        all_success = []

        for i in range(all_results.size(0)):
            if all_masks[i] > 0.5:
                ep_idx = int(all_results[i, 0].item())
                state_idx = int(all_results[i, 1].item())
                success = bool(all_results[i, 2].item() > 0.5)
                per_state[state_idx].append(success)
                all_success.append(success)

        overall_sr = np.mean(all_success) if all_success else 0.0
        per_state_sr = {sid: np.mean(v) for sid, v in sorted(per_state.items())}

        eval_result = {
            "eval_sr": overall_sr,
            "per_state": per_state_sr,
            "n_episodes": len(all_success),
            "video_paths": [],
        }

    accelerator.wait_for_everyone()
    return eval_result
