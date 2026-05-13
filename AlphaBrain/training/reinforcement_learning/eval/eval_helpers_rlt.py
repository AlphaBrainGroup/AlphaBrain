"""Deterministic eval helper for `--encoder_mode rlt`.

The shared `eval_helpers._eval_deterministic_local` hard-codes the
action-token encoder input (`encoder.encode(action_queries)`), which is
wrong for encoders trained via the rlt path — rollout/training feed
those encoders compacted image hidden states, not action_queries. Using
the action-token path at eval time produces garbage RL tokens and
systematically low eval SR even when rollout SR is high.

This helper mirrors `_eval_deterministic_local` one-for-one but swaps
the RL token construction to match the rlt rollout path in
``action_token_trainer.BatchInferenceServer._loop`` (encoder_mode branch):

    last_hidden, encoder_mask, _, _, vla_actions = \
        get_vla_hidden_states_and_action(vla, ..., image_only=True)
    dense, kp_mask = compact_by_mask(last_hidden, encoder_mask)
    rl_token = encoder.encode(dense.float(), key_padding_mask=kp_mask)

It is drop-in compatible: same signature, same return type, so the
trainer can dispatch on ``args.encoder_mode`` without further edits.
"""
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def _eval_deterministic_local_rlt(
    frozen_vla,
    encoder,   # RLTokenEncoderDecoder
    actor,     # ActionTokenActor
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
    """rlt twin of eval_helpers._eval_deterministic_local.

    Returns: list of (ep_idx, state_idx, success) tuples, one per episode
    in `episode_indices`.
    """
    # Use the socket-IPC env (same as rollout) instead of the legacy
    # pipe-IPC LiberoEnv. The pipe IPC deadlocks on some container
    # configurations (cluster jobs) where stderr buffering or fd
    # inheritance differs from local. Socket IPC has settimeout()
    # protection so a hung worker can't freeze the whole eval thread.
    from AlphaBrain.training.reinforcement_learning.envs.persistent_env_pool import (
        _FastLiberoEnv as LiberoEnv,
    )
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
    report_every = max(1, n_eps_total // 5)

    print(f"  [eval-rlt] task {task_id} rank {rank}: start — {n_eps_total} eps",
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
                seed=seed,
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
                    prop_state = torch.tensor(
                        np.array(obs["state"], dtype=np.float32)
                    ).unsqueeze(0).to(device)

                    from AlphaBrain.training.reinforcement_learning.algos.RLT.pi05_inference import (
                        run_rlt_inference,
                    )
                    rl_token, vla_actions = run_rlt_inference(
                        frozen_vla, encoder, images, [task_desc], prop_state,
                    )

                    if vla_actions.size(1) > chunk_len:
                        vla_actions = vla_actions[:, :chunk_len, :]
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

            done_i = local_i + 1
            if done_i % report_every == 0 or done_i == n_eps_total:
                running_sr = n_success / done_i
                print(f"  [eval-rlt] task {task_id} rank {rank}: {done_i}/{n_eps_total}"
                      f"  running SR={running_sr:.2%}  (last ep {ep_idx} "
                      f"{'SUCCESS' if success else 'fail'})",
                      flush=True)
        finally:
            env.close()

    return results
