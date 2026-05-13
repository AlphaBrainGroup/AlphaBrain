#!/usr/bin/env python3
"""Offline eval for an RL Token (rlt) iter checkpoint on LIBERO.

This is the rlt twin of ``eval_libero.py``. The in-training eval path
(``eval_helpers._eval_deterministic_local``) hard-codes the action-token
encoder input (``encoder.encode(action_queries)``) and therefore produces
meaningless numbers for encoders trained via ``--encoder_mode rlt``,
whose rollout path feeds compacted image hidden states through
``get_vla_hidden_states_and_action(image_only=True) -> compact_by_mask
-> encoder.encode(dense, key_padding_mask=kp_mask)``.

This script reproduces the *rollout* encoder path exactly, so the numbers
it prints correspond to what the actor actually saw during training.

Loads:
  * frozen QwenOFT VLA from ``--vla_ckpt``
  * ``RLTokenEncoderDecoder`` from ``<action_token_ckpt>/encoder.pt``
  * ``ActionTokenActor`` from ``<action_token_ckpt>/actor.pt``
"""
from AlphaBrain.training.reinforcement_learning._bootstrap import setup

setup()

import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.training.reinforcement_learning.envs.libero_env import MAX_STEPS, get_suite_info
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import ActionTokenActor
from AlphaBrain.training.reinforcement_learning.algos.RLT import (
    RLTokenEncoderDecoder,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vla_ckpt", required=True, help="QwenOFT SFT base checkpoint dir")
    p.add_argument("--action_token_ckpt", required=True,
                   help="rlt iter ckpt dir containing encoder.pt and actor.pt")
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--n_eps_per_task", type=int, default=50)
    p.add_argument("--gpu", type=int, default=0)
    # rlt arch defaults — match scripts/run_rl_scripts/run_rlt_rl_task0_release.sh
    p.add_argument("--bottleneck_dim", type=int, default=2048,
                   help="rlt encoder hidden_dim (must equal VLA hidden_size)")
    p.add_argument("--encoder_layers", type=int, default=2)
    p.add_argument("--encoder_heads", type=int, default=8)
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--ref_dropout", type=float, default=0.5)
    p.add_argument("--fixed_std", type=float, default=0.1)
    p.add_argument("--prop_dim", type=int, default=8)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video_dir", default=None,
                   help="If set, save eval rollout videos here")
    p.add_argument("--task_ids", default=None,
                   help="Comma-separated task ids to eval (default: all tasks in suite)")
    p.add_argument("--results_json", default=None,
                   help="If set, append per-task SR to this JSON file")
    p.add_argument("--num_workers", type=int, default=1,
                   help="Parallel env threads per shard (match training eval)")
    return p.parse_args()


@torch.no_grad()
def _eval_one_task_rlt(
    frozen_vla,
    encoder: RLTokenEncoderDecoder,
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
    """Deterministic eval for rlt — mirrors _eval_deterministic_local but
    builds ``rl_token`` via the image-only / compact-by-mask path so the
    encoder sees the same inputs it saw during training rollouts.
    """
    # Use socket-IPC env (matches rollout). The legacy pipe-IPC LiberoEnv
    # deadlocks on stderr buffering in some container configurations.
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


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    print(f"Loading frozen VLA from {args.vla_ckpt}")
    frozen_vla = BaseFramework.from_pretrained(args.vla_ckpt)
    frozen_vla = frozen_vla.to(torch.bfloat16).to(device).eval()
    for p in frozen_vla.parameters():
        p.requires_grad_(False)

    from AlphaBrain.training.reinforcement_learning.algos.RLT.pi05_inference import (
        is_pi05, resolve_vla_metadata,
    )
    hidden_dim, action_norm_stats, chunk_len, action_dim = resolve_vla_metadata(frozen_vla)
    if is_pi05(frozen_vla):
        print("  Pi05 detected: using identity action_norm_stats")
    print(f"  hidden_dim={hidden_dim} chunk_len={chunk_len} action_dim={action_dim}")

    if args.bottleneck_dim != hidden_dim:
        print(f"WARNING: --bottleneck_dim={args.bottleneck_dim} != VLA hidden_dim={hidden_dim}; "
              f"rlt encoder hidden_dim must equal VLA hidden_dim. Overriding.")
        args.bottleneck_dim = hidden_dim

    print(f"Loading encoder from {args.action_token_ckpt}/encoder.pt")
    encoder = RLTokenEncoderDecoder(
        hidden_dim=hidden_dim,
        num_heads=args.encoder_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.encoder_layers,
        max_len=args.max_len,
    ).to(device)
    enc_state = torch.load(os.path.join(args.action_token_ckpt, "encoder.pt"),
                           map_location=device)
    encoder.load_state_dict(enc_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    print(f"Loading actor from {args.action_token_ckpt}/actor.pt")
    actor = ActionTokenActor(
        bottleneck_dim=args.bottleneck_dim,
        action_dim=action_dim,
        chunk_len=chunk_len,
        hidden_dim=args.actor_hidden_dim,
        ref_dropout=args.ref_dropout,
        fixed_std=args.fixed_std,
        prop_dim=args.prop_dim,
    ).to(device)
    actor_state = torch.load(os.path.join(args.action_token_ckpt, "actor.pt"),
                             map_location=device)
    actor.load_state_dict(actor_state)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)

    suite_info = get_suite_info(args.suite,
                                libero_python=os.environ.get("LIBERO_PYTHON"))
    n_tasks = suite_info["n_tasks"]
    task_names = suite_info["task_names"]
    max_steps = MAX_STEPS[args.suite]
    print(f"Suite={args.suite} n_tasks={n_tasks} max_steps={max_steps} "
          f"eps_per_task={args.n_eps_per_task}")

    if args.task_ids is not None:
        tids_to_eval = [int(x) for x in args.task_ids.split(",") if x.strip() != ""]
    else:
        tids_to_eval = list(range(n_tasks))

    jobs = []
    for tid in tids_to_eval:
        ep_indices = list(range(args.n_eps_per_task))
        chunks = [[] for _ in range(args.num_workers)]
        for i, ep in enumerate(ep_indices):
            chunks[i % args.num_workers].append(ep)
        for chunk in chunks:
            if chunk:
                jobs.append((tid, chunk))

    print(f"Running {len(jobs)} chunks across {args.num_workers} workers "
          f"({len(tids_to_eval)} tasks × {args.n_eps_per_task} eps)")

    task_results = defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {}
        for tid, chunk in jobs:
            video_dir_t = (os.path.join(args.video_dir, f"task_{tid:02d}")
                           if args.video_dir else None)
            fut = pool.submit(
                _eval_one_task_rlt,
                frozen_vla=frozen_vla,
                encoder=encoder,
                actor=actor,
                suite_name=args.suite,
                task_id=tid,
                action_norm_stats=action_norm_stats,
                max_steps=max_steps,
                chunk_len=chunk_len,
                episode_indices=chunk,
                num_steps_wait=args.num_steps_wait,
                seed=args.seed,
                device=device,
                rank=tid,
                video_dir=video_dir_t,
            )
            futures[fut] = tid
        for fut in as_completed(futures):
            tid = futures[fut]
            task_results[tid].extend(fut.result())

    per_task_sr = {}
    for tid in tids_to_eval:
        results = task_results[tid]
        n_success = sum(1 for _, _, s in results if s)
        sr = n_success / len(results) if results else 0.0
        per_task_sr[tid] = sr
        print(f"Task {tid} ({task_names[tid][:40]}): {n_success}/{len(results)} = {sr:.2%}")

    overall_sr = sum(per_task_sr.values()) / len(per_task_sr)
    print("\n" + "=" * 60)
    print(f"Overall SR ({args.suite}) on {len(per_task_sr)} tasks: {overall_sr:.2%}")
    for tid, sr in per_task_sr.items():
        print(f"  task_{tid:02d} ({task_names[tid][:40]}): {sr:.2%}")
    print("=" * 60)

    if args.results_json is not None:
        import json
        os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
        payload = {
            "action_token_ckpt": args.action_token_ckpt,
            "vla_ckpt": args.vla_ckpt,
            "suite": args.suite,
            "n_eps_per_task": args.n_eps_per_task,
            "encoder_mode": "rlt",
            "per_task_sr": {int(k): float(v) for k, v in per_task_sr.items()},
            "overall_sr": float(overall_sr),
        }
        existing = []
        if os.path.exists(args.results_json):
            try:
                with open(args.results_json) as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except Exception:
                existing = []
        existing.append(payload)
        with open(args.results_json, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Saved results to {args.results_json}")


if __name__ == "__main__":
    main()
