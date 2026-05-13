#!/usr/bin/env python3
"""Standalone offline eval for an RLT_a iter checkpoint on LIBERO.

Loads:
  - frozen QwenOFT VLA from --vla_ckpt
  - ActionTokenEncoderDecoder from <action_token_ckpt>/encoder.pt
  - ActionTokenActor          from <action_token_ckpt>/actor.pt

Runs deterministic eval across all (or selected) tasks of the suite, prints
per-task SR, and optionally appends the result to a JSON file.

The eval protocol is shared with training via
`AlphaBrain.training.reinforcement_learning.eval.eval_helpers._eval_deterministic_local`.
"""
from AlphaBrain.training.reinforcement_learning._bootstrap import setup

setup()  # load .env, configure logging — before heavy imports

import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.training.reinforcement_learning.eval.eval_helpers import _eval_deterministic_local
from AlphaBrain.training.reinforcement_learning.envs.libero_env import MAX_STEPS, get_suite_info
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import ActionTokenActor
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import ActionTokenEncoderDecoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vla_ckpt", required=True, help="QwenOFT SFT base checkpoint dir")
    p.add_argument("--action_token_ckpt", required=True,
                   help="RLT_a iter checkpoint dir containing encoder.pt and actor.pt")
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--n_eps_per_task", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bottleneck_dim", type=int, default=256)
    p.add_argument("--encoder_layers", type=int, default=2)
    p.add_argument("--encoder_heads", type=int, default=4)
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
                   help="Number of parallel env threads (matches training eval pattern)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    print(f"Loading frozen VLA from {args.vla_ckpt}")
    frozen_vla = BaseFramework.from_pretrained(args.vla_ckpt)
    frozen_vla = frozen_vla.to(torch.bfloat16).to(device).eval()
    for p in frozen_vla.parameters():
        p.requires_grad_(False)

    hidden_dim = frozen_vla.qwen_vl_interface.model.config.hidden_size
    chunk_len = frozen_vla.chunk_len
    action_dim = frozen_vla.config.framework.action_model.action_dim
    norm_stats = frozen_vla.norm_stats
    unnorm_key = next(iter(norm_stats.keys()))
    action_norm_stats = norm_stats[unnorm_key]["action"]
    print(f"  hidden_dim={hidden_dim} chunk_len={chunk_len} action_dim={action_dim}")

    print(f"Loading encoder from {args.action_token_ckpt}/encoder.pt")
    encoder = ActionTokenEncoderDecoder(
        input_dim=hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        chunk_len=chunk_len,
        num_heads=args.encoder_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.encoder_layers,
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

    # Build (tid, [ep_chunk]) jobs; split episodes into num_workers chunks per task
    # so workers run tasks concurrently AND parallelize within a task.
    jobs = []
    for tid in tids_to_eval:
        ep_indices = list(range(args.n_eps_per_task))
        # Round-robin split into num_workers chunks for even load.
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
                _eval_deterministic_local,
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
