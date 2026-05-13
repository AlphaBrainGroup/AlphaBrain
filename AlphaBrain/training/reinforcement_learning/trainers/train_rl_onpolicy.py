"""Phase 2 (legacy on-policy variant): multi-GPU rollout + PPO update.

NOTE: this is the legacy on-policy training path. The off-policy TD3 variant
(train_rl_offpolicy.run_rl_offpolicy) is the production code path used by every
release run; this file is kept for reference.

TODO: implement proper PPO / GRPO updates here. The current `action_token_ppo_loss` is
a placeholder that mixes value-loss + clipped surrogate; before relying on this
phase for new experiments, port a battle-tested PPO loop (importance sampling,
KL early stop, value clipping, gradient accumulation, group-relative
normalization for GRPO, etc.) from a reference implementation.
"""
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.training.reinforcement_learning.common.ckpt_io import save_rlt_checkpoint
from AlphaBrain.training.reinforcement_learning.eval.eval_helpers import _eval_distributed
from AlphaBrain.training.reinforcement_learning.envs.libero_env import MAX_STEPS, get_suite_info
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import ActionTokenActor, ActionTokenCritic
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import ActionTokenEncoderDecoder
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import action_token_collect_group, action_token_ppo_loss

logger = logging.getLogger(__name__)


def run_rl(args):
    """Phase 2 on-policy: multi-GPU parallel rollout + PPO update.

    Each GPU loads a frozen VLA copy and runs its own env workers to collect
    episodes in parallel. All episodes are gathered, then the tiny RLT_a
    network update happens on every rank (identical, since network is tiny).

    6 GPUs = 6× rollout throughput (the actual bottleneck is CPU env.step).
    """
    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    # Each rank loads its own frozen VLA copy (~12GB per GPU)
    logger.info(f"[rank {rank}/{world_size}] Loading frozen VLA from {args.ckpt_path}")
    frozen_vla = BaseFramework.from_pretrained(args.ckpt_path)
    frozen_vla = frozen_vla.to(torch.bfloat16).to(device).eval()
    for param in frozen_vla.parameters():
        param.requires_grad_(False)

    hidden_dim = frozen_vla.qwen_vl_interface.model.config.hidden_size
    chunk_len = frozen_vla.chunk_len
    action_dim = frozen_vla.config.framework.action_model.action_dim

    _norm_stats = frozen_vla.norm_stats
    _unnorm_key = next(iter(_norm_stats.keys()))
    action_norm_stats = _norm_stats[_unnorm_key]["action"]

    suite_info = get_suite_info(args.suite)
    n_tasks = suite_info["n_tasks"]
    max_steps = MAX_STEPS[args.suite]

    # Create RLT_a modules (tiny, same on all ranks)
    enc_dec = ActionTokenEncoderDecoder(
        input_dim=hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        chunk_len=chunk_len,
        num_heads=args.encoder_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.encoder_layers,
    ).to(device)

    if args.encoder_path:
        logger.info(f"[rank {rank}] Loading pretrained encoder from {args.encoder_path}")
        state = torch.load(args.encoder_path, map_location=device)
        enc_dec.load_state_dict(state)

    actor = ActionTokenActor(
        bottleneck_dim=args.bottleneck_dim,
        action_dim=action_dim,
        chunk_len=chunk_len,
        hidden_dim=args.actor_hidden_dim,
        ref_dropout=args.ref_dropout,
    ).to(device)

    critic = ActionTokenCritic(
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.critic_hidden_dim,
    ).to(device)

    if is_main:
        enc_params = sum(p.numel() for p in enc_dec.parameters())
        actor_params = sum(p.numel() for p in actor.parameters())
        critic_params = sum(p.numel() for p in critic.parameters())
        vla_params = sum(p.numel() for p in frozen_vla.parameters())
        logger.info(f"Frozen VLA: {vla_params / 1e9:.2f}B params × {world_size} GPUs")
        logger.info(f"RLT_a trainable: encoder={enc_params / 1e6:.2f}M, "
                    f"actor={actor_params / 1e6:.2f}M, critic={critic_params / 1e6:.2f}M")
        logger.info(f"Rollout parallelism: {world_size} ranks × {args.num_envs} envs × "
                    f"{args.G} episodes/rank = {world_size * args.G} episodes/iter")

    # Optimizer
    param_groups = [
        {"params": actor.parameters(), "lr": args.lr_actor},
        {"params": critic.parameters(), "lr": args.lr_critic},
    ]
    if args.lr_encoder > 0:
        param_groups.append({"params": enc_dec.parameters(), "lr": args.lr_encoder})
    else:
        for p in enc_dec.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=1e-8)

    # WandB (main rank only)
    if args.use_wandb and is_main:
        run_name = args.run_name or f"action_token_rl_{args.suite}_task{args.task_id}"
        wandb.init(project=args.wandb_project, name=run_name,
                   config={**vars(args), "chunk_len": chunk_len,
                           "hidden_dim": hidden_dim, "action_dim": action_dim,
                           "world_size": world_size})

    video_dir = Path(args.output_dir) / "videos"
    metrics_history = []
    best_sr = 0.0
    best_eval_sr = 0.0
    running_sr = []
    total_env_steps = 0  # cumulative environment steps (sample steps)

    # ── Training loop ──────────────────────────────────────
    for iteration in range(1, args.max_iter + 1):
        if is_main:
            logger.info(f"{'='*60}")
            logger.info(f"[iter {iteration}/{args.max_iter}] Collecting "
                         f"{args.G}×{world_size}={args.G * world_size} episodes across {world_size} GPUs...")

        save_video = (args.save_video_interval > 0 and
                      (iteration == 1 or iteration % args.save_video_interval == 0))
        iter_video_dir = (str(video_dir / f"iter_{iteration:05d}")
                          if save_video and is_main else None)

        task_id = args.task_id if args.task_id >= 0 else random.randint(0, n_tasks - 1)

        # ── Each rank collects G episodes in parallel ────────
        group_seed = args.seed + iteration * 1000 + rank * 100
        local_episodes = action_token_collect_group(
            frozen_vla=frozen_vla,
            encoder=enc_dec,
            actor=actor,
            critic=critic,
            suite_name=args.suite,
            task_id=task_id,
            n_initial_states=50,
            action_norm_stats=action_norm_stats,
            max_steps=max_steps,
            chunk_len=chunk_len,
            G=args.G,
            libero_python=os.environ.get("LIBERO_PYTHON"),
            seed=group_seed,
            num_steps_wait=args.num_steps_wait,
            device=str(device),
            video_dir=iter_video_dir,
            num_envs=args.num_envs,
            group_idx=iteration * world_size + rank,
            group_size=args.group_size,
            reward_coef=args.reward_coef,
        )

        # Gather rewards from all ranks for global stats
        local_rewards = torch.tensor(
            [ep.reward for ep in local_episodes], device=device, dtype=torch.float32)
        global_rewards = accelerator.gather(local_rewards).cpu().numpy()

        success_rate = float(np.mean(global_rewards > 0.5))
        mean_reward = float(np.mean(global_rewards))
        mean_steps = np.mean([ep.finish_step for ep in local_episodes])
        # Accumulate env steps: gather local env_steps across all ranks
        local_env_steps = torch.tensor(
            sum(ep.env_steps for ep in local_episodes), device=device, dtype=torch.long)
        global_env_steps = accelerator.reduce(local_env_steps, reduction="sum").item()
        total_env_steps += int(global_env_steps)
        running_sr.append(success_rate)
        if len(running_sr) > 20:
            running_sr.pop(0)
        running_sr_avg = np.mean(running_sr)
        best_sr = max(best_sr, success_rate)

        if is_main:
            logger.info(f"[iter {iteration}] SR={success_rate:.2f} (best={best_sr:.2f}, "
                         f"running_avg={running_sr_avg:.2f}) reward={mean_reward:.2f} "
                         f"steps={mean_steps:.1f} ({len(global_rewards)} total episodes)")

        # ── PPO update on local episodes (each rank independently) ────
        # Since the network is tiny and identical across ranks, each rank
        # computes gradients on its own local episodes. We average gradients
        # across ranks for consistency.
        if is_main:
            logger.info(f"[iter {iteration}] PPO update ({args.ppo_epochs} epochs)...")
        actor.train()
        critic.train()
        if args.lr_encoder > 0:
            enc_dec.train()

        epoch_stats = []
        for ppo_epoch in range(args.ppo_epochs):
            optimizer.zero_grad()
            loss, stats = action_token_ppo_loss(
                encoder=enc_dec,
                actor=actor,
                critic=critic,
                episodes=local_episodes,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_eps=args.clip_eps,
                vf_coef=args.vf_coef,
                recon_loss_coef=args.recon_loss_coef,
                frozen_vla=frozen_vla,
                device=str(device),
            )
            loss.backward()
            # Average gradients across ranks
            for p in list(actor.parameters()) + list(critic.parameters()):
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
            if args.lr_encoder > 0:
                for p in enc_dec.parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
            if args.max_grad_norm > 0:
                all_params = list(actor.parameters()) + list(critic.parameters())
                if args.lr_encoder > 0:
                    all_params += list(enc_dec.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
            optimizer.step()
            epoch_stats.append(stats)

        # ── Deterministic Eval (distributed across all ranks) ──────
        eval_sr = None
        eval_result = None
        do_eval = (args.eval_interval > 0
                   and iteration % args.eval_interval == 0)
        if do_eval:
            if is_main:
                logger.info(f"[iter {iteration}] Running distributed eval "
                             f"({args.eval_n_episodes} episodes across {world_size} GPUs)...")
            eval_video_dir = str(video_dir / f"eval_iter_{iteration:05d}") if save_video else None
            eval_result = _eval_distributed(
                accelerator=accelerator,
                frozen_vla=frozen_vla,
                encoder=enc_dec,
                actor=actor,
                suite_name=args.suite,
                task_id=task_id,
                action_norm_stats=action_norm_stats,
                max_steps=max_steps,
                chunk_len=chunk_len,
                n_episodes=args.eval_n_episodes,
                num_steps_wait=args.num_steps_wait,
                seed=args.seed,
                device=str(device),
                video_dir=eval_video_dir,
            )
            if is_main and eval_result:
                eval_sr = eval_result["eval_sr"]
                best_eval_sr = max(best_eval_sr, eval_sr)
                logger.info(f"  [eval] SR={eval_sr:.2%} (best_eval={best_eval_sr:.2%})")
                for sid, sr in eval_result["per_state"].items():
                    logger.info(f"    state {sid}: {sr:.2%}")

        # ── Logging (main rank only) ──────────────────────────
        if iteration % args.log_interval == 0 and is_main:
            avg = lambda k: float(np.mean([s[k] for s in epoch_stats if k in s]))
            log_entry = {
                "iter": iteration,
                "total_env_steps": total_env_steps,
                "success_rate": success_rate,
                "best_success_rate": best_sr,
                "running_avg_sr": running_sr_avg,
                "mean_reward": mean_reward,
                "loss": avg("loss"),
                "pg_loss": avg("pg_loss"),
                "vf_loss": avg("vf_loss"),
                "ratio_mean": avg("ratio_mean"),
                "clip_frac": avg("clip_frac"),
                "advantage_mean": avg("advantage_mean"),
                "value_mean": avg("value_mean"),
                "n_steps": avg("n_steps"),
            }
            if eval_sr is not None:
                log_entry["eval_sr"] = eval_sr
                log_entry["best_eval_sr"] = best_eval_sr
            metrics_history.append(log_entry)
            logger.info(f"  loss={log_entry['loss']:.4f} pg={log_entry['pg_loss']:.4f} "
                         f"vf={log_entry['vf_loss']:.4f} ratio={log_entry['ratio_mean']:.3f} "
                         f"clip_frac={log_entry['clip_frac']:.3f} "
                         f"total_env_steps={total_env_steps}")

            if args.use_wandb:
                wandb_log = {
                    "rollout/success_rate": success_rate,
                    "rollout/best_success_rate": best_sr,
                    "rollout/running_avg_sr": running_sr_avg,
                    "rollout/mean_reward": mean_reward,
                    "rollout/total_env_steps": total_env_steps,
                    "rollout/iter_env_steps": int(global_env_steps),
                    "train/loss": log_entry["loss"],
                    "train/pg_loss": log_entry["pg_loss"],
                    "train/vf_loss": log_entry["vf_loss"],
                    "train/ratio_mean": log_entry["ratio_mean"],
                    "train/clip_frac": log_entry["clip_frac"],
                    "train/advantage_mean": log_entry["advantage_mean"],
                    "train/value_mean": log_entry["value_mean"],
                    "train/n_steps": log_entry["n_steps"],
                }
                if eval_sr is not None:
                    wandb_log["eval/success_rate"] = eval_sr
                    wandb_log["eval/best_success_rate"] = best_eval_sr
                    for sid, sr in eval_result["per_state"].items():
                        wandb_log[f"eval/state_{sid:02d}"] = sr
                for ep in sorted(local_episodes, key=lambda e: -e.success):
                    if ep.video_path and os.path.exists(ep.video_path):
                        status = "success" if ep.success else "fail"
                        wandb_log[f"video/{status}"] = wandb.Video(
                            ep.video_path, fps=10, format="mp4")
                        break
                wandb.log(wandb_log, step=iteration)

        # ── Checkpoint (main rank only) ──────────────────────
        if iteration % args.save_interval == 0 and is_main:
            save_rlt_checkpoint(enc_dec, actor, critic,
                                iteration, args.output_dir, phase="rl")

        # Sync all ranks before next iteration
        accelerator.wait_for_everyone()

    # Final save
    if is_main:
        save_rlt_checkpoint(enc_dec, actor, critic,
                            args.max_iter, args.output_dir, phase="rl")
        metrics_path = Path(args.output_dir) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)
        logger.info(f"Done. Metrics -> {metrics_path}")

    if args.use_wandb and is_main:
        wandb.finish()
