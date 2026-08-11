"""Phase 2 (on-policy GRPO): group-relative policy optimization.

GRPO drops the value critic entirely. Per-episode advantage is the
group-normalized return where a "group" = episodes sharing the same
initial state (state_idx). A frozen reference actor regularizes policy
drift through a KL penalty (k3 estimator).

Compared to train_rl_onpolicy.run_rl (PPO):
  - No ActionTokenCritic, no V/GAE, no value loss
  - Reference actor maintained as a deepcopy (no grad)
  - Uses action_token_grpo_loss

Reuses the rest: collector, encoder, actor, eval, ckpt I/O.
"""
import copy
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
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import (
    ActionTokenActor, ActionTokenCritic,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import (
    ActionTokenEncoderDecoder,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import (
    action_token_collect_group, action_token_grpo_loss,
)
from AlphaBrain.training.trainer_utils.local_metrics import append_local_metrics
from AlphaBrain.training.trainer_utils.wandb_integration import configure_wandb_module

configure_wandb_module(wandb)

logger = logging.getLogger(__name__)


def run_rl_grpo(args):
    """GRPO trainer entry point. Mirrors run_rl (PPO) but with group-relative
    advantage, KL-to-ref penalty, and no value function."""
    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    logger.info(f"[rank {rank}/{world_size}] Loading frozen VLA from {args.ckpt_path}")
    frozen_vla = BaseFramework.from_pretrained(args.ckpt_path)
    frozen_vla = frozen_vla.to(torch.bfloat16).to(device).eval()
    for p in frozen_vla.parameters():
        p.requires_grad_(False)

    hidden_dim = frozen_vla.qwen_vl_interface.model.config.hidden_size
    chunk_len = frozen_vla.chunk_len
    action_dim = frozen_vla.config.framework.action_model.action_dim

    _norm = frozen_vla.norm_stats
    action_norm_stats = _norm[next(iter(_norm.keys()))]["action"]

    suite_info = get_suite_info(args.suite)
    n_tasks = suite_info["n_tasks"]
    max_steps = MAX_STEPS[args.suite]

    # Encoder
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
        enc_dec.load_state_dict(torch.load(args.encoder_path, map_location=device))

    # Actor (stochastic Gaussian, fixed_std)
    actor = ActionTokenActor(
        bottleneck_dim=args.bottleneck_dim,
        action_dim=action_dim,
        chunk_len=chunk_len,
        hidden_dim=args.actor_hidden_dim,
        ref_dropout=args.ref_dropout,
    ).to(device)

    # Reference actor — frozen snapshot for KL penalty
    ref_actor = copy.deepcopy(actor).eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    # A dummy critic kept only so collect_group / ckpt-save signatures stay
    # the same as PPO. Its value head is unused by GRPO loss.
    dummy_critic = ActionTokenCritic(
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.critic_hidden_dim,
    ).to(device).eval()
    for p in dummy_critic.parameters():
        p.requires_grad_(False)

    if is_main:
        actor_params = sum(p.numel() for p in actor.parameters())
        enc_params = sum(p.numel() for p in enc_dec.parameters())
        logger.info(f"Frozen VLA: {sum(p.numel() for p in frozen_vla.parameters()) / 1e9:.2f}B × {world_size} GPU")
        logger.info(f"GRPO trainable: encoder={enc_params / 1e6:.2f}M  actor={actor_params / 1e6:.2f}M  (no critic)")
        logger.info(f"Rollout: {world_size} ranks × {args.G} ep/rank = {world_size * args.G} ep/iter; "
                    f"group_size={args.group_size} → ~{args.G // max(args.group_size, 1)} groups/rank")

    # Optimizer — no critic params
    param_groups = [{"params": actor.parameters(), "lr": args.lr_actor}]
    if args.lr_encoder > 0:
        param_groups.append({"params": enc_dec.parameters(), "lr": args.lr_encoder})
    else:
        for p in enc_dec.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=1e-8)

    if args.use_wandb and is_main:
        run_name = args.run_name or f"grpo_{args.suite}_task{args.task_id}"
        wandb.init(project=args.wandb_project, name=run_name,
                   config={**vars(args), "chunk_len": chunk_len,
                           "hidden_dim": hidden_dim, "action_dim": action_dim,
                           "world_size": world_size, "algo": "grpo"})

    video_dir = Path(args.output_dir) / "videos"
    metrics_history = []
    best_sr = 0.0
    best_eval_sr = 0.0
    running_sr = []
    total_env_steps = 0

    for iteration in range(1, args.max_iter + 1):
        if is_main:
            logger.info("=" * 60)
            logger.info(f"[iter {iteration}/{args.max_iter}] collecting "
                        f"{args.G}×{world_size}={args.G * world_size} ep")

        save_video = (args.save_video_interval > 0 and
                      (iteration == 1 or iteration % args.save_video_interval == 0))
        iter_video_dir = (str(video_dir / f"iter_{iteration:05d}")
                         if save_video and is_main else None)

        task_id = args.task_id if args.task_id >= 0 else random.randint(0, n_tasks - 1)
        group_seed = args.seed + iteration * 1000 + rank * 100
        local_episodes = action_token_collect_group(
            frozen_vla=frozen_vla, encoder=enc_dec, actor=actor, critic=dummy_critic,
            suite_name=args.suite, task_id=task_id,
            n_initial_states=50,
            action_norm_stats=action_norm_stats,
            max_steps=max_steps, chunk_len=chunk_len, G=args.G,
            libero_python=os.environ.get("LIBERO_PYTHON"),
            seed=group_seed,
            num_steps_wait=args.num_steps_wait,
            device=str(device), video_dir=iter_video_dir,
            num_envs=args.num_envs,
            group_idx=iteration * world_size + rank,
            group_size=args.group_size,
            reward_coef=args.reward_coef,
        )

        local_rewards = torch.tensor([ep.reward for ep in local_episodes],
                                     device=device, dtype=torch.float32)
        global_rewards = accelerator.gather(local_rewards).cpu().numpy()
        success_rate = float(np.mean(global_rewards > 0.5))
        mean_reward = float(np.mean(global_rewards))
        mean_steps = float(np.mean([ep.finish_step for ep in local_episodes]))
        local_env_steps = torch.tensor(sum(ep.env_steps for ep in local_episodes),
                                       device=device, dtype=torch.long)
        global_env_steps = accelerator.reduce(local_env_steps, reduction="sum").item()
        total_env_steps += int(global_env_steps)
        running_sr.append(success_rate)
        if len(running_sr) > 20:
            running_sr.pop(0)
        running_sr_avg = float(np.mean(running_sr))
        best_sr = max(best_sr, success_rate)

        if is_main:
            logger.info(f"[iter {iteration}] SR={success_rate:.2f} (best={best_sr:.2f}, "
                        f"avg={running_sr_avg:.2f}) reward={mean_reward:.2f} "
                        f"steps={mean_steps:.1f} ({len(global_rewards)} ep)")

        # ── GRPO update ─────────────────────────────────────────
        if is_main:
            logger.info(f"[iter {iteration}] GRPO update ({args.ppo_epochs} epochs, "
                        f"kl_coef={args.grpo_kl_coef})")
        actor.train()
        if args.lr_encoder > 0:
            enc_dec.train()

        epoch_stats = []
        for grpo_epoch in range(args.ppo_epochs):
            optimizer.zero_grad()
            loss, stats = action_token_grpo_loss(
                encoder=enc_dec, actor=actor, ref_actor=ref_actor,
                episodes=local_episodes,
                clip_eps=args.clip_eps, kl_coef=args.grpo_kl_coef,
                device=str(device),
            )
            loss.backward()
            _ddp_active = torch.distributed.is_available() and torch.distributed.is_initialized()
            if _ddp_active:
                for p in actor.parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
                if args.lr_encoder > 0:
                    for p in enc_dec.parameters():
                        if p.grad is not None:
                            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
            if args.max_grad_norm > 0:
                all_params = list(actor.parameters())
                if args.lr_encoder > 0:
                    all_params += list(enc_dec.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
            optimizer.step()
            epoch_stats.append(stats)

        # Update reference actor periodically (snapshot of current actor)
        ref_update = getattr(args, "ref_update_interval", 0)
        if ref_update > 0 and iteration % ref_update == 0:
            if is_main:
                logger.info(f"[iter {iteration}] refreshing reference actor")
            ref_actor.load_state_dict(actor.state_dict())

        # ── Eval ────────────────────────────────────────────────
        eval_sr = None
        eval_result = None
        do_eval = (args.eval_interval > 0 and iteration % args.eval_interval == 0)
        if do_eval:
            if is_main:
                logger.info(f"[iter {iteration}] distributed eval ({args.eval_n_episodes} ep)")
            eval_video_dir = str(video_dir / f"eval_iter_{iteration:05d}") if save_video else None
            eval_result = _eval_distributed(
                accelerator=accelerator, frozen_vla=frozen_vla,
                encoder=enc_dec, actor=actor,
                suite_name=args.suite, task_id=task_id,
                action_norm_stats=action_norm_stats,
                max_steps=max_steps, chunk_len=chunk_len,
                n_episodes=args.eval_n_episodes,
                num_steps_wait=args.num_steps_wait, seed=args.seed,
                device=str(device), video_dir=eval_video_dir,
            )
            if is_main and eval_result:
                eval_sr = eval_result["eval_sr"]
                best_eval_sr = max(best_eval_sr, eval_sr)
                logger.info(f"  [eval] SR={eval_sr:.2%} (best={best_eval_sr:.2%})")

        if iteration % args.log_interval == 0 and is_main:
            avg = lambda k: float(np.mean([s[k] for s in epoch_stats if k in s]))
            entry = {
                "iter": iteration, "total_env_steps": total_env_steps,
                "success_rate": success_rate, "best_success_rate": best_sr,
                "running_avg_sr": running_sr_avg, "mean_reward": mean_reward,
                "loss": avg("loss"), "pg_loss": avg("pg_loss"), "kl": avg("kl"),
                "ratio_mean": avg("ratio_mean"), "clip_frac": avg("clip_frac"),
                "advantage_mean": avg("advantage_mean"),
                "advantage_std": avg("advantage_std"),
                "n_groups_with_signal": avg("n_groups_with_signal"),
                "n_steps": avg("n_steps"),
            }
            if eval_sr is not None:
                entry["eval_sr"] = eval_sr
                entry["best_eval_sr"] = best_eval_sr
            metrics_history.append(entry)
            logger.info(f"  loss={entry['loss']:.4f} pg={entry['pg_loss']:.4f} "
                        f"kl={entry['kl']:.4f} ratio={entry['ratio_mean']:.3f} "
                        f"clip_frac={entry['clip_frac']:.3f} "
                        f"groups_signal={entry['n_groups_with_signal']:.1f}")

            append_local_metrics(
                args.output_dir,
                entry,
                phase="grpo",
                iteration=iteration,
            )

            if args.use_wandb:
                wandb_log = {
                    "rollout/success_rate": success_rate,
                    "rollout/best_success_rate": best_sr,
                    "rollout/running_avg_sr": running_sr_avg,
                    "rollout/mean_reward": mean_reward,
                    "rollout/total_env_steps": total_env_steps,
                    "rollout/iter_env_steps": int(global_env_steps),
                    "train/loss": entry["loss"], "train/pg_loss": entry["pg_loss"],
                    "train/kl": entry["kl"], "train/ratio_mean": entry["ratio_mean"],
                    "train/clip_frac": entry["clip_frac"],
                    "train/advantage_mean": entry["advantage_mean"],
                    "train/advantage_std": entry["advantage_std"],
                    "train/n_groups_with_signal": entry["n_groups_with_signal"],
                    "train/n_steps": entry["n_steps"],
                }
                if eval_sr is not None:
                    wandb_log["eval/success_rate"] = eval_sr
                    wandb_log["eval/best_success_rate"] = best_eval_sr
                wandb.log(wandb_log, step=iteration)

        if iteration % args.save_interval == 0 and is_main:
            save_rlt_checkpoint(enc_dec, actor, dummy_critic,
                                iteration, args.output_dir, phase="grpo")
        accelerator.wait_for_everyone()

    if is_main:
        save_rlt_checkpoint(enc_dec, actor, dummy_critic,
                            args.max_iter, args.output_dir, phase="grpo")
        metrics_path = Path(args.output_dir) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)
        logger.info(f"Done. Metrics -> {metrics_path}")

    if args.use_wandb and is_main:
        wandb.finish()
