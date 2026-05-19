"""Vanilla VLA + PPO trainer — full-parameter fine-tune via PG.

Architecture: VLA's action head IS the policy. Each PPO update epoch
re-forwards the VLA over every transition (in micro-batches) to compute
new_log_prob with gradients. Memory-heavy; relies on gradient
checkpointing on the VLM backbone.
"""
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.training.reinforcement_learning.envs.libero_env import MAX_STEPS, get_suite_info
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO import (
    VLAPolicy, VLAValueHead, vla_ppo_collect, vla_ppo_loss,
)

logger = logging.getLogger(__name__)


def run_rl_vla_ppo(args):
    """Vanilla VLA-PPO entry point. Single-GPU only (no Accelerate)."""
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    if args.train_gpu is not None:
        device = f"cuda:{args.train_gpu}"
    else:
        device = "cuda:0"
    logger.info(f"=== Vanilla VLA + PPO (full finetune) on {device} ===")

    # Load VLA TRAINABLE (full FT)
    logger.info(f"Loading VLA from {args.ckpt_path} (trainable, bf16)")
    vla = BaseFramework.from_pretrained(args.ckpt_path)
    vla = vla.to(torch.bfloat16).to(device).train()
    # Enable gradient checkpointing on the language model
    if hasattr(vla, "qwen_vl_interface") and hasattr(vla.qwen_vl_interface, "model"):
        try:
            vla.qwen_vl_interface.model.gradient_checkpointing_enable()
            logger.info("gradient_checkpointing enabled on qwen_vl_interface.model")
        except Exception as e:
            logger.warning(f"gradient_checkpointing_enable failed: {e}")
    # All VLA params trainable
    n_train = 0
    for p in vla.parameters():
        p.requires_grad_(True)
        n_train += p.numel()
    logger.info(f"VLA trainable params: {n_train / 1e9:.3f}B")

    hidden_dim = vla.qwen_vl_interface.model.config.hidden_size
    chunk_len = vla.chunk_len
    action_dim = vla.config.framework.action_model.action_dim

    _norm = vla.norm_stats
    action_norm_stats = _norm[next(iter(_norm.keys()))]["action"]

    suite_info = get_suite_info(args.suite)
    n_tasks = suite_info["n_tasks"]
    max_steps = MAX_STEPS[args.suite]

    # Policy wrapper + value head
    policy = VLAPolicy(vla, fixed_std=args.fixed_std)
    value_head = VLAValueHead(vla_hidden_dim=hidden_dim,
                              hidden_dim=args.critic_hidden_dim).to(device)
    vh_params = sum(p.numel() for p in value_head.parameters())
    logger.info(f"Value head: {vh_params / 1e6:.2f}M params")

    # Optimizer: VLA full + value head
    optimizer = torch.optim.AdamW(
        [{"params": vla.parameters(), "lr": args.lr_vla},
         {"params": value_head.parameters(), "lr": args.lr_critic}],
        betas=(0.9, 0.95), weight_decay=1e-8,
    )

    if args.use_wandb:
        run_name = args.run_name or f"vla_ppo_qwen_task{args.task_id}"
        wandb.init(project=args.wandb_project, name=run_name,
                   config={**vars(args), "chunk_len": chunk_len,
                           "hidden_dim": hidden_dim, "action_dim": action_dim,
                           "algo": "vla_ppo_full"})

    video_dir = Path(args.output_dir) / "videos"
    metrics_history = []
    best_sr = 0.0
    running_sr = []
    total_env_steps = 0

    for iteration in range(1, args.max_iter + 1):
        logger.info("=" * 60)
        logger.info(f"[iter {iteration}/{args.max_iter}] collecting {args.G} ep on task {args.task_id}")

        task_id = args.task_id if args.task_id >= 0 else random.randint(0, n_tasks - 1)
        group_seed = args.seed + iteration * 1000

        # ── Rollout (no grad, eval-mode within wrapper) ───────────────
        local_episodes = vla_ppo_collect(
            policy=policy, value_head=value_head,
            suite_name=args.suite, task_id=task_id,
            n_initial_states=50,
            action_norm_stats=action_norm_stats,
            max_steps=max_steps, chunk_len=chunk_len, G=args.G,
            libero_python=os.environ.get("LIBERO_PYTHON"),
            seed=group_seed,
            num_steps_wait=args.num_steps_wait,
            device=device, num_envs=args.num_envs,
            group_idx=iteration, group_size=args.group_size,
            reward_coef=args.reward_coef,
        )

        # Stats
        ep_rewards = np.array([ep.reward for ep in local_episodes])
        sr = float(np.mean(ep_rewards > 0.5))
        mean_r = float(np.mean(ep_rewards))
        mean_steps = float(np.mean([ep.finish_step for ep in local_episodes]))
        iter_env_steps = sum(ep.env_steps for ep in local_episodes)
        total_env_steps += iter_env_steps
        running_sr.append(sr)
        if len(running_sr) > 20: running_sr.pop(0)
        best_sr = max(best_sr, sr)

        logger.info(f"[iter {iteration}] SR={sr:.2f} (best={best_sr:.2f}, "
                    f"avg={np.mean(running_sr):.2f}) reward={mean_r:.2f} "
                    f"steps={mean_steps:.1f} env_steps={iter_env_steps}")

        # ── PPO update (VLA in train mode, re-forward every transition) ──
        logger.info(f"[iter {iteration}] PPO update ({args.ppo_epochs} epochs, "
                    f"micro_batch={args.micro_batch})")
        vla.train()
        value_head.train()
        epoch_stats = []
        for ppo_epoch in range(args.ppo_epochs):
            optimizer.zero_grad()
            loss, stats = vla_ppo_loss(
                policy=policy, value_head=value_head,
                episodes=local_episodes,
                gamma=args.gamma, gae_lambda=args.gae_lambda,
                clip_eps=args.clip_eps, vf_coef=args.vf_coef,
                micro_batch=args.micro_batch,
                device=device,
            )
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(vla.parameters()) + list(value_head.parameters()),
                    args.max_grad_norm,
                )
            optimizer.step()
            epoch_stats.append(stats)

        avg = lambda k: float(np.mean([s[k] for s in epoch_stats if k in s]))
        log_entry = {
            "iter": iteration, "total_env_steps": total_env_steps,
            "success_rate": sr, "best_success_rate": best_sr,
            "running_avg_sr": float(np.mean(running_sr)), "mean_reward": mean_r,
            "loss": avg("loss"), "pg_loss": avg("pg_loss"), "vf_loss": avg("vf_loss"),
            "ratio_mean": avg("ratio_mean"), "clip_frac": avg("clip_frac"),
            "advantage_mean": avg("advantage_mean"), "return_mean": avg("return_mean"),
            "n_steps": avg("n_steps"),
        }
        metrics_history.append(log_entry)
        logger.info(f"  loss={log_entry['loss']:.4f} pg={log_entry['pg_loss']:.4f} "
                    f"vf={log_entry['vf_loss']:.4f} ratio={log_entry['ratio_mean']:.3f} "
                    f"clip_frac={log_entry['clip_frac']:.3f} "
                    f"return={log_entry['return_mean']:.3f}")

        if args.use_wandb:
            wandb.log({
                "rollout/success_rate": sr, "rollout/best": best_sr,
                "rollout/mean_reward": mean_r,
                "rollout/total_env_steps": total_env_steps,
                "train/loss": log_entry["loss"], "train/pg_loss": log_entry["pg_loss"],
                "train/vf_loss": log_entry["vf_loss"],
                "train/ratio_mean": log_entry["ratio_mean"],
                "train/clip_frac": log_entry["clip_frac"],
                "train/advantage_mean": log_entry["advantage_mean"],
                "train/return_mean": log_entry["return_mean"],
                "train/n_steps": log_entry["n_steps"],
            }, step=iteration)

        # Checkpoint: save VLA + value head
        if iteration % args.save_interval == 0:
            ckpt_dir = Path(args.output_dir) / "checkpoints" / f"vla_ppo_iter_{iteration:05d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(value_head.state_dict(), ckpt_dir / "value_head.pt")
            # Full VLA save via framework's own save (handles HF + safetensors)
            try:
                vla.save_pretrained(str(ckpt_dir / "vla"))
            except Exception as e:
                logger.warning(f"vla.save_pretrained failed: {e}; falling back to state_dict")
                torch.save(vla.state_dict(), ckpt_dir / "vla_state_dict.pt")
            logger.info(f"Saved ckpt → {ckpt_dir}")

    # Final
    final_dir = Path(args.output_dir) / "checkpoints" / f"vla_ppo_iter_{args.max_iter:05d}_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(value_head.state_dict(), final_dir / "value_head.pt")
    try:
        vla.save_pretrained(str(final_dir / "vla"))
    except Exception:
        torch.save(vla.state_dict(), final_dir / "vla_state_dict.pt")

    metrics_path = Path(args.output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    logger.info(f"Done. Metrics → {metrics_path}")

    if args.use_wandb:
        wandb.finish()
