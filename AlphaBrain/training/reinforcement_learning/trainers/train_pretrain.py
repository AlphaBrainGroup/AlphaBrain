"""Phase 1: encoder-decoder pretraining via reconstruction loss."""
import logging
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate.utils import set_seed

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.training.reinforcement_learning.envs.libero_env import MAX_STEPS, get_suite_info
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import ActionTokenEncoderDecoder
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import (
    collect_observations_fast,
    extract_action_queries_from_obs,
)

logger = logging.getLogger(__name__)


def run_pretrain(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen VLA
    logger.info(f"Loading frozen VLA from {args.ckpt_path}")
    frozen_vla = BaseFramework.from_pretrained(args.ckpt_path)
    frozen_vla = frozen_vla.to(torch.bfloat16).to(device).eval()
    for param in frozen_vla.parameters():
        param.requires_grad_(False)

    hidden_dim = frozen_vla.qwen_vl_interface.model.config.hidden_size
    chunk_len = frozen_vla.chunk_len

    # Action norm stats
    _norm_stats = frozen_vla.norm_stats
    _unnorm_key = next(iter(_norm_stats.keys()))
    action_norm_stats = _norm_stats[_unnorm_key]["action"]

    # Suite info
    suite_info = get_suite_info(args.suite)
    n_tasks = suite_info["n_tasks"]
    task_names = suite_info["task_names"]
    max_steps = MAX_STEPS[args.suite]
    logger.info(f"Suite: {args.suite} | hidden_dim={hidden_dim} | chunk_len={chunk_len}")

    # Create encoder-decoder
    enc_dec = ActionTokenEncoderDecoder(
        input_dim=hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        chunk_len=chunk_len,
        num_heads=args.encoder_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.encoder_layers,
    ).to(device)

    n_params = sum(p.numel() for p in enc_dec.parameters())
    logger.info(f"RLT_a Encoder-Decoder: {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(enc_dec.parameters(), lr=args.pretrain_lr)

    # WandB
    if args.use_wandb:
        run_name = args.run_name or f"action_token_pretrain_{args.suite}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Phase 1a: Fast observation collection (no VLA forward, just env resets + random steps)
    if args.all_tasks:
        logger.info(f"Collecting {args.pretrain_n_obs} observations from ALL {n_tasks} tasks...")
        observations = []
        obs_per_task = max(1, args.pretrain_n_obs // n_tasks)
        for tid in range(n_tasks):
            logger.info(f"  Collecting from task {tid}/{n_tasks}: {task_names[tid]}")
            task_obs = collect_observations_fast(
                suite_name=args.suite,
                task_id=tid,
                n_observations=obs_per_task,
                steps_per_env=args.pretrain_steps_per_reset,
                num_envs=args.num_envs,
                n_initial_states=50,
                libero_python=os.environ.get("LIBERO_PYTHON"),
                seed=args.seed + tid * 100,
            )
            observations.extend(task_obs)
        logger.info(f"  Total observations collected: {len(observations)}")
    else:
        task_id = args.task_id if args.task_id >= 0 else 0
        logger.info(f"Collecting {args.pretrain_n_obs} observations via fast random exploration...")
        observations = collect_observations_fast(
            suite_name=args.suite,
            task_id=task_id,
            n_observations=args.pretrain_n_obs,
            steps_per_env=args.pretrain_steps_per_reset,
            num_envs=args.num_envs,
            n_initial_states=50,
            libero_python=os.environ.get("LIBERO_PYTHON"),
            seed=args.seed,
        )

    # Phase 1b: Batch-extract all action_queries via frozen VLA (one-time GPU work)
    logger.info(f"Extracting action_queries from {len(observations)} observations (batch VLA forward)...")
    all_queries = extract_action_queries_from_obs(
        frozen_vla=frozen_vla,
        observations=observations,
        batch_size=args.vla_extract_batch_size,
        device=str(device),
    )  # (N, chunk_len, H) on GPU
    del observations  # free CPU memory
    n_samples = all_queries.size(0)
    logger.info(f"Extracted {n_samples} action_queries tensors, shape={tuple(all_queries.shape)}")

    # Free VLA from GPU — no longer needed for pretraining
    del frozen_vla
    torch.cuda.empty_cache()
    logger.info("Freed frozen VLA from GPU memory")

    # Phase 1c: Train encoder-decoder purely on cached tensors (fast, high GPU util)
    enc_dec.train()
    best_loss = float("inf")
    bs = args.pretrain_batch_size
    global_step = 0

    for epoch in range(args.pretrain_epochs):
        perm = torch.randperm(n_samples)
        epoch_losses = []
        n_batches = (n_samples + bs - 1) // bs

        for b_idx, start in enumerate(range(0, n_samples, bs)):
            idx = perm[start:start + bs]
            batch_aq = all_queries[idx]  # (B, chunk_len, H) already on GPU

            optimizer.zero_grad()
            _, recon_loss = enc_dec(batch_aq)
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_dec.parameters(), 1.0)
            optimizer.step()

            loss_val = recon_loss.item()
            epoch_losses.append(loss_val)
            global_step += 1

            # Step-level wandb logging
            if args.use_wandb and global_step % 10 == 0:
                wandb.log({
                    "pretrain/recon_loss_step": loss_val,
                    "pretrain/lr": optimizer.param_groups[0]["lr"],
                    "pretrain/global_step": global_step,
                }, step=global_step)

            # Console progress every 20% of epoch
            if (b_idx + 1) % max(1, n_batches // 5) == 0:
                running_avg = np.mean(epoch_losses[-max(1, n_batches // 5):])
                logger.info(f"  epoch {epoch + 1}/{args.pretrain_epochs} "
                             f"batch {b_idx + 1}/{n_batches} "
                             f"loss={running_avg:.6f}")

        avg_loss = np.mean(epoch_losses)
        logger.info(f"Pretrain epoch {epoch + 1}/{args.pretrain_epochs}: "
                     f"recon_loss={avg_loss:.6f} ({n_batches} batches x {bs})")

        if args.use_wandb:
            wandb.log({
                "pretrain/recon_loss_epoch": avg_loss,
                "pretrain/best_loss": min(best_loss, avg_loss),
                "pretrain/epoch": epoch + 1,
            }, step=global_step)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = Path(args.output_dir) / "checkpoints" / "pretrain_best"
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save(enc_dec.state_dict(), str(best_dir / "encoder.pt"))
            logger.info(f"  -> New best loss: {best_loss:.6f}, saved checkpoint")

    logger.info(f"Encoder pretraining complete. Best loss: {best_loss:.6f}")
    if args.use_wandb:
        wandb.finish()
