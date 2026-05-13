"""Phase-1 encoder-decoder pretraining that follows the RL Token reference.

Algorithm 1, lines 1-3 of the reference:

    ϕ, θ_vla = argmin L_ro(ϕ) + α L_vla(θ_vla)

where
  * ``z_i = f_i(s, ℓ; θ_vla)`` are the VLA's final-layer token embeddings,
  * ``z_rl = g_ϕ([z_{1:M}, e_rl])_{M+1}`` is the RL token (Eq. 1),
  * ``L_ro = E_D[ Σ_i ‖ h_ϕ(d_ϕ([z_rl, sg(z_{1:i-1})]))_i − sg(z_i) ‖² ]`` (Eq. 2),
  * ``L_vla`` is the VLA's own imitation loss (L1 regression here),
  * ``D`` is a task-specific demonstration dataset,
  * ``α ≥ 0`` controls whether the VLA is jointly fine-tuned (``α = 0`` →
    VLA stays frozen; only the encoder-decoder trains).

This trainer prefers demonstration data (the reference's choice); if no
demo config is provided it falls back to the random-rollout observation
collector shared with the sibling ``RLT_a`` track, in which case
only ``α = 0`` is meaningful.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import wandb
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.training.reinforcement_learning.envs.libero_env import (
    MAX_STEPS,
    get_suite_info,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT import (
    RLTokenEncoderDecoder,
    get_vla_hidden_states,
    pad_mask_from_attention,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import (
    collect_observations_fast,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data providers
# -----------------------------------------------------------------------------

def _build_demo_loader(demo_config: str, batch_size: int):
    """Build a ``LeRobotMixtureDataset`` DataLoader from a config YAML.

    The config must contain ``datasets.vla_data`` in the same shape the SFT
    trainer consumes. Returns an infinite-cycling iterator over samples
    compatible with ``Qwenvl_OFT.forward`` (dicts with ``image``, ``lang``,
    ``action`` keys).
    """
    from AlphaBrain.dataloader.lerobot_datasets import (
        get_vla_dataset,
        collate_fn,
    )

    cfg = OmegaConf.load(demo_config)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info(f"Loaded demo dataset from {demo_config}: {len(dataset)} samples")
    return loader, len(dataset)


def _rollout_observations(args, n_tasks, task_names):
    """Fallback path: collect random-rollout observations (sibling track)."""
    observations = []
    if args.all_tasks:
        logger.info(
            f"Collecting {args.pretrain_n_obs} observations across {n_tasks} tasks"
        )
        obs_per_task = max(1, args.pretrain_n_obs // n_tasks)
        for tid in range(n_tasks):
            logger.info(f"  task {tid}/{n_tasks}: {task_names[tid]}")
            observations.extend(
                collect_observations_fast(
                    suite_name=args.suite,
                    task_id=tid,
                    n_observations=obs_per_task,
                    steps_per_env=args.pretrain_steps_per_reset,
                    num_envs=args.num_envs,
                    n_initial_states=50,
                    libero_python=os.environ.get("LIBERO_PYTHON"),
                    seed=args.seed + tid * 100,
                )
            )
    else:
        tid = args.task_id if args.task_id >= 0 else 0
        observations = collect_observations_fast(
            suite_name=args.suite,
            task_id=tid,
            n_observations=args.pretrain_n_obs,
            steps_per_env=args.pretrain_steps_per_reset,
            num_envs=args.num_envs,
            n_initial_states=50,
            libero_python=os.environ.get("LIBERO_PYTHON"),
            seed=args.seed,
        )
    return observations


# -----------------------------------------------------------------------------
# Loss terms
# -----------------------------------------------------------------------------

def _forward_recon_loss(
    vla,
    enc_dec: RLTokenEncoderDecoder,
    batch_images,
    instructions,
    image_only: bool,
    drop_action_tokens: bool,
) -> torch.Tensor:
    """Run VLA → gather z_{1:M} → encoder-decoder → L_ro (Eq. 2).

    The encoder-decoder always sees stop-gradient VLA embeddings, per Eq. 2.

    With ``image_only=True`` the attention mask is 1 only at image-token
    positions (Fig. 2 / footnote 1); variable-length image token counts
    per sample are handled by ``key_padding_mask`` throughout, and by
    gathering a dense ``(B, M_max, H)`` tensor indexed by each sample's
    kept positions.
    """
    # Framework dispatch: Qwen path keeps the original behavior; Pi05/PaliGemma
    # routes through the local adapter (no in-stream action tokens, prefix-only
    # representation). See vla_features_pi05.py for caveats.
    if hasattr(vla, "qwen_vl_interface"):
        last_hidden, attention_mask, _action_mask = get_vla_hidden_states(
            vla,
            batch_images=batch_images,
            instructions=instructions,
            image_only=image_only,
            drop_action_tokens=drop_action_tokens,
        )
    else:
        from AlphaBrain.training.reinforcement_learning.algos.RLT.vla_features_pi05 import (
            get_vla_hidden_states_pi05,
        )
        last_hidden, attention_mask, _action_mask = get_vla_hidden_states_pi05(
            vla,
            batch_images=batch_images,
            instructions=instructions,
            image_only=image_only,
            drop_action_tokens=drop_action_tokens,
        )
    # Compact the (B, L, H) sequence to (B, M_max, H) by gathering the kept
    # positions per sample. Different samples can have different kept
    # counts (variable image-token counts per image resolution); we pad to
    # the batch max and use key_padding_mask to hide the pad slots.
    dense_hidden, kp_mask = _compact_by_mask(last_hidden, attention_mask)
    _, recon_loss = enc_dec(dense_hidden.detach().float(), key_padding_mask=kp_mask)
    return recon_loss


def _compact_by_mask(
    last_hidden: torch.Tensor,   # (B, L, H)
    attention_mask: torch.Tensor,  # (B, L) int 0/1
):
    """Gather kept positions per sample into a dense (B, M_max, H) tensor.

    Returns:
        dense: (B, M_max, H) — positions marked 1 in ``attention_mask``,
            packed left-aligned. Padding slots hold zeros.
        kp_mask: (B, M_max) bool — True at pad slots (for transformer's
            ``key_padding_mask``).
    """
    B, L, H = last_hidden.shape
    mask = attention_mask.bool()
    counts = mask.sum(dim=1)               # (B,)
    M_max = int(counts.max().item()) if counts.numel() > 0 else 0
    if M_max == 0:
        # No image tokens at all — shouldn't happen in practice; return an
        # empty (B, 1, H) stub to keep shapes well-defined.
        return (
            torch.zeros(B, 1, H, device=last_hidden.device, dtype=last_hidden.dtype),
            torch.ones(B, 1, device=last_hidden.device, dtype=torch.bool),
        )

    dense = torch.zeros(B, M_max, H, device=last_hidden.device, dtype=last_hidden.dtype)
    kp_mask = torch.ones(B, M_max, device=last_hidden.device, dtype=torch.bool)
    for i in range(B):
        idx = mask[i].nonzero(as_tuple=False).squeeze(-1)  # (M_i,)
        m_i = idx.numel()
        if m_i == 0:
            continue
        dense[i, :m_i] = last_hidden[i].index_select(0, idx)
        kp_mask[i, :m_i] = False
    return dense, kp_mask


def _forward_vla_loss(vla, demo_batch) -> torch.Tensor:
    """Run the VLA's own imitation loss L_vla on a demo batch.

    The batch format is whatever the framework's ``forward`` consumes
    (here: a list of dicts with ``image``, ``lang``, ``action`` keys).
    """
    out = vla(examples=demo_batch)
    return out["action_loss"]


# -----------------------------------------------------------------------------
# Top-level phase
# -----------------------------------------------------------------------------

def run_rlt_pretrain(args):
    """Entry point for ``--phase pretrain_rlt``."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha_vla: float = float(getattr(args, "alpha_vla", 0.0))
    finetune_vla = alpha_vla > 0.0
    demo_config: Optional[str] = getattr(args, "demo_config", None)

    if finetune_vla and demo_config is None:
        raise ValueError(
            "alpha_vla > 0 requires --demo_config (need action labels for L_vla). "
            "Either set alpha_vla=0 or provide a demo dataset YAML."
        )

    # Load VLA
    logger.info(f"Loading VLA from {args.ckpt_path}")
    vla = BaseFramework.from_pretrained(args.ckpt_path)
    vla = vla.to(device)
    if not finetune_vla:
        vla = vla.to(torch.bfloat16)
        vla.eval()
        for p in vla.parameters():
            p.requires_grad_(False)
        logger.info("VLA frozen (alpha_vla = 0)")
    else:
        vla.train()
        logger.info(f"VLA will be jointly fine-tuned with alpha_vla = {alpha_vla}")

    # Hidden-dim source dispatches on framework type (matches the dispatch in
    # _forward_recon_loss above). chunk_len is uniform across frameworks.
    if hasattr(vla, "qwen_vl_interface"):
        hidden_dim = vla.qwen_vl_interface.model.config.hidden_size
    elif hasattr(vla, "vlm_interface") and hasattr(vla.vlm_interface, "hidden_size"):
        # Pi05/PaliGemma exposes hidden_size directly on the interface
        # (custom PaliGemmaVLM, not an HF model with .config.hidden_size).
        hidden_dim = vla.vlm_interface.hidden_size
    else:
        hidden_dim = getattr(vla, "_get_vlm_hidden_size", lambda: None)()
        if hidden_dim is None:
            raise RuntimeError(
                f"Cannot determine VLM hidden_size for {type(vla).__name__}; "
                f"add explicit branch in train_rlt_pretrain.run_rlt_pretrain."
            )
    chunk_len = vla.chunk_len

    # Encoder-decoder at the VLA hidden dim (reference: no extra bottleneck proj)
    enc_dec = RLTokenEncoderDecoder(
        hidden_dim=hidden_dim,
        num_heads=args.encoder_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=getattr(args, "decoder_layers", args.encoder_layers),
        dropout=getattr(args, "dropout", 0.0),
        max_len=getattr(args, "max_len", 4096),
    ).to(device)

    n_params = sum(p.numel() for p in enc_dec.parameters())
    logger.info(
        f"RLT Encoder-Decoder: {n_params / 1e6:.2f}M params, "
        f"hidden_dim={hidden_dim}, chunk_len={chunk_len}"
    )

    # Optimizer(s): one param group for enc-dec, one for VLA if jointly fine-tuning
    param_groups = [
        {"params": enc_dec.parameters(), "lr": args.pretrain_lr}
    ]
    if finetune_vla:
        param_groups.append(
            {"params": [p for p in vla.parameters() if p.requires_grad is None or p.requires_grad],
             "lr": getattr(args, "lr_vla", 5e-6)}
        )
        # Ensure VLA params require grad
        for p in vla.parameters():
            p.requires_grad_(True)
    optimizer = torch.optim.AdamW(param_groups)

    # WandB
    if args.use_wandb:
        run_name = args.run_name or f"rlt_pretrain_{args.suite}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Suite metadata (used only by rollout fallback)
    suite_info = get_suite_info(args.suite)
    n_tasks = suite_info["n_tasks"]
    task_names = suite_info["task_names"]
    _ = MAX_STEPS[args.suite]  # sanity

    max_steps = int(getattr(args, "pretrain_max_steps", 0) or 0)
    if max_steps > 0:
        logger.info(
            f"Step budget: pretrain_max_steps={max_steps} (epochs cap at {args.pretrain_epochs})"
        )

    # -------------------------------------------------------------------------
    # Data path A: demonstration dataset (paper-faithful, enables L_vla)
    # -------------------------------------------------------------------------
    if demo_config is not None:
        logger.info(f"Pretraining on demonstrations from {demo_config}")
        loader, n_samples = _build_demo_loader(
            demo_config=demo_config, batch_size=args.pretrain_batch_size
        )

        best_loss = float("inf")
        global_step = 0
        stop = False
        enc_dec.train()

        for epoch in range(args.pretrain_epochs):
            if stop:
                break
            epoch_ro = []
            epoch_vla = []

            for b_idx, batch in enumerate(loader):
                # batch is a list of sample dicts; extract (images, instr, action)
                batch_images = [s["image"] for s in batch]
                instructions = [s["lang"] for s in batch]

                optimizer.zero_grad(set_to_none=True)

                # L_ro — encoder-decoder always sees sg(z)
                recon_loss = _forward_recon_loss(
                    vla=vla,
                    enc_dec=enc_dec,
                    batch_images=batch_images,
                    instructions=instructions,
                    image_only=getattr(args, "image_only", True),
                    drop_action_tokens=getattr(args, "drop_action_tokens", True),
                )
                loss = recon_loss

                vla_loss_val = 0.0
                if finetune_vla:
                    vla_loss = _forward_vla_loss(vla, batch)
                    loss = recon_loss + alpha_vla * vla_loss
                    vla_loss_val = vla_loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc_dec.parameters(), 1.0)
                if finetune_vla:
                    torch.nn.utils.clip_grad_norm_(vla.parameters(), 1.0)
                optimizer.step()

                epoch_ro.append(recon_loss.item())
                if finetune_vla:
                    epoch_vla.append(vla_loss_val)
                global_step += 1

                if args.use_wandb and global_step % 10 == 0:
                    wandb.log(
                        {
                            "pretrain/recon_loss_step": recon_loss.item(),
                            "pretrain/vla_loss_step": vla_loss_val,
                            "pretrain/total_loss_step": loss.item(),
                            "pretrain/lr": optimizer.param_groups[0]["lr"],
                            "pretrain/global_step": global_step,
                        },
                        step=global_step,
                    )

                if (b_idx + 1) % max(1, len(loader) // 10) == 0:
                    logger.info(
                        f"  epoch {epoch+1}/{args.pretrain_epochs} "
                        f"step {global_step}"
                        + (f"/{max_steps}" if max_steps > 0 else "")
                        + f" batch {b_idx+1}/{len(loader)} "
                        f"L_ro={np.mean(epoch_ro):.6f}"
                        + (f" L_vla={np.mean(epoch_vla):.6f}" if finetune_vla else "")
                    )

                if max_steps > 0 and global_step >= max_steps:
                    logger.info(
                        f"Hit pretrain_max_steps={max_steps} — stopping inside epoch "
                        f"{epoch+1} at batch {b_idx+1}/{len(loader)}"
                    )
                    stop = True
                    break

            avg_ro = float(np.mean(epoch_ro)) if epoch_ro else float("inf")
            avg_vla = float(np.mean(epoch_vla)) if epoch_vla else 0.0
            logger.info(
                f"epoch {epoch+1}: L_ro={avg_ro:.6f}"
                + (f", L_vla={avg_vla:.6f}" if finetune_vla else "")
            )
            if args.use_wandb:
                wandb.log(
                    {
                        "pretrain/recon_loss_epoch": avg_ro,
                        "pretrain/vla_loss_epoch": avg_vla,
                        "pretrain/epoch": epoch + 1,
                    },
                    step=global_step,
                )

            if avg_ro < best_loss:
                best_loss = avg_ro
                _save_checkpoint(args, enc_dec, vla if finetune_vla else None, tag="pretrain_best")
                logger.info(f"  -> new best L_ro={best_loss:.6f}; checkpoint saved")

        logger.info(f"Pretraining done. Best L_ro={best_loss:.6f}")
        if args.use_wandb:
            wandb.finish()
        return

    # -------------------------------------------------------------------------
    # Data path B: random-rollout observations (fallback, L_vla not supported)
    # -------------------------------------------------------------------------
    logger.warning(
        "No --demo_config given: falling back to random-rollout observations. "
        "This deviates from the reference's demonstration-driven pretraining."
    )
    observations = _rollout_observations(args, n_tasks, task_names)
    logger.info(f"Collected {len(observations)} observations")

    best_loss = float("inf")
    global_step = 0
    stop = False
    enc_dec.train()
    bs = args.pretrain_batch_size
    n = len(observations)

    for epoch in range(args.pretrain_epochs):
        if stop:
            break
        perm = np.random.permutation(n)
        epoch_losses = []
        n_batches = (n + bs - 1) // bs

        for b_idx, start in enumerate(range(0, n, bs)):
            idx = perm[start:start + bs]
            batch_images = [observations[i][0] for i in idx]
            instructions = [observations[i][1] for i in idx]

            optimizer.zero_grad(set_to_none=True)
            recon_loss = _forward_recon_loss(
                vla=vla,
                enc_dec=enc_dec,
                batch_images=batch_images,
                instructions=instructions,
                image_only=getattr(args, "image_only", True),
                drop_action_tokens=getattr(args, "drop_action_tokens", True),
            )
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_dec.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(recon_loss.item())
            global_step += 1
            if args.use_wandb and global_step % 10 == 0:
                wandb.log(
                    {
                        "pretrain/recon_loss_step": recon_loss.item(),
                        "pretrain/lr": optimizer.param_groups[0]["lr"],
                        "pretrain/global_step": global_step,
                    },
                    step=global_step,
                )
            if (b_idx + 1) % max(1, n_batches // 5) == 0:
                logger.info(
                    f"  epoch {epoch+1}/{args.pretrain_epochs} "
                    f"step {global_step}"
                    + (f"/{max_steps}" if max_steps > 0 else "")
                    + f" batch {b_idx+1}/{n_batches} "
                    f"L_ro={np.mean(epoch_losses):.6f}"
                )

            if max_steps > 0 and global_step >= max_steps:
                logger.info(
                    f"Hit pretrain_max_steps={max_steps} — stopping inside epoch "
                    f"{epoch+1} at batch {b_idx+1}/{n_batches}"
                )
                stop = True
                break

        avg = float(np.mean(epoch_losses))
        logger.info(f"epoch {epoch+1}: L_ro={avg:.6f}")
        if args.use_wandb:
            wandb.log(
                {"pretrain/recon_loss_epoch": avg, "pretrain/epoch": epoch + 1},
                step=global_step,
            )

        if avg < best_loss:
            best_loss = avg
            _save_checkpoint(args, enc_dec, None, tag="pretrain_best")
            logger.info(f"  -> new best L_ro={best_loss:.6f}; checkpoint saved")

    logger.info(f"Pretraining done. Best L_ro={best_loss:.6f}")
    if args.use_wandb:
        wandb.finish()


def _save_checkpoint(args, enc_dec, vla, tag: str):
    out = Path(args.output_dir) / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)
    torch.save(enc_dec.state_dict(), str(out / "encoder.pt"))
    if vla is not None:
        # Joint fine-tune: snapshot the VLA as well so downstream RL uses it
        torch.save(vla.state_dict(), str(out / "vla_finetuned.pt"))
