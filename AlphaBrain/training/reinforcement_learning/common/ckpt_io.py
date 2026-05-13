"""Checkpoint save helper shared by all RLT_a training phases."""
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_rlt_checkpoint(encoder, actor, critic, iteration, output_dir, phase="rl"):
    ckpt_dir = Path(output_dir) / "checkpoints" / f"{phase}_iter_{iteration:05d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), str(ckpt_dir / "encoder.pt"))
    if actor is not None:
        torch.save(actor.state_dict(), str(ckpt_dir / "actor.pt"))
    if critic is not None:
        torch.save(critic.state_dict(), str(ckpt_dir / "critic.pt"))
    logger.info(f"Saved RLT_a checkpoint -> {ckpt_dir}")
    return str(ckpt_dir)
