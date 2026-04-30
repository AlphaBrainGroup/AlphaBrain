"""
Continual Learning Trainer for AlphaBrain.

Trains a VLA model sequentially on a stream of tasks, delegating the CL
strategy (Experience Replay / EWC / RETAIN / DWE / …) to a pluggable
``CLAlgorithm`` instance built by
:func:`AlphaBrain.training.continual_learning.algorithms.build_cl_algorithm`.

Design:
- Follows the framework convention of one trainer file per training strategy.
- Reuses existing build_framework, build_dataloader, and TrainerUtils.
- Adds an outer loop over tasks.
- Invokes the CL algorithm via ``observe`` / ``modify_batch`` /
  ``compute_penalty`` / ``on_task_start`` / ``on_task_end`` hooks — so
  adding new methods doesn't require touching this file.

Config:
    Add a ``continual_learning`` section to your YAML.  Two config styles
    are supported; see ``algorithms/__init__.py::build_cl_algorithm`` for
    details.  Replay-style (existing / back-compat) example::

        continual_learning:
          task_sequence: libero_spatial     # CL sequence name
          steps_per_task: 10000             # training steps per task
          save_checkpoint_per_task: true    # save after each task

          replay:
            enabled: true
            method: experience_replay       # currently only ER
            buffer_size_per_task: 500       # samples per past task
            replay_batch_ratio: 0.3         # fraction of each batch from replay
            balanced_sampling: false        # equal per task vs. uniform
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from AlphaBrain.dataloader import build_dataloader
from AlphaBrain.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
from AlphaBrain.training.continual_learning.algorithms import (
    CLContext,
    build_cl_algorithm,
)
from AlphaBrain.training.continual_learning.datasets.task_sequences import (
    CL_TASK_SEQUENCES,
    build_dataset_task_map,
    build_episode_task_map,
    get_task_sequence,
    TaskFilteredDataset,
)
from AlphaBrain.model.framework import build_framework
from AlphaBrain.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from AlphaBrain.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


# ============================================================================
# Data utilities
# ============================================================================

def build_full_dataset(cfg):
    """Build the full (unfiltered) VLA dataset from config."""
    vla_dataset_cfg = cfg.datasets.vla_data
    return get_vla_dataset(data_cfg=vla_dataset_cfg)


def build_task_dataloader(
    full_dataset, task_index, episode_task_map, cfg, *, task_stream_mode="by_task_index"
):
    """Build a DataLoader for a specific task by filtering the full dataset."""
    filtered_dataset = TaskFilteredDataset(
        base_dataset=full_dataset,
        task_indices=[task_index],
        episode_task_map=episode_task_map,
        task_stream_mode=task_stream_mode,
    )
    # num_workers configurable per-yaml so heavy 3-camera embodiments (Robocasa)
    # can fall back to 0 (synchronous main-thread reads) when libav workers
    # accumulate state over long runs and segfault at task transitions.
    num_workers = int(cfg.datasets.vla_data.get("num_workers", 4))
    dataloader = DataLoader(
        filtered_dataset,
        batch_size=cfg.datasets.vla_data.per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
        # persistent_workers requires num_workers > 0 (PyTorch raises otherwise).
        persistent_workers=num_workers > 0,
    )
    return dataloader, filtered_dataset


# ============================================================================
# ContinualVLATrainer
# ============================================================================

class ContinualVLATrainer(TrainerUtils):
    """Sequential task trainer with experience replay support.

    Outer loop: iterate over tasks in the CL sequence.
    Inner loop: standard VLA training on the current task + replay samples.
    """

    def __init__(self, cfg, model, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.cl_cfg = cfg.continual_learning
        self.completed_steps = 0
        self.total_batch_size = (
            cfg.datasets.vla_data.per_device_batch_size
            * accelerator.num_processes
            * getattr(cfg.trainer, "gradient_accumulation_steps", 1)
        )

        # LoRA
        from AlphaBrain.training.trainer_utils.peft import is_lora_enabled
        self.use_lora = is_lora_enabled(cfg)

        # Continual-learning algorithm (ER / MIR / EWC / ...).
        # None means "plain sequential baseline — no CL intervention".
        self.cl_algorithm = build_cl_algorithm(cfg, seed=cfg.get("seed", 42))

    def prepare_training(self):
        """Initialize training state (checkpoints, freezing, distributed setup)."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._init_checkpointing()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if hasattr(self.config.trainer, "freeze_modules")
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        # NOTE: we prepare model and optimizer here, dataloaders are prepared per-task.
        # DeepSpeed requires train_micro_batch_size_per_gpu when no dataloader is passed
        # to accelerator.prepare(), so set it explicitly from config.
        if hasattr(self.accelerator.state, "deepspeed_plugin") and self.accelerator.state.deepspeed_plugin is not None:
            ds_cfg = self.accelerator.state.deepspeed_plugin.deepspeed_config
            micro_bs = self.config.datasets.vla_data.per_device_batch_size
            if ds_cfg.get("train_micro_batch_size_per_gpu") == "auto":
                ds_cfg["train_micro_batch_size_per_gpu"] = micro_bs
            if ds_cfg.get("gradient_accumulation_steps") == "auto":
                ds_cfg["gradient_accumulation_steps"] = getattr(self.config.trainer, "gradient_accumulation_steps", 1)
            if ds_cfg.get("train_batch_size") == "auto":
                grad_acc = ds_cfg.get("gradient_accumulation_steps", 1)
                ds_cfg["train_batch_size"] = micro_bs * self.accelerator.num_processes * grad_acc
        self.model, self.optimizer = self.setup_distributed_training(
            self.accelerator, self.model, self.optimizer
        )

        self._init_wandb()

    def train(self, full_dataset, episode_task_map):
        """Execute the continual learning training loop."""
        seq_cfg = get_task_sequence(self.cl_cfg.task_sequence)
        num_tasks = seq_cfg["num_tasks"]
        task_order = seq_cfg.get("task_order", list(range(num_tasks)))
        task_stream_mode = seq_cfg.get("task_stream_mode", "by_task_index")
        steps_per_task = self.cl_cfg.steps_per_task
        save_per_task = self.cl_cfg.get("save_checkpoint_per_task", True)

        # Determine which tasks to skip based on completed steps
        start_task_idx = self.completed_steps // steps_per_task
        if start_task_idx > 0:
            logger.info(
                f"Resuming from step {self.completed_steps}: "
                f"skipping {start_task_idx} completed tasks"
            )
            # Rebuild CL algorithm state by replaying on_task_end for each
            # completed task (ER re-populates its buffer, EWC re-computes
            # Fisher from snapshots, RETAIN re-applies merges, etc.).
            if self.cl_algorithm is not None:
                for skip_idx in range(start_task_idx):
                    skip_task_id = task_order[skip_idx]
                    _, skip_dataset = build_task_dataloader(
                        full_dataset, skip_task_id, episode_task_map, self.config,
                        task_stream_mode=task_stream_mode,
                    )
                    skip_ctx = CLContext(
                        task_id=skip_task_id,
                        model=self.model,
                        task_dataset=skip_dataset,
                        accelerator=self.accelerator,
                    )
                    self.cl_algorithm.on_task_end(skip_ctx)
                logger.info(
                    f"Rebuilt CL algorithm state for {start_task_idx} completed tasks: "
                    f"{self.cl_algorithm.describe()}"
                )

        self._log_cl_config(num_tasks, steps_per_task)

        # Outer tqdm bar over the CL task sequence (1..num_tasks).
        # disable on non-main ranks to avoid duplicate bars under DeepSpeed.
        task_pbar = tqdm(
            enumerate(task_order),
            total=num_tasks,
            desc="CL tasks",
            disable=not self.accelerator.is_local_main_process,
            initial=start_task_idx,
        )
        for task_idx_in_seq, task_id in task_pbar:
            # Skip already completed tasks
            if task_idx_in_seq < start_task_idx:
                task_pbar.write(
                    f"Skipping Task {task_idx_in_seq + 1}/{num_tasks} "
                    f"(task_index={task_id}) — already completed"
                )
                continue

            task_pbar.set_postfix(task_id=task_id, step=self.completed_steps)
            task_pbar.write(f"{'='*60}")
            task_pbar.write(
                f"Starting Task {task_idx_in_seq + 1}/{num_tasks} "
                f"(task_index={task_id})"
            )
            task_pbar.write(f"{'='*60}")

            # Build per-task dataloader
            task_dataloader, task_dataset = build_task_dataloader(
                full_dataset, task_id, episode_task_map, self.config,
                task_stream_mode=task_stream_mode,
            )

            # Prepare dataloader for distributed training
            self.accelerator.dataloader_config.dispatch_batches = False
            task_dataloader = self.accelerator.prepare(task_dataloader)
            dist.barrier()

            # Pre-task CL hook (DWE expands model, LwF snapshots teacher, …)
            if self.cl_algorithm is not None:
                task_start_ctx = CLContext(
                    task_id=task_id,
                    model=self.model,
                    task_dataset=task_dataset,
                    task_dataloader=task_dataloader,
                    accelerator=self.accelerator,
                )
                self.cl_algorithm.on_task_start(task_start_ctx)

            # Reset LR scheduler for new task
            # Unwrap through AcceleratedOptimizer → DeepSpeedZeroOptimizer → base AdamW
            base_optimizer = self.optimizer
            while hasattr(base_optimizer, "optimizer"):
                base_optimizer = base_optimizer.optimizer
            task_lr_scheduler = get_scheduler(
                name=self.config.trainer.lr_scheduler_type,
                optimizer=base_optimizer,
                num_warmup_steps=self.config.trainer.num_warmup_steps,
                num_training_steps=steps_per_task,
                scheduler_specific_kwargs=self.config.trainer.scheduler_specific_kwargs,
            )

            # Train on current task
            self._train_single_task(
                task_id=task_id,
                task_idx_in_seq=task_idx_in_seq,
                num_tasks=num_tasks,
                task_dataloader=task_dataloader,
                lr_scheduler=task_lr_scheduler,
                steps_per_task=steps_per_task,
            )

            # Post-task CL hook (ER populates, EWC computes Fisher, RETAIN merges, …)
            if self.cl_algorithm is not None:
                logger.info(
                    f"Running {self.cl_algorithm.name}.on_task_end for task {task_id}..."
                )
                task_end_ctx = CLContext(
                    task_id=task_id,
                    model=self.model,
                    task_dataset=task_dataset,
                    task_dataloader=task_dataloader,
                    accelerator=self.accelerator,
                )
                self.cl_algorithm.on_task_end(task_end_ctx)
                logger.info(
                    f"CL state after task {task_id}: {self.cl_algorithm.describe()}"
                )

            # Save checkpoint after task
            if save_per_task:
                self._save_task_checkpoint(task_id, task_idx_in_seq)

            dist.barrier()

        self._finalize_training()

    def _train_single_task(
        self,
        task_id: int,
        task_idx_in_seq: int,
        num_tasks: int,
        task_dataloader: DataLoader,
        lr_scheduler,
        steps_per_task: int,
    ):
        """Train on a single task for the specified number of steps."""
        data_iter = iter(task_dataloader)
        epoch_count = 0
        task_step = 0

        progress_bar = tqdm(
            range(steps_per_task),
            desc=f"Task {task_idx_in_seq + 1}/{num_tasks} (id={task_id})",
            disable=not self.accelerator.is_local_main_process,
        )

        while task_step < steps_per_task:
            # Get current task batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter, epoch_count = TrainerUtils._reset_dataloader(
                    task_dataloader, epoch_count
                )
                batch = next(data_iter)

            # CL per-step hooks: algorithm may inject replay samples, record
            # path-integrals, etc.  Default implementations are no-ops.
            if self.cl_algorithm is not None:
                self.cl_algorithm.observe(batch, task_id)
                batch = self.cl_algorithm.modify_batch(batch, task_id)

            # Training step
            t_start = time.perf_counter()
            step_metrics = self._train_step(batch, lr_scheduler)
            t_end = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                task_step += 1
                self.completed_steps += 1

            # Logging
            step_metrics["task_id"] = task_id
            step_metrics["task_step"] = task_step
            step_metrics["model_time"] = t_end - t_start
            if self.cl_algorithm is not None:
                step_metrics.update(self.cl_algorithm.metrics())
            self._log_metrics(step_metrics)

            # Periodic save within task
            save_interval = self.config.trainer.save_interval
            if self.completed_steps % save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

        progress_bar.close()

    def _train_step(self, batch, lr_scheduler):
        """Execute a single training step.

        The task loss is the model's own `action_loss` plus an optional
        algorithm-provided penalty (EWC / SI regularizer).  Algorithms that
        don't need a penalty return ``None`` from :meth:`compute_penalty`.
        """
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

                penalty_value: float = 0.0
                if self.cl_algorithm is not None:
                    penalty = self.cl_algorithm.compute_penalty(self.model)
                    if penalty is not None:
                        total_loss = total_loss + penalty
                        # Detach for logging — no grad needed on the scalar copy.
                        penalty_value = float(penalty.detach().item())

            self.accelerator.backward(total_loss)

            # After-backward hook: param.grad is populated and reduced;
            # optimizer hasn't consumed it yet.  MIR snapshots the current
            # gradient here to drive its next refresh's virtual step (doing
            # the backward inside MIR directly would break DeepSpeed ZeRO-2's
            # single-reduction-per-micro-batch assumption).
            if self.cl_algorithm is not None:
                self.cl_algorithm.after_backward(self.model)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )

            self.optimizer.step()
            lr_scheduler.step()

        # Split loss components so dashboards / grep can see each part
        # of the sum that drove this step's optimizer update.  For
        # algorithms that don't return a penalty (ER / MIR) the `cl_penalty`
        # column stays at 0.0 and `total_loss` == `action_dit_loss`.
        return {
            "action_dit_loss": action_loss.item(),
            "cl_penalty_loss": penalty_value,
            "total_loss": action_loss.item() + penalty_value,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _init_checkpointing(self):
        """Initialize checkpoint directory and load pretrained if specified."""
        cfg = self.config
        cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
        output_dir = Path(cfg.output_dir)

        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_dir / "checkpoints", exist_ok=True)

        self.checkpoint_dir = str(output_dir / "checkpoints")

        pretrained_checkpoint = getattr(cfg.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(cfg.trainer, "is_resume", False)

        if is_resume:
            resume_path, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_path:
                self.model = self.load_pretrained_backbones(
                    self.model, resume_path, reload_modules=None
                )
                logger.info(f"Resumed from {resume_path}, step {self.completed_steps}")
                return
            logger.warning("No checkpoint found, starting from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(cfg.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(
                self.model, pretrained_checkpoint, reload_modules=reload_modules
            )
            logger.info(f"Loaded pretrained: {pretrained_checkpoint}")

    def _save_lora_checkpoint(self, checkpoint_path: str):
        """Save LoRA adapter + non-VLM weights (delegated to trainer_utils.peft)."""
        from AlphaBrain.training.trainer_utils.peft import save_lora_checkpoint
        save_lora_checkpoint(
            accelerator=self.accelerator,
            model=self.model,
            base_path=checkpoint_path,
            cfg=self.config,
        )

    def _save_checkpoint(self):
        """Save model checkpoint at current step."""
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"steps_{self.completed_steps}"
            )

            if self.use_lora:
                self._save_lora_checkpoint(checkpoint_path)
            else:
                save_format = getattr(self.config.trainer, "save_format", "pt")
                state_dict = self.accelerator.get_state_dict(self.model)
                if save_format == "safetensors":
                    from safetensors.torch import save_file
                    save_file(state_dict, checkpoint_path + "_model.safetensors")
                else:
                    torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.accelerator.wait_for_everyone()

    def _save_task_checkpoint(self, task_id: int, task_idx_in_seq: int):
        """Save a named checkpoint after completing a task."""
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"task_{task_idx_in_seq}_id{task_id}_steps_{self.completed_steps}",
            )

            if self.use_lora:
                self._save_lora_checkpoint(checkpoint_path)
            else:
                save_format = getattr(self.config.trainer, "save_format", "pt")
                state_dict = self.accelerator.get_state_dict(self.model)
                if save_format == "safetensors":
                    from safetensors.torch import save_file
                    save_file(state_dict, checkpoint_path + "_model.safetensors")
                else:
                    torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # Save CL algorithm state (ER buffer metadata, EWC Fisher stats, …)
            if self.cl_algorithm is not None:
                cl_state = self.cl_algorithm.state_dict()
                with open(checkpoint_path + "_cl_state.json", "w") as f:
                    json.dump(cl_state, f, indent=2)

            logger.info(
                f"Task checkpoint saved: {checkpoint_path} "
                f"(task {task_idx_in_seq + 1}, id={task_id})"
            )

        self.accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            trackers = getattr(self.config, "trackers", ["jsonl", "wandb"])
            if "wandb" not in trackers:
                self._use_wandb = False
                logger.info("Wandb disabled (not in trackers list)")
                return
            self._use_wandb = True
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity or None,
                group="vla-continual",
            )

    def _log_metrics(self, metrics):
        log_freq = self.config.trainer.logging_frequency
        if self.completed_steps % log_freq == 0 and dist.get_rank() == 0:
            metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            metrics["global_step"] = self.completed_steps
            if getattr(self, "_use_wandb", False):
                wandb.log(metrics, step=self.completed_steps)
            # Tag every step line with the active CL method so a glance at
            # stdout tells you which algorithm produced the numbers.
            # Tag goes to stdout only — we don't send it to wandb so
            # dashboards stay numeric.
            cl_tag = (
                f"[{self.cl_algorithm.name}] " if self.cl_algorithm is not None else ""
            )
            logger.info(f"{cl_tag}Step {self.completed_steps}: {metrics}")

    def _log_cl_config(self, num_tasks, steps_per_task):
        if not self.accelerator.is_main_process:
            return

        algo_name = self.cl_algorithm.name if self.cl_algorithm is not None else "NONE"
        bar = "═" * 72
        logger.info(bar)
        logger.info(f"  Continual Learning Configuration · CL METHOD = {algo_name}")
        logger.info(bar)
        logger.info(f"  Task sequence       : {self.cl_cfg.task_sequence}")
        logger.info(f"  Number of tasks     : {num_tasks}")
        logger.info(f"  Steps per task      : {steps_per_task}")
        logger.info(f"  Total steps         : {num_tasks * steps_per_task}")
        logger.info(f"  LoRA enabled        : {self.use_lora}")
        if self.use_lora:
            lora_cfg = self.config.get("lora", {})
            logger.info(
                f"  LoRA rank / alpha   : {lora_cfg.get('rank', 32)} / "
                f"{lora_cfg.get('alpha', 16)}"
            )
        if self.cl_algorithm is not None:
            logger.info(f"  CL algorithm        : {algo_name}")
            logger.info(f"  Algorithm params    :")
            for k, v in self.cl_algorithm.describe().items():
                if k == "algorithm":
                    continue
                logger.info(f"    {k:<22s}: {v}")
        else:
            logger.info("  CL algorithm        : (none — plain sequential baseline)")
        logger.info(bar)

    def _finalize_training(self):
        if self.accelerator.is_main_process:
            final_path = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_path, exist_ok=True)

            if self.use_lora:
                self._save_lora_checkpoint(os.path.join(final_path, "final"))
            else:
                save_format = getattr(self.config.trainer, "save_format", "pt")
                state_dict = self.accelerator.get_state_dict(self.model)
                if save_format == "safetensors":
                    from safetensors.torch import save_file
                    save_file(state_dict, os.path.join(final_path, "model.safetensors"))
                else:
                    torch.save(state_dict, os.path.join(final_path, "pytorch_model.pt"))

            logger.info(f"Continual learning complete. Final model saved at {final_path}")
            if getattr(self, "_use_wandb", False):
                wandb.finish()

        self.accelerator.wait_for_everyone()


# ============================================================================
# Main entry point
# ============================================================================

def main(cfg) -> None:
    # Peek at the CL method before wrapping the config so the very first
    # log line tells you which algorithm this run uses.
    _cl_block = cfg.get("continual_learning", {}) if hasattr(cfg, "get") else {}
    _replay = _cl_block.get("replay", None) if hasattr(_cl_block, "get") else None
    _algo = _cl_block.get("algorithm", None) if hasattr(_cl_block, "get") else None
    if _replay is not None and _replay.get("enabled", False):
        _m = _replay.get("method", "experience_replay")
        _cl_method = "ER" if _m == "experience_replay" else str(_m).upper()
    elif _algo is not None and _algo.get("name", None) is not None:
        _cl_method = str(_algo.get("name")).upper()
    else:
        _cl_method = "NONE (plain sequential)"
    logger.info(
        f"==== Continual Learning VLA Training — Warming Up "
        f"(CL method = {_cl_method}) ===="
    )

    cfg = wrap_config(cfg)

    # ── Build model ─────────────────────────────────────────────────────
    fw_name = getattr(cfg.framework, 'name', '<unknown>')
    base_vlm = None
    for vlm_key in ('qwenvl', 'llamavl', 'paligemma'):
        sub = getattr(cfg.framework, vlm_key, None)
        if sub is not None and hasattr(sub, 'base_vlm'):
            base_vlm = sub.base_vlm; break
    logger.info(f"[1/6] Building framework `{fw_name}` (base VLM: {os.path.basename(str(base_vlm))})")
    t0 = time.time()
    vla = build_framework(cfg)
    n_total = sum(p.numel() for p in vla.parameters())
    logger.info(f"      → built in {time.time()-t0:.1f}s · total params = {n_total/1e6:.1f}M")

    # ── LoRA wrapping (before optimizer, after model build) ────────────
    from AlphaBrain.training.trainer_utils.peft import apply_lora, is_lora_enabled
    if is_lora_enabled(cfg):
        logger.info(f"[2/6] Applying LoRA (rank={cfg.lora.get('rank', 32)}, "
                    f"alpha={cfg.lora.get('alpha', 16)})...")
        apply_lora(vla, cfg)
    else:
        logger.info(f"[2/6] LoRA disabled — full-param training mode")

    # ── Build full dataset and episode-task mapping (one-time cost) ────
    logger.info(f"[3/6] Building full dataset (mix={cfg.datasets.vla_data.dataset_mix})...")
    t0 = time.time()
    full_dataset = build_full_dataset(cfg)

    # CL task-partitioning strategy depends on the sequence.
    #   by_task_index: LIBERO-style (one parquet, task_index column).
    #   by_dataset:    Robocasa-style (each sub-dataset is one CL task).
    seq_cfg = get_task_sequence(cfg.continual_learning.task_sequence)
    task_stream_mode = seq_cfg.get("task_stream_mode", "by_task_index")
    if task_stream_mode == "by_dataset":
        episode_task_map = build_dataset_task_map(full_dataset)
    elif task_stream_mode == "by_task_index":
        base_ds = full_dataset.datasets[0] if hasattr(full_dataset, "datasets") else full_dataset
        episode_task_map = build_episode_task_map(base_ds)
    else:
        raise ValueError(
            f"Unknown task_stream_mode={task_stream_mode!r} in CL sequence "
            f"{cfg.continual_learning.task_sequence!r}; expected "
            f"'by_task_index' or 'by_dataset'."
        )
    n_eps = sum(len(v) for v in episode_task_map.values())
    n_tasks_in_data = len(episode_task_map)
    logger.info(
        f"      → dataset built in {time.time()-t0:.1f}s · "
        f"{n_eps} episodes across {n_tasks_in_data} tasks "
        f"(task_stream_mode={task_stream_mode})"
    )

    # ── Build optimizer ─────────────────────────────────────────────────
    logger.info(f"[4/6] Building AdamW optimizer (base lr={cfg.trainer.learning_rate.base})...")
    param_groups = build_param_lr_groups(model=vla, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # ── Build trainer ───────────────────────────────────────────────────
    logger.info(f"[5/6] Building ContinualVLATrainer...")
    trainer = ContinualVLATrainer(
        cfg=cfg,
        model=vla,
        optimizer=optimizer,
        lr_scheduler=None,  # created per-task
        accelerator=accelerator,
    )

    logger.info(f"[6/6] Preparing training (DeepSpeed ZeRO-2, NCCL init)...")
    trainer.prepare_training()
    logger.info("====  Setup complete — entering training loop  ====")
    trainer.train(full_dataset, episode_task_map)

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more YAML config paths merged left-to-right (later files "
            "override earlier ones).  Single-file usage is unchanged.  "
            "Composable usage: --config_yaml cl_base.yaml models/qwengr00t.yaml"
        ),
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml[0])
    for _extra_yaml in args.config_yaml[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(_extra_yaml))
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
