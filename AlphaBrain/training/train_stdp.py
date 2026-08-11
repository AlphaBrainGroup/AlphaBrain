"""
STDP Fine-tuning Training Script for NeuroVLA.

This script loads a pretrained NeuroVLA checkpoint and fine-tunes the SNN action
head using Reward-Modulated STDP (R-STDP), optionally blended with standard
backpropagation gradients.

Modes:
  - hybrid:     Δw = α·Δw_backprop + β·Δw_rstdp  (default)
  - pure_stdp:  Δw = Δw_rstdp only (no backprop for SNN weights)

Usage:
  accelerate launch AlphaBrain/training/train_stdp.py \
      --config_yaml configs/finetune_config.yaml \
      --mode neuro_vla_stdp
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
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
from tqdm import tqdm
from transformers import get_scheduler

from AlphaBrain.training.trainer_utils.trainer_tools import (
    normalize_dotlist_args,
    TrainerUtils,
    build_param_lr_groups,
)
from AlphaBrain.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig
from AlphaBrain.training.trainer_utils.finetune_config import build_config_from_finetune
from AlphaBrain.training.trainer_utils.wandb_integration import configure_wandb_module
from AlphaBrain.model.framework import build_framework
from AlphaBrain.dataloader import build_dataloader

# STDP modules
from AlphaBrain.model.modules.action_model.stdp import (
    SpikeMonitor,
    STDPLearner,
    RSTDPOptimizer,
)

configure_wandb_module(wandb)

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)


def setup_file_logging(output_dir: str, rank: int = 0):
    if rank != 0:
        return None
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_stdp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    return log_file


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.output_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        log_file = setup_file_logging(str(output_dir), rank=0)
        if log_file:
            logger.info(f"STDP training logs: {log_file}")
    return output_dir


class STDPTrainer(TrainerUtils):
    """
    Trainer for R-STDP fine-tuning of NeuroVLA.

    Extends the standard training loop with:
    1. SpikeMonitor to record spike timing from LIF layers
    2. STDPLearner to compute STDP weight updates
    3. RSTDPOptimizer to blend backprop and STDP updates
    """

    def __init__(self, cfg, model, dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

        # STDP configuration
        stdp_cfg = cfg.stdp if hasattr(cfg, "stdp") else OmegaConf.create({})
        self.stdp_enabled = getattr(stdp_cfg, "enabled", True)
        self.stdp_mode = getattr(stdp_cfg, "mode", "hybrid")
        self.alpha = getattr(stdp_cfg, "alpha", 0.7)
        self.beta = getattr(stdp_cfg, "beta", 0.3)

        # STDP components (initialized in prepare_training)
        self.spike_monitor = None
        self.stdp_learner = None
        self.rstdp_optimizer = None

        # EMA reward tracker for smoother R-STDP signal
        self._ema_loss: float = None
        self._ema_decay: float = 0.95

    def _calculate_total_batch_size(self):
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Load pretrained checkpoint
        self._init_checkpointing()

        # Adjust LR scheduler for resume
        self._adjust_lr_scheduler_for_resume()

        # Freeze VLM + QFormer (only train SNN action head + edit model)
        freeze_modules = getattr(self.config.trainer, "freeze_modules", "")
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        # Initialize STDP components on the SNN action model
        self._init_stdp()

        # Distributed training setup
        self.model, self.optimizer, self.dataloader = self.setup_distributed_training(
            self.accelerator, self.model, self.optimizer, self.dataloader
        )

        self._init_wandb()

    def _init_stdp(self):
        """Initialize STDP monitoring and learning components."""
        if not self.stdp_enabled:
            logger.info("STDP is disabled, using standard backprop only.")
            return

        stdp_cfg = self.config.stdp if hasattr(self.config, "stdp") else OmegaConf.create({})

        # Find the SNN action model
        snn_model = self.model.action_model.model  # MLPResNet
        logger.info(f"Attaching SpikeMonitor to SNN action model: {type(snn_model).__name__}")

        # Create SpikeMonitor
        self.spike_monitor = SpikeMonitor(snn_model)
        self.spike_monitor.enable()
        logger.info(f"SpikeMonitor enabled. Monitoring {len(self.spike_monitor.layer_names)} LIF layers: "
                     f"{self.spike_monitor.layer_names}")

        # Create STDPLearner
        self.stdp_learner = STDPLearner(
            A_plus=getattr(stdp_cfg, "A_plus", 0.01),
            A_minus=getattr(stdp_cfg, "A_minus", 0.012),
            tau_plus=getattr(stdp_cfg, "tau_plus", 20.0),
            tau_minus=getattr(stdp_cfg, "tau_minus", 20.0),
            trace_decay=getattr(stdp_cfg, "trace_decay", 0.95),
            weight_clip=getattr(stdp_cfg, "weight_clip", 1.0),
            binarize_pre=getattr(stdp_cfg, "binarize_pre", True),
        )

        # Create RSTDPOptimizer wrapping the base optimizer
        self.rstdp_optimizer = RSTDPOptimizer(
            snn_model=snn_model,
            base_optimizer=self.optimizer,
            spike_monitor=self.spike_monitor,
            stdp_learner=self.stdp_learner,
            alpha=self.alpha,
            beta=self.beta,
            mode=self.stdp_mode,
            reward_baseline_momentum=getattr(stdp_cfg, "reward_baseline_momentum", 0.99),
            stdp_lr=getattr(stdp_cfg, "stdp_lr", 1e-4),
            max_update_norm=getattr(stdp_cfg, "max_update_norm", 0.1),
            warmup_steps=getattr(stdp_cfg, "warmup_steps", 500),
            align_with_grad=getattr(stdp_cfg, "align_with_grad", True),
        )

        logger.info(f"R-STDP initialized: mode={self.stdp_mode}, alpha={self.alpha}, beta={self.beta}")
        logger.info(f"STDP params: A+={self.stdp_learner.A_plus}, A-={self.stdp_learner.A_minus}, "
                     f"tau+={self.stdp_learner.tau_plus}, tau-={self.stdp_learner.tau_minus}")

    def _adjust_lr_scheduler_for_resume(self):
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(f"LR scheduler at step {self.completed_steps}, LR: {self.lr_scheduler.get_last_lr()}")

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            if hasattr(self.config, "environment") and self.config.environment is not None:
                wandb_project = self.config.environment.wandb_project
                wandb_entity = self.config.environment.wandb_entity
            else:
                wandb_project = getattr(self.config, "wandb_project", "vla-engine-stdp")
                wandb_entity = getattr(self.config, "wandb_entity", "")
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=wandb_project,
                entity=wandb_entity or None,
                group="vla-stdp-train",
            )

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        explicit_resume_checkpoint = getattr(self.config.trainer, "resume_checkpoint", None)
        self.resume_from_checkpoint = explicit_resume_checkpoint or pretrained_checkpoint

        if is_resume:
            if explicit_resume_checkpoint:
                resume_from_checkpoint = os.path.abspath(os.path.expanduser(str(explicit_resume_checkpoint)))
                if not os.path.exists(resume_from_checkpoint):
                    raise RuntimeError(f"Explicit resume checkpoint does not exist: {resume_from_checkpoint}")
                match = re.search(r"steps[_-](\d+)", os.path.basename(resume_from_checkpoint))
                self.completed_steps = int(match.group(1)) if match else 0
            else:
                resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(f"Resuming STDP training from: {self.resume_from_checkpoint}, steps: {self.completed_steps}")
                return
            else:
                logger.warning(f"No checkpoint found in {self.checkpoint_dir}. Starting from pretrained.")
                self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}")
        else:
            logger.info("No pretrained checkpoint provided. Starting from scratch.")
            self.completed_steps = 0

    def _create_data_iterators(self):
        self.data_iter = iter(self.dataloader)

    def _get_next_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            if not hasattr(self, "epoch_count"):
                self.epoch_count = 0
            self.data_iter, self.epoch_count = TrainerUtils._reset_dataloader(
                self.dataloader, self.epoch_count
            )
            batch = next(self.data_iter)
        return batch

    def _train_step(self, batch):
        """Execute one training step with R-STDP."""
        with self.accelerator.accumulate(self.model):
            # Reset spike monitor for this step
            if self.spike_monitor is not None:
                self.spike_monitor.reset()
                self.stdp_learner.reset_traces()

            # Zero gradients
            if self.rstdp_optimizer is not None:
                self.rstdp_optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()

            # Forward pass (spikes are recorded by monitor)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            # Backward pass (compute gradients)
            self.accelerator.backward(total_loss)

            # Gradient clipping
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )

            # Optimizer step: hybrid backprop + STDP
            if self.rstdp_optimizer is not None:
                loss_val = action_loss.item()
                # Use EMA-smoothed loss for more stable reward signal
                if self._ema_loss is None:
                    self._ema_loss = loss_val
                else:
                    self._ema_loss = self._ema_decay * self._ema_loss + (1.0 - self._ema_decay) * loss_val
                reward = -self._ema_loss
                self.rstdp_optimizer.step(reward=reward)
            else:
                self.optimizer.step()

            self.lr_scheduler.step()

        # Collect metrics
        metrics = {"action_dit_loss": action_loss.item()}

        # Add STDP-specific metrics
        if self.rstdp_optimizer is not None:
            stdp_stats = self.rstdp_optimizer.get_stats()
            metrics.update(stdp_stats)

        if self.spike_monitor is not None:
            spike_rates = self.spike_monitor.get_spike_rates()
            for name, rate in spike_rates.items():
                safe_name = name.replace(".", "_")
                metrics[f"spike_rate/{safe_name}"] = rate

        return metrics

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                metrics["epoch"] = round(self.completed_steps / max(len(self.dataloader), 1), 2)
                metrics["step"] = self.completed_steps

                wandb.log(metrics, step=self.completed_steps)

                metrics_file = os.path.join(self.config.output_dir, "metrics.jsonl")
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                loss_val = metrics.get("action_dit_loss", float("nan"))
                lr_val = metrics.get("learning_rate", float("nan"))
                reward_val = metrics.get("stdp/reward_normalized", 0.0)
                stdp_norm = metrics.get("stdp/update_norm", 0.0)
                eff_beta = metrics.get("stdp/effective_beta", 0.0)
                logger.info(
                    f"step {self.completed_steps:>6d}  "
                    f"loss={loss_val:.5f}  lr={lr_val:.2e}  "
                    f"reward={reward_val:.4f}  stdp_norm={stdp_norm:.6f}  "
                    f"beta={eff_beta:.4f}"
                )

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        examples = self._get_next_batch()
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        states = [ex["state"] for ex in examples] if "state" in examples[0] else None

        output_dict = self.model.predict_action(
            batch_images=batch_images, instructions=instructions, states=states,
            use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_elements = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_elements

        del examples
        dist.barrier()
        return step_metrics

    def _save_checkpoint(self):
        if self.accelerator.is_main_process:
            import shutil
            save_format = getattr(self.config.trainer, "save_format", "safetensors")
            checkpoint_dir_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            os.makedirs(checkpoint_dir_path, exist_ok=True)

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file
                save_file(state_dict, os.path.join(checkpoint_dir_path, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(checkpoint_dir_path, "pytorch_model.pt"))

            # Save config
            if isinstance(self.config, AccessTrackedConfig):
                self.config.save_accessed_config(
                    Path(checkpoint_dir_path) / "framework_config.yaml",
                    use_original_values=False
                )
            else:
                OmegaConf.save(self.config, os.path.join(checkpoint_dir_path, "framework_config.yaml"))

            # Save dataset statistics
            dataset_stats_src = os.path.join(self.config.output_dir, "dataset_statistics.json")
            if os.path.exists(dataset_stats_src):
                shutil.copy2(dataset_stats_src, os.path.join(checkpoint_dir_path, "dataset_statistics.json"))

            # Save Qwen config + processor
            qwen_pretrained_dir = os.path.join(checkpoint_dir_path, "qwen_pretrained")
            os.makedirs(qwen_pretrained_dir, exist_ok=True)
            if hasattr(self.model, "qwen_vl_interface"):
                self.model.qwen_vl_interface.model.config.save_pretrained(qwen_pretrained_dir)
                self.model.qwen_vl_interface.processor.save_pretrained(qwen_pretrained_dir)

            # Save STDP optimizer state
            if self.rstdp_optimizer is not None:
                torch.save(
                    self.rstdp_optimizer.state_dict(),
                    os.path.join(checkpoint_dir_path, "rstdp_optimizer.pt")
                )

            summary_data = {"steps": self.completed_steps, "training_mode": "stdp"}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")

            self.accelerator.print(f"STDP checkpoint saved at {checkpoint_dir_path}")

        self.accelerator.wait_for_everyone()

    def train(self):
        self._log_training_config()
        self._create_data_iterators()

        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix({
                    "loss": f"{step_metrics.get('action_dit_loss', 0):.4f}",
                    "stdp": f"{step_metrics.get('stdp/update_norm', 0):.4f}",
                    "data": f"{t_end_data - t_start_data:.3f}s",
                    "fwd": f"{t_end_model - t_start_model:.3f}s",
                })

            # Evaluate
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            # Log
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            # Save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _log_training_config(self):
        if self.accelerator.is_main_process:
            sep = "=" * 56
            logger.info(sep)
            logger.info("  R-STDP Fine-tuning Configuration")
            logger.info(f"  STDP mode:         {self.stdp_mode}")
            logger.info(f"  Alpha (backprop):  {self.alpha}")
            logger.info(f"  Beta (STDP):       {self.beta}")
            logger.info(f"  Total steps:       {self.config.trainer.max_train_steps}")
            logger.info(f"  Batch / device:    {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Grad accumulation: {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size:  {self.total_batch_size}")
            logger.info(sep)

    def _finalize_training(self):
        if self.accelerator.is_main_process:
            import shutil
            save_format = getattr(self.config.trainer, "save_format", "safetensors")
            final_dir = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file
                save_file(state_dict, os.path.join(final_dir, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_dir, "pytorch_model.pt"))

            if isinstance(self.config, AccessTrackedConfig):
                self.config.save_accessed_config(
                    Path(final_dir) / "framework_config.yaml", use_original_values=False
                )
            else:
                OmegaConf.save(self.config, os.path.join(final_dir, "framework_config.yaml"))

            dataset_stats_src = os.path.join(self.config.output_dir, "dataset_statistics.json")
            if os.path.exists(dataset_stats_src):
                shutil.copy2(dataset_stats_src, os.path.join(final_dir, "dataset_statistics.json"))

            qwen_dir = os.path.join(final_dir, "qwen_pretrained")
            os.makedirs(qwen_dir, exist_ok=True)
            if hasattr(self.model, "qwen_vl_interface"):
                self.model.qwen_vl_interface.model.config.save_pretrained(qwen_dir)
                self.model.qwen_vl_interface.processor.save_pretrained(qwen_dir)

            if self.rstdp_optimizer is not None:
                torch.save(
                    self.rstdp_optimizer.state_dict(),
                    os.path.join(final_dir, "rstdp_optimizer.pt")
                )

            logger.info(f"STDP training complete. Final model saved at {final_dir}")

        # Cleanup
        if self.spike_monitor is not None:
            self.spike_monitor.disable()

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("NeuroVLA R-STDP Fine-tuning :: Starting")

    cfg = wrap_config(cfg)

    output_dir = setup_directories(cfg)
    model = build_framework(cfg)
    dataloader = build_dataloader(cfg=cfg, dataloader_module=cfg.datasets.vla_data.dataloader_module)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    # Build optimizer and scheduler
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    trainer = STDPTrainer(
        cfg=cfg,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("R-STDP training complete!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="configs/finetune_config.yaml")
    parser.add_argument("--mode", type=str, default="neuro_vla_stdp")
    args, extra_cli_args = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    if "modes" in cfg:
        mode = args.mode or OmegaConf.to_container(cfg.get("defaults", {}), resolve=False).get("model")
        if not mode:
            raise ValueError("--mode required when using finetune_config.yaml")
        cfg = build_config_from_finetune(cfg, mode)
    elif "defaults" in cfg:
        defaults = cfg.pop("defaults")
        base_cfgs = []
        if "model" in defaults:
            base_cfgs.append(OmegaConf.load(f"configs/models/{defaults.model}.yaml"))
        if "dataset" in defaults:
            base_cfgs.append(OmegaConf.load(f"configs/datasets/{defaults.dataset}.yaml"))
        if "trainer" in defaults:
            base_cfgs.append(OmegaConf.load(f"configs/trainer/{defaults.trainer}.yaml"))
        cfg = OmegaConf.merge(*base_cfgs, cfg)

    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(extra_cli_args))
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
