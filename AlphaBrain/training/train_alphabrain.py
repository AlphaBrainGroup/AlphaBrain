# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
AlphaBrain’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from AlphaBrain.training.trainer_utils.trainer_tools import normalize_dotlist_args
import copy
from AlphaBrain.model.framework import build_framework
from AlphaBrain.training.trainer_utils.trainer_tools import TrainerUtils
from AlphaBrain.training.trainer_utils.trainer_tools import build_param_lr_groups
from AlphaBrain.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig
from AlphaBrain.training.trainer_utils.finetune_config import build_config_from_finetune


def _build_accelerator(gradient_accumulation_steps: int = 1) -> Accelerator:
    """Create Accelerator with correct gradient_accumulation_steps from config."""
    import os as _os
    if _os.environ.get('USE_DDP') == '1':
        # WM mode: DDP with find_unused_parameters=True
        # Required because V2 2-frame DiT forward creates non-deterministic
        # autograd paths (per-rank random sigma) that DDP/DeepSpeed can't handle.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        acc = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
        )
    else:
        # Standard VLM mode: DeepSpeed ZeRO-2
        deepspeed_plugin = DeepSpeedPlugin()
        acc = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            deepspeed_plugin=deepspeed_plugin,
        )
    acc.print(acc.state)
    return acc


# Suppress DeepSpeed verbose INFO logs (config dump, optimizer init details, etc.)
import logging as _logging
_logging.getLogger("deepspeed").setLevel(_logging.WARNING)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def setup_file_logging(output_dir: str, rank: int = 0):
    """Add a FileHandler to root logger so all log messages are saved to a local file.
    Only the main process (rank 0) writes to avoid multi-process file conflicts.
    """
    if rank != 0:
        return None
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    return log_file


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.output_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # setup file logging to save logs locally
        log_file = setup_file_logging(str(output_dir), rank=0)
        if log_file:
            logger.info(f"Training logs will be saved to: {log_file}")

        # # save config
        # OmegaConf.save(cfg, output_dir / "config.yaml")
        # with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
        #     yaml_cfg = yaml.safe_load(f_yaml)
        #     json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    if hasattr(cfg.framework, 'qwenvl') and hasattr(cfg.framework.qwenvl, 'base_vlm'):
        logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    else:
        logger.info(f"Building framework: {cfg.framework.name}")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from AlphaBrain.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    dataset_mix = getattr(cfg.datasets.vla_data, 'dataset_mix', 'N/A')
    logger.info(f"Creating VLA Dataset with Mixture `{dataset_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataloader_module=cfg.datasets.vla_data.dataloader_module)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()

    return vla_train_dataloader


def _build_lambda_linear_scheduler(optimizer, cycle_lengths, warm_up_steps, f_start, f_max, f_min):
    """
    Multi-cycle LambdaWarmUpCosine scheduler matching the original cosmos_policy.

    Each cycle i has:
      - warm_up_steps[i] steps: linear ramp from f_start[i] -> f_max[i]
      - remaining steps: cosine decay from f_max[i] -> f_min[i]
    The last cycle repeats indefinitely at f_min[-1].
    """
    import math

    def lr_lambda(step: int) -> float:
        cumulative = 0
        for i, cycle_len in enumerate(cycle_lengths):
            cycle_start = cumulative
            if i == len(cycle_lengths) - 1:
                local_step = step - cycle_start
            else:
                if step < cumulative + cycle_len:
                    local_step = step - cycle_start
                else:
                    cumulative += cycle_len
                    continue

            wu = warm_up_steps[i]
            fs, fm, fn = f_start[i], f_max[i], f_min[i]

            if wu > 0 and local_step < wu:
                # Linear warmup: fs -> fm
                return fs + (fm - fs) * local_step / wu
            else:
                # Cosine decay: fm -> fn (matching official LambdaWarmUpCosineScheduler)
                decay_steps = cycle_len - wu
                if decay_steps <= 0:
                    return fn
                decay_step = local_step - wu
                if i == len(cycle_lengths) - 1:
                    progress = min(decay_step / max(decay_steps, 1), 1.0)
                else:
                    progress = decay_step / decay_steps
                return fn + 0.5 * (fm - fn) * (1.0 + math.cos(math.pi * progress))

        return f_min[-1]

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    scheduler_type = getattr(cfg.trainer, 'scheduler_type', None)
    if scheduler_type == 'lambda_linear':
        from omegaconf import OmegaConf
        t = cfg.trainer
        cycle_lengths = list(OmegaConf.to_container(t.cycle_lengths, resolve=True))
        warm_up_steps = list(OmegaConf.to_container(t.warm_up_steps, resolve=True))
        f_start = list(OmegaConf.to_container(t.f_start, resolve=True))
        f_max = list(OmegaConf.to_container(t.f_max, resolve=True))
        f_min = list(OmegaConf.to_container(t.f_min, resolve=True))
        lr_scheduler = _build_lambda_linear_scheduler(
            optimizer, cycle_lengths, warm_up_steps, f_start, f_max, f_min
        )
    else:
        lr_scheduler = get_scheduler(
            name=cfg.trainer.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.trainer.num_warmup_steps,
            num_training_steps=cfg.trainer.max_train_steps,
            scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
        )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # LoRA
        from AlphaBrain.training.trainer_utils.peft import is_lora_enabled
        self.use_lora = is_lora_enabled(cfg)

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

        # EMA (Exponential Moving Average)
        ema_cfg = getattr(cfg.trainer, 'ema', None)
        self.use_ema = ema_cfg is not None and getattr(ema_cfg, 'enabled', False)
        self.ema_decay = getattr(ema_cfg, 'decay', 0.99) if ema_cfg else 0.99
        self.ema_model = None  # initialized after distributed setup

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        self._init_checkpointing() # TODO merge with load pretrained weights

        # 根据  resume 调整 lr_scheduler
        self._adjust_lr_scheduler_for_resume()

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        # torch.compile for kernel fusion speedup (B200/Blackwell friendly)
        use_compile = getattr(self.config, 'torch_compile', False) or os.environ.get('VLAE_TORCH_COMPILE', '') == '1'
        if use_compile:
            compile_mode = getattr(self.config, 'compile_mode', 'max-autotune')
            if self.accelerator.is_main_process:
                print(f"[info] Applying torch.compile (mode={compile_mode})...")
            self.model = torch.compile(self.model, mode=compile_mode)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_wandb()

        # Initialize EMA after distributed setup
        if self.use_ema and self.accelerator.is_main_process:
            logger.info(f"[EMA] Initializing EMA with decay={self.ema_decay}")
            unwrapped = self.accelerator.unwrap_model(self.model)
            self.ema_model = copy.deepcopy(unwrapped).cpu()
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)


    def _adjust_lr_scheduler_for_resume(self):
        """根据已完成的步数调整学习率调度器状态"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")

            # 方法1: 直接模拟已完成的步数（适用于大多数调度器）
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()

            # 或者方法2: 对于某些调度器，可以直接设置最后步数
            # if hasattr(self.lr_scheduler, '_step_count'):
            #     self.lr_scheduler._step_count = self.completed_steps

            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        wandb_mode = getattr(self.config, 'wandb_mode', None) or getattr(getattr(self.config, 'environment', None), 'wandb_mode', None) or os.environ.get('WANDB_MODE', 'online')
        if wandb_mode == 'disabled':
            os.environ['WANDB_MODE'] = 'disabled'
            return
        if self.accelerator.is_main_process:
            # Support both nested (environment.wandb_project) and flat (wandb_project) config layouts
            if hasattr(self.config, 'environment') and self.config.environment is not None:
                wandb_project = self.config.environment.wandb_project
                wandb_entity = self.config.environment.wandb_entity
                wandb_base_url = getattr(self.config.environment, 'wandb_base_url', None)
            else:
                wandb_project = getattr(self.config, 'wandb_project', 'vla-engine')
                wandb_entity = getattr(self.config, 'wandb_entity', '')
                wandb_base_url = getattr(self.config, 'wandb_base_url', None)
            # Set proxy/mirror base URL if configured
            if wandb_base_url:
                os.environ["WANDB_BASE_URL"] = wandb_base_url
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=wandb_project,
                entity=wandb_entity or None,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            # === Resume: restore full training state ===
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)

            if resume_from_checkpoint:
                # Validate GPU count
                import json as _json
                meta_path = os.path.join(resume_from_checkpoint, "resume_meta.json")
                with open(meta_path) as f:
                    meta = _json.load(f)

                saved_gpus = meta["num_gpus"]
                current_gpus = self.accelerator.num_processes
                if saved_gpus != current_gpus:
                    raise RuntimeError(
                        f"GPU count mismatch! Checkpoint saved with {saved_gpus} GPUs, "
                        f"current {current_gpus} GPUs. Resume requires exact GPU count match."
                    )

                # Validate framework
                saved_framework = meta.get("framework_name", "unknown")
                current_framework = getattr(self.config.framework, "name", "unknown")
                if saved_framework != current_framework and saved_framework != "unknown":
                    raise RuntimeError(
                        f"Framework mismatch! Checkpoint: {saved_framework}, current: {current_framework}"
                    )

                # Load full training state (model + optimizer + scheduler + RNG)
                training_state_dir = os.path.join(resume_from_checkpoint, "training_state")
                # Check training_state has actual files (not just empty dir from failed save_state)
                has_state = os.path.isdir(training_state_dir) and len(os.listdir(training_state_dir)) > 0
                has_model_state = has_state and any(
                    f.endswith((".pt", ".bin", ".safetensors", ".pkl"))
                    for root, dirs, files in os.walk(training_state_dir) for f in files
                )
                if has_model_state:
                    self.accelerator.load_state(training_state_dir)
                    self.resume_from_checkpoint = resume_from_checkpoint
                    logger.info(
                        f"Resume OK: step={self.completed_steps}, GPUs={saved_gpus}, "
                        f"path={resume_from_checkpoint}"
                    )
                    return None
                else:
                    # Fallback: checkpoint without valid training_state — warm restart
                    logger.warning(
                        f"Checkpoint {resume_from_checkpoint} has no valid training_state. "
                        f"Loading model weights only (optimizer/scheduler reset)."
                    )
                    # load_pretrained_backbones expects a file path, not directory
                    ckpt_file = os.path.join(resume_from_checkpoint, "model.safetensors")
                    if not os.path.isfile(ckpt_file):
                        ckpt_file = resume_from_checkpoint  # fallback to original path
                    self.model = self.load_pretrained_backbones(
                        self.model, ckpt_file, reload_modules=None
                    )
                    self.resume_from_checkpoint = resume_from_checkpoint
                    logger.info(f"Weights-only resume from step {self.completed_steps}")
                    return None
            else:
                logger.warning(f"is_resume=True but no resumable checkpoint in {self.checkpoint_dir}")
                self.completed_steps = 0

        # Load pretrained weights (not resume, start from step 0)
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}")
        else:
            logger.info("No pretrained checkpoint. Starting from scratch.")
            self.completed_steps = 0


    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")


    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest resumable checkpoint in checkpoint_dir.

        Returns:
            (checkpoint_path, completed_steps) or (None, 0)
        """
        import json as _json
        if not os.path.isdir(checkpoint_dir):
            return None, 0

        candidates = []
        for d in sorted(os.listdir(checkpoint_dir)):
            if d.startswith("steps_") and os.path.isdir(os.path.join(checkpoint_dir, d)):
                meta_path = os.path.join(checkpoint_dir, d, "resume_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = _json.load(f)
                    candidates.append((d, meta["completed_steps"], meta))

        if not candidates:
            return None, 0

        candidates.sort(key=lambda x: x[1], reverse=True)
        latest_dir, latest_steps, _ = candidates[0]
        return os.path.join(checkpoint_dir, latest_dir), latest_steps


    def _get_vl_interface(self, model=None):
        """Get the VLM interface from model (supports Qwen, Llama, PaliGemma)."""
        if model is None:
            model = self.accelerator.unwrap_model(self.model)
        for attr_name in ('qwen_vl_interface', 'llama_vl_interface', 'paligemma_vl_interface'):
            if hasattr(model, attr_name):
                return getattr(model, attr_name), attr_name
        raise RuntimeError("No VLM interface found on model")

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
        """save current training state"""

        if self.accelerator.is_main_process:
            if self.use_lora:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
                self._save_lora_checkpoint(checkpoint_path)
                self.accelerator.wait_for_everyone()
                return

            # # TODO 这里没有把save_format读进去, 是直接写死
            # save_format = getattr(self.config.trainer, "save_format", "pt")
            # checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # # save model state
            # state_dict = self.accelerator.get_state_dict(self.model)
            # if save_format == "safetensors":
            #     from safetensors.torch import save_file

            #     save_file(state_dict, checkpoint_path + "_model.safetensors")
            # elif save_format == "pt":
            #     torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            # else:
            #     raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            # # origin0309: 原始保存方式，仅保存权重文件（缺少架构信息和processor）
            # save_format = getattr(self.config.trainer, "save_format", "safetensors")
            # checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # # save model state
            # state_dict = self.accelerator.get_state_dict(self.model)
            # if save_format == "safetensors":
            #     from safetensors.torch import save_file
            #     save_file(state_dict, checkpoint_path + "_model.safetensors")
            # elif save_format == "pt":
            #     torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            # else:
            #     raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            # # save training metadata
            # summary_data = {
            #     "steps": self.completed_steps,
            # }
            # with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
            #     f.write(json.dumps(summary_data) + "\n")
            # self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            # # ✅ Save accessed configuration only
            # if isinstance(self.config, AccessTrackedConfig):
            #     logger.info("📊 Saving accessed configuration...")
            #     output_dir = Path(self.config.output_dir)
            #     self.config.save_accessed_config(
            #         output_dir / "config.yaml",
            #         use_original_values=False
            #     )
            #     logger.info("✅ Configuration files saved")

            # lpt0309: 保存自包含checkpoint目录（权重 + 架构config + processor + 训练config + norm_stats）
            import shutil
            save_format = getattr(self.config.trainer, "save_format", "safetensors")
            checkpoint_dir_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            os.makedirs(checkpoint_dir_path, exist_ok=True)

            # lpt0309: 保存模型权重
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file
                # Move to CPU and clone shared tensors to avoid OOM + safetensors error
                import gc; torch.cuda.empty_cache(); gc.collect()
                state_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
                save_file(state_dict, os.path.join(checkpoint_dir_path, "model.safetensors"))
                del state_dict; gc.collect(); torch.cuda.empty_cache()
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(checkpoint_dir_path, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            # lpt0309: 保存框架训练config到checkpoint目录
            # 注意：不用 save_accessed_config，因为它只保存被访问过的叶节点，
            # 会丢失 getattr(cfg, key, default) 方式读取的配置（如 paligemma 参数）
            OmegaConf.save(self.config, os.path.join(checkpoint_dir_path, "framework_config.yaml"))

            # lpt0309: 保存dataset_statistics到checkpoint目录
            dataset_stats_src = os.path.join(self.config.output_dir, "dataset_statistics.json")
            if os.path.exists(dataset_stats_src):
                shutil.copy2(dataset_stats_src, os.path.join(checkpoint_dir_path, "dataset_statistics.json"))


            # lpt0309: 保存模型特定文件到checkpoint目录
            if hasattr(self.model, 'qwen_vl_interface'):
                # Qwen-based models: 保存config和processor到qwen_pretrained/子目录
                qwen_pretrained_dir = os.path.join(checkpoint_dir_path, "qwen_pretrained")
                os.makedirs(qwen_pretrained_dir, exist_ok=True)
                self.model.qwen_vl_interface.model.config.save_pretrained(qwen_pretrained_dir)
                self.model.qwen_vl_interface.processor.save_pretrained(qwen_pretrained_dir)
                logger.info(f"[lpt0309] Saved Qwen config + processor to {qwen_pretrained_dir}")
            elif hasattr(self.model, 'llama_vl_interface'):
                qwen_pretrained_dir = os.path.join(checkpoint_dir_path, "qwen_pretrained")
                os.makedirs(qwen_pretrained_dir, exist_ok=True)
                self.model.llama_vl_interface.model.config.save_pretrained(qwen_pretrained_dir)
                self.model.llama_vl_interface.processor.save_pretrained(qwen_pretrained_dir)
            elif hasattr(self.model, 'paligemma_vl_interface'):
                qwen_pretrained_dir = os.path.join(checkpoint_dir_path, "qwen_pretrained")
                os.makedirs(qwen_pretrained_dir, exist_ok=True)
                self.model.paligemma_vl_interface.model.config.save_pretrained(qwen_pretrained_dir)
                self.model.paligemma_vl_interface.processor.save_pretrained(qwen_pretrained_dir)
            elif getattr(self.config.framework, 'name', '') in ('CosmosPolicy',):
                # CosmosPolicy: 保存config.json和t5_embeddings
                self._save_cosmos_policy_extras(checkpoint_dir_path)

            # lpt0309: 保存training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            # === Save resume metadata (rank 0 only — file I/O) ===
            import json as _json
            resume_meta = {
                "completed_steps": self.completed_steps,
                "num_gpus": self.accelerator.num_processes,
                "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
                "per_device_batch_size": int(getattr(self.config.datasets.vla_data, "per_device_batch_size", 0)),
                "effective_batch_size": int(self._calculate_total_batch_size()),
                "framework_name": str(getattr(self.config.framework, "name", "unknown")),
            }
            with open(os.path.join(checkpoint_dir_path, "resume_meta.json"), "w") as f:
                _json.dump(resume_meta, f, indent=2)

            self.accelerator.print(f"✅ [lpt0309] Self-contained checkpoint saved at {checkpoint_dir_path}")

            # lpt0309: 同时保存config到output_dir（兼容旧逻辑）
            logger.info("📊 Saving full configuration...")
            output_dir = Path(self.config.output_dir)
            OmegaConf.save(self.config, str(output_dir / "config.yaml"))
            logger.info("✅ Configuration files saved")

        # === Collective operations — ALL ranks must participate ===
        self.accelerator.wait_for_everyone()

        # === Optionally save full training state (optimizer + scheduler + RNG) ===
        # Controlled by trainer.save_training_state (default: false)
        # When enabled, saves ~50GB per checkpoint for full resume capability.
        # When disabled, only model weights + resume_meta are saved (warm restart on resume).
        save_training_state = getattr(self.config.trainer, "save_training_state", False)
        checkpoint_dir_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        if save_training_state:
            training_state_dir = os.path.join(checkpoint_dir_path, "training_state")
            try:
                self.accelerator.save_state(training_state_dir)
                self.accelerator.print(f"Training state saved ({self.accelerator.num_processes} GPUs)")
            except Exception as e:
                self.accelerator.print(f"Warning: save_state failed: {e}. Resume will use model weights only.")
        else:
            self.accelerator.print("Skipping training state save (save_training_state=false). Resume will use warm restart.")

        self.accelerator.wait_for_everyone()

    def _save_cosmos_policy_extras(self, checkpoint_dir_path):
        """Save CosmosPolicy-specific files for self-contained inference checkpoint.
        Matches pretrained format so server_policy_cosmos.py can load directly.
        """
        import shutil

        # config.json
        cosmos_config = {
            "model_type": "cosmos-policy",
            "architecture": "diffusion-transformer",
            "action_dim": self.model.action_dim,
            "chunk_size": self.model.chunk_size,
            "proprio_dim": self.model.proprio_dim,
            "state_t": self.model.state_t,
            "training_steps": self.completed_steps,
        }
        with open(os.path.join(checkpoint_dir_path, "config.json"), "w") as f:
            json.dump(cosmos_config, f, indent=2)

        # T5 embeddings (save as libero_t5_embeddings.pkl to match server expectation)
        t5_path = getattr(self.model, 't5_embeddings_path', None)
        if t5_path and os.path.exists(t5_path):
            shutil.copy2(t5_path, os.path.join(checkpoint_dir_path, "libero_t5_embeddings.pkl"))

        # Dataset statistics (save as libero_dataset_statistics.json to match server)
        stats_candidates = [
            os.path.join(self.config.output_dir, "dataset_statistics.json"),
            getattr(self.config, 'dataset_stats_path', ''),
        ]
        # Also check the pretrained model dir
        try:
            pretrained_dir = self.model.config["framework"]["cosmos_policy"]["checkpoint"]["pretrained_dir"]
        except (KeyError, TypeError, AttributeError):
            pretrained_dir = ''
        if pretrained_dir:
            stats_candidates.append(os.path.join(os.path.dirname(pretrained_dir),
                "Cosmos-Policy-LIBERO-Predict2-2B", "libero_dataset_statistics.json"))
        for src in stats_candidates:
            if src and os.path.exists(src):
                shutil.copy2(src, os.path.join(checkpoint_dir_path, "libero_dataset_statistics.json"))
                break

        # Save DIT-only weights as cosmos_dit.pt (net. prefix for server compatibility)
        dit_state = {}
        for k, v in self.model.dit.state_dict().items():
            dit_state[f"net.{k}"] = v
        torch.save(dit_state, os.path.join(checkpoint_dir_path, "cosmos_dit.pt"))

        logger.info(f"[CosmosPolicy] Saved cosmos checkpoint extras to {checkpoint_dir_path}")

    def _log_metrics(self, metrics):
        """record training metrics"""

        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:

                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0] # see lr group in yaml.trainer.learning_rate

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # add step info
                metrics["step"] = self.completed_steps

                # record to W&B
                if wandb.run is not None:
                    wandb.log(metrics, step=self.completed_steps)

                # record to local metrics.jsonl
                metrics_file = os.path.join(self.config.output_dir, "metrics.jsonl")
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                # debug output
                loss_val = metrics.get("action_dit_loss", float("nan"))
                lr_val   = metrics.get("learning_rate", float("nan"))
                mse_val  = metrics.get("mse_score", None)
                act_mse  = metrics.get("action_mse", None)
                cond_mse = metrics.get("cond_mse", None)
                step_str = f"[bold cyan]step {self.completed_steps:>6d}[/bold cyan]"
                loss_str = f"[bold yellow]action_dit_loss={loss_val:.5f}[/bold yellow]"
                lr_str   = f"[cyan]lr={lr_val:.2e}[/cyan]"
                mse_str  = f"  [bold green]mse={mse_val:.6f}[/bold green]" if mse_val is not None else ""
                act_str  = f"  act_mse={act_mse:.4f}" if act_mse is not None and act_mse > 0 else ""
                cond_str = f"  cond_mse={cond_mse:.6f}" if cond_mse is not None else ""
                # V2: show video_loss and total_loss in console
                vid_loss = metrics.get("video_loss", None)
                tot_loss = metrics.get("total_loss", None)
                vid_str  = f"  [magenta]video_loss={vid_loss:.4f}[/magenta]" if vid_loss is not None else ""
                tot_str  = f"  [bold]total_loss={tot_loss:.4f}[/bold]" if tot_loss is not None else ""
                print(f"  step {self.completed_steps:>6d}  action_dit_loss={loss_val:.5f}" + (f"  video_loss={vid_loss:.4f}" if vid_loss is not None else "") + (f"  total_loss={tot_loss:.4f}" if tot_loss is not None else "") + f"  lr={lr_val:.2e}" + (f"  mse={mse_val:.6f}" if mse_val is not None else "") + (f"  act_mse={act_mse:.4f}" if act_mse is not None and act_mse > 0 else "") + (f"  cond_mse={cond_mse:.6f}" if cond_mse is not None else ""), flush=True)

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                _postfix = {
                    "action_dit_loss": f"{step_metrics.get('action_dit_loss', 0):.4f}",
                    "data": f"{t_end_data - t_start_data:.3f}s",
                    "fwd": f"{t_end_model - t_start_model:.3f}s",
                }
                if "video_loss" in step_metrics:
                    _postfix["video_loss"] = f"{step_metrics['video_loss']:.4f}"
                    _postfix["total_loss"] = f"{step_metrics.get('total_loss', 0):.4f}"
                progress_bar.set_postfix(_postfix)

            # evaluate model

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                try:
                    step_metrics = self.eval_action_model(step_metrics)
                except Exception as e:
                    if self.accelerator.is_main_process:
                        logger.warning(f"eval_action_model failed: {e}, skipping")


            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        examples = self._get_next_batch()
        if examples is None:
            logger.warning('eval_action_model: got None batch, skipping')
            return step_metrics if step_metrics else {}
        score = 0.0
        num_samples = len(examples)
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]  # label
        states = [example["state"] for example in examples] if "state" in examples[0] else None
        # Predict actions using the model
        output_dict = self.model.predict_action(
            batch_images=batch_images, instructions=instructions, states=states,
            use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            actions = np.array(actions)  # convert actions to numpy.ndarray
            # B, Chunk, dim = actions.shape
            num_elements = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_elements
            step_metrics["mse_score"] = average_score

        del examples
        if dist.is_initialized():
            dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            sep = "━" * 56
            logger.info(f"[bold cyan]{sep}[/bold cyan]")
            logger.info(f"  [bold cyan]▶  Training Configuration[/bold cyan]")
            logger.info(f"  [dim]Total steps:[/dim]        [bold yellow]{self.config.trainer.max_train_steps}[/bold yellow]")
            logger.info(f"  [dim]Batch / device:[/dim]     [yellow]{self.config.datasets.vla_data.per_device_batch_size}[/yellow]")
            logger.info(f"  [dim]Grad accumulation:[/dim]  [yellow]{self.config.trainer.gradient_accumulation_steps}[/yellow]")
            logger.info(f"  [dim]Total batch size:[/dim]   [bold yellow]{self.total_batch_size}[/bold yellow]")
            logger.info(f"[bold cyan]{sep}[/bold cyan]")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        # NOTE: Do NOT use accelerator.accumulate() — it calls no_sync() which is
        # incompatible with DeepSpeed ZeRO-2 reduce_scatter. Instead, DeepSpeed
        # handles gradient accumulation internally via its own gradient_accumulation_steps.
        self.optimizer.zero_grad()

        # VLA task forward propagation
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)

            action_loss = output_dict["action_loss"]
            # V2: use total_loss (action + video) if present, else action only
            if "total_loss" in output_dict:
                total_loss = output_dict["total_loss"]
            else:
                total_loss = action_loss

        # Loss spike protection — MUST NOT skip backward in distributed training
        # or NCCL all_reduce will deadlock (some GPUs skip, others wait forever).
        # Instead, zero-out the loss so backward still runs with zero gradients.
        loss_val = total_loss.detach().item()
        _loss_spiked = torch.isnan(total_loss) or torch.isinf(total_loss) or loss_val > 100.0
        if _loss_spiked:
            print(f"[step {self.completed_steps}] Loss spike: {loss_val:.4f}, zeroing loss", flush=True)
            total_loss = total_loss * 0.0  # keeps graph alive for backward

        # VLA backward propagation
        self.accelerator.backward(total_loss)

        # NaN-to-num protection (skip for CosmosPolicy which manages its own gradient stability)
        gradient_clipping = getattr(self.config.trainer, 'gradient_clipping', 1.0)
        if gradient_clipping is not None and gradient_clipping != 0:
            for param in self.model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)
            # gradient clipping
            self.accelerator.clip_grad_norm_(self.model.parameters(), gradient_clipping)

        # optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()

        # EMA update
        if self.use_ema and self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(self.model)
            with torch.no_grad():
                for ema_p, model_p in zip(self.ema_model.parameters(), unwrapped.parameters()):
                    ema_p.data.mul_(self.ema_decay).add_(model_p.data.cpu(), alpha=1 - self.ema_decay)

        log_dict = {
            "action_dit_loss": action_loss.item(),
            "action_mse": output_dict.get("action_mse", torch.tensor(0.0)).item() if isinstance(output_dict.get("action_mse"), torch.Tensor) else output_dict.get("action_mse", 0.0),
            "cond_mse": output_dict.get("cond_mse", torch.tensor(0.0)).item() if isinstance(output_dict.get("cond_mse"), torch.Tensor) else output_dict.get("cond_mse", 0.0),
        }
        # V2: log video_loss and total_loss if present
        if "video_loss" in output_dict:
            vl = output_dict["video_loss"]
            log_dict["video_loss"] = vl.item() if isinstance(vl, torch.Tensor) else vl
            log_dict["total_loss"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return log_dict

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            if self.use_lora:
                final_path = os.path.join(self.config.output_dir, "final_model")
                os.makedirs(final_path, exist_ok=True)
                self._save_lora_checkpoint(os.path.join(final_path, "final"))
                logger.info(f"LoRA training complete. Final model saved at {final_path}")
                if self.accelerator.is_main_process:
                    wandb.finish()
                self.accelerator.wait_for_everyone()
                return

            # lpt0309: 保存自包含final checkpoint目录
            import shutil
            save_format = getattr(self.config.trainer, "save_format", "safetensors")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)

            # lpt0309: 保存模型权重 (use EMA weights if available)
            if self.use_ema and self.ema_model is not None:
                logger.info("[EMA] Saving EMA model weights as final model")
                state_dict = {k: v.clone() for k, v in self.ema_model.state_dict().items()}
            else:
                state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file
                import gc; torch.cuda.empty_cache(); gc.collect()
                state_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
                del state_dict; gc.collect(); torch.cuda.empty_cache()
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            # lpt0309: 保存框架训练config
            # 同上：不用 save_accessed_config，直接保存完整 config
            OmegaConf.save(self.config, os.path.join(final_checkpoint, "framework_config.yaml"))

            # lpt0309: 保存dataset_statistics
            dataset_stats_src = os.path.join(self.config.output_dir, "dataset_statistics.json")
            if os.path.exists(dataset_stats_src):
                shutil.copy2(dataset_stats_src, os.path.join(final_checkpoint, "dataset_statistics.json"))

            # lpt0309: 保存模型特定文件
            if hasattr(self.model, 'qwen_vl_interface'):
                # Qwen-based models: 保存config和processor到qwen_pretrained/子目录
                qwen_pretrained_dir = os.path.join(final_checkpoint, "qwen_pretrained")
                os.makedirs(qwen_pretrained_dir, exist_ok=True)
                self.model.qwen_vl_interface.model.config.save_pretrained(qwen_pretrained_dir)
                self.model.qwen_vl_interface.processor.save_pretrained(qwen_pretrained_dir)
                logger.info(f"[lpt0309] Saved Qwen config + processor to {qwen_pretrained_dir}")

            elif getattr(self.config.framework, 'name', '') in ('CosmosPolicy',):
                # CosmosPolicy: 保存config.json和t5_embeddings
                self._save_cosmos_policy_extras(final_checkpoint)


            elif hasattr(self.model, 'llama_vl_interface'):
                llama_pretrained_dir = os.path.join(final_checkpoint, "qwen_pretrained")
                os.makedirs(llama_pretrained_dir, exist_ok=True)
                self.model.llama_vl_interface.model.config.save_pretrained(llama_pretrained_dir)
                self.model.llama_vl_interface.processor.save_pretrained(llama_pretrained_dir)
                logger.info(f"[lpt0319] Saved Llama config + processor to {llama_pretrained_dir}")

            elif hasattr(self.model, 'paligemma_vl_interface'):
                paligemma_pretrained_dir = os.path.join(final_checkpoint, "qwen_pretrained")
                os.makedirs(paligemma_pretrained_dir, exist_ok=True)
                self.model.paligemma_vl_interface.model.config.save_pretrained(paligemma_pretrained_dir)
                self.model.paligemma_vl_interface.processor.save_pretrained(paligemma_pretrained_dir)
                logger.info(f"Saved PaliGemma config + processor to {paligemma_pretrained_dir}")

            logger.info(f"[lpt0309] Training complete. Self-contained final model saved at {final_checkpoint}")

        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)

    # Build accelerator FIRST so logger (accelerate.logging) can be used
    grad_accum = getattr(cfg.trainer, "gradient_accumulation_steps", 1)
    accelerator = _build_accelerator(gradient_accumulation_steps=grad_accum)

    logger.info("VLA Training :: Warming Up")
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model

    model = build_model(cfg)

    # LoRA wrapping (before optimizer, after model build)
    from AlphaBrain.training.trainer_utils.peft import apply_lora
    apply_lora(model, cfg)


    # Freeze modules BEFORE optimizer creation (so frozen params get no optimizer states)
    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if freeze_modules and isinstance(freeze_modules, str) and freeze_modules.strip():
        freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]
        for freeze_path in freeze_patterns:
            module = model
            try:
                for attr in freeze_path.split("."):
                    module = getattr(module, attr)
                for p in module.parameters():
                    p.requires_grad_(False)
                frozen_count = sum(1 for p in module.parameters())
                logger.info(f"[early_freeze] Froze {freeze_path} ({frozen_count} param tensors)")
            except AttributeError:
                logger.warning(f"[early_freeze] Path not found: {freeze_path}")

    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    if dist.is_initialized(): dist.barrier()
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="configs/finetune_config.yaml", help="Path to YAML config")
    parser.add_argument("--mode", type=str, default=None, help="Training mode (e.g. qwen_oft) for finetune_config.yaml")
    args, extra_cli_args = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    # WM modes use DDP (DeepSpeed can't handle V2 2-frame DiT backward)
    import os as _os
    if args.mode and args.mode.startswith('world_model_'):
        _os.environ['USE_DDP'] = '1'

    if 'modes' in cfg:
        # finetune_config.yaml path — highest priority config file
        mode = args.mode or OmegaConf.to_container(cfg.get('defaults', {}), resolve=False).get('model')
        if not mode:
            raise ValueError("--mode required when using finetune_config.yaml")
        cfg = build_config_from_finetune(cfg, mode)

    elif 'defaults' in cfg:
        # train_recipe.yaml with defaults section
        defaults = cfg.pop('defaults')
        base_cfgs = []
        if 'model' in defaults:
            base_cfgs.append(OmegaConf.load(f"configs/models/{defaults.model}.yaml"))
        if 'dataset' in defaults:
            base_cfgs.append(OmegaConf.load(f"configs/datasets/{defaults.dataset}.yaml"))
        if 'trainer' in defaults:
            base_cfgs.append(OmegaConf.load(f"configs/trainer/{defaults.trainer}.yaml"))
        cfg = OmegaConf.merge(*base_cfgs, cfg)

    elif '_model_config_' in cfg:
        # Backward compat: old _model_config_ mechanism
        model_cfg = OmegaConf.load(cfg.pop('_model_config_'))
        cfg = OmegaConf.merge(model_cfg, cfg)

    # CLI args have the highest priority
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(extra_cli_args))
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
