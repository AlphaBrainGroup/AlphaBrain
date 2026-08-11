"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

import os
import torch.nn as nn
from typing import List

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from typing import List

from pathlib import Path
from typing import Dict, List
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
import numpy as np
from AlphaBrain.model.tools import auto_get_trainable_modules

from AlphaBrain.model.framework.config_utils import read_mode_config
from AlphaBrain.training.trainer_utils import initialize_overwatch
from AlphaBrain.model.framework.config_utils import dict_to_namespace
from AlphaBrain.model.framework.__init__ import build_framework

logger = initialize_overwatch(__name__)


# PreTrainedModel, AutoModel, PretrainedConfig,  are so good, find sometime to study them
# TODO: find sometime to merge yaml config with transformer config

# ── VLM config key discovery ─────────────────────────────────────────────
# All known (config_attr, model_attr) pairs for VLM interfaces.
# Order matters: first match wins.
_VLM_REGISTRY = [
    # (framework config key,  model attribute name)
    ("paligemma",            "vlm_interface"),
    ("llamavl",              "llama_vl_interface"),
    ("qwenvl",               "qwen_vl_interface"),
]


def _detect_vlm_cfg_key(framework_cfg) -> str | None:
    """Return the first framework config key that exists (e.g. 'paligemma')."""
    for cfg_key, _ in _VLM_REGISTRY:
        if hasattr(framework_cfg, cfg_key):
            return cfg_key
    return None


def _detect_vlm_interface(model) -> object | None:
    """Return the VLM interface submodule from a model instance."""
    for _, attr in _VLM_REGISTRY:
        if hasattr(model, attr):
            return getattr(model, attr)
    return None


def _get_base_vlm_path(framework_cfg) -> str | None:
    """Extract base_vlm path from whichever VLM config block exists."""
    cfg_key = _detect_vlm_cfg_key(framework_cfg)
    if cfg_key is None:
        return None
    vlm_block = getattr(framework_cfg, cfg_key)
    if hasattr(vlm_block, 'get'):
        return vlm_block.get('base_vlm', None)
    return getattr(vlm_block, 'base_vlm', None)


def _apply_runtime_vlm_overrides(config, base_vlm_path=None, attention_backend=None) -> None:
    """Apply in-memory deployment overrides without mutating checkpoint files."""
    cfg_key = _detect_vlm_cfg_key(config.framework)
    if cfg_key is None:
        if base_vlm_path or attention_backend:
            logger.warning("VLM runtime override ignored: framework has no registered VLM block")
        return
    vlm_block = getattr(config.framework, cfg_key)
    if base_vlm_path:
        vlm_block.base_vlm = str(Path(base_vlm_path).expanduser().resolve())
        logger.info("Using deployment base VLM override: %s", vlm_block.base_vlm)
    if attention_backend:
        vlm_block.attn_implementation = str(attention_backend)
        logger.info("Using deployment attention backend: %s", attention_backend)


class BaseFramework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(
        self,
        hf_config = PretrainedConfig()
    ) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """
        
        super().__init__(hf_config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        base_vlm_path: str | None = None,
        attention_backend: str | None = None,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict strictly (reports missing/unexpected keys)
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt/.safetensors file or self-contained checkpoint directory.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            BaseFramework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)

        # lpt0309: 支持自包含目录格式checkpoint（单一路径推断，无需base_vlm）
        if pretrained_checkpoint.is_dir():
            logger.info(f"[lpt0309] Loading from self-contained checkpoint directory: {pretrained_checkpoint}")
            model_config, norm_stats = read_mode_config(pretrained_checkpoint)  # lpt0309: 从目录读取config和norm_stats

            config = dict_to_namespace(model_config)
            config.trainer.pretrained_checkpoint = None
            _apply_runtime_vlm_overrides(config, base_vlm_path, attention_backend)

            # 单次加载优化 - 如果checkpoint目录包含vlm_pretrained/（兼容旧名qwen_pretrained/），
            # 直接从中加载tokenizer/config，用meta device创建模型骨架，由后续load_state_dict一次性加载所有权重
            vlm_pretrained_dir = pretrained_checkpoint / "vlm_pretrained"
            legacy_dir = pretrained_checkpoint / "qwen_pretrained"
            if not (vlm_pretrained_dir.is_dir() and any(vlm_pretrained_dir.iterdir())):
                # 兼容旧格式 qwen_pretrained/
                if legacy_dir.is_dir() and any(legacy_dir.iterdir()):
                    vlm_pretrained_dir = legacy_dir

            if vlm_pretrained_dir.is_dir() and any(vlm_pretrained_dir.iterdir()):
                logger.info(f"Found {vlm_pretrained_dir.name}/ in checkpoint, using single-read loading")
                cfg_key = _detect_vlm_cfg_key(config.framework)
                if cfg_key is not None:
                    vlm_block = getattr(config.framework, cfg_key)
                    original_base_vlm = getattr(vlm_block, 'base_vlm', "") if not hasattr(vlm_block, 'get') else vlm_block.get('base_vlm', "")
                    vlm_block.vlm_type = original_base_vlm
                    vlm_block.base_vlm = str(vlm_pretrained_dir)
                    vlm_block._meta_device_init = True
                else:
                    logger.warning("vlm_pretrained/ found but no VLM config key detected, skipping single-read optimization")
            else:
                logger.warning(f"No vlm_pretrained/ found (or empty), falling back to two-read loading from original base_vlm")

            FrameworkModel = build_framework(cfg=config)
            FrameworkModel.norm_stats = norm_stats

            # lpt0309: 从目录中找到权重文件
            weights_path = pretrained_checkpoint / "model.safetensors"
            if not weights_path.exists():
                weights_path = pretrained_checkpoint / "pytorch_model.pt"
            assert weights_path.exists(), f"[lpt0309] No weights file found in {pretrained_checkpoint}"

            if weights_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                model_state_dict = load_file(str(weights_path))
            else:
                model_state_dict = torch.load(weights_path, map_location="cpu")

            logger.info(f"[lpt0309] Loading weights from {weights_path}")
            
            # Key remapping: old checkpoints use 'vlm.' prefix, new model uses 'qwen_vl_interface.'
            remapped = {}
            for k, v in model_state_dict.items():
                new_k = k.replace('vlm.', 'qwen_vl_interface.', 1) if k.startswith('vlm.') else k
                remapped[new_k] = v
            if len(remapped) != len(model_state_dict):
                logger.warning(f"Key remapping changed key count: {len(model_state_dict)} -> {len(remapped)}")
            else:
                n_remapped = sum(1 for k in model_state_dict if k.startswith('vlm.'))
                if n_remapped > 0:
                    logger.info(f"Remapped {n_remapped} keys from vlm.* to qwen_vl_interface.*")
            model_state_dict = remapped
            
            model_keys = set(FrameworkModel.state_dict().keys())
            checkpoint_keys = set(model_state_dict.keys())
            # Try strict first; fall back to non-strict if only non-critical keys mismatch
            try:
                FrameworkModel.load_state_dict(model_state_dict, strict=True)
            except RuntimeError as e:
                common_keys = model_keys.intersection(checkpoint_keys)
                missing_keys = model_keys - common_keys
                unexpected_keys = checkpoint_keys - common_keys
                if missing_keys:
                    logger.warning(f"Missing keys in state_dict ({len(missing_keys)}): {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in state_dict ({len(unexpected_keys)}): {unexpected_keys}")
                # Fall back to non-strict loading for cross-framework weight loading (e.g. openpi → AlphaBrain)
                logger.warning(f"Strict loading failed, falling back to non-strict (missing={len(missing_keys)}, unexpected={len(unexpected_keys)})")
                FrameworkModel.load_state_dict(model_state_dict, strict=False)

            logger.info(
                "[lpt0324] Successfully loaded model from self-contained checkpoint "
                "with legacy two-stage loading"
            )
            return FrameworkModel

        # origin0309: 原始文件格式加载（需要base_vlm路径，存在冗余权重读取）
        else:
            model_config, norm_stats = read_mode_config(pretrained_checkpoint)  # read config and norm_stats

            config = dict_to_namespace(model_config)
            model_config = config
            model_config.trainer.pretrained_checkpoint = None
            _apply_runtime_vlm_overrides(model_config, base_vlm_path, attention_backend)
            # FrameworkModel = cls(config=model_config, **kwargs) # TODO find cls by config
            FrameworkModel = build_framework(cfg=model_config)
            # set for action un-norm
            FrameworkModel.norm_stats = norm_stats
            # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
            if pretrained_checkpoint.suffix == ".safetensors":
                from safetensors.torch import load_file
                # TODO pretrained_checkpoint 这里先转成了path后面又用str, 存在冗余
                model_state_dict = load_file(str(pretrained_checkpoint))
            else:
                model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
            # logger.info(f"Loading model weights from `{pretrained_checkpoint}`")
            model_keys = set(FrameworkModel.state_dict().keys())
            checkpoint_keys = set(model_state_dict.keys())  # TODO 为什么会存在重复?
            try:
                FrameworkModel.load_state_dict(model_state_dict, strict=True)
            except RuntimeError as e:
                # must keep all keys matched
                common_keys = model_keys.intersection(checkpoint_keys)
                missing_keys = model_keys - common_keys
                unexpected_keys = checkpoint_keys - common_keys
                if missing_keys:
                    logger.warning(f"Missing keys in state_dict: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

                raise e

            # **ensure model is on GPU**
            FrameworkModel = FrameworkModel
            return FrameworkModel

    # lpt0309: 将旧格式checkpoint转换为新的自包含目录格式
    @staticmethod
    def convert_checkpoint_to_dir(
        old_ckpt_path: str,
        output_dir: str = None,
        base_vlm_path: str = None,
    ):
        """
        Convert an old-format file checkpoint to the new self-contained directory format.

        Args:
            old_ckpt_path: Path to old .safetensors/.pt checkpoint file.
            output_dir: Output directory path. If None, creates a directory alongside the file.
            base_vlm_path: Path to Qwen base model (for saving config + processor).
                          If None, reads from the checkpoint's config.yaml.
        """
        import shutil
        old_ckpt_path = Path(old_ckpt_path)
        assert old_ckpt_path.is_file(), f"Old checkpoint not found: {old_ckpt_path}"

        # Determine output directory
        if output_dir is None:
            output_dir = old_ckpt_path.parent / old_ckpt_path.stem.replace("_model", "").replace("_pytorch", "")
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Copy weights
        weights_name = "model.safetensors" if old_ckpt_path.suffix == ".safetensors" else "pytorch_model.pt"
        shutil.copy2(str(old_ckpt_path), str(output_dir / weights_name))
        logger.info(f"[lpt0309] Copied weights to {output_dir / weights_name}")

        # Copy config.yaml and dataset_statistics.json from run dir
        run_dir = old_ckpt_path.parents[1]
        for fname, target_name in [("config.yaml", "framework_config.yaml"), ("dataset_statistics.json", "dataset_statistics.json")]:
            src = run_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(output_dir / target_name))
                logger.info(f"[lpt0309] Copied {fname} -> {output_dir / target_name}")

        # Save VLM config + processor (auto-detect VLM type from config)
        if base_vlm_path is None:
            config_yaml = run_dir / "config.yaml"
            if config_yaml.exists():
                from omegaconf import OmegaConf
                cfg = OmegaConf.load(str(config_yaml))
                base_vlm_path = _get_base_vlm_path(cfg.framework)

        if base_vlm_path:
            vlm_pretrained_dir = output_dir / "vlm_pretrained"
            os.makedirs(vlm_pretrained_dir, exist_ok=True)
            try:
                from transformers import AutoConfig, AutoProcessor
                vlm_config = AutoConfig.from_pretrained(base_vlm_path, trust_remote_code=True)
                vlm_config.save_pretrained(str(vlm_pretrained_dir))
                processor = AutoProcessor.from_pretrained(base_vlm_path)
                processor.save_pretrained(str(vlm_pretrained_dir))
                logger.info(f"[lpt0309] Saved VLM config + processor to {vlm_pretrained_dir}")
            except Exception as e:
                logger.warning(f"[lpt0309] Could not save VLM config/processor from {base_vlm_path}: {e}")

        logger.info(f"[lpt0309] Conversion complete: {output_dir}")

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Infer or validate the dataset stats key used for un-normalization.

        Args:
            norm_stats: Dict[str, dict] mapping dataset key -> stats block.
            unnorm_key: Optional explicit dataset key.

        Returns:
            str: Resolved key.

        Raises:
            AssertionError: If multiple datasets present and key not provided,
                            or provided key not found.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @classmethod
    def get_action_stats(self, unnorm_key=None):
        """
        Retrieve raw action normalization statistics.

        Args:
            unnorm_key: Optional dataset stats key.

        Returns:
            dict: Stats structure (e.g. q01, q99, mask).
        """
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    @property
    def trainable_module_keys(self, max_depth=1) -> List[str]:
        """
        Enumerate trainable submodule names up to a depth.

        Args:
            max_depth: Descent depth when traversing module tree.

        Returns:
            List[str]: Module path names considered trainable.
        """
        keys = auto_get_trainable_modules(self, max_depth=max_depth)  # auto check which modules are trainable
        return keys

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Map normalized actions (≈[-1, 1]) back to original value range.

        Auto-detects normalization mode via the optional 'norm_mode' key in
        action_norm_stats (defaults to 'q99' for backward compatibility):
            - 'q99'     → uses q01 / q99 bounds
            - 'min_max' → uses min / max bounds

        Steps:
            - Clamp values to [-1, 1]
            - Threshold channel index 6 to {0,1} (binary semantic)
            - Apply linear scaling for masked dimensions

        Args:
            normalized_actions: Array shape [T, D] (or chunk length × action_dim).
            action_norm_stats: Dict containing stat arrays and optional 'norm_mode'.

        Returns:
            np.ndarray: Unnormalized actions (same shape as input).
        """
        norm_mode = action_norm_stats.get("norm_mode", "q99")
        if norm_mode == "min_max":
            ref_key_high, ref_key_low = "max", "min"
        else:
            ref_key_high, ref_key_low = "q99", "q01"
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats[ref_key_low], dtype=bool))
        action_high = np.array(action_norm_stats[ref_key_high])
        action_low = np.array(action_norm_stats[ref_key_low])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions
