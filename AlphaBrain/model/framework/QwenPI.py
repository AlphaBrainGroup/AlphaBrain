# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].

"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0

Extended (2026-04): World model backbone support (V-JEPA, Cosmos, Wan).
When a world model VLM is used, ``forward_all_layers()`` extracts per-backbone-
block features so each PI action head layer cross-attends to a DIFFERENT
backbone layer (true layerwise cross-attention, no replication).
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from types import SimpleNamespace



from AlphaBrain.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.modules.vlm import get_vlm_model
try:
    from AlphaBrain.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model, LayerwiseFlowmatchingActionHead
except ImportError:
    get_action_model = None
    LayerwiseFlowmatchingActionHead = None
from AlphaBrain.training.trainer_utils.trainer_tools import resize_images
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY

####################################################
# Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################


def _is_world_model_vlm(vlm_interface) -> bool:
    """Check whether the VLM interface is a WorldModelVLMInterface instance.

    Uses class-name matching so we do not need to import the world model module
    at the top level (it may pull in heavy dependencies).
    """
    cls_names = [c.__name__ for c in type(vlm_interface).__mro__]
    return "WorldModelVLMInterface" in cls_names


@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(BaseFramework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise cross DiT diffusion head

    World model mode:
      When the VLM is a world model (Cosmos, Wan), per-DiT-block features are
      extracted via ``forward_all_layers()`` so each PI action head layer
      cross-attends to a DIFFERENT backbone layer.
      ``framework.qwenvl.num_vl_layers`` must match the backbone block count
      (28 for Cosmos, 30 for Wan).

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # ----- Detect VLM backend type -----
        self._world_model_mode = _is_world_model_vlm(self.qwen_vl_interface)

        if self._world_model_mode:
            llm_hidden_size = self.qwen_vl_interface.model.config.hidden_size
            num_vl_layers = getattr(config.framework.qwenvl, "num_vl_layers", 28)
            _backend = "unknown"
            if hasattr(self.qwen_vl_interface, "wm_config"):
                _backend = getattr(self.qwen_vl_interface.wm_config, "backend", "unknown")
            logger.info(
                "[QwenPI] World model mode: backend=%s, hidden_size=%d, "
                "num_vl_layers(backbone blocks for layerwise XAttn)=%d",
                _backend, llm_hidden_size, num_vl_layers,
            )
        else:
            # Standard Qwen2.5-VL path (original behaviour)
            num_vl_layers, llm_hidden_size = 36, self.qwen_vl_interface.model.config.hidden_size

        # Ensure qwenvl namespace exists for action model config (LayerwiseFM reads from it)
        if not hasattr(self.config.framework, 'qwenvl'):
            from omegaconf import OmegaConf, DictConfig
            # Handle AccessTrackedConfig wrapper
            fw_cfg = self.config.framework
            if hasattr(fw_cfg, '_cfg') and isinstance(fw_cfg._cfg, DictConfig):
                fw_cfg._cfg.qwenvl = OmegaConf.create({"vl_hidden_dim": llm_hidden_size, "num_vl_layers": num_vl_layers})
                fw_cfg._children.pop('qwenvl', None)  # clear cached child
            else:
                fw_cfg.qwenvl = OmegaConf.create({"vl_hidden_dim": llm_hidden_size, "num_vl_layers": num_vl_layers})
        else:
            self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
            self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Determine whether state was used during training
        self.use_state = getattr(
            getattr(getattr(config, "datasets", None), "vla_data", None),
            "include_state", False,
        )
        if self.use_state in ["False", False, None, "false", ""]:
            self.use_state = False
        else:
            self.use_state = True
        logger.info("[QwenPI] use_state=%s", self.use_state)

        # Video loss is always enabled (unified mode)
        self._video_loss_weight = float(
            getattr(
                getattr(getattr(config, 'framework', None), 'world_model', None),
                'video_loss_weight', 1.0,
            )
        ) if (hasattr(config, 'framework') and hasattr(config.framework, 'world_model')) else 1.0
        logger.info("[QwenPI] video_loss_weight=%.3f", self._video_loss_weight)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_vlm_inputs(self, batch_images, instructions):
        return self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

    def _extract_vl_embs(self, qwenvl_outputs):
        """Extract per-layer VL embeddings from the VLM output.

        For Qwen2.5-VL the output already contains one hidden state per
        transformer layer.  For world models in forward_all_layers mode,
        hidden_states is a tuple with one tensor per backbone DiT block so
        each PI action head layer cross-attends to a DIFFERENT backbone layer.

        Returns:
            (vl_embs_list, base_hidden)
        """
        all_hidden = qwenvl_outputs.hidden_states
        expected_layers = len(self.action_model.model.transformer_blocks)

        if self._world_model_mode:
            if len(all_hidden) > 1:
                # Per-layer features from forward_all_layers() (correct PI mode).
                # Resample/truncate to match the action model DiT depth.
                if len(all_hidden) >= expected_layers:
                    # Take last expected_layers (deepest backbone layers)
                    vl_embs_list = list(all_hidden[-expected_layers:])
                else:
                    # Backbone has fewer blocks than DiT layers — replicate last
                    vl_embs_list = list(all_hidden) + [all_hidden[-1]] * (expected_layers - len(all_hidden))
                base_hidden = vl_embs_list[-1]
            else:
                # Fallback: single fused output (backward compat), replicate
                fused = all_hidden[-1]  # [B, L, H]
                vl_embs_list = [fused] * expected_layers
                base_hidden = fused
        else:
            # Qwen2.5-VL: take the last N layers matching DiT depth.
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

        return vl_embs_list, base_hidden

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B,[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 7]

        state = [example["state"] for example in examples] if (self.use_state and "state" in examples[0]) else None

        video_loss = None

        # ===================================================================
        # Video loss path (training)
        # ===================================================================
        has_next_images = (
            len(examples) > 0
            and "next_image" in examples[0]
            and examples[0]["next_image"] is not None
        )
        has_visual_encoder = hasattr(self.qwen_vl_interface, "visual_encoder")

        if has_next_images and has_visual_encoder:
            wm_encoder = self.qwen_vl_interface.visual_encoder
            next_images_raw = [example.get("next_image") for example in examples]
            valid_mask = [img is not None for img in next_images_raw]

            if any(valid_mask):
                curr_images_flat = [
                    imgs[0] if isinstance(imgs, (list, tuple)) else imgs
                    for imgs in batch_images
                ]
                dummy_next = [
                    next_images_raw[i] if valid_mask[i] else curr_images_flat[i]
                    for i in range(len(examples))
                ]

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    curr_pv = wm_encoder.preprocess(curr_images_flat)
                    next_pv = wm_encoder.preprocess(dummy_next)
                    qwenvl_outputs, video_loss_raw = self.qwen_vl_interface.forward_with_video_loss(
                        curr_pv, instructions, next_pv
                    )
                    last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

                    # Single-layer features from video loss path; replicate for all PI DiT layers
                    expected_layers = len(self.action_model.model.transformer_blocks)
                    vl_embs_list = [last_hidden] * expected_layers
                    base_hidden = last_hidden

                if not all(valid_mask):
                    valid_count = sum(valid_mask)
                    scale = len(valid_mask) / max(valid_count, 1)
                    video_loss = video_loss_raw * scale
                else:
                    video_loss = video_loss_raw
            else:
                pass  # fall through to inference path

        if not (has_next_images and has_visual_encoder):
            # ===============================================================
            # Standard encode (inference / no next_image)
            # ===============================================================
            # Step 1: QWenVL input format
            qwen_inputs = self._build_vlm_inputs(batch_images, instructions)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self._world_model_mode and hasattr(self.qwen_vl_interface, 'forward_all_layers'):
                    # World model: extract per-backbone-block features for layerwise XAttn
                    qwenvl_outputs = self.qwen_vl_interface.forward_all_layers(
                        qwen_inputs["pixel_values"], instructions
                    )
                else:
                    qwenvl_outputs = self.qwen_vl_interface(
                        **qwen_inputs,
                        output_attentions=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                vl_embs_list, base_hidden = self._extract_vl_embs(qwenvl_outputs)
            video_loss = None

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            repeated_diffusion_steps = 2  # NO repeat for big action FM
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                # Truncate state to match model's expected state_dim (e.g. LIBERO sends 8-dim, model expects 7-dim)
                expected_state_dim = getattr(
                    getattr(getattr(self.config, 'framework', None), 'action_model', None),
                    'state_dim', None
                )
                if expected_state_dim and state.shape[-1] > expected_state_dim:
                    state = state[..., :expected_state_dim]
                # Ensure state is 3D (B, T, state_dim) for action model
                if state.ndim == 2:
                    state = state.unsqueeze(1)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)

        result = {"action_loss": action_loss}
        if video_loss is not None:
            result["video_loss"] = video_loss
            result["total_loss"] = action_loss + self._video_loss_weight * video_loss
        return result

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        batch_images: List = None,
        instructions: List[str] = None,
        states=None,
        **kwargs,
    ) -> np.ndarray:
        """
        Inference: single forward pass to regress future actions via flow-matching
        sampling through the layerwise DiT.

        Supports two input formats:
          - examples: List[dict] with keys "image", "lang", "state" (legacy)
          - batch_images + instructions: direct arguments

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim].
        """
        if examples is not None:
            if type(examples) is not list:
                examples = [examples]
            from deployment.model_server.tools.image_tools import to_pil_preserve
            batch_images = [to_pil_preserve(example["image"]) for example in examples]
            instructions = [example["lang"] for example in examples]
            state = [example["state"] for example in examples] if (self.use_state and "state" in examples[0]) else None
        else:
            assert batch_images is not None and instructions is not None, \
                "Either examples or both batch_images and instructions must be provided"
            if isinstance(batch_images[0][0], np.ndarray):
                batch_images = [[Image.fromarray(img) for img in seq] for seq in batch_images]
            state = states if self.use_state else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self._build_vlm_inputs(batch_images, instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self._world_model_mode and hasattr(self.qwen_vl_interface, 'forward_all_layers'):
                # World model: extract per-backbone-block features for layerwise XAttn
                qwenvl_outputs = self.qwen_vl_interface.forward_all_layers(
                    qwen_inputs["pixel_values"], instructions
                )
            else:
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            vl_embs_list, base_hidden = self._extract_vl_embs(qwenvl_outputs)

        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            # Truncate state to match model expected state_dim
            expected_state_dim = getattr(
                getattr(getattr(self.config, "framework", None), "action_model", None),
                "state_dim", None,
            )
            if expected_state_dim and state_tensor.shape[-1] > expected_state_dim:
                state_tensor = state_tensor[..., :expected_state_dim]
            if state_tensor.ndim == 2:
                state_tensor = state_tensor.unsqueeze(1)

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state_tensor)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./configs/train_recipes/QwenFM_OXE.yaml", help="Path to YAML config")
    args, extra_cli_args = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./data/pretrained_models/Qwen3-VL-4B-Instruct"

    model = Qwen_PI(cfg)
    print(model)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }

    batch = [sample, sample]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")
