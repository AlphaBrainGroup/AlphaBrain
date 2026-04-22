# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from AlphaBrain.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.modules.vlm import get_vlm_model
from AlphaBrain.model.modules.action_model.groot_action_header import get_action_model, FlowmatchingActionHead
from AlphaBrain.training.trainer_utils.trainer_tools import resize_images
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(BaseFramework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        # align dims: use world_model hidden_size (post-fusion) if available, else VLM hidden_size
        wm_cfg = getattr(self.config.framework, "world_model", None)
        if wm_cfg is not None and getattr(wm_cfg, "hidden_size", None) is not None:
            self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = wm_cfg.hidden_size
        else:
            self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Determine whether state was used during training (controls state_encoder usage)
        self.use_state = getattr(
            getattr(getattr(config, 'datasets', None), 'vla_data', None),
            'include_state', False
        )
        if self.use_state in ["False", False, None, "false", ""]:
            self.use_state = False
        else:
            self.use_state = True
        logger.info(f"[QwenGR00T] use_state={self.use_state} (from config.datasets.vla_data.include_state)")

        # Freeze state_encoder if state is not used (prevents DeepSpeed gradient deadlock)
        if not self.use_state and hasattr(self, 'action_model'):
            if hasattr(self.action_model, 'state_encoder') and self.action_model.state_encoder is not None:
                self.action_model.state_encoder.requires_grad_(False)
                logger.info("[QwenGR00T] Froze state_encoder (include_state=false)")

        # Video loss weight (hyperparameter, not a toggle).
        # WM backbones auto-enable video loss via has_visual_encoder check in forward().
        self._video_loss_weight = float(
            getattr(
                getattr(getattr(config, 'framework', None), 'world_model', None),
                'video_loss_weight', 1.0,
            )
        ) if (hasattr(config, 'framework') and hasattr(config.framework, 'world_model')) else 1.0
        logger.info("[QwenGR00T] video_loss_weight=%.3f", self._video_loss_weight)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """Run a full training forward pass, with video loss when next_image is available.

        When next_image is available, performs
        a SINGLE DiT forward that simultaneously yields action visual tokens and the
        next-frame video prediction loss.  Both share the same backward graph so the
        DiT backbone receives gradients from both losses without a redundant forward pass.

        During inference (no next_image): the standard encode path is used and no video
        loss is computed.
        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL]]
        instructions = [example["lang"] for example in examples]   # [B, str]
        actions = [example["action"] for example in examples]      # [B, T, action_dim]

        state = [example["state"] for example in examples] if (self.use_state and "state" in examples[0]) else None

        video_loss = None

        # ===================================================================
        # V2 video loss path (training with next_image)
        # Single DiT pass → layer 18 features for action + final output for video
        # ===================================================================
        has_next_images = (
            len(examples) > 0
            and "next_image" in examples[0]
            and "next_image" in examples[0]
        )
        has_visual_encoder = hasattr(self.qwen_vl_interface, "visual_encoder")

        if has_next_images and has_visual_encoder:
            wm_encoder = self.qwen_vl_interface.visual_encoder
            next_images_raw = [example.get("next_image") for example in examples]
            valid_mask = [img is not None for img in next_images_raw]

            if True:  # always go V2 path to avoid NCCL deadlock
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

                valid_count = sum(valid_mask)
                if valid_count == 0:
                    video_loss = video_loss_raw * 0.0  # no valid next_image, zero out video loss
                elif valid_count < len(valid_mask):
                    scale = len(valid_mask) / valid_count
                    video_loss = video_loss_raw * scale
                else:
                    video_loss = video_loss_raw

        if not (has_next_images and has_visual_encoder) or video_loss is None:
            # ===============================================================
            # Standard encode path (inference or training without video loss)
            # ===============================================================
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, instructions=instructions
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]
            video_loss = None

        # ===================================================================
        # Action Expert Forward and Loss
        # ===================================================================
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4)
                if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                expected_state_dim = getattr(
                    getattr(getattr(self.config, "framework", None), "action_model", None),
                    "state_dim", None,
                )
                if expected_state_dim and state.shape[-1] > expected_state_dim:
                    state = state[..., :expected_state_dim]
                if state.ndim == 2:
                    state = state.unsqueeze(1)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated
            )

        # ===================================================================
        # Combine losses and build output dict
        # ===================================================================
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
        return_predicted_frame: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Supports two input formats:
          - examples: List[dict] with keys "image", "lang", "state" (legacy format)
          - batch_images + instructions: direct arguments (consistent with NeuroVLA/QwenOFT)

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if examples is not None:
            # Legacy examples format
            if type(examples) is not list:
                examples = [examples]
            batch_images = [to_pil_preserve(example["image"]) for example in examples]
            instructions = [example["lang"] for example in examples]  # [B, str]
            state = [example["state"] for example in examples] if (self.use_state and "state" in examples[0]) else None
        else:
            # Direct batch_images/instructions format (from websocket client)
            assert batch_images is not None and instructions is not None, \
                "Either 'examples' or both 'batch_images' and 'instructions' must be provided"
            if isinstance(batch_images[0][0], np.ndarray):
                batch_images = [[Image.fromarray(img) for img in seq] for seq in batch_images]
            state = states if self.use_state else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None

        # Truncate state to match model's expected state_dim (e.g. LIBERO sends 8-dim, model expects 7-dim)
        if state is not None:
            expected_state_dim = getattr(
                getattr(getattr(self.config, 'framework', None), 'action_model', None),
                'state_dim', None
            )
            if expected_state_dim and state.shape[-1] > expected_state_dim:
                state = state[..., :expected_state_dim]
            # Ensure state is 3D (B, T, state_dim) for action model
            if state.ndim == 2:
                state = state.unsqueeze(1)

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        result = {"normalized_actions": normalized_actions}
        if return_predicted_frame and hasattr(self.qwen_vl_interface, "visual_encoder") and hasattr(self.qwen_vl_interface.visual_encoder, "denoise_future_frame"):
            try:
                predicted_frame = self.predict_future_frame(
                    batch_images=batch_images, instructions=instructions,
                    num_steps=5, sigma_min=0.002, sigma_max=80.0,
                )
                result["predicted_frame"] = predicted_frame
            except Exception as e:
                import logging
                logging.warning("predict_future_frame failed: %s", e)
        return result

    @torch.inference_mode()
    def predict_future_frame(
        self,
        batch_images: List = None,
        instructions: List[str] = None,
        num_steps: int = 5,
        sigma_min: float = 4.0,
        sigma_max: float = 80.0,
    ) -> np.ndarray:
        """Predict future frame using DiT denoising.

        Returns:
            future_frames: np.ndarray [B, H, W, 3] uint8 predicted future frames
        """
        if not hasattr(self.qwen_vl_interface, 'visual_encoder'):
            raise ValueError("predict_future_frame requires world model visual encoder")

        wm_encoder = self.qwen_vl_interface.visual_encoder

        # Preprocess images
        curr_images = [imgs[0] if isinstance(imgs, (list, tuple)) else imgs for imgs in batch_images]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            curr_pv = wm_encoder.preprocess(curr_images)

            # Get text embeddings
            text_embeds = wm_encoder.encode_text(instructions, curr_pv.device) if hasattr(wm_encoder, 'encode_text') else None

            # Encode current frame to latent
            latent_t = wm_encoder.encode_to_latent(curr_pv)

            # Denoise future frame
            future_latent = wm_encoder.denoise_future_frame(
                latent_t, text_embeds, num_steps=num_steps,
                sigma_min=sigma_min, sigma_max=sigma_max,
            )

            # Decode to pixels
            future_video = wm_encoder.decode_latent(future_latent)

        # Convert to uint8 numpy
        future_frames = ((future_video[:, :, 0] + 1) * 127.5).clamp(0, 255)
        future_frames = future_frames.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

        return future_frames



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./benchmarks/Robotwin/train/alphabrain_cotrain_robotwin.yaml", help="Path to YAML config")
    args, extra_cli_args = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    # cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = "./data/pretrained_models/Florence-2-large"


    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample, but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from AlphaBrain.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # forward model with dataloader
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # try get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model(batch)
        # break

    action = model.predict_action(examples=batch)
    print("Finished")
