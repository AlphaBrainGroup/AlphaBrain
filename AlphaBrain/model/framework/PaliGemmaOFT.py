# Copyright 2025 VLA-Engine. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
PaliGemmaOFT v2 — Multi-image aligned with QwenOFT

Key differences from v1:
  - Multi-view images are processed independently through SigLIP (not concatenated)
  - Each image gets its own vision token segment (like QwenOFT / Pi0)
  - Manual embedding assembly (image tokens + language tokens → Gemma LM)
  - Action tokens extracted from last hidden states for L1 regression

Architecture:
  [img1_tokens(256)] [img2_tokens(256)] [text_tokens] [action_tokens(chunk_len)]
       ↓                   ↓                 ↓              ↓
                    Gemma Language Model (shared attention)
                                                            ↓
                                                  MLP → action prediction
"""

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.modules.action_model.mlp_action_header import get_action_model
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY
from AlphaBrain.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def _load_paligemma_components(paligemma_cfg):
    """
    Load PaliGemma components: SigLIP vision tower + Gemma language model + projector.
    
    Loads from HF PaliGemmaForConditionalGeneration, then decomposes into parts.
    This ensures we get the correct pretrained weights.
    """
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor, AutoConfig

    model_id = paligemma_cfg.get("base_vlm") or os.path.join(
        os.environ.get("PRETRAINED_MODELS_DIR", "data/pretrained_models"),
        "paligemma-3b-pt-224",
    )
    
    attn_impl = paligemma_cfg.get("attn_implementation", "flash_attention_2")
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            attn_impl = "sdpa"
            logger.info("flash_attn not available, falling back to sdpa")

    # Load full PaliGemma
    full_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        attn_implementation=attn_impl,
    )
    logger.info(f"PaliGemma loaded with attn_implementation={attn_impl}")

    # Load processor (tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    # Extract components
    vision_tower = full_model.vision_tower        # SiglipVisionModel
    projector = full_model.multi_modal_projector   # Linear(1152 → 2048)
    language_model = full_model.language_model      # GemmaForCausalLM or GemmaModel
    # Handle both: GemmaForCausalLM.model.embed_tokens and GemmaModel.embed_tokens
    if hasattr(language_model, 'model') and hasattr(language_model.model, 'embed_tokens'):
        embed_tokens = language_model.model.embed_tokens
    elif hasattr(language_model, 'embed_tokens'):
        embed_tokens = language_model.embed_tokens
    else:
        raise AttributeError(f"Cannot find embed_tokens in language_model ({type(language_model).__name__})")

    config = full_model.config
    hidden_size = config.text_config.hidden_size  # 2048

    # Detach from full_model to avoid double memory
    del full_model

    return vision_tower, projector, language_model, embed_tokens, processor, hidden_size


@FRAMEWORK_REGISTRY.register("PaliGemmaOFT_v2")
class PaliGemmaOFT_v2(BaseFramework):
    """
    PaliGemmaOFT v2: Multi-image support aligned with QwenOFT.
    
    Each view gets independent vision tokens. Language model sees:
    [view1_tokens][view2_tokens]...[text_tokens][action_tokens]
    
    Action tokens (🔍) are appended to the text, and their hidden states
    are extracted for L1 regression via MLP action head.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config

        paligemma_cfg = config.framework.get("paligemma", {})

        # Load PaliGemma components
        (self.vision_tower, self.projector, self.language_model,
         self.embed_tokens, self.processor, self.hidden_size) = \
            _load_paligemma_components(paligemma_cfg)

        # Freeze vision tower's pooling head (not used, same as Pi0)
        if hasattr(self.vision_tower, 'head'):
            for p in self.vision_tower.head.parameters():
                p.requires_grad = False

        # Image processing config
        self.image_size = getattr(paligemma_cfg, 'image_size', 224)
        # SigLIP: 224x224 → 16x16 patches → 256 vision tokens per image
        self.num_vision_tokens_per_image = (self.image_size // 14) ** 2  # 256

        # Action model (MLP head)
        config.framework.action_model.action_hidden_dim = self.hidden_size
        self.action_model = get_action_model(config=config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Action token
        self.action_token = "🔍"
        action_token_ids = self.processor.tokenizer(
            self.action_token, add_special_tokens=False
        )["input_ids"]
        if len(action_token_ids) == 1:
            self.action_token_id = action_token_ids[0]
        else:
            self.action_token_id = action_token_ids
        logger.info(f"Action token '{self.action_token}' → id: {self.action_token_id}")

        # L1 loss
        self.l1_loss = nn.L1Loss()

        # Embedding scaling factor (matches HF PaliGemma / openpi)
        self.embed_scale = math.sqrt(self.hidden_size)

        logger.info(
            f"PaliGemmaOFT_v2 initialized: hidden={self.hidden_size}, "
            f"vision_tokens/img={self.num_vision_tokens_per_image}, "
            f"chunk_len={self.chunk_len}"
        )

    # ─── Image Processing ───────────────────────────────────────────

    def _preprocess_images(self, batch_images: List[List[Image.Image]]) -> List[List[torch.Tensor]]:
        """
        Process multi-view images through SigLIP image processor.
        
        Args:
            batch_images: [B, num_views] list of PIL images
            
        Returns:
            List[List[Tensor]]: [B, num_views] pixel_values tensors, each [1, 3, 224, 224]
        """
        batch_pixel_values = []
        for img_list in batch_images:
            if not isinstance(img_list, (list, tuple)):
                img_list = [img_list]
            views = []
            for img in img_list:
                # Use PaliGemma's image processor for proper normalization
                processed = self.processor.image_processor(
                    images=img, return_tensors="pt"
                )
                views.append(processed["pixel_values"])  # [1, 3, 224, 224]
            batch_pixel_values.append(views)
        return batch_pixel_values

    # ─── Embedding Assembly ─────────────────────────────────────────

    def _build_embeddings(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build combined embedding sequence: [vision_tokens...] [text_tokens] [action_tokens]
        
        Returns:
            embeddings: [B, total_seq_len, H]
            attention_mask: [B, total_seq_len]
            input_ids_for_action_gather: [B, total_seq_len] (with action token IDs at correct positions)
        """
        device = next(self.parameters()).device
        B = len(instructions)
        
        # 1. Process images → vision embeddings
        batch_pixel_values = self._preprocess_images(batch_images)
        
        # Determine max number of views
        num_views = [len(views) for views in batch_pixel_values]
        max_views = max(num_views)
        
        # 2. Encode each view through SigLIP + projector
        # batch_vision_embs[b][v] = [num_vision_tokens, H]
        batch_vision_embs = []
        for b in range(B):
            view_embs = []
            for v in range(len(batch_pixel_values[b])):
                pixel_values = batch_pixel_values[b][v].to(device)  # [1, 3, 224, 224]
                with torch.no_grad() if not self.vision_tower.training else torch.enable_grad():
                    vision_out = self.vision_tower(pixel_values=pixel_values)
                img_features = vision_out.last_hidden_state  # [1, 256, 1152]
                img_emb = self.projector(img_features)       # [1, 256, 2048]
                view_embs.append(img_emb.squeeze(0))         # [256, 2048]
            batch_vision_embs.append(view_embs)
        
        # 3. Tokenize text (instruction + action tokens)
        action_tokens_str = self.action_token * self.chunk_len
        prompts = [
            f"{inst} Please predict the next {self.chunk_len} robot actions: <action>{action_tokens_str}<action>."
            for inst in instructions
        ]
        
        tokenized = self.processor.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)
        text_input_ids = tokenized["input_ids"]       # [B, text_len]
        text_attention_mask = tokenized["attention_mask"]  # [B, text_len]
        
        # 4. Embed text tokens
        text_emb = self.embed_tokens(text_input_ids)  # [B, text_len, H]
        text_emb = text_emb * self.embed_scale         # Gemma embedding scaling
        
        # 5. Assemble: [vision_view1] [vision_view2] ... [text_tokens]
        all_embs = []
        all_masks = []
        all_ids = []  # For action token gathering
        
        vision_pad_token_id = 0  # Dummy ID for vision token positions
        
        for b in range(B):
            sample_embs = []
            sample_masks = []
            sample_ids = []
            
            # Vision tokens for each view
            for v in range(max_views):
                if v < len(batch_vision_embs[b]):
                    # Real view
                    sample_embs.append(batch_vision_embs[b][v])  # [256, H]
                    sample_masks.append(torch.ones(self.num_vision_tokens_per_image, device=device))
                    sample_ids.append(torch.full(
                        (self.num_vision_tokens_per_image,), vision_pad_token_id,
                        dtype=torch.long, device=device
                    ))
                else:
                    # Padded view (zeros)
                    sample_embs.append(torch.zeros(
                        self.num_vision_tokens_per_image, self.hidden_size, device=device
                    ))
                    sample_masks.append(torch.zeros(self.num_vision_tokens_per_image, device=device))
                    sample_ids.append(torch.full(
                        (self.num_vision_tokens_per_image,), vision_pad_token_id,
                        dtype=torch.long, device=device
                    ))
            
            # Text tokens
            sample_embs.append(text_emb[b])
            sample_masks.append(text_attention_mask[b].float())
            sample_ids.append(text_input_ids[b])
            
            all_embs.append(torch.cat(sample_embs, dim=0))
            all_masks.append(torch.cat(sample_masks, dim=0))
            all_ids.append(torch.cat(sample_ids, dim=0))
        
        # Pad to same length across batch
        max_len = max(e.shape[0] for e in all_embs)
        
        embeddings = torch.zeros(B, max_len, self.hidden_size, device=device)
        attention_mask = torch.zeros(B, max_len, device=device)
        input_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        
        for b in range(B):
            seq_len = all_embs[b].shape[0]
            # Right-align (left-pad) to match PaliGemma convention
            offset = max_len - seq_len
            embeddings[b, offset:] = all_embs[b]
            attention_mask[b, offset:] = all_masks[b]
            input_ids[b, offset:] = all_ids[b]
        
        return embeddings, attention_mask, input_ids

    # ─── Forward Pass ───────────────────────────────────────────────

    def _run_language_model(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run Gemma language model on pre-assembled embeddings.
        
        Handles both GemmaForCausalLM and GemmaModel (openpi-patched env).
        """
        # Get the actual GemmaModel (handles both CausalLM wrapper and direct Model)
        lm = self.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            # GemmaForCausalLM → use .model (GemmaModel)
            lm = lm.model

        outputs = lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id=None,
    ) -> torch.Tensor:
        """Extract action token embeddings — same logic as QwenOFT."""
        if action_token_id is None:
            raise ValueError("action_token_id cannot be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)

        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Action tokens insufficient for chunk_len {self.chunk_len}: "
                f"samples {insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values

        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        action_queries = last_hidden.gather(dim=1, index=expanded_index)
        return action_queries

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        """Training forward: multi-image → Gemma LM → action token L1 regression."""
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        # Build multi-image embeddings
        embeddings, attention_mask, input_ids = self._build_embeddings(
            batch_images, instructions
        )

        # Run through Gemma language model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self._run_language_model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
            )

        # Extract action tokens and predict
        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List = None,
        instructions: List[str] = None,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        """Inference: predict actions from images + instruction."""
        from deployment.model_server.tools.image_tools import to_pil_preserve
        from AlphaBrain.training.trainer_utils.trainer_tools import resize_images

        if examples is not None:
            batch_images = [to_pil_preserve(example["image"]) for example in examples]
            instructions = [example["lang"] for example in examples]
        else:
            batch_images = [to_pil_preserve(imgs) for imgs in batch_images]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        embeddings, attention_mask, input_ids = self._build_embeddings(
            batch_images, instructions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self._run_language_model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
            )

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
