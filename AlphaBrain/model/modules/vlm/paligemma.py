"""
PaliGemma VLM Interface for VLA-Engine (Pi0/Pi0.5)

Manually assembles PaliGemma from SiglipVisionModel + GemmaForCausalLM + Linear projector,
avoiding the broken PaliGemmaForConditionalGeneration import in some transformers versions.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from transformers import SiglipVisionModel, SiglipVisionConfig
from transformers import GemmaForCausalLM
from transformers.models.auto import CONFIG_MAPPING

import logging
logger = logging.getLogger(__name__)


class PaliGemmaVLM(nn.Module):
    """
    Manual PaliGemma assembly: SigLIP vision + Gemma language + linear projector.
    
    Equivalent to PaliGemmaForConditionalGeneration but without the import issues.
    """
    
    def __init__(self, gemma_config_hf, vision_config_hf=None):
        super().__init__()
        
        # Vision tower (SigLIP)
        if vision_config_hf is None:
            vision_config_hf = SiglipVisionConfig(
                hidden_size=1152,
                intermediate_size=4304,
                num_hidden_layers=27,
                num_attention_heads=16,
                patch_size=14,
                image_size=224,
                projection_dim=2048,
            )
        self.vision_tower = SiglipVisionModel(vision_config_hf)
        
        # Multi-modal projector (vision → language dim)
        self.multi_modal_projector = nn.Linear(
            vision_config_hf.hidden_size, gemma_config_hf.hidden_size
        )
        
        # Language model (Gemma)
        self.language_model = GemmaForCausalLM(gemma_config_hf).model  # .model = GemmaModel
        
        # LM head (tied with embeddings typically)
        self.lm_head = nn.Linear(gemma_config_hf.hidden_size, gemma_config_hf.vocab_size, bias=False)
        
        # Embed tokens shortcut
        self.embed_tokens = self.language_model.embed_tokens

    def get_image_features(self, pixel_values):
        """SigLIP vision encoding → projected to language dim."""
        vision_outputs = self.vision_tower(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state
        return self.multi_modal_projector(image_features)


class _PaliGemma_VL_Interface(nn.Module):
    """
    PaliGemma VLM interface for VLA-Engine.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config
        paligemma_cfg = config.framework.paligemma

        # Build Gemma config
        attn_impl = getattr(paligemma_cfg, 'attn_implementation', 'flash_attention_2')
        if attn_impl == 'flash_attention_2':
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                attn_impl = 'sdpa'
                logger.info("flash_attn not available, falling back to sdpa")
        gemma_config = CONFIG_MAPPING["gemma"](
            hidden_size=getattr(paligemma_cfg, 'width', 2048),
            intermediate_size=getattr(paligemma_cfg, 'mlp_dim', 16384),
            num_attention_heads=getattr(paligemma_cfg, 'num_heads', 8),
            head_dim=getattr(paligemma_cfg, 'head_dim', 256),
            num_hidden_layers=getattr(paligemma_cfg, 'depth', 18),
            num_key_value_heads=getattr(paligemma_cfg, 'num_kv_heads', 1),
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            attn_implementation=attn_impl,
        )
        logger.info(f"PaliGemma VLM: attn_implementation={attn_impl}")

        self.model = PaliGemmaVLM(gemma_config)
        self.hidden_size = gemma_config.hidden_size
        self.processor = None  # Tokenizer loaded separately

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(image)

    def embed_language(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.model.embed_tokens(tokens)
        # openpi's embed_prefix does: lang_emb * math.sqrt(lang_emb_dim)
        # This matches HF PaliGemma's Gemma embedding scaling.
        return emb * math.sqrt(emb.shape[-1])

    def encode_prefix(
        self,
        images: List[torch.Tensor],
        image_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs, pad_masks, att_mask_list = [], [], []

        for img, img_mask in zip(images, image_masks):
            img_emb = self.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_mask_list += [0] * num_img_embs

        lang_emb = self.embed_language(lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_mask_list += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_mask_list, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], -1)
        return embs, pad_masks, att_masks

    def get_language_model(self):
        return self.model.language_model
