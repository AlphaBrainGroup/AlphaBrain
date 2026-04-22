"""
PaliGemma VLM Interface for OFT (action-token regression).

Uses HF PaliGemmaForConditionalGeneration directly (not the manual assembly in paligemma.py
which is designed for Pi0's flow-matching head).

Mirrors the _Llama_VL_Interface / _QWen_VL_Interface pattern:
  - build_paligemma_inputs()  → tokenize images + text → BatchFeature
  - forward()                 → run model, return hidden_states
"""

import torch
import torch.nn as nn
from typing import Optional, List
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor, AutoConfig

import logging
logger = logging.getLogger(__name__)

_ACTION_TOKEN = "🔍"


class _PaliGemma_OFT_VL_Interface(nn.Module):
    """PaliGemma VLM interface for OFT-style action token regression."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config

        paligemma_cfg = config.framework.get("paligemma", {})
        model_id = paligemma_cfg.get("base_vlm", "/datasets/peligemma")

        use_meta_device = paligemma_cfg.get("_meta_device_init", False)
        if use_meta_device:
            hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            with torch.device("meta"):
                model = PaliGemmaForConditionalGeneration(hf_config)
            model = model.to_empty(device="cpu")
            # Rebuild non-persistent buffers (inv_freq etc.)
            for name, module in model.named_modules():
                if hasattr(module, 'inv_freq'):
                    if hasattr(module, 'rope_init_fn') and hasattr(module, 'config'):
                        inv_freq, attention_scaling = module.rope_init_fn(module.config, device='cpu')
                        module.inv_freq = inv_freq
                        module.original_inv_freq = inv_freq
                        module.attention_scaling = attention_scaling
                    elif module.inv_freq is not None:
                        dim = module.inv_freq.shape[0] * 2
                        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
                        module.inv_freq = inv_freq
            logger.info(f"[meta_device_init] Created PaliGemma model via meta device, model_id={model_id}")
        else:
            attn_impl = paligemma_cfg.get("attn_implementation", "flash_attention_2")
            # Auto-fallback: if flash_attention_2 requested but flash_attn not installed, use sdpa
            if attn_impl == "flash_attention_2":
                try:
                    import flash_attn  # noqa: F401
                except ImportError:
                    attn_impl = "sdpa"
                    logger.info("flash_attn not available, falling back to sdpa")
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",
                attn_implementation=attn_impl,
            )
            logger.info(f"PaliGemma loaded with attn_implementation={attn_impl}")

        processor = AutoProcessor.from_pretrained(model_id)
        # PaliGemma processor uses suffix for the text prompt
        # Padding side left for batch inference consistency
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor

        # Action token
        self._action_token = _ACTION_TOKEN
        action_token_ids = self.processor.tokenizer(_ACTION_TOKEN, add_special_tokens=False)["input_ids"]
        self._action_token_id = action_token_ids[0] if len(action_token_ids) == 1 else action_token_ids
        logger.info(f"PaliGemma action token '{_ACTION_TOKEN}' -> id(s): {self._action_token_id}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        token_type_ids=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        return outputs

    @staticmethod
    def _concat_images_horizontal(img_list):
        """Concatenate multiple PIL images horizontally into one."""
        from PIL import Image
        if len(img_list) == 1:
            return img_list[0]
        # Resize all to same height, then concat
        target_h = min(img.size[1] for img in img_list)
        resized = []
        for img in img_list:
            if img.size[1] != target_h:
                ratio = target_h / img.size[1]
                new_w = int(img.size[0] * ratio)
                img = img.resize((new_w, target_h), Image.BILINEAR)
            resized.append(img)
        total_w = sum(img.size[0] for img in resized)
        merged = Image.new("RGB", (total_w, target_h))
        x_offset = 0
        for img in resized:
            merged.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        return merged

    def build_paligemma_inputs(self, images, instructions, **kwargs):
        """
        Build batched inputs for PaliGemma with multi-image support.

        PaliGemma natively supports single image. For multi-view inputs,
        images are concatenated horizontally into one composite image.

        Parameters:
            images: List[List[PIL.Image]] - [B, num_views]
            instructions: List[str] - [B] text prompts (including action tokens)

        Returns:
            BatchFeature with input_ids, attention_mask, pixel_values, token_type_ids
        """
        assert len(images) == len(instructions)

        processed_images = []
        for img_list in images:
            if isinstance(img_list, (list, tuple)) and len(img_list) > 1:
                # Multi-view: concatenate horizontally
                processed_images.append(self._concat_images_horizontal(img_list))
            elif isinstance(img_list, (list, tuple)):
                processed_images.append(img_list[0])
            else:
                processed_images.append(img_list)

        # PaliGemma processor: images + text (suffix)
        batch_input = self.processor(
            text=instructions,
            images=processed_images,
            padding=True,
            return_tensors="pt",
        )

        return batch_input
