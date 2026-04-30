# Copyright 2025 VLA-Engine. All rights reserved.
# Llama 3.2 Vision interface for VLA-Engine

import os

import torch
import torch.nn as nn
from typing import Optional, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoConfig
from transformers import BatchFeature

import logging

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Llama 3.2 Vision action token range — will be set after tokenizer adds special tokens
_ACTION_TOKEN = "🔍"

# Llama 3.2 Vision chat template (base model doesn't include one)
_LLAMA3_VISION_CHAT_TEMPLATE = """{{- bos_token }}
{%- for message in messages %}
{%- if message.role == 'user' %}
<|start_header_id|>user<|end_header_id|>

{%- for item in message.content %}
{%- if item.type == 'image' %}
<|image|>
{%- elif item.type == 'text' %}
{{ item.text }}
{%- endif %}
{%- endfor %}<|eot_id|>
{%- elif message.role == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>

{%- for item in message.content %}
{%- if item.type == 'text' %}
{{ item.text }}
{%- endif %}
{%- endfor %}<|eot_id|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>

{%- endif %}"""



class _Llama_VL_Interface(nn.Module):
    """
    Lightweight wrapper around Llama 3.2 Vision (MllamaForConditionalGeneration).
    Mirrors _QWen_VL_Interface but for Llama 3.2 Vision.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        llamavl_config = config.framework.get("llamavl", {})
        model_id = llamavl_config.get("base_vlm") or os.path.join(
            os.environ.get("PRETRAINED_MODELS_DIR", "data/pretrained_models"),
            "Llama-3.2-11B-Vision-Instruct",
        )

        use_meta_device = llamavl_config.get("_meta_device_init", False)
        if use_meta_device:
            llama_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            with torch.device("meta"):
                model = MllamaForConditionalGeneration(llama_config)
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
            print(f"[meta_device_init] Created Llama 3.2 Vision model via meta device, model_id={model_id}")
        else:
            attn_impl = getattr(llamavl_config, 'attn_implementation', 'sdpa')
            model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",
                attn_implementation=attn_impl,
            )
            logger.info(f"Llama loaded with attn_implementation={attn_impl}")

        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        # Set chat template if not present
        if not getattr(processor, "chat_template", None):
            processor.chat_template = _LLAMA3_VISION_CHAT_TEMPLATE

        self.model = model
        self.processor = processor
        self.config = config

        # Get the action token id
        self._action_token = _ACTION_TOKEN
        action_token_ids = self.processor.tokenizer(_ACTION_TOKEN, add_special_tokens=False)["input_ids"]
        self._action_token_id = action_token_ids[0] if len(action_token_ids) == 1 else action_token_ids
        logger.info(f"Llama action token '{_ACTION_TOKEN}' -> id(s): {self._action_token_id}")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                cross_attention_mask=cross_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        return outputs

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(**kwargs)
        return generation_output

    def build_llama_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Construct and tokenize multimodal chat-style inputs for Llama 3.2 Vision (batched).

        Parameters:
            images: List[List[PIL.Image.Image]] - [B, num_images]
            instructions: List[str] - [B]
            solutions: Optional[List[str]] - [B]

        Returns:
            BatchFeature with input_ids, attention_mask, pixel_values, 
            aspect_ratio_ids, aspect_ratio_mask, cross_attention_mask, etc.
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        messages_list = []
        all_images = []  # nested: [[imgs_for_sample_0], [imgs_for_sample_1], ...]

        for imgs, instruction in zip(images, instructions):
            # Build chat message with image tokens
            content = []
            sample_images = []
            for img in imgs:
                content.append({"type": "image"})
                sample_images.append(img)
            all_images.append(sample_images)

            if "CoT_prompt" in self.config.datasets.vla_data:
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages_list)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})

            messages_list.append(msg)

        # Apply chat template to get text prompts
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]

        # Process with the Llama processor
        if all_images:
            batch_input = self.processor(
                text=texts,
                images=all_images,
                padding=True,
                return_tensors="pt",
            )
        else:
            batch_input = self.processor(
                text=texts,
                padding=True,
                return_tensors="pt",
            )

        # Build labels if solutions provided (mask non-action tokens)
        if solutions is not None:
            action_token_id = self._action_token_id
            labels = batch_input['input_ids'].clone()
            for i in range(labels.size(0)):
                seq = labels[i]
                if isinstance(action_token_id, (list, tuple)):
                    mask_seq = torch.zeros_like(seq, dtype=torch.bool)
                    for tid in action_token_id:
                        mask_seq |= (seq == tid)
                else:
                    mask_seq = (seq == action_token_id)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    seq[:] = IGNORE_INDEX

            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_input['labels'] = labels

        return batch_input.to(self.model.device)
