"""VLA feature extraction for the RL Token encoder.

The RL Token encoder (Eq. 1 in the reference) expects the VLA's
final-layer token embeddings ``z_{1:M}``. Two consumption modes are
supported here:

* **Strict (``image_only=True``, default):** only the image-token
  positions are kept. This matches Fig. 2 of the reference
  ("image embeddings N × 2048") and the experimental choice described in
  footnote 1 (language embeddings dropped because each task has a fixed
  instruction).
* **General (``image_only=False``):** all non-padding input-token
  positions are kept, optionally excluding the action placeholders. This
  corresponds to the paper's more general claim in Sec. IV-A that "the
  construction applies to all VLA embeddings."

The production framework classes (e.g. ``Qwenvl_OFT``) only expose the
action-query slice via ``get_action_queries``; this helper runs the same
VLM forward pass and returns the full sequence (plus a mask) without
modifying the framework.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

# Qwen2.5-VL special token indices — mirrors
# ``AlphaBrain/model/modules/vlm/qwen2_5.py``.
QWEN_IMAGE_TOKEN_INDEX = 151655
QWEN_VIDEO_TOKEN_INDEX = 151656


@torch.no_grad()
def get_vla_hidden_states(
    vla,
    batch_images: List,
    instructions: List[str],
    image_only: bool = True,
    drop_action_tokens: bool = True,
    image_token_id: int = QWEN_IMAGE_TOKEN_INDEX,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the VLA's full last-layer token embeddings for a batch.

    Mirrors the forward-pass logic of ``Qwenvl_OFT.get_action_queries`` but
    yields the *entire* last-layer hidden state sequence rather than the
    action-token slice, so the RL Token encoder can consume ``z_{1:M}``
    exactly as in Eq. 1 of the reference.

    Args:
        vla: a loaded ``Qwenvl_OFT`` (or compatible) framework instance;
            only its public attributes are used (``qwen_vl_interface``,
            ``action_token``, ``action_token_id``, ``chunk_len``,
            ``config``).
        batch_images: list of multi-view images per sample — same shape
            the framework accepts in ``predict_action``.
        instructions: list of language instructions per sample (no action
            prompt suffix — this helper adds it, matching the framework's
            own wrapping).
        image_only: when True (the strict-reference default), the returned
            ``attention_mask`` is 1 **only** at image-token positions
            (matches Fig. 2 and footnote 1). In this mode
            ``drop_action_tokens`` is irrelevant because action placeholder
            positions are also excluded.
        drop_action_tokens: only used when ``image_only=False``. When True
            the mask additionally zeros out action placeholder positions
            from the VLA's own input sequence.
        image_token_id: the input-id value that marks an image patch token
            in ``input_ids``. Defaults to Qwen2.5-VL's 151655.

    Returns:
        last_hidden: ``(B, L, H)`` final-layer VLM embeddings.
        attention_mask: ``(B, L)`` int mask — 1 for positions to include
            in downstream attention, 0 for everything else (padding,
            language, action placeholders, BOS/EOS, etc., depending on
            mode).
        action_mask: ``(B, L)`` bool — True at action-token positions.
            Exposed so callers can, for example, keep action tokens but
            weight their loss differently.
    """
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from AlphaBrain.training.trainer_utils.trainer_tools import resize_images

    batch_images = [to_pil_preserve(imgs) for imgs in batch_images]

    train_obs_image_size = getattr(vla.config.datasets.vla_data, "image_size", None)
    if train_obs_image_size:
        batch_images = resize_images(batch_images, target_size=train_obs_image_size)

    action_tokens = vla.action_token * vla.chunk_len
    prompt_suffix = (
        f" Please predict the next {vla.chunk_len} robot actions:"
        f" <action>{action_tokens}<action>."
    )
    instructions = [inst + prompt_suffix for inst in instructions]

    qwen_inputs = vla.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images, instructions=instructions
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        qwen_outputs = vla.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = qwen_outputs.hidden_states[-1]  # (B, L, H)

    input_ids = qwen_inputs.get("input_ids")
    attention_mask = qwen_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    action_mask = input_ids == vla.action_token_id  # (B, L) bool

    if image_only:
        # Strict-reference mode (Fig. 2 / footnote 1): keep only image tokens.
        image_mask = input_ids == image_token_id
        keep = attention_mask.bool() & image_mask
        attention_mask = keep.to(attention_mask.dtype)
    elif drop_action_tokens:
        keep = attention_mask.bool() & (~action_mask)
        attention_mask = keep.to(attention_mask.dtype)

    return last_hidden, attention_mask, action_mask


def pad_mask_from_attention(attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert an HF-style attention_mask (1=keep, 0=pad) to a
    ``key_padding_mask`` for ``nn.TransformerEncoder`` / ``nn.TransformerDecoder``
    (True = ignore).
    """
    return attention_mask == 0
