"""Pi05 (PaliGemmaPi05) inference adapter for Phase-2 RL rollout.

Phase-2 RL needs ``(rl_token, vla_action_chunk)`` per step. For Pi05 the
proven-working inference path is ``vla.predict_action`` (achieves 84% SR on
LIBERO standalone for the openpi 5traj ckpt). We mirror that path's prefix
construction byte-for-byte so the action expert sees the same KV cache it
was tuned to consume — and we share that single prefix forward with the
rl_token extraction (image-only slice → encoder).

Why we don't reuse ``_prepare_prefix_paligemma`` (which is what
``forward()`` calls during training): in this codebase Pi05 is finetuned
from openpi base weights, and the proven-working inference uses an
**inline openpi-style prefix** (3 image slots with -1.0 dummy padding for
absent views, 200-token language padding, BOS+text+newline). The training
``_prepare_prefix_paligemma`` instead uses config-driven settings
(default 48-token padding, no view padding); the model's diffusion
empirically only works on the openpi-style prefix at inference. Using the
training builder gives correct shapes but the wrong KV cache distribution
for the action expert → rollout SR=0.

Phase-1 encoder pretrain used ``_prepare_prefix_paligemma`` and so it was
trained on a shorter, no-dummy prefix. We bridge the gap by: building the
openpi-style prefix here, but in the image-only mask we keep ONLY the
real-view image positions (filter dummy via ``prefix_pad_masks``). The
encoder then sees the same V × 256 real image tokens it saw at Phase-1.
"""

from __future__ import annotations

import math as _math
from typing import List, Optional, Tuple

import torch


# SigLIP-So400m at 224x224 with patch 14 → (224/14)^2 = 256 image embeds per view.
_PALIGEMMA_IMG_TOKENS_PER_VIEW = 256
# openpi-style language token padding length. Mirrors PaliGemmaPi05.py:677
# (`_PREDICT_MAX_LEN = 200`).
_PALIGEMMA_LANG_PAD_LEN = 200
# openpi-style image-slot padding count. Mirrors PaliGemmaPi05.py:711
# (`while len(img_tensors) < 3`).
_PALIGEMMA_NUM_IMG_SLOTS = 3
# Sentinel for HF-style additive 4D attention mask.
_NEG_INF_ADDITIVE = -2.3819763e38


def _build_openpi_prefix(vla, batch_images_normed, instructions):
    """Replicate PaliGemmaPi05._predict_action's inline openpi-style prefix.

    Mirrors lines 666-737 of PaliGemmaPi05.py exactly so the action expert
    sees identical KV cache content. Returns
    ``(prefix_embs, prefix_pad_masks, prefix_att_masks, num_real_views_per_sample)``.
    """
    import torchvision.transforms.functional as TF

    device = next(vla.parameters()).device
    dtype = next(vla.parameters()).dtype

    if vla._tokenizer is None and not hasattr(vla, "_hf_tokenizer"):
        vla._init_tokenizer()

    def _tokenize(text):
        cleaned = str(text).strip().replace("_", " ").replace("\n", " ")
        if hasattr(vla, "_hf_tokenizer") and vla._hf_tokenizer is not None:
            bos_id = vla._hf_tokenizer.bos_token_id
            text_ids = vla._hf_tokenizer.encode(cleaned, add_special_tokens=False)
            newline_ids = vla._hf_tokenizer.encode("\n", add_special_tokens=False)
            ids = ([bos_id] if bos_id is not None else []) + text_ids + newline_ids
        else:
            ids = vla._tokenizer.encode(cleaned, add_bos=True) + vla._tokenizer.encode("\n")
        max_len = _PALIGEMMA_LANG_PAD_LEN
        tokens_len = len(ids)
        if tokens_len < max_len:
            mask = [True] * tokens_len + [False] * (max_len - tokens_len)
            ids = ids + [0] * (max_len - tokens_len)
        else:
            ids = ids[:max_len]
            mask = [True] * max_len
        return ids, mask

    def _proc_img(im):
        """uint8 HWC → float CHW resized to 224 → [-1, 1] normalized."""
        if isinstance(im, torch.Tensor):
            t = im.float()
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
        else:
            import numpy as _np
            t = torch.from_numpy(_np.asarray(im).copy()).float()
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
        if t.max() > 1.0:
            t = t / 255.0
        t = TF.resize(t, [224, 224], antialias=True)
        t = TF.normalize(t, mean=[0.5] * 3, std=[0.5] * 3)
        return t

    bsize = len(batch_images_normed)
    num_real_views = len(batch_images_normed[0])  # uniform per batch

    # Build per-slot image tensors (B, 3, 224, 224) for slot 0..num_slots-1
    slot_tensors = []  # length = _PALIGEMMA_NUM_IMG_SLOTS, each (B, 3, 224, 224)
    slot_masks = []    # length = _PALIGEMMA_NUM_IMG_SLOTS, each (B,) bool
    for slot in range(_PALIGEMMA_NUM_IMG_SLOTS):
        per_sample = []
        per_mask = []
        for b in range(bsize):
            views = batch_images_normed[b]
            if slot < len(views):
                per_sample.append(_proc_img(views[slot]))
                per_mask.append(True)
            else:
                per_sample.append(torch.full((3, 224, 224), -1.0))
                per_mask.append(False)
        slot_tensors.append(torch.stack(per_sample).to(device=device, dtype=dtype))
        slot_masks.append(torch.tensor(per_mask, device=device, dtype=torch.bool))

    # Tokenize each instruction
    all_ids, all_masks = [], []
    for inst in instructions:
        ids, mask = _tokenize(inst)
        all_ids.append(ids)
        all_masks.append(mask)
    tokens_t = torch.tensor(all_ids, dtype=torch.long, device=device)
    token_masks_t = torch.tensor(all_masks, dtype=torch.bool, device=device)

    # Concatenate slot embeddings + lang embeddings
    embs_list, pad_list, att_list = [], [], []
    for img_t, img_m in zip(slot_tensors, slot_masks):
        img_emb = vla.vlm_interface.model.get_image_features(img_t)  # (B, 256, H)
        n_embs = img_emb.shape[1]
        embs_list.append(img_emb)
        pad_list.append(img_m[:, None].expand(bsize, n_embs))
        att_list += [0] * n_embs

    lang_emb = vla.vlm_interface.model.embed_tokens(tokens_t)
    lang_emb = lang_emb * _math.sqrt(lang_emb.shape[-1])  # Gemma scaling
    embs_list.append(lang_emb)
    pad_list.append(token_masks_t)
    att_list += [0] * lang_emb.shape[1]

    prefix_embs = torch.cat(embs_list, dim=1)               # (B, L, H)
    prefix_pad_masks = torch.cat(pad_list, dim=1)           # (B, L) bool
    att_t = torch.tensor(att_list, dtype=torch.bool, device=device)
    prefix_att_masks = att_t[None, :].expand(bsize, -1)     # (B, L) bool

    return prefix_embs, prefix_pad_masks, prefix_att_masks, num_real_views


@torch.no_grad()
def get_pi05_rl_state_and_action(
    vla,
    encoder,                       # RLTokenEncoderDecoder, frozen
    batch_images: List,            # list of [view1, view2] per sample
    instructions: List[str],
    batch_props: Optional[torch.Tensor] = None,  # (B, prop_dim) on device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One Pi05 prefix forward → rl_token via encoder + action via diffusion.

    Returns:
        rl_tokens   : (B, 1, H_vlm) — encoder output (paper's z_rl)
        vla_actions : (B, action_horizon, action_dim) — env-space actions
                      with gripper kept in dataset {0=close, 1=open}
                      convention (rollout's _unnormalize binarize +
                      _postprocess flip handle the LIBERO sign conversion).
    """
    from AlphaBrain.training.reinforcement_learning.algos.RLT_ori.vla_features import (
        compact_by_mask,
    )
    from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.openpi_inference import (
        make_att_2d_masks,
    )

    device = next(vla.parameters()).device

    # One-shot memory profiling: emit allocated/reserved at each stage of
    # the helper for the first call only. Lets us see where the live tensor
    # footprint actually grows vs what nvidia-smi shows as reserved (which
    # includes PyTorch's caching allocator overhead).
    _PROFILE = not getattr(get_pi05_rl_state_and_action, "_profiled", False)
    def _mem_dump(tag):
        if not _PROFILE:
            return
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(device) / 1e9
            reserv = torch.cuda.memory_reserved(device) / 1e9
            print(f"  [pi05 mem][{tag}] allocated={alloc:.2f} GB  reserved={reserv:.2f} GB",
                  flush=True)
    _mem_dump("entry")

    # ── Step 1: openpi-style prefix (matches predict_action's proven path) ──
    batch_images_normed = [
        list(imgs) if isinstance(imgs, (list, tuple)) else [imgs]
        for imgs in batch_images
    ]
    prefix_embs, prefix_pad_masks, prefix_att_masks, num_real_views = (
        _build_openpi_prefix(vla, batch_images_normed, instructions)
    )
    bsize = prefix_embs.shape[0]
    _mem_dump("after_prefix_build")

    # ── Step 2: VLM language model (Gemma) over prefix → last_hidden + KV cache ──
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_att_4d = prefix_att_2d[:, None, :, :]
    prefix_att_4d = torch.where(prefix_att_4d, 0.0, _NEG_INF_ADDITIVE)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    vlm_lm = vla._get_vlm_language_model()
    # 4D additive mask is incompatible with flash_attention_2.
    vlm_lm.config._attn_implementation = "eager"
    prefix_output = vlm_lm.forward(
        inputs_embeds=prefix_embs,
        attention_mask=prefix_att_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        use_cache=True,
    )
    last_hidden = prefix_output.last_hidden_state                # (B, L, H)
    past_key_values = prefix_output.past_key_values
    _mem_dump("after_prefix_forward")

    if hasattr(past_key_values, "to_legacy_cache"):
        _vlm_kv_legacy = past_key_values.to_legacy_cache()
        _use_dynamic_cache = True
    else:
        _vlm_kv_legacy = past_key_values
        _use_dynamic_cache = False

    # ── Step 3: image-only mask + compact + encoder.encode → rl_tokens ──
    # Take only REAL image-token positions (drop dummy slots and lang).
    # AND-ing with prefix_pad_masks already drops dummy slots (their pad
    # mask is False), but be explicit by limiting n_img to num_real_views.
    B, L, _H = last_hidden.shape
    n_real_img = min(num_real_views * _PALIGEMMA_IMG_TOKENS_PER_VIEW, L)
    encoder_mask = torch.zeros(B, L, dtype=torch.long, device=last_hidden.device)
    encoder_mask[:, :n_real_img] = 1
    encoder_mask = encoder_mask & prefix_pad_masks.long()
    dense, kp_mask = compact_by_mask(last_hidden, encoder_mask)
    rl_tokens = encoder.encode(dense.float(), key_padding_mask=kp_mask)  # (B, 1, H)
    _mem_dump("after_encoder_encode")

    fmh = vla.flow_matching_head
    state_dim = getattr(getattr(vla.config.framework, "action_model", None), "state_dim", None)
    fmh_dtype = next(fmh.parameters()).dtype

    # State for diffusion conditioning: Pi05 (pi05=True) ignores state in
    # embed_suffix, but pass it through (sliced to state_dim) for symmetry
    # with non-pi05 paths and future-proofing.
    if batch_props is None:
        diffusion_state = torch.zeros(bsize, state_dim or 7, device=device, dtype=fmh_dtype)
    else:
        diffusion_state = batch_props
        if diffusion_state.ndim == 1:
            diffusion_state = diffusion_state.unsqueeze(0)
        if state_dim is not None and diffusion_state.shape[-1] > state_dim:
            diffusion_state = diffusion_state[..., :state_dim]
        diffusion_state = diffusion_state.to(device=device, dtype=fmh_dtype)

    expert_model = fmh.action_expert.model.model
    num_steps = fmh.num_inference_steps
    dt = -1.0 / num_steps
    dt_t = torch.tensor(dt, dtype=torch.float32, device=device)

    # x_t in fmh dtype so embed_suffix's nn.Linear weights match input dtype.
    noise = torch.randn(bsize, vla.action_horizon, vla.action_dim,
                        dtype=fmh_dtype, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)

    import contextlib
    if fmh_dtype in (torch.float16, torch.bfloat16):
        autocast_ctx = torch.autocast("cuda", dtype=fmh_dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    with autocast_ctx:
        while time >= -dt_t / 2:
            expanded_time = time.expand(bsize)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = fmh.embed_suffix(
                diffusion_state, x_t, expanded_time
            )
            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            full_att_4d = full_att_2d[:, None, :, :]
            full_att_4d = torch.where(full_att_4d, 0.0, _NEG_INF_ADDITIVE)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            expert_model.config._attn_implementation = "eager"
            if _use_dynamic_cache:
                from transformers import DynamicCache
                _kv_for_step = DynamicCache.from_legacy_cache(_vlm_kv_legacy)
            else:
                _kv_for_step = _vlm_kv_legacy

            suffix_output = expert_model.forward(
                inputs_embeds=suffix_embs,
                attention_mask=full_att_4d,
                position_ids=suffix_position_ids,
                past_key_values=_kv_for_step,
                use_cache=False,
                adarms_cond=adarms_cond,
            )
            suffix_out = suffix_output.last_hidden_state[:, -vla.action_horizon:]
            v_t = fmh.action_out_proj(suffix_out)
            x_t = x_t + dt_t * v_t.to(x_t.dtype)
            time = time + dt_t

    actions = x_t.to(torch.float32)

    # ── Step 5: Pi05's MEAN_STD unnormalize (NO gripper sign flip) ──
    # We deliberately omit Pi05's `actions[:, :, 6] = 1 - 2*x` because the
    # rollout pipeline (_unnormalize binarize + _postprocess flip) already
    # converts dataset-convention gripper to LIBERO sign. Applying both
    # flips inverts the gripper command — exactly the SR=0 we observed
    # in an earlier iteration of this helper.
    if getattr(vla, "use_action_norm", False):
        actions = actions * vla.action_std.to(actions.device) + vla.action_mean.to(actions.device)

    _mem_dump("after_diffusion+unnorm (return)")
    if _PROFILE:
        # Set the flag so subsequent calls don't print.
        get_pi05_rl_state_and_action._profiled = True

    return rl_tokens, actions


def is_pi05(vla) -> bool:
    """Cheap framework-type check for trainer/rollout dispatch."""
    return (
        hasattr(vla, "_prepare_prefix_paligemma")
        and hasattr(vla, "flow_matching_head")
        and hasattr(vla, "_get_vlm_language_model")
    )


def make_pi05_identity_action_norm_stats(action_dim: int = 7) -> dict:
    """Identity q01/q99 stats so rollout's `_unnormalize` is a no-op for Pi05."""
    return {"q01": [-1.0] * action_dim, "q99": [1.0] * action_dim}


def resolve_vla_metadata(vla):
    """Backbone-agnostic accessors used by both RL trainer and offline eval.

    Returns: (hidden_dim, action_norm_stats, chunk_len, action_dim)

    - hidden_dim: VLM last-hidden width. Qwen exposes it on
      qwen_vl_interface.model.config; Pi05 (PaliGemma) exposes it on
      vlm_interface or via _get_vlm_hidden_size.
    - action_norm_stats: Qwen uses VLA-internal q01/q99 (rollout's
      _unnormalize maps actor output back to env space). Pi05 outputs
      env-space actions directly from the flow-matching head, so identity
      stats turn _unnormalize into a no-op.
    """
    chunk_len = vla.chunk_len
    action_dim = vla.config.framework.action_model.action_dim

    if hasattr(vla, "qwen_vl_interface"):
        hidden_dim = vla.qwen_vl_interface.model.config.hidden_size
    elif hasattr(vla, "vlm_interface") and hasattr(vla.vlm_interface, "hidden_size"):
        hidden_dim = vla.vlm_interface.hidden_size
    else:
        hidden_dim = getattr(vla, "_get_vlm_hidden_size", lambda: None)()
        if hidden_dim is None:
            raise RuntimeError(
                f"Cannot determine VLM hidden_size for {type(vla).__name__}; "
                f"add explicit branch in resolve_vla_metadata."
            )

    if is_pi05(vla):
        action_norm_stats = make_pi05_identity_action_norm_stats(action_dim=action_dim)
    else:
        norm_stats = vla.norm_stats
        unnorm_key = next(iter(norm_stats.keys()))
        action_norm_stats = norm_stats[unnorm_key]["action"]

    return hidden_dim, action_norm_stats, chunk_len, action_dim
