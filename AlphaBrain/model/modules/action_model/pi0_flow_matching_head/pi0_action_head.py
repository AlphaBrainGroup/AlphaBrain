"""
Pi0 Flow Matching Action Head

Decomposed from openpi's PI0Pytorch. This module handles:
  1. Action Expert (independent Gemma model for action token processing)
  2. Flow Matching (noise scheduling, denoising, loss computation)
  3. Prefix-Suffix attention (shared attention between VLM prefix and action suffix)

The VLM backbone is NOT included here - it's provided by the VLM interface
(paligemma.py, qwen2_5.py, etc.) through the framework layer.

Reference: openpi/src/openpi/models_pytorch/pi0_pytorch.py
           openpi/src/openpi/models_pytorch/gemma_pytorch.py
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal

# Apply adaRMS patch before importing Gemma
# adarms_patch is no longer needed when using openpi's patched transformers 4.53.2
# The patched transformers already has adaRMS, gated residual, and KV cache support built-in
from .adarms_patch import patch_gemma_for_adarms
patch_gemma_for_adarms()
from .adarms_patch import _gated_residual  # still need the helper function

from transformers import GemmaForCausalLM
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from .pi0_utils import (
    create_sinusoidal_pos_embedding,
    sample_beta,
    make_att_2d_masks,
)

logger = logging.getLogger(__name__)


class GemmaActionExpert(nn.Module):
    """
    Independent Gemma model that serves as the action expert in π0.5.

    In the π0 architecture, the VLM (PaliGemma) processes image+language as prefix,
    and this expert processes action tokens as suffix. They share attention via
    layer-wise cross computation.

    This is separate from the VLM so that:
    - The expert can be smaller than the VLM (e.g., 300M vs 2B)
    - The expert uses adaRMS conditioning on timestep (for π0.5)
    - Future: the expert can be paired with different VLM backbones
    """

    def __init__(
        self,
        width: int = 1024,
        depth: int = 18,
        mlp_dim: int = 4096,
        num_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        use_adarms: bool = True,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()

        config_hf = CONFIG_MAPPING["gemma"](
            head_dim=head_dim,
            hidden_size=width,
            intermediate_size=mlp_dim,
            num_attention_heads=num_heads,
            num_hidden_layers=depth,
            num_key_value_heads=num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms,
            adarms_cond_dim=width if use_adarms else None,
        )

        self.model = GemmaForCausalLM(config=config_hf)
        self.model.model.embed_tokens = None  # Don't need token embeddings

        self.width = width
        self.depth = depth
        self.use_adarms = use_adarms

        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
            # Keep norms in float32
            for name, param in self.named_parameters():
                if "layernorm" in name.lower() or "norm" in name.lower():
                    param.data = param.data.to(dtype=torch.float32)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        adarms_cond: Optional[torch.Tensor] = None,
    ):
        """Forward pass through the action expert."""
        output = self.model.model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )
        if hasattr(output, 'last_hidden_state'):
            # Fix: AdaRMSNorm may return (tensor, gate) tuple for last_hidden_state
            lhs = output.last_hidden_state
            if isinstance(lhs, tuple):
                lhs = lhs[0]
            return lhs, getattr(output, 'past_key_values', None)
        elif isinstance(output, tuple):
            return output[0], output[1] if len(output) > 1 else None
        else:
            return output, None


class Pi0FlowMatchingHead(nn.Module):
    """
    Flow matching action head for Pi0/Pi0.5.

    Handles:
    - Action/state/time embedding (suffix construction)
    - Flow matching loss (training)
    - Multi-step denoising (inference)
    - Prefix-suffix shared attention orchestration

    Does NOT contain:
    - VLM backbone (provided externally)
    - Image/language processing (handled by VLM interface)
    """

    def __init__(
        self,
        action_dim: int = 32,
        action_horizon: int = 50,
        action_expert_width: int = 1024,
        action_expert_depth: int = 18,
        action_expert_mlp_dim: int = 4096,
        action_expert_num_heads: int = 8,
        action_expert_num_kv_heads: int = 1,
        action_expert_head_dim: int = 256,
        pi05: bool = True,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        num_inference_steps: int = 10,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        state_dim: int = None,  # state dimension for π₀ mode state_proj
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.pi05 = pi05
        self.num_inference_steps = num_inference_steps
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta

        expert_width = action_expert_width

        self.action_expert = GemmaActionExpert(
            width=action_expert_width,
            depth=action_expert_depth,
            mlp_dim=action_expert_mlp_dim,
            num_heads=action_expert_num_heads,
            num_kv_heads=action_expert_num_kv_heads,
            head_dim=action_expert_head_dim,
            use_adarms=pi05,
            precision=precision,
        )

        # Projection layers
        self.action_in_proj = nn.Linear(action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, action_dim)

        if pi05:
            # π0.5: timestep injected via adaRMS conditioning
            self.time_mlp_in = nn.Linear(expert_width, expert_width)
            self.time_mlp_out = nn.Linear(expert_width, expert_width)
        else:
            # π0: state as continuous input, timestep concatenated with action
            _state_dim = state_dim if state_dim is not None else action_dim
            self.state_proj = nn.Linear(_state_dim, expert_width)
            self.action_time_mlp_in = nn.Linear(2 * expert_width, expert_width)
            self.action_time_mlp_out = nn.Linear(expert_width, expert_width)

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(self.noise_beta_alpha, self.noise_beta_beta, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_suffix(
        self,
        state: Optional[torch.Tensor],
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ):
        """
        Embed state + noisy actions + timestep into suffix tokens.

        Returns:
            embs: [B, suffix_len, hidden] suffix embeddings
            pad_masks: [B, suffix_len] padding masks
            att_masks: [B, suffix_len] attention pattern masks
            adarms_cond: [B, hidden] or None, conditioning for adaRMS (π0.5 only)
        """
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            # π0: state as continuous input
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            # If state is [B, T, state_dim], take last timestep
            if state.dim() == 3:
                state = state[:, -1, :]  # [B, state_dim]
            state_emb = self.state_proj(state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        # Timestep encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Action embedding
        action_emb = self.action_in_proj(noisy_actions)

        if not self.pi05:
            # π0: concat time + action, then MLP
            time_emb_expanded = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=2)
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            action_time_emb = self.action_time_mlp_out(x)
            adarms_cond = None
        else:
            # π₀.5: time goes through MLP for adaRMS conditioning
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            time_emb = F.silu(x)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device))
        att_masks += [1] + ([0] * (self.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, -1)

        return embs, pad_masks, att_masks, adarms_cond

    def compute_loss(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        vlm_language_model: nn.Module,
        state: Optional[torch.Tensor],
        actions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching training loss.

        Args:
            prefix_embs: [B, prefix_len, H] from VLM encode_prefix()
            prefix_pad_masks, prefix_att_masks: from VLM encode_prefix()
            vlm_language_model: the VLM's language model (for shared layer-wise attention)
            state: [B, state_dim] robot proprioceptive state (None for π0.5 discrete state)
            actions: [B, action_horizon, action_dim] ground truth actions

        Returns:
            loss: scalar MSE flow matching loss
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        # Cast to match VLM dtype
        if prefix_embs.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Build combined attention masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        att_2d_masks_4d = torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

        # Shared forward: VLM language model + action expert
        suffix_out = self._shared_forward(
            vlm_language_model,
            prefix_embs, suffix_embs,
            att_2d_masks_4d, position_ids,
            [None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.action_horizon:]
        suffix_out = suffix_out.float()
        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def compute_loss_prefix_cache(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        state: Optional[torch.Tensor],
        actions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching training loss using prefix-cache mode.
        
        For non-Gemma VLMs (Llama, etc.), runs everything through action expert
        instead of joint attention. Prefix embeddings have already been projected
        to action expert width.
        
        Args:
            prefix_embs: [B, prefix_len, 1024] projected prefix embeddings
            prefix_pad_masks, prefix_att_masks: prefix attention masks
            state: [B, state_dim] robot state (None for π0.5 discrete state)
            actions: [B, action_horizon, action_dim] ground truth actions
            
        Returns:
            loss: scalar MSE flow matching loss
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Get suffix embeddings
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        # Cast to match prefix dtype
        if prefix_embs.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Concat prefix + suffix, run EVERYTHING through action expert
        all_embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_4d = att_2d[:, None, :, :]
        att_4d = torch.where(att_4d, 0.0, -2.3819763e38)

        # Prepare adarms_cond for action expert
        # prefix tokens don't need timestep conditioning, suffix tokens do
        prefix_len = prefix_pad_masks.shape[1]
        total_len = all_embs.shape[1]
        
        if adarms_cond is not None:
            # Create a batched adarms_cond where prefix positions get None (zero conditioning)
            # and suffix positions get the timestep conditioning
            batch_size = all_embs.shape[0]
            # Use zero conditioning for prefix, timestep conditioning for suffix
            # Note: The patched GemmaModel handles per-token conditioning
            expert_adarms_cond = adarms_cond
        else:
            expert_adarms_cond = None

        expert_model = self.action_expert.model.model
        expert_model.config._attn_implementation = "eager"

        output = expert_model(
            inputs_embeds=all_embs,
            attention_mask=att_4d,
            position_ids=position_ids,
            adarms_cond=expert_adarms_cond,
        )

        # Extract suffix output (last suffix_len tokens)
        suffix_len = suffix_pad_masks.shape[1]
        suffix_out = output.last_hidden_state[:, -suffix_len:]
        suffix_out = suffix_out[:, -self.action_horizon:]
        suffix_out = suffix_out.float()
        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        vlm_language_model: nn.Module,
        state: Optional[torch.Tensor],
        device: torch.device,
        noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Multi-step denoising inference using KV cache (matches openpi's inference path).

        Step 1: Run prefix through VLM language_model once, cache KV.
        Step 2: For each denoise step, run suffix through action_expert with KV cache.

        Returns:
            [B, action_horizon, action_dim] predicted actions
        """
        num_steps = num_steps or self.num_inference_steps
        bsize = prefix_pad_masks.shape[0]

        if noise is None:
            actions_shape = (bsize, self.action_horizon, self.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Use float32 for noise/time (matches openpi's precision control)
        # model_dtype is used for attention masks only
        model_dtype = next(self.parameters()).dtype

        # ── Step 1: Compute prefix KV cache using standard GemmaModel.forward ──
        # openpi's patched transformers supports adarms_cond + past_key_values natively
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_len = prefix_pad_masks.shape[1]

        prefix_att_4d = prefix_att_2d[:, None, :, :]  # [B, 1, seq, seq]
        prefix_att_4d = torch.where(prefix_att_4d, 0.0, -2.3819763e38).to(dtype=model_dtype)

        vlm_language_model.config._attn_implementation = "eager"
        prefix_output = vlm_language_model(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
            adarms_cond=None,  # VLM doesn't use adaRMS
        )
        past_key_values = prefix_output.past_key_values
        # transformers >= 4.45 returns a stateful DynamicCache. Snapshot the
        # prefix K,V as legacy tuples so each denoise step can run against a
        # fresh cache instead of polluting the prefix with expert K,V.
        if hasattr(past_key_values, "to_legacy_cache"):
            prefix_kv_legacy = past_key_values.to_legacy_cache()
            use_dynamic_cache = True
        else:
            prefix_kv_legacy = None
            use_dynamic_cache = False

        # ── Step 2: Iterative denoising with KV cache ──
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise  # keep float32, openpi uses float32 for noise/actions
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        # Force eager attention on action expert
        expert_model = self.action_expert.model.model
        expert_model.config._attn_implementation = "eager"

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Embed suffix
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, x_t, expanded_time
            )
            suffix_len = suffix_pad_masks.shape[1]

            # Build attention mask: suffix attends to prefix (via KV cache) + suffix self-attention
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)  # [B, suffix_len, prefix_len+suffix_len]

            full_att_4d = full_att_2d[:, None, :, :]  # [B, 1, suffix_len, prefix_len+suffix_len]
            full_att_4d = torch.where(full_att_4d, 0.0, -2.3819763e38).to(dtype=model_dtype)

            # Suffix position IDs (continue from prefix)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            # Run suffix through action expert with prefix KV cache.
            # Rebuild a fresh DynamicCache from the legacy prefix snapshot each
            # step so expert K,V from prior steps don't accumulate into it.
            if use_dynamic_cache:
                from transformers import DynamicCache
                kv_for_step = DynamicCache.from_legacy_cache(prefix_kv_legacy)
            else:
                kv_for_step = past_key_values

            expert_output = expert_model(
                inputs_embeds=suffix_embs,
                attention_mask=full_att_4d,
                position_ids=suffix_position_ids,
                past_key_values=kv_for_step,
                use_cache=False,
                adarms_cond=adarms_cond,
            )
            suffix_out = expert_output.last_hidden_state

            # Extract action tokens and project. Keep float32 throughout the
            # denoise update — matches openpi's precision control and the
            # sibling sample_actions_prefix_cache path.
            suffix_out = suffix_out[:, -self.action_horizon:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)

            x_t = x_t + dt * v_t
            time = time + dt

        return x_t

    @torch.no_grad()
    def sample_actions_prefix_cache(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        state: Optional[torch.Tensor],
        device: torch.device,
        noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Multi-step denoising inference using prefix-cache mode.
        
        For non-Gemma VLMs: prefix is processed through action expert to build KV cache,
        then suffix is iteratively denoised using that cache.
        
        Args:
            prefix_embs: [B, prefix_len, 1024] projected prefix embeddings
            prefix_pad_masks, prefix_att_masks: prefix attention masks
            state: [B, state_dim] robot state (None for π0.5)
            device: target device
            
        Returns:
            [B, action_horizon, action_dim] predicted actions
        """
        num_steps = num_steps or self.num_inference_steps
        bsize = prefix_pad_masks.shape[0]

        if noise is None:
            actions_shape = (bsize, self.action_horizon, self.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Use float32 for noise/time (matches openpi's precision control)
        model_dtype = next(self.parameters()).dtype

        # Step 1: Compute prefix KV cache using action expert
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_len = prefix_pad_masks.shape[1]

        prefix_att_4d = prefix_att_2d[:, None, :, :]  # [B, 1, seq, seq]
        prefix_att_4d = torch.where(prefix_att_4d, 0.0, -2.3819763e38).to(dtype=model_dtype)

        # Use action expert to build KV cache (no timestep conditioning for prefix)
        expert_model = self.action_expert.model.model
        expert_model.config._attn_implementation = "eager"
        
        prefix_output = expert_model(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
            adarms_cond=None,  # No timestep conditioning for prefix
        )
        past_key_values = prefix_output.past_key_values

        # Step 2: Iterative denoising with KV cache
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise  # keep float32
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Embed suffix
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, x_t, expanded_time
            )
            suffix_len = suffix_pad_masks.shape[1]

            # Build attention mask: suffix attends to prefix (via KV cache) + suffix self-attention
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)  # [B, suffix_len, prefix_len+suffix_len]

            full_att_4d = full_att_2d[:, None, :, :]  # [B, 1, suffix_len, prefix_len+suffix_len]
            full_att_4d = torch.where(full_att_4d, 0.0, -2.3819763e38).to(dtype=model_dtype)

            # Suffix position IDs (continue from prefix)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            # Run suffix through action expert with prefix KV cache
            expert_output = expert_model(
                inputs_embeds=suffix_embs,
                attention_mask=full_att_4d,
                position_ids=suffix_position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                adarms_cond=adarms_cond,  # Timestep conditioning for suffix
            )
            suffix_out = expert_output.last_hidden_state

            # Extract action tokens and project
            suffix_out = suffix_out[:, -self.action_horizon:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)

            x_t = x_t + dt * v_t
            time = time + dt

        return x_t

    def _shared_forward(
        self,
        vlm_language_model: nn.Module,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: list,
    ) -> torch.Tensor:
        """
        Layer-wise shared forward between VLM language model and action expert.

        Each layer:
        1. Both models compute Q/K/V independently
        2. Q/K/V are concatenated for joint attention
        3. Output is split back and passed through respective FFNs

        This is the core of the π0 prefix-suffix architecture.
        """
        models = [vlm_language_model, self.action_expert.model.model]
        inputs_embeds = [prefix_embs, suffix_embs]
        num_layers = vlm_language_model.config.num_hidden_layers

        use_gc = (
            hasattr(self.action_expert.model.model, "gradient_checkpointing")
            and self.action_expert.model.model.gradient_checkpointing
            and self.training
        )

        for layer_idx in range(num_layers):
            def compute_layer(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                query_states, key_states, value_states, gates = [], [], [], []

                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states_normed, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
                    gates.append(gate)

                    input_shape = hidden_states_normed.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_states.append(layer.self_attn.q_proj(hidden_states_normed).view(hidden_shape).transpose(1, 2))
                    key_states.append(layer.self_attn.k_proj(hidden_states_normed).view(hidden_shape).transpose(1, 2))
                    value_states.append(layer.self_attn.v_proj(hidden_states_normed).view(hidden_shape).transpose(1, 2))

                # Joint attention
                q = torch.cat(query_states, dim=2)
                k = torch.cat(key_states, dim=2)
                v = torch.cat(value_states, dim=2)

                dummy = torch.zeros(q.shape[0], q.shape[2], q.shape[-1], device=q.device, dtype=q.dtype)
                # CRITICAL: must use VLM's rotary_emb for both prefix and suffix — expert RoPE config
                # (max_position_embeddings, rope_scaling) may differ; sharing keeps training/inference aligned.
                cos, sin = vlm_language_model.rotary_emb(dummy, position_ids)
                q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

                scaling = vlm_language_model.layers[layer_idx].self_attn.scaling
                att_output, _ = modeling_gemma.eager_attention_forward(
                    vlm_language_model.layers[layer_idx].self_attn, q, k, v, attention_mask, scaling
                )

                head_dim = vlm_language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(q.shape[0], -1, q.shape[1] * head_dim)

                # Split back and apply per-model FFN
                outputs = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    attn_slice = att_output[:, start_pos:end_pos]
                    if attn_slice.dtype != layer.self_attn.o_proj.weight.dtype:
                        attn_slice = attn_slice.to(layer.self_attn.o_proj.weight.dtype)
                    out = layer.self_attn.o_proj(attn_slice)

                    out = modeling_gemma._gated_residual(hidden_states, out, gates[i])
                    after_first_residual = out.clone()
                    out, gate = layer.post_attention_layernorm(out, cond=adarms_cond[i])
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out = out.to(dtype=torch.bfloat16)
                    out = layer.mlp(out)
                    out = modeling_gemma._gated_residual(after_first_residual, out, gate)
                    outputs.append(out)
                    start_pos = end_pos

                return outputs

            if use_gc:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer, layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                inputs_embeds = compute_layer(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond)

        # Final norms
        outputs = []
        for i, hidden_states in enumerate(inputs_embeds):
            out, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
            outputs.append(out)

        return outputs[1]  # Return suffix output only
