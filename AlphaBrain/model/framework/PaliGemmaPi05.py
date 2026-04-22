# Copyright 2025 VLA-Engine. All rights reserved.
# Pi0/Pi0.5 OFT Framework — Flow Matching VLA with swappable VLM backbone

"""
PaliGemmaOFT Framework

Integrates the π₀/π₀.₅ flow matching architecture into VLA-Engine.
Key innovation: the VLM backbone is swappable — you can use PaliGemma (original),
Qwen2.5-VL, Llama 3.2 Vision, or any future VLM backend.

Architecture:
  VLM (any) → prefix embedding → [KV cache] → Action Expert (Gemma) + Flow Matching → actions

Components:
  - VLM interface: reuses VLAE's existing get_vlm_model() factory
  - Action Expert: independent Gemma transformer (from openpi)
  - Flow Matching Head: multi-step denoising action generation

Training: flow matching loss (MSE between predicted and target velocity fields)
Inference: iterative denoising from Gaussian noise (default 10 steps)
"""

from typing import List, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.modules.vlm import get_vlm_model
from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.pi0_action_head import Pi0FlowMatchingHead
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY
from AlphaBrain.training.trainer_utils.trainer_tools import resize_images

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@FRAMEWORK_REGISTRY.register("PaliGemmaPi05")
@FRAMEWORK_REGISTRY.register("LlamaPi0")
class PaliGemma_Pi05(BaseFramework):
    """
    Pi0/Pi0.5 framework with swappable VLM backbone.
    
    Config structure:
        framework:
          name: PaliGemmaOFT
          pi05: true                    # true for π₀.₅, false for π₀
          paligemma:                     # or qwenvl/llamavl — uses get_vlm_model()
            base_vlm: google/paligemma-3b-pt-224
          action_expert:
            width: 1024
            depth: 18
            ...
          action_model:
            action_dim: 7
            action_horizon: 50
            num_inference_steps: 10
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        
        pi05 = getattr(config.framework, 'pi05', True)
        self.pi05 = pi05
        
        # ── VLM backbone (swappable) ──
        # Determine which VLM to use based on config
        vlm_type = self._detect_vlm_type(config)
        
        if vlm_type == "paligemma":
            from AlphaBrain.model.modules.vlm.paligemma import _PaliGemma_VL_Interface
            self.vlm_interface = _PaliGemma_VL_Interface(config=config)
        else:
            # Use existing VLM factory for Qwen/Llama/Florence etc.
            self.vlm_interface = get_vlm_model(config=config)
        
        # ── Action Expert + Flow Matching Head ──
        expert_cfg = config.framework.action_expert
        action_cfg = config.framework.action_model
        
        # Determine expert type based on framework name and config
        expert_type = "gemma"  # default
        if config.framework.name == "LlamaPi0":
            expert_type = "llama"
        elif hasattr(expert_cfg, 'type'):
            expert_type = expert_cfg.type
        
        self._tokenizer = None
        self.flow_matching_head = Pi0FlowMatchingHead(
            action_dim=action_cfg.action_dim,
            action_horizon=action_cfg.action_horizon,
            action_expert_width=getattr(expert_cfg, 'width', 1024),
            action_expert_depth=getattr(expert_cfg, 'depth', 18),
            action_expert_mlp_dim=getattr(expert_cfg, 'mlp_dim', 4096),
            action_expert_num_heads=getattr(expert_cfg, 'num_heads', 8 if expert_type == "gemma" else 32),
            action_expert_num_kv_heads=getattr(expert_cfg, 'num_kv_heads', 1 if expert_type == "gemma" else 8),
            action_expert_head_dim=getattr(expert_cfg, 'head_dim', 256 if expert_type == "gemma" else 128),
            pi05=pi05,
            precision=getattr(expert_cfg, 'precision', 'bfloat16'),
            num_inference_steps=getattr(action_cfg, 'num_inference_steps', 10),
            noise_beta_alpha=getattr(action_cfg, 'noise_beta_alpha', 1.5),
            noise_beta_beta=getattr(action_cfg, 'noise_beta_beta', 1.0),
            expert_type=expert_type,
            state_dim=getattr(action_cfg, 'state_dim', None),
        )
        
        # ── Prefix projection (for VLM hidden_size != action_expert_width) ──
        expert_width = getattr(expert_cfg, 'width', 1024)
        vlm_hidden_size = self._get_vlm_hidden_size()
        if vlm_hidden_size is not None and vlm_hidden_size != expert_width and vlm_type != "paligemma":
            # PaliGemma uses its own encode_prefix which outputs expert_width directly
            # For Qwen/Llama, VLM hidden states need projection to match action expert
            self.prefix_proj = nn.Linear(vlm_hidden_size, expert_width, bias=False)
            logger.info(f"[prefix_proj] Added projection: VLM hidden={vlm_hidden_size} → expert_width={expert_width}")
        else:
            self.prefix_proj = None
        
        # ── Action dimension settings ──
        self.action_dim = action_cfg.action_dim
        self.action_horizon = action_cfg.action_horizon
        self.future_action_window_size = getattr(action_cfg, 'future_action_window_size', action_cfg.action_horizon)
        self.past_action_window_size = getattr(action_cfg, 'past_action_window_size', 0)
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Store VLM rotary_emb reference for inference consistency
        vlm_lm = self._get_vlm_language_model()
        if vlm_lm is not None and hasattr(vlm_lm, 'rotary_emb'):
            self.flow_matching_head._vlm_rotary_emb = vlm_lm.rotary_emb

        logger.info(
            f"PaliGemmaOFT initialized: pi05={pi05}, vlm={vlm_type}, "
            f"action_dim={self.action_dim}, horizon={self.action_horizon}"
        )

        # Enable gradient checkpointing on action expert only.
        # _shared_forward handles VLM layers via use_gc (compute_layer checkpoint).
        # Enabling HF-level GC on VLM GemmaModel would cause double-checkpoint conflict
        # with DeepSpeed ZeRO: same parameters reduced twice -> AssertionError.
        for m in self.flow_matching_head.modules():
            if hasattr(m, "gradient_checkpointing_enable"):
                m.gradient_checkpointing_enable()
        # Disable HF-level GC on VLM language model to prevent double-backward
        try:
            vlm_lm = self._get_vlm_language_model()
            if hasattr(vlm_lm, "gradient_checkpointing_disable"):
                vlm_lm.gradient_checkpointing_disable()
        except Exception:
            pass

        # Freeze VLM modules based on config (empty string = no freeze = full finetune)
        freeze_modules = getattr(config.trainer, 'freeze_modules', 'vlm_interface.model.language_model,vlm_interface.model.lm_head')
        if freeze_modules:
            freeze_list = [m.strip() for m in str(freeze_modules).split(',') if m.strip()]
            if hasattr(self.vlm_interface, "model"):
                for name, param in self.vlm_interface.model.named_parameters():
                    if any(fm.replace('vlm_interface.model.', '') in name for fm in freeze_list):
                        param.requires_grad = False
            frozen_p = sum(p.numel() for p in self.parameters() if not p.requires_grad)
            train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"Frozen: {frozen_p/1e6:.0f}M, Trainable: {train_p/1e6:.0f}M (freeze={freeze_modules})")
        else:
            total_p = sum(p.numel() for p in self.parameters())
            logger.info(f"Full finetune: all {total_p/1e6:.0f}M parameters trainable")

        # ── Action/State Normalization (MEAN_STD, matches openpi) ──
        norm_cfg = getattr(config.framework, 'normalization', None)
        self.use_action_norm = norm_cfg is not None and getattr(norm_cfg, 'enabled', False)
        if self.use_action_norm:
            import json as _json
            action_mean = torch.tensor(getattr(norm_cfg, 'action_mean', [0.0]*self.action_dim), dtype=torch.float32)
            action_std = torch.tensor(getattr(norm_cfg, 'action_std', [1.0]*self.action_dim), dtype=torch.float32)
            self.register_buffer('action_mean', action_mean)
            self.register_buffer('action_std', action_std)
            if hasattr(norm_cfg, 'state_mean'):
                state_mean = torch.tensor(norm_cfg.state_mean, dtype=torch.float32)
                state_std = torch.tensor(norm_cfg.state_std, dtype=torch.float32)
                self.register_buffer('state_mean', state_mean)
                self.register_buffer('state_std', state_std)
            logger.info(f"[norm] Action MEAN_STD normalization enabled (action_dim={self.action_dim})")
        else:
            logger.info("[norm] No action normalization (raw actions)")

    def _init_tokenizer(self):
        """Initialize PaliGemma tokenizer (HF AutoTokenizer or sentencepiece fallback)."""
        import os
        # Try HF tokenizer first (has proper special tokens)
        # Optional local tokenizer cache via env var; falls back to HF hub
        _local_tok = os.environ.get("PALIGEMMA_TOKENIZER_PATH")
        tokenizer_dirs = ([_local_tok] if _local_tok else []) + [
            "google/paligemma-3b-pt-224",
        ]
        for td in tokenizer_dirs:
            try:
                from transformers import AutoTokenizer
                self._hf_tokenizer = AutoTokenizer.from_pretrained(td)
                self._tokenizer = None  # signal to use HF tokenizer
                logger.info(f"[PaliGemmaOFT] Loaded HF tokenizer from {td}")
                return
            except Exception:
                continue
        
        # Fallback: sentencepiece
        import sentencepiece as spm
        # Sentencepiece fallback: env var first, then project-local tokenizer.model
        _local_sp = os.environ.get("PALIGEMMA_TOKENIZER_MODEL")
        candidates = ([_local_sp] if _local_sp else []) + [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../tokenizer.model"),
            "./tokenizer.model",
        ]
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                self._tokenizer = spm.SentencePieceProcessor(model_file=c)
                self._hf_tokenizer = None
                logger.info(f"[PaliGemmaOFT] Loaded sentencepiece tokenizer from {c}")
                return
        raise FileNotFoundError("Cannot find PaliGemma tokenizer")

    def _get_vlm_hidden_size(self) -> Optional[int]:
        """Get the hidden size of the VLM's language model."""
        vlm = self.vlm_interface
        if hasattr(vlm, 'model'):
            model = vlm.model
            if hasattr(model, 'config'):
                cfg = model.config
                # Llama / multi-modal models with text_config
                if hasattr(cfg, 'text_config') and hasattr(cfg.text_config, 'hidden_size'):
                    return cfg.text_config.hidden_size
                # Direct hidden_size
                if hasattr(cfg, 'hidden_size'):
                    return cfg.hidden_size
        return None

    def _detect_vlm_type(self, config) -> str:
        """Detect which VLM backend to use from config."""
        if hasattr(config.framework, 'paligemma'):
            return "paligemma"
        elif hasattr(config.framework, 'qwenvl'):
            return "qwen"
        elif hasattr(config.framework, 'llamavl'):
            return "llama"
        else:
            logger.warning("No VLM config found in framework config, defaulting to paligemma")
            return "paligemma"

    def _get_vlm_language_model(self):
        """Get the underlying language model from the VLM for shared attention."""
        vlm = self.vlm_interface
        if hasattr(vlm, 'get_language_model'):
            return vlm.get_language_model()
        elif hasattr(vlm, 'model'):
            # Qwen/Llama style
            if hasattr(vlm.model, 'language_model'):
                return vlm.model.language_model
            elif hasattr(vlm.model, 'model'):
                return vlm.model.model
        raise AttributeError("Cannot find language model in VLM interface")

    def _prepare_prefix(self, examples):
        """
        Prepare prefix embeddings from examples.
        
        For PaliGemma: use the native encode_prefix() method
        For Qwen/Llama: extract hidden states from VLM forward, treat as prefix
        """
        vlm_type = self._detect_vlm_type(self.config)
        
        if vlm_type == "paligemma":
            return self._prepare_prefix_paligemma(examples)
        else:
            return self._prepare_prefix_generic(examples)

    def _prepare_prefix_paligemma(self, examples):
        """Prepare prefix using PaliGemma's native encode_prefix.
        
        Pipeline: images -> SigLIP -> projector -> image embeds
                  text -> sentencepiece -> embed_tokens -> text embeds
                  concat -> prefix
        
        Configurable via framework.paligemma:
            num_images: total number of image slots (default: auto-detect from input)
            image_mask: list of bools per slot, False = zero-padded dummy (default: all True)
            tokenize_format: "openpi" (BOS + text + \n) or "raw" (text only) (default: "openpi")
            max_token_len: max token length for instructions (default: 48)
        """
        import torchvision.transforms.functional as TF
        
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        B = len(examples)
        paligemma_cfg = getattr(self.config.framework, 'paligemma', None)
        
        # --- Config-driven image settings ---
        cfg_num_images = getattr(paligemma_cfg, 'num_images', None) if paligemma_cfg else None
        cfg_image_mask = getattr(paligemma_cfg, 'image_mask', None) if paligemma_cfg else None
        
        # --- Image preprocessing (multi-view support) ---
        def _process_single_img(img):
            """Process a single image to [3, 224, 224] normalized to [-1, 1]."""
            if not isinstance(img, torch.Tensor):
                import numpy as np
                from PIL import Image as PILImage
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img.copy()).float()
                    if img.ndim == 3 and img.shape[-1] == 3:
                        img = img.permute(2, 0, 1)  # HWC -> CHW
                elif isinstance(img, PILImage.Image):
                    img = TF.to_tensor(img)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            elif img.max() > 1.0:
                img = img.float() / 255.0
            img = TF.resize(img, [224, 224], antialias=True)
            img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1, 1]
            return img
        
        all_view_tensors = []  # list of [num_views, 3, 224, 224] per sample
        for ex in examples:
            img = ex["image"]
            if isinstance(img, (list, tuple)):
                views = [_process_single_img(v) for v in img]
            else:
                views = [_process_single_img(img)]
            
            # Pad to cfg_num_images with zero images if configured
            # Note: zero uint8 image [0,0,0] after normalize (x/255 - 0.5)/0.5 = -1.0
            # So padding images should be all -1.0 to match openpi's np.zeros_like behavior
            if cfg_num_images is not None and len(views) < cfg_num_images:
                pad_img = torch.full((3, 224, 224), -1.0, dtype=views[0].dtype)
                while len(views) < cfg_num_images:
                    views.append(pad_img)
            
            all_view_tensors.append(torch.stack(views))  # [V, 3, 224, 224]
        
        num_views = all_view_tensors[0].shape[0]
        # Stack all views across batch: [B, V, 3, 224, 224]
        pixel_values = torch.stack(all_view_tensors, dim=0).to(device=device, dtype=dtype)
        
        # Build per-view image masks from config
        if cfg_image_mask is not None:
            # cfg_image_mask: list of bools, e.g. [true, true, false] for 3 views
            mask_vals = cfg_image_mask
            if len(mask_vals) < num_views:
                mask_vals = list(mask_vals) + [True] * (num_views - len(mask_vals))
            masks_list = [torch.full((B,), bool(mask_vals[v]), dtype=torch.bool, device=device)
                          for v in range(num_views)]
        else:
            masks_list = [torch.ones(B, dtype=torch.bool, device=device) for _ in range(num_views)]
        
        # --- Text tokenization ---
        instructions = [ex["lang"] for ex in examples]
        
        if self._tokenizer is None and not hasattr(self, '_hf_tokenizer'):
            self._init_tokenizer()
        
        # Tokenize with configurable format
        tokenize_fmt = getattr(paligemma_cfg, 'tokenize_format', 'openpi') if paligemma_cfg else 'openpi'
        max_len = getattr(paligemma_cfg, 'max_token_len', 48) if paligemma_cfg else 48
        discrete_state = getattr(paligemma_cfg, 'discrete_state_input', False) if paligemma_cfg else False
        all_ids = []
        for i, text in enumerate(instructions):
            cleaned = text.strip().replace("_", " ").replace("\n", " ")
            
            # Build prompt text based on discrete_state_input setting
            if discrete_state and "state" in examples[i]:
                # π₀.₅ mode: discretize state into 256 bins and embed in prompt
                state = examples[i]["state"]
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                import numpy as np
                # Normalize state with MEAN_STD if available, then discretize to [-1, 1] → 256 bins
                if hasattr(self, 'state_mean'):
                    state_np = (state - self.state_mean.cpu().numpy()) / (self.state_std.cpu().numpy() + 1e-6)
                else:
                    state_np = state
                state_np = np.clip(state_np, -1.0, 1.0)
                discretized = np.digitize(state_np, bins=np.linspace(-1, 1, 257)[:-1]) - 1
                discretized = np.clip(discretized, 0, 255)
                state_str = " ".join(map(str, discretized.astype(int)))
                prompt_text = f"Task: {cleaned}, State: {state_str};\n"
            else:
                # π₀ mode: state as continuous input (not in prompt)
                prompt_text = cleaned
            
            if tokenize_fmt == "openpi":
                # openpi format: BOS + prompt_text (+ "\n" if not already ending with it)
                if hasattr(self, '_hf_tokenizer') and self._hf_tokenizer is not None:
                    bos_id = self._hf_tokenizer.bos_token_id
                    if discrete_state:
                        # prompt_text already ends with ";\n"
                        text_ids = self._hf_tokenizer.encode(prompt_text, add_special_tokens=False)
                        ids = ([bos_id] if bos_id is not None else []) + text_ids
                    else:
                        text_ids = self._hf_tokenizer.encode(prompt_text, add_special_tokens=False)
                        newline_ids = self._hf_tokenizer.encode("\n", add_special_tokens=False)
                        ids = ([bos_id] if bos_id is not None else []) + text_ids + newline_ids
                else:
                    if discrete_state:
                        ids = self._tokenizer.encode(prompt_text)
                    else:
                        ids = self._tokenizer.encode(prompt_text) + self._tokenizer.encode("\n")
            else:
                # raw format: text only, no special tokens
                if hasattr(self, '_hf_tokenizer') and self._hf_tokenizer is not None:
                    ids = self._hf_tokenizer.encode(prompt_text, add_special_tokens=False)
                else:
                    ids = self._tokenizer.encode(prompt_text)
            if len(ids) > max_len:
                ids = ids[:max_len]
            all_ids.append(ids)
        
        # Pad token sequences to fixed length (openpi pads to max_token_len for consistent position ids)
        max_actual = max(len(ids) for ids in all_ids)
        pad_len = max(max_actual, max_len)  # use max_token_len as minimum pad length
        token_ids = torch.zeros(B, pad_len, dtype=torch.long, device=device)
        token_masks = torch.zeros(B, pad_len, dtype=torch.bool, device=device)
        for i, ids in enumerate(all_ids):
            token_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            token_masks[i, :len(ids)] = True
        
        # Encode prefix through VLM (multi-view: pass each view separately)
        view_tensors = pixel_values  # [B, V, 3, 224, 224]
        images_list = [view_tensors[:, v] for v in range(num_views)]  # list of [B, 3, 224, 224]
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.vlm_interface.encode_prefix(
            images=images_list,
            image_masks=masks_list,
            lang_tokens=token_ids,
            lang_masks=token_masks,
        )
        
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _prepare_prefix_llama_raw(self, examples):
        """
        For LlamaPi0: run full VLM forward, get final hidden states, project to expert width.
        
        Llama 3.2 Vision has mixed self-attention and cross-attention layers
        (cross-attention layers process vision features). True layer-wise joint 
        attention is not feasible because cross-attention layers lack self_attn.
        
        Instead, we run the full VLM, project the output, and feed it to the
        action expert as prefix context. The action expert then does all 18 layers
        of self-attention over [projected_prefix | action_suffix].
        """
        # This is the same as _prepare_prefix_generic — reuse it
        return self._prepare_prefix_generic(examples)

    def _prepare_prefix_generic(self, examples):
        """
        Prepare prefix using Qwen/Llama VLM.
        
        Strategy: Run VLM forward to get hidden states, use as prefix embeddings.
        This allows any VLM that produces hidden states to serve as the prefix encoder.
        """
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        
        # Build VLM inputs using existing interface
        if hasattr(self.vlm_interface, 'build_qwenvl_inputs'):
            vlm_inputs = self.vlm_interface.build_qwenvl_inputs(
                images=batch_images, instructions=instructions
            )
        elif hasattr(self.vlm_interface, 'build_llama_inputs'):
            vlm_inputs = self.vlm_interface.build_llama_inputs(
                images=batch_images, instructions=instructions
            )
        else:
            raise NotImplementedError(f"VLM interface does not have a known input builder")
        
        # Forward through VLM to get hidden states
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm_interface.model(
                **vlm_inputs, output_hidden_states=True
            )
        
        # Use last hidden state as prefix embedding
        hidden_states = outputs.hidden_states[-1]  # [B, seq_len, H_vlm]
        bsize, seq_len, _ = hidden_states.shape
        
        # Project VLM hidden states to action expert width if needed
        if self.prefix_proj is not None:
            hidden_states = self.prefix_proj(hidden_states.float()).to(hidden_states.dtype)
        
        # Create masks (all valid, bidirectional attention for prefix)
        pad_masks = vlm_inputs.get("attention_mask", torch.ones(bsize, seq_len, dtype=torch.bool, device=hidden_states.device))
        att_masks = torch.zeros(bsize, seq_len, dtype=torch.bool, device=hidden_states.device)
        
        return hidden_states, pad_masks, att_masks

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        """
        Training forward pass.
        
        Args:
            examples: list of dicts with keys: image, lang, action, (state)
            
        Returns:
            (loss, metrics_dict)
        """
        actions = torch.stack([torch.tensor(ex["action"], dtype=torch.float32) for ex in examples])
        actions = actions.to(next(self.parameters()).device)
        
        # Normalize actions (MEAN_STD) if enabled
        if self.use_action_norm:
            actions = (actions - self.action_mean.to(actions.device)) / (self.action_std.to(actions.device) + 1e-8)
        
        # Pad action feature dim to model action_dim (e.g., robot 8-dim → base model 32-dim)
        if actions.shape[-1] < self.action_dim:
            actions = torch.nn.functional.pad(actions, (0, self.action_dim - actions.shape[-1]))
        
        # Truncate/pad actions to action_horizon
        if actions.shape[1] > self.action_horizon:
            actions = actions[:, :self.action_horizon]
        elif actions.shape[1] < self.action_horizon:
            pad = torch.zeros(
                actions.shape[0], self.action_horizon - actions.shape[1], actions.shape[2],
                device=actions.device, dtype=actions.dtype
            )
            actions = torch.cat([actions, pad], dim=1)
        
        # Get state if available (None for π₀.₅ with discrete state)
        state = None
        if not self.pi05 and "state" in examples[0]:
            state = torch.stack([torch.tensor(ex["state"], dtype=torch.float32) for ex in examples])
            state = state.to(actions.device)
            # Pad state feature dim to model action_dim
            if state.shape[-1] < self.action_dim:
                state = torch.nn.functional.pad(state, (0, self.action_dim - state.shape[-1]))
        
        # Encode prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._prepare_prefix(examples)
        
        # Check VLM type and framework to determine forward path
        vlm_type = self._detect_vlm_type(self.config)
        framework_name = self.config.framework.name
        
        if framework_name == "LlamaPi0" and vlm_type == "llama":
            # LlamaPi0: Need VLM input embeddings (not full forward output)
            # because _shared_forward_llama handles layer-by-layer processing
            prefix_embs_raw, prefix_pad_masks_raw, prefix_att_masks_raw = self._prepare_prefix_llama_raw(examples)
            vlm_lm = self._get_vlm_language_model()
            loss = self.flow_matching_head.compute_loss_llama(
                prefix_embs=prefix_embs_raw,
                prefix_pad_masks=prefix_pad_masks_raw,
                prefix_att_masks=prefix_att_masks_raw,
                vlm_language_model=vlm_lm,
                state=state,
                actions=actions,
            )
        elif vlm_type == "paligemma":
            # Traditional joint attention path for PaliGemma
            vlm_lm = self._get_vlm_language_model()
            loss = self.flow_matching_head.compute_loss(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                vlm_language_model=vlm_lm,
                state=state,
                actions=actions,
            )
        else:
            # Prefix cache path for non-PaliGemma VLMs (Llama, etc.)
            loss = self.flow_matching_head.compute_loss_prefix_cache(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                state=state,
                actions=actions,
            )
        
        loss_mean = loss.mean()
        return {"action_loss": loss_mean, "flow_matching_loss": loss_mean.item()}

    @torch.no_grad()
    @torch.amp.autocast('cuda', dtype=torch.bfloat16)
    def predict_action(self, batch_images: List = None, instructions: List[str] = None,
                       examples: List[dict] = None, unnorm_key=None, **kwargs):
        """
        Inference: predict actions via multi-step denoising.
        
        Returns:
            np.ndarray: [B, action_horizon, action_dim] unnormalized actions
        """
        # CRITICAL: disable gradient checkpointing for inference
        # GC is incompatible with KV cache (corrupts cached key/values)
        for m in self.modules():
            if hasattr(m, 'gradient_checkpointing'):
                m.gradient_checkpointing = False
        
        # Support both flat format (batch_images/instructions) and legacy examples format
        if examples is None and batch_images is not None:
            from PIL import Image
            states = kwargs.get("states", None)
            examples = []
            for i, imgs in enumerate(batch_images):
                # Pass all views as list for multi-view support
                img = list(imgs) if isinstance(imgs, (list, tuple)) else imgs
                lang = instructions[i] if instructions else ""
                ex = {"image": img, "lang": lang}
                if states is not None and i < len(states):
                    ex["state"] = states[i]
                examples.append(ex)
        
        device = next(self.parameters()).device
        
        state = None
        if not self.pi05 and "state" in examples[0]:
            state = torch.stack([torch.tensor(ex["state"], dtype=torch.float32) for ex in examples])
            state = state.to(device)
        
        # Check VLM type and framework to determine inference path
        vlm_type = self._detect_vlm_type(self.config)
        framework_name = self.config.framework.name
        
        if framework_name == "LlamaPi0" and vlm_type == "llama":
            # LlamaPi0: Llama VLM + Llama Action Expert with joint attention
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._prepare_prefix_generic(examples)
            vlm_lm = self._get_vlm_language_model()
            actions = self.flow_matching_head.sample_actions_llama(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                vlm_language_model=vlm_lm,
                state=state,
                device=device,
            )
            
            # Unnormalize actions if MEAN_STD normalization was used
            if self.use_action_norm:
                actions = actions * self.action_std.to(actions.device) + self.action_mean.to(actions.device)
                # Map gripper (dim 6) from [0,1] to [+1,-1] so that client-side
                # -gripper + binarize(>0.5 → open) produces correct LIBERO commands
                actions[:, :, 6] = 1.0 - 2.0 * actions[:, :, 6]
            
            actions_np = actions.cpu().float().numpy()
            return {"normalized_actions": actions_np.tolist()}
        elif vlm_type != "paligemma":
            # Use prefix cache mode for non-PaliGemma VLMs
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._prepare_prefix_generic(examples)
            actions = self.flow_matching_head.sample_actions_prefix_cache(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                state=state,
                device=device,
            )
            
            # Unnormalize actions if MEAN_STD normalization was used
            if self.use_action_norm:
                actions = actions * self.action_std.to(actions.device) + self.action_mean.to(actions.device)
                # Map gripper (dim 6) from [0,1] to [+1,-1] so that client-side
                # -gripper + binarize(>0.5 → open) produces correct LIBERO commands
                actions[:, :, 6] = 1.0 - 2.0 * actions[:, :, 6]
            
            actions_np = actions.cpu().float().numpy()
            return {"normalized_actions": actions_np.tolist()}
        
        # Traditional PaliGemma inference path below
        # Bypass _prepare_prefix — use openpi-style embed_prefix directly
        # Inline PaligemmaTokenizer logic (matches openpi's PaligemmaTokenizer)
        # to avoid importing openpi which pulls in jax as a top-level dependency.
        import torchvision.transforms.functional as TF
        import numpy as np_
        import math as _math
        
        # Ensure tokenizer is initialized (reuse existing _init_tokenizer)
        if self._tokenizer is None and not hasattr(self, '_hf_tokenizer'):
            self._init_tokenizer()
        
        _PREDICT_MAX_LEN = 200  # match openpi PaligemmaTokenizer default
        
        def _tokenize_openpi_style(text, max_len=_PREDICT_MAX_LEN):
            """Tokenize text in openpi PaligemmaTokenizer format: BOS + cleaned_text + newline, padded to max_len."""
            cleaned = str(text).strip().replace("_", " ").replace("\n", " ")
            if hasattr(self, '_hf_tokenizer') and self._hf_tokenizer is not None:
                bos_id = self._hf_tokenizer.bos_token_id
                text_ids = self._hf_tokenizer.encode(cleaned, add_special_tokens=False)
                newline_ids = self._hf_tokenizer.encode("\n", add_special_tokens=False)
                ids = ([bos_id] if bos_id is not None else []) + text_ids + newline_ids
            else:
                ids = self._tokenizer.encode(cleaned, add_bos=True) + self._tokenizer.encode("\n")
            tokens_len = len(ids)
            if tokens_len < max_len:
                mask = [True] * tokens_len + [False] * (max_len - tokens_len)
                ids = ids + [0] * (max_len - tokens_len)
            else:
                ids = ids[:max_len]
                mask = [True] * max_len
            return np_.asarray(ids), np_.asarray(mask)
        
        def _proc_img(im):
            t = torch.from_numpy(im.copy()).float()
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
            t = t / 255.0
            t = TF.resize(t, [224, 224], antialias=True)
            t = TF.normalize(t, mean=[0.5]*3, std=[0.5]*3)
            return t
        
        ex = examples[0]
        imgs_raw = ex['image'] if isinstance(ex['image'], list) else [ex['image']]
        _dtype = next(self.parameters()).dtype
        img_tensors = [_proc_img(im).unsqueeze(0).to(device).to(_dtype) for im in imgs_raw]
        while len(img_tensors) < 3:
            img_tensors.append(torch.full((1, 3, 224, 224), -1.0, device=device, dtype=_dtype))
        img_masks_list = [torch.tensor([True], device=device)] * len(imgs_raw) + \
                         [torch.tensor([False], device=device)] * (3 - len(imgs_raw))
        
        tokens, masks = _tokenize_openpi_style(ex['lang'])
        tokens_t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        masks_t = torch.tensor(masks, dtype=torch.bool).unsqueeze(0).to(device)
        
        embs_list, pad_list, att_list = [], [], []
        for img_t, img_m in zip(img_tensors, img_masks_list):
            img_emb = self.vlm_interface.model.get_image_features(img_t)
            bsize, n_embs = img_emb.shape[:2]
            embs_list.append(img_emb)
            pad_list.append(img_m[:, None].expand(bsize, n_embs))
            att_list += [0] * n_embs
        
        lang_emb = self.vlm_interface.model.embed_tokens(tokens_t)
        lang_emb = lang_emb * _math.sqrt(lang_emb.shape[-1])
        embs_list.append(lang_emb)
        pad_list.append(masks_t)
        att_list += [0] * lang_emb.shape[1]
        
        prefix_embs = torch.cat(embs_list, dim=1)
        prefix_pad_masks = torch.cat(pad_list, dim=1)
        att_tensor = torch.tensor(att_list, dtype=torch.bool, device=device)
        prefix_att_masks = att_tensor[None, :].expand(bsize, -1)
        
        # DEBUG: log prefix stats for comparison with openpi
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            # Log image info from examples
            ex0 = examples[0]
            img0 = ex0['image']
            if isinstance(img0, list):
                for vi, v in enumerate(img0):
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        print(f"[DEBUG] image[{vi}]: type=ndarray, dtype={v.dtype}, shape={v.shape}, range=[{v.min()}, {v.max()}]")
                    elif isinstance(v, torch.Tensor):
                        print(f"[DEBUG] image[{vi}]: type=tensor, dtype={v.dtype}, shape={v.shape}, range=[{v.min()}, {v.max()}]")
                    else:
                        print(f"[DEBUG] image[{vi}]: type={type(v)}")
            print(f"[DEBUG] prefix shape={prefix_embs.shape}, "
                  f"mean={prefix_embs.float().mean():.6f}, std={prefix_embs.float().std():.6f}, "
                  f"img[0:4]={prefix_embs[0,0,:4].float().cpu().tolist()}, "
                  f"lang[768:772]={prefix_embs[0,768,:4].float().cpu().tolist()}")
            import sys; sys.stdout.flush(); sys.stderr.flush()
            self._debug_count += 1
        
        vlm_lm = self._get_vlm_language_model()
        
        # Use openpi-style KV cache inference directly (proven to work in AB test)
        from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.openpi_inference import make_att_2d_masks
        from transformers.models.gemma import modeling_gemma
        
        bsize = prefix_pad_masks.shape[0]
        
        # Step 1: prefix → KV cache through VLM language model
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = prefix_att_2d[:, None, :, :]
        prefix_att_4d = torch.where(prefix_att_4d, 0.0, -2.3819763e38)
        
        vlm_lm.config._attn_implementation = "eager"
        prefix_output = vlm_lm.forward(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = prefix_output.past_key_values
        # For transformers >= 4.45: DynamicCache accumulates K,V across denoising steps.
        # Convert to legacy tuple format so we can recreate a fresh cache each step.
        if hasattr(past_key_values, 'to_legacy_cache'):
            _vlm_kv_legacy = past_key_values.to_legacy_cache()
            _use_dynamic_cache = True
        else:
            _use_dynamic_cache = False
        
        # Step 2: iterative denoising through action expert
        expert_model = self.flow_matching_head.action_expert.model.model
        num_steps = self.flow_matching_head.num_inference_steps
        dt = -1.0 / num_steps
        dt_t = torch.tensor(dt, dtype=torch.float32, device=device)
        
        noise = torch.randn(bsize, self.action_horizon, self.action_dim, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        while time >= -dt_t / 2:
            expanded_time = time.expand(bsize)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.flow_matching_head.embed_suffix(
                state, x_t, expanded_time
            )
            
            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks.shape[1]
            
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            full_att_4d = full_att_2d[:, None, :, :]
            full_att_4d = torch.where(full_att_4d, 0.0, -2.3819763e38)
            
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            
            expert_model.config._attn_implementation = "eager"
            # Rebuild fresh DynamicCache each step: prevents expert K,V from
            # accumulating across denoising steps (DynamicCache is stateful in 4.45+).
            if _use_dynamic_cache:
                from transformers import DynamicCache
                _kv_for_step = DynamicCache.from_legacy_cache(_vlm_kv_legacy)
            else:
                _kv_for_step = past_key_values
            suffix_output = expert_model.forward(
                inputs_embeds=suffix_embs,
                attention_mask=full_att_4d,
                position_ids=suffix_position_ids,
                past_key_values=_kv_for_step,
                use_cache=False,
                adarms_cond=adarms_cond,
            )
            
            suffix_out = suffix_output.last_hidden_state
            suffix_out = suffix_out[:, -self.action_horizon:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.flow_matching_head.action_out_proj(suffix_out)
            
            x_t = x_t + dt_t * v_t
            time = time + dt_t
        
        actions = x_t
        
        # Unnormalize actions if MEAN_STD normalization was used
        if self.use_action_norm:
            actions = actions * self.action_std.to(actions.device) + self.action_mean.to(actions.device)
            # Map gripper (dim 6) from [0,1] to [+1,-1] so that client-side
            # -gripper + binarize(>0.5 → open) produces correct LIBERO commands
            actions[:, :, 6] = 1.0 - 2.0 * actions[:, :, 6]
        
        actions_np = actions.cpu().float().numpy()
        
        # Return actions (unnormalized if norm was enabled, raw otherwise)
        return {"normalized_actions": actions_np.tolist()}
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.vlm_interface, 'model'):
            if hasattr(self.vlm_interface.model, 'gradient_checkpointing_enable'):
                self.vlm_interface.model.gradient_checkpointing_enable()
            # For PaliGemmaVLM: enable on sub-models
            if hasattr(self.vlm_interface.model, 'language_model') and hasattr(self.vlm_interface.model.language_model, 'gradient_checkpointing_enable'):
                self.vlm_interface.model.language_model.gradient_checkpointing_enable()
            if hasattr(self.vlm_interface.model, 'vision_tower') and hasattr(self.vlm_interface.model.vision_tower, 'gradient_checkpointing_enable'):
                self.vlm_interface.model.vision_tower.gradient_checkpointing_enable()
        # Freeze VLM language model and lm_head to save GPU memory
        # Only train: vision_tower + projector + action_expert + flow matching head
        if hasattr(self.vlm_interface, 'model'):
            for name, param in self.vlm_interface.model.named_parameters():
                if 'language_model' in name or 'lm_head' in name:
                    param.requires_grad = False
            frozen = sum(1 for p in self.vlm_interface.model.parameters() if not p.requires_grad)
            total = sum(1 for p in self.vlm_interface.model.parameters())
            trainable_m = sum(p.numel() for p in self.vlm_interface.model.parameters() if p.requires_grad) / 1e6
            print(f'[PaliGemmaOFT] VLM: frozen {frozen}/{total} params, trainable {trainable_m:.0f}M (vision+projector)')

        if hasattr(self.flow_matching_head, 'action_expert') and hasattr(self.flow_matching_head.action_expert.model, 'gradient_checkpointing_enable'):
            self.flow_matching_head.action_expert.model.gradient_checkpointing_enable()

