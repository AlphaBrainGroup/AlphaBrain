"""WorldModelVLMInterface: unified WM encoder wrapper for VLA framework."""
import logging
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import WorldModelEncoderConfig
from .text_encoder import LightweightTextEncoder
from .fusion import CrossAttentionFusion
from .encoder import BaseWorldModelEncoder

logger = logging.getLogger(__name__)


class WorldModelVLMInterface(nn.Module):
    """Unified entry point that plugs world-model encoders into the VLA framework.

    Usage in config::

        framework:
          world_model:
            backend: "vjepa2"
            checkpoint_path: "/path/to/vjepa2.pth"
            hidden_size: 1024
            ...
    """

    def __init__(self, config):
        super().__init__()
        self.full_config = config

        # Parse world-model sub-config
        wm_cfg = config.framework.world_model
        self.wm_config = WorldModelEncoderConfig(
            backend=getattr(wm_cfg, "backend", "vjepa2"),
            checkpoint_path=getattr(wm_cfg, "checkpoint_path", ""),
            hidden_size=getattr(wm_cfg, "hidden_size", 1024),
            text_encoder_type=getattr(wm_cfg, "text_encoder_type", "t5-small"),
            text_encoder_path=getattr(wm_cfg, "text_encoder_path", ""),
            text_hidden_size=getattr(wm_cfg, "text_hidden_size", 512),
            fusion_type=getattr(wm_cfg, "fusion_type", "cross_attention"),
            num_fusion_layers=getattr(wm_cfg, "num_fusion_layers", 2),
            image_size=getattr(wm_cfg, "image_size", 384),
            use_intermediate_features=getattr(wm_cfg, "use_intermediate_features", False),
            intermediate_layer_ids=getattr(wm_cfg, "intermediate_layer_ids", None),
            feature_layer_id=getattr(wm_cfg, "feature_layer_id", None),
            pretrained_dir=getattr(wm_cfg, "pretrained_dir", ""),
            reason1_path=getattr(wm_cfg, "reason1_path", ""),
            sigma_min=getattr(wm_cfg, "sigma_min", 0.002),
            sigma_max=getattr(wm_cfg, "sigma_max", 80.0),
            sigma_data=getattr(wm_cfg, "sigma_data", 1.0),
            sigma_conditional=getattr(wm_cfg, "sigma_conditional", 0.0001),
            vjepa_num_frames=getattr(wm_cfg, "vjepa_num_frames", 16),
            vjepa_patch_size=getattr(wm_cfg, "vjepa_patch_size", 16),
            vjepa_tubelet_size=getattr(wm_cfg, "vjepa_tubelet_size", 2),
            vjepa_use_rope=getattr(wm_cfg, "vjepa_use_rope", True),
            vjepa_interpolate_rope=getattr(wm_cfg, "vjepa_interpolate_rope", True),
            wan_variant=getattr(wm_cfg, "wan_variant", None),
        )

        # Build components
        self.visual_encoder: BaseWorldModelEncoder = self._build_visual_encoder()
        self.text_encoder = LightweightTextEncoder(
            encoder_type=self.wm_config.text_encoder_type,
            encoder_path=self.wm_config.text_encoder_path,
            output_dim=self.wm_config.text_hidden_size,
        )
        self.fusion = self._build_fusion()

        # Expose compatibility shim for framework internals that expect
        # self.model.config.hidden_size (e.g. action head wiring).
        # Add save_pretrained to config/processor for compatibility with trainer checkpoint saving
        def _save_pretrained_noop(path, **kwargs):
            import json, os
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump({"hidden_size": self.wm_config.hidden_size, "backend": self.wm_config.backend}, f)

        _config_ns = SimpleNamespace(
            hidden_size=self.wm_config.hidden_size,
            save_pretrained=_save_pretrained_noop,
        )
        _processor_ns = SimpleNamespace(
            save_pretrained=_save_pretrained_noop,
        )
        self.processor = _processor_ns

        self.model = SimpleNamespace(
            config=_config_ns,
        )

        # Freeze text_encoder.projection — unused in V2 video loss path.
        # V2 forward_with_video_loss passes native text embeddings directly,
        # bypassing the lightweight text encoder's projection layer.
        if hasattr(self.text_encoder, 'projection') and self.text_encoder.projection is not None:
            self.text_encoder.projection.requires_grad_(False)
            logger.info("WorldModelVLMInterface: froze text_encoder.projection (unused in V2)")

        logger.info(
            "WorldModelVLMInterface initialized: backend=%s, hidden_size=%d, "
            "fusion=%s (%d layers)",
            self.wm_config.backend,
            self.wm_config.hidden_size,
            self.wm_config.fusion_type,
            self.wm_config.num_fusion_layers,
        )

    # -- factory helpers ----------------------------------------------------

    def _build_visual_encoder(self) -> BaseWorldModelEncoder:
        """Lazy-import and instantiate the backend encoder."""
        backend = self.wm_config.backend.lower().strip()

        if backend in ("vjepa2", "vjepa", "v-jepa"):
            from ..vjepa.encoder import VJEPAEncoder
            return VJEPAEncoder(self.wm_config)
        elif backend in ("wan2.2", "wan22", "wan"):
            from ..wan.encoder import WanEncoder
            return WanEncoder(self.wm_config)
        elif backend in ("cosmos2-diffusers", "cosmos2-diff", "predict2-diffusers", "cosmos2.5-diffusers", "cosmos25-diff", "predict2.5-diffusers"):
            from ..cosmos.encoder import Cosmos2DiffusersEncoder
            return Cosmos2DiffusersEncoder(self.wm_config)
        elif backend in ("cosmos2.5", "cosmos", "predict2.5"):
            from ..cosmos.legacy_encoder import CosmosEncoder
            return CosmosEncoder(self.wm_config)
        else:
            raise ValueError(f"Unknown world-model backend: {backend}")

    def _build_fusion(self) -> nn.Module:
        """Build the fusion module."""
        if self.wm_config.fusion_type == "cross_attention":
            return CrossAttentionFusion(
                visual_dim=self.visual_encoder.encoder_dim,
                text_dim=self.wm_config.text_hidden_size,
                output_dim=self.wm_config.hidden_size,
                num_layers=self.wm_config.num_fusion_layers,
            )
        else:
            raise ValueError(
                f"Unknown fusion type: {self.wm_config.fusion_type}"
            )

    # -- public API ---------------------------------------------------------

    def build_vlm_inputs(
        self,
        images: torch.Tensor,
        instructions: Union[List[str], torch.Tensor],
    ) -> BatchFeature:
        """Preprocess images and return a BatchFeature for framework compatibility.

        Args:
            images: raw images tensor.
            instructions: text instructions or precomputed embeddings.

        Returns:
            BatchFeature containing ``pixel_values`` and ``instructions``.
        """
        # images is List[List[PIL.Image]] from dataloader (batch of sample image lists)
        # Flatten to list of images, take first image per sample for single-frame encoding
        flat_images = []
        for sample_imgs in images:
            if isinstance(sample_imgs, (list, tuple)):
                flat_images.append(sample_imgs[0] if len(sample_imgs) > 0 else sample_imgs)
            else:
                flat_images.append(sample_imgs)
        pixel_values = self.visual_encoder.preprocess(flat_images)
        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "instructions": instructions,
            }
        )

    # Compat alias for framework code that calls the name used by real VLM interfaces
    # (Florence / QwenVL / CosmosReason). Keeps WM-based checkpoints loadable by
    # QwenGR00T.predict_action / forward without branching on interface type.
    build_qwenvl_inputs = build_vlm_inputs

    def forward(
        self,
        pixel_values: torch.Tensor,
        instructions: Union[List[str], torch.Tensor, None] = None,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Run the full encode-fuse pipeline.

        Args:
            pixel_values: preprocessed image tensor.
            instructions: text strings or precomputed embeddings.
            output_hidden_states: if True, include fused tokens in hidden_states.

        Returns:
            CausalLMOutputWithPast with hidden_states=(fused_tokens,).
        """
        # Text encoding: use backbone's native text encoder if available,
        # otherwise fall back to the external lightweight text encoder.
        if instructions is None:
            instructions = [""] * pixel_values.shape[0]

        # Check if backbone has native text encoder (e.g. Cosmos T5-XXL, Wan UMT5-XXL)
        native_text_embeds = None
        if hasattr(self.visual_encoder, 'encode_text'):
            native_text_embeds = self.visual_encoder.encode_text(
                instructions, pixel_values.device,
            )

        # Visual encoding (pass native text embeds to DiT cross-attention)
        if native_text_embeds is not None:
            visual_tokens = self.visual_encoder.encode_images(
                pixel_values, text_embeds=native_text_embeds,
            )
        else:
            visual_tokens = self.visual_encoder.encode_images(pixel_values)
        # [B, N_v, encoder_dim]

        # Text for fusion: use native text encoder output (projected) or lightweight encoder
        if native_text_embeds is not None:
            text_tokens = self.text_encoder.encode(native_text_embeds, visual_tokens.device)
        else:
            text_tokens = self.text_encoder.encode(instructions, visual_tokens.device)
        # [B, N_t, text_hidden_size]

        # Fusion — cast inputs to fusion layer dtype to handle fp32 encoder outputs
        fusion_dtype = next(self.fusion.parameters()).dtype
        fused_tokens = self.fusion(
            visual_tokens.to(dtype=fusion_dtype),
            text_tokens.to(dtype=fusion_dtype),
        )
        # [B, N_v, hidden_size]

        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
        if output_hidden_states:
            hidden_states = (fused_tokens,)

        return CausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )

    def forward_with_video_loss(
        self,
        pixel_values: torch.Tensor,
        instructions: Union[List[str], torch.Tensor, None],
        next_pixel_values: torch.Tensor,
    ) -> Tuple['CausalLMOutputWithPast', torch.Tensor]:
        """V2 single forward: returns fused visual tokens AND video loss.

        Performs a single DiT forward pass that simultaneously computes
        action features (layer 18) and the next-frame video prediction loss.
        Both share the same backward graph so the DiT backbone receives
        gradients from both losses without a redundant forward.

        Args:
            pixel_values: preprocessed current-frame tensor.
            instructions: text strings or precomputed embeddings.
            next_pixel_values: preprocessed next-frame tensor.

        Returns:
            Tuple of:
              - CausalLMOutputWithPast with hidden_states=(fused_tokens,)
              - video_loss scalar tensor (with gradients)
        """
        if instructions is None:
            instructions = [""] * pixel_values.shape[0]

        device = pixel_values.device

        # Native text encoder (e.g. Cosmos T5-XXL)
        native_text_embeds = None
        if hasattr(self.visual_encoder, 'encode_text'):
            native_text_embeds = self.visual_encoder.encode_text(instructions, device)

        # Backbones that support a meaningful next-frame video loss implement
        # both `encode_to_latent` and `encode_images_with_video_loss`
        # (Cosmos / WAN diffusion DiTs). JEPA-style encoders (V-JEPA) do not:
        # their ViT features lack generative / reconstruction supervision, so
        # we gracefully fall back to the plain encoding path with video_loss=0
        # instead of polluting training with a meaningless MSE term.
        supports_video_loss = (
            hasattr(self.visual_encoder, 'encode_to_latent')
            and hasattr(self.visual_encoder, 'encode_images_with_video_loss')
        )

        if supports_video_loss:
            # Encode both frames to latent space
            latent_t = self.visual_encoder.encode_to_latent(pixel_values)
            with torch.no_grad():
                latent_t1 = self.visual_encoder.encode_to_latent(next_pixel_values)

            # Single DiT forward -> action visual tokens + video loss
            visual_tokens, video_loss = self.visual_encoder.encode_images_with_video_loss(
                latent_t, latent_t1, native_text_embeds,
            )
        else:
            # JEPA-style fallback: no generative signal -> zero video loss.
            if native_text_embeds is not None:
                visual_tokens = self.visual_encoder.encode_images(
                    pixel_values, text_embeds=native_text_embeds,
                )
            else:
                visual_tokens = self.visual_encoder.encode_images(pixel_values)
            video_loss = torch.zeros((), device=device, dtype=visual_tokens.dtype)
        # visual_tokens: [B, N, encoder_dim], video_loss: scalar

        # Text for fusion layer
        if native_text_embeds is not None:
            text_tokens = self.text_encoder.encode(native_text_embeds, device)
        else:
            text_tokens = self.text_encoder.encode(instructions, device)

        # Fusion
        fusion_param = next(self.fusion.parameters())
        fusion_dtype = fusion_param.dtype
        fusion_device = fusion_param.device
        fused_tokens = self.fusion(
            visual_tokens.to(device=fusion_device, dtype=fusion_dtype),
            text_tokens.to(device=fusion_device, dtype=fusion_dtype),
        )

        return CausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=(fused_tokens,),
            attentions=None,
        ), video_loss

    def forward_all_layers(
        self,
        pixel_values: torch.Tensor,
        instructions: Union[List[str], torch.Tensor, None] = None,
    ) -> 'CausalLMOutputWithPast':
        """PI-style forward: returns per-layer features from all DiT layers.

        Each layer's features are fused with text tokens independently,
        yielding a tuple of fused tensors (one per DiT layer) in
        hidden_states.  The PI action head can then perform layer-wise
        weighted aggregation.

        Args:
            pixel_values: preprocessed image tensor.
            instructions: text strings or precomputed embeddings.

        Returns:
            CausalLMOutputWithPast with hidden_states as a tuple of
            [B, N, hidden_size] tensors, one per DiT layer (28 total for
            Cosmos-DiT).
        """
        if instructions is None:
            instructions = [""] * pixel_values.shape[0]

        device = pixel_values.device

        # Native text encoder
        native_text_embeds = None
        if hasattr(self.visual_encoder, 'encode_text'):
            native_text_embeds = self.visual_encoder.encode_text(instructions, device)

        # Get features from all DiT layers
        # Fallback to standard forward() if backend doesn't support per-layer extraction
        if not hasattr(self.visual_encoder, 'encode_images_all_layers'):
            return self.forward(pixel_values, instructions, output_hidden_states=True)
        all_layer_features = self.visual_encoder.encode_images_all_layers(
            pixel_values, native_text_embeds,
        )
        # all_layer_features: list of [B, N, encoder_dim], one per layer

        # Text for fusion
        if native_text_embeds is not None:
            text_tokens = self.text_encoder.encode(native_text_embeds, device)
        else:
            text_tokens = self.text_encoder.encode(instructions, device)

        fusion_dtype = next(self.fusion.parameters()).dtype
        fused_list = []
        for layer_feat in all_layer_features:
            fused = self.fusion(
                layer_feat.to(dtype=fusion_dtype),
                text_tokens.to(dtype=fusion_dtype),
            )
            fused_list.append(fused)

        return CausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=tuple(fused_list),
            attentions=None,
        )

    # -- checkpoint management ----------------------------------------------

    def save_world_model_checkpoint(
        self,
        save_dir: str,
        step: int,
    ) -> str:
        """Save only trainable parameters (fusion, projection, text encoder trainable parts).

        The frozen visual encoder is NOT saved to avoid redundant large files.

        Args:
            save_dir: directory to write the checkpoint into.
            step: current training step (used in filename).

        Returns:
            Path to the saved checkpoint file.
        """
        os.makedirs(save_dir, exist_ok=True)

        trainable_state: Dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state[name] = param.data.cpu()

        save_path = os.path.join(save_dir, f"world_model_step_{step}.pt")
        torch.save(
            {
                "step": step,
                "config": self.wm_config,
                "trainable_state_dict": trainable_state,
            },
            save_path,
        )
        logger.info(
            "Saved world-model checkpoint (%d trainable params) to %s",
            len(trainable_state),
            save_path,
        )
        return save_path

    def load_world_model_checkpoint(self, checkpoint_path: str) -> None:
        """Load trainable parameters from a previously saved checkpoint.

        Args:
            checkpoint_path: path to the .pt checkpoint file.
        """
        logger.info("Loading world-model checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        trainable_state = ckpt["trainable_state_dict"]

        missing, unexpected = [], []
        own_state = self.state_dict()
        for key, value in trainable_state.items():
            if key in own_state:
                own_state[key].copy_(value)
            else:
                unexpected.append(key)

        for name, param in self.named_parameters():
            if param.requires_grad and name not in trainable_state:
                missing.append(name)

        if missing:
            logger.warning("Missing keys in checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)
        logger.info(
            "Loaded world-model checkpoint (step=%d).", ckpt.get("step", -1)
        )



# Backward-compat alias (Phase 2b-A naming); kept to protect older
# checkpoints that may pickle this class name.
WorldModelEncoderInterface = WorldModelVLMInterface
