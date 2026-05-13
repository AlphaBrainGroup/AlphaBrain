"""
ActionToken Encoder-Decoder: Information bottleneck between frozen VLA and
small RL network.

Inspired by the "RL Token" paper (Physical Intelligence, 2026), but with
deviations from the paper's construction:
  - Encoder input is the VLA's action-query hidden states (M × H) gathered at
    the action-token positions, not the full image-token sequence (N × H) as
    in the paper's Fig. 2.
  - An extra `Linear(H → bottleneck_dim)` projection compresses per-token dim
    (e.g. 2048 → 256); the paper keeps the RL token at the VLA hidden dim.
  - The decoder is a self-attention transformer with a causal mask and a
    prefix token, not the encoder-decoder cross-attention structure of the
    paper's Eq. 2.
A faithful paper-accurate reimplementation is still under test.

Paper Eq. 1 — Encoder:
  z_rl = g_φ([z_{1:M}, e_rl])_{M+1}
  Append learnable embedding e_rl to VLA token sequence, run through
  self-attention encoder transformer, take the e_rl position output.

Paper Eq. 2 — Decoder (autoregressive reconstruction):
  L_ro = E[ Σ_{i=1}^{M} ‖h_φ(d_φ([z_rl, sg(z_{1:i-1})]))_i − sg(z_i)‖² ]
  Reconstruct VLA tokens autoregressively from z_rl to enforce information
  preservation in the bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionTokenEncoder(nn.Module):
    """
    Paper Eq. 1: Compress VLA action_queries (B, M, H) → rl_token (B, 1, D).

    Appends a learnable e_rl to the token sequence and processes with
    self-attention (TransformerEncoderLayer). The output at the e_rl
    position is projected to the bottleneck dimension.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Learnable RL embedding e_rl (appended to token sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        # Self-attention encoder layers (paper: g_φ processes [z_{1:M}, e_rl])
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=input_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.bottleneck_proj = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, action_queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_queries: (B, M, H) from frozen VLA
        Returns:
            rl_token: (B, 1, D_bottleneck)
        """
        action_queries = action_queries.float()
        B = action_queries.size(0)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, H)
        seq = torch.cat([action_queries, cls], dim=1)         # (B, M+1, H)
        for layer in self.self_attn_layers:
            seq = layer(seq)                                  # (B, M+1, H)
        rl_token = self.bottleneck_proj(seq[:, -1:, :])       # (B, 1, D)
        return rl_token


class ActionTokenDecoder(nn.Module):
    """
    Paper Eq. 2: Autoregressive reconstruction of VLA tokens from z_rl.

    L_ro = E[ Σ_i ‖h_φ(d_φ([z_rl, sg(z_{1:i-1})]))_i − sg(z_i)‖² ]

    The decoder takes z_rl as prefix, and autoregressively reconstructs
    each VLA token conditioned on z_rl and previously reconstructed tokens.
    A causal mask ensures position i can only attend to positions < i
    (plus the z_rl prefix which is always visible).
    """

    def __init__(
        self,
        bottleneck_dim: int = 256,
        output_dim: int = 2048,
        chunk_len: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.output_dim = output_dim
        self.expand_proj = nn.Linear(bottleneck_dim, output_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_len, output_dim) * 0.02)

        # Self-attention decoder layers with causal masking
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=num_heads,
                dim_feedforward=output_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        rl_token: torch.Tensor,
        target_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            rl_token: (B, 1, D_bottleneck)
            target_tokens: (B, M, H) stop-gradient VLA tokens for teacher forcing.
                           If None, uses learned positional embeddings (inference mode).
        Returns:
            reconstructed: (B, M, H)
        """
        B = rl_token.size(0)
        prefix = self.expand_proj(rl_token)                   # (B, 1, H)

        if target_tokens is not None:
            # Training: teacher forcing with stop-gradient targets
            # Sequence: [z_rl, sg(z_1), sg(z_2), ..., sg(z_{M-1})]
            # Target:   [z_1,  z_2,     z_3,     ...,  z_M        ]
            # Shifted input: z_rl is position 0, z_1 is position 1, etc.
            shifted_input = target_tokens[:, :-1, :].detach()  # (B, M-1, H)
            seq = torch.cat([prefix, shifted_input], dim=1)    # (B, M, H)
            seq = seq + self.pos_embed                         # add positional info
        else:
            # Inference: use positional embeddings (no teacher forcing)
            seq = prefix.expand(-1, self.chunk_len, -1) + self.pos_embed  # (B, M, H)

        # Causal mask: position i can only attend to positions <= i
        # This ensures autoregressive structure
        M = seq.size(1)
        causal_mask = torch.triu(
            torch.ones(M, M, device=seq.device, dtype=torch.bool), diagonal=1
        )  # True = masked out

        for layer in self.self_attn_layers:
            seq = layer(seq, src_mask=causal_mask, is_causal=True)

        return seq  # (B, M, H) — each position predicts the corresponding target


class ActionTokenEncoderDecoder(nn.Module):
    """
    Combined Encoder-Decoder for ActionToken pretraining.

    Training: autoregressive reconstruction with teacher forcing.
    Inference: encoder only (decoder not used during RL).
    """

    def __init__(
        self,
        input_dim: int = 2048,
        bottleneck_dim: int = 256,
        chunk_len: int = 8,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = ActionTokenEncoder(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            num_layers=encoder_layers,
            dropout=dropout,
        )
        self.decoder = ActionTokenDecoder(
            bottleneck_dim=bottleneck_dim,
            output_dim=input_dim,
            chunk_len=chunk_len,
            num_heads=num_heads,
            num_layers=decoder_layers,
            dropout=dropout,
        )

    def encode(self, action_queries: torch.Tensor) -> torch.Tensor:
        return self.encoder(action_queries)

    def decode(self, rl_token: torch.Tensor, target_tokens=None) -> torch.Tensor:
        return self.decoder(rl_token, target_tokens)

    def forward(self, action_queries: torch.Tensor):
        """
        Full encode-decode pass with autoregressive reconstruction loss.

        Paper Eq. 2:
          L_ro = E[ Σ_i ‖reconstructed_i − sg(z_i)‖² ]

        Returns:
            rl_token: (B, 1, D)
            recon_loss: scalar MSE reconstruction loss
        """
        action_queries = action_queries.float()
        rl_token = self.encoder(action_queries)
        # Autoregressive decode with teacher forcing
        reconstructed = self.decoder(rl_token, target_tokens=action_queries.detach())
        recon_loss = F.mse_loss(reconstructed, action_queries.detach())
        return rl_token, recon_loss
