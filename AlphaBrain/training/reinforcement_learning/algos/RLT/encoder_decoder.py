"""RL Token encoder-decoder — faithful to the reference construction.

Reference: "RL Token: Bootstrapping Online RL with Vision-Language-Action
Models" (Physical Intelligence, 2026).

Eq. 1 (Encoder):
    z_rl = g_φ([z_{1:M}, e_rl])_{M+1}

    Take the VLA's final-layer token embeddings ``z_{1:M}`` (one per input
    token to the VLA), append a learned embedding ``e_rl``, process the
    augmented sequence with a small self-attention encoder ``g_φ``, and
    take the output at the ``e_rl`` position as the RL token.

    The RL token has the **same hidden dim as the VLA backbone** — the
    bottleneck comes from collapsing M tokens to 1 token, not from reducing
    per-token width. No extra projection.

Eq. 2 (Decoder, autoregressive reconstruction):
    L_ro = E_D[ Σ_{i=1}^{M}  ‖ h_φ( d_φ([z_rl, sg(z_{1:i-1})]) )_i  −  sg(z_i) ‖² ]

    A decoder transformer ``d_φ`` with a linear output projection ``h_φ``
    is trained to autoregressively reconstruct each ``z_i`` from the RL
    token plus the stop-gradient previous tokens. Enforcing that ``z_rl``
    is sufficient for reconstruction ensures it preserves task-relevant
    information from the VLA embeddings.

The decoder here uses cross-attention against ``z_rl`` as memory (the
reference's encoder-decoder construction), with a causally-masked target
stream of shifted stop-gradient VLA tokens. This contrasts with the
sibling ``RLT_a`` decoder, which is a self-attention-only causal
transformer with ``z_rl`` as a prefix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RLTokenEncoder(nn.Module):
    """Eq. 1: compress VLA embeddings ``(B, M, H)`` → ``z_rl`` ``(B, 1, H)``.

    Appends a learnable ``e_rl`` to the token sequence and processes with a
    bidirectional self-attention encoder. The output at the ``e_rl``
    position is the RL token, kept at the VLA hidden dim ``H``.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        dim_feedforward = dim_feedforward or hidden_dim * 4

        # e_rl: learnable embedding appended to the VLA token sequence
        self.rl_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # g_φ: small self-attention encoder over [z_{1:M}, e_rl]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        vla_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            vla_embeddings: ``(B, M, H)`` — the VLA's final-layer token
                embeddings ``z_{1:M}``.
            key_padding_mask: optional ``(B, M)`` bool mask; ``True`` marks
                padding positions to be ignored by attention. The appended
                ``e_rl`` position is always attended to.
        Returns:
            z_rl: ``(B, 1, H)`` — the RL token.
        """
        B = vla_embeddings.size(0)
        vla_embeddings = vla_embeddings.to(self.rl_embed.dtype)

        e_rl = self.rl_embed.expand(B, -1, -1)                  # (B, 1, H)
        seq = torch.cat([vla_embeddings, e_rl], dim=1)          # (B, M+1, H)

        if key_padding_mask is not None:
            # The e_rl position is never padded.
            pad_for_rl = torch.zeros(
                (B, 1), dtype=torch.bool, device=vla_embeddings.device
            )
            kp_mask = torch.cat([key_padding_mask, pad_for_rl], dim=1)
        else:
            kp_mask = None

        seq = self.encoder(seq, src_key_padding_mask=kp_mask)   # (B, M+1, H)
        z_rl = seq[:, -1:, :]                                   # (B, 1, H)
        return z_rl


class RLTokenDecoder(nn.Module):
    """Eq. 2: autoregressive reconstruction of ``z_{1:M}`` from ``z_rl``.

    Implemented with ``nn.TransformerDecoderLayer`` — the reference
    encoder-decoder construction — where:

      * memory = ``z_rl`` (``B, 1, H``)
      * tgt    = ``[BOS, sg(z_1), ..., sg(z_{M-1})]`` with causal mask
      * output = ``h_φ(d_φ(...))``; position ``i`` predicts ``sg(z_i)``

    A learned BOS embedding seeds the target stream so position 0 has a
    well-defined input that is independent of any ``z_i``. ``h_φ`` is a
    final ``Linear(H, H)``.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        max_len: int = 4096,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        dim_feedforward = dim_feedforward or hidden_dim * 4

        self.bos = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.tgt_pos_embed = nn.Parameter(
            torch.randn(1, max_len, hidden_dim) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # h_φ: final linear projection to VLA hidden dim
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        z_rl: torch.Tensor,
        vla_embeddings_sg: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced autoregressive reconstruction.

        Args:
            z_rl: ``(B, 1, H)`` — the RL token (gradients flow back through
                the encoder via this).
            vla_embeddings_sg: ``(B, M, H)`` — stop-gradient VLA
                embeddings. The caller must pass the detached version.
        Returns:
            reconstructed: ``(B, M, H)``. Position ``i`` is the prediction
            of ``sg(z_i)``.
        """
        B, M, H = vla_embeddings_sg.shape
        assert M + 1 <= self.tgt_pos_embed.size(1), (
            f"RLTokenDecoder max_len={self.tgt_pos_embed.size(1)} too small "
            f"for sequence length {M}; increase max_len."
        )

        # Shifted-right target stream: [BOS, sg(z_1), ..., sg(z_{M-1})]
        bos = self.bos.expand(B, -1, -1)                         # (B, 1, H)
        tgt = torch.cat([bos, vla_embeddings_sg[:, :-1, :]], dim=1)  # (B, M, H)
        tgt = tgt + self.tgt_pos_embed[:, :M, :]

        # Causal mask over the target stream (position i attends to ≤ i)
        causal_mask = torch.triu(
            torch.ones(M, M, device=tgt.device, dtype=torch.bool),
            diagonal=1,
        )

        out = self.decoder(
            tgt=tgt,
            memory=z_rl,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
        )                                                        # (B, M, H)
        reconstructed = self.output_proj(out)
        return reconstructed


class RLTokenEncoderDecoder(nn.Module):
    """Combined encoder-decoder used during Phase-1 reconstruction training.

    Forward pass returns ``(z_rl, L_ro)`` where ``L_ro`` is the
    reference's Eq. 2 MSE loss against stop-gradient VLA embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        max_len: int = 4096,
    ):
        super().__init__()
        self.encoder = RLTokenEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = RLTokenDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

    def encode(
        self,
        vla_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``z_rl`` only (used at downstream RL inference time)."""
        return self.encoder(vla_embeddings, key_padding_mask=key_padding_mask)

    def forward(
        self,
        vla_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ):
        """Full encode + autoregressive reconstruction.

        ``vla_embeddings`` should already be detached from the VLA
        computation graph (``L_ro`` is defined against stop-gradient
        targets). The encoder receives the same inputs; its gradients flow
        through the RL token only, never back into the VLA.
        """
        vla_embeddings = vla_embeddings.detach()
        z_rl = self.encoder(vla_embeddings, key_padding_mask=key_padding_mask)
        reconstructed = self.decoder(z_rl, vla_embeddings)

        if key_padding_mask is not None:
            # Only compute the loss over non-padding positions.
            valid = (~key_padding_mask).to(reconstructed.dtype).unsqueeze(-1)
            sq_err = (reconstructed - vla_embeddings) ** 2 * valid
            denom = valid.sum().clamp_min(1.0) * reconstructed.size(-1)
            recon_loss = sq_err.sum() / denom
        else:
            recon_loss = F.mse_loss(reconstructed, vla_embeddings)

        return z_rl, recon_loss
