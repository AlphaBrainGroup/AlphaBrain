"""VLA policy + value head wrappers for vanilla VLA-PPO.

Policy:
    π(a | obs) = N(VLA(images, instr).action_mean, σ² · I)
    σ is a fixed scalar (default 0.1). action ∈ ℝ^{chunk_len × action_dim}.

Value head:
    V(obs) = MLP(pool(action_queries))
    Pool: mean over chunk_len dim → (B, H) → MLP → scalar.

These wrap an existing QwenOFT (no monkey-patch). The VLA must be
loaded in TRAIN mode with all params requires_grad=True for backprop
through forward; gradient checkpointing is recommended (set by trainer).
"""
from typing import List, Tuple

import torch
import torch.nn as nn


_LOG_2PI = 1.8378770664093453  # log(2π)


def _gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, std: float) -> torch.Tensor:
    """Per-sample log-prob of a chunked action under an isotropic Gaussian.

    action / mean: (B, chunk_len, action_dim). Returns (B,) — sum over all
    action dims and chunk steps (joint Gaussian over chunk_len·action_dim
    independent components).
    """
    diff = (action - mean) ** 2
    # log p(x) = -0.5 [(x-μ)²/σ² + log(2π) + 2 log σ]   per dim
    log_var = 2.0 * torch.log(torch.as_tensor(std, dtype=action.dtype, device=action.device))
    log_p_per_dim = -0.5 * (diff / (std ** 2) + _LOG_2PI + log_var)
    return log_p_per_dim.sum(dim=(-2, -1))


class VLAPolicy:
    """Thin wrapper around a QwenOFT VLA exposing PPO-friendly hooks.

    Stateless — does not own parameters; the VLA is the policy.

    Usage:
        policy = VLAPolicy(vla, fixed_std=0.1)
        # Rollout (no grad if you want):
        with torch.no_grad():
            action_mean = policy.forward_mean(images, instr)
            sampled, lp = policy.sample(action_mean)
        # PPO update (grad on):
        new_mean = policy.forward_mean(images, instr)
        new_lp = policy.log_prob_of_with_mean(new_mean, taken_action)
    """

    def __init__(self, vla, fixed_std: float = 0.1):
        self.vla = vla
        self.fixed_std = float(fixed_std)

    # ── Inference paths ──────────────────────────────────────────────

    def forward_mean(self, batch_images: List, instructions: List[str]) -> torch.Tensor:
        """Run the full VLA forward and return the deterministic action mean.

        Grad flows iff self.vla parameters have requires_grad=True AND no
        outer torch.no_grad() context.
        Returns: (B, chunk_len, action_dim) float tensor.
        """
        # Use the grad-enabled inner methods (vla.get_vla_action() itself
        # is @torch.no_grad-decorated — we skip that wrapper).
        action_queries = self.vla.get_action_queries(batch_images, instructions)
        with torch.autocast("cuda", dtype=torch.float32):
            action_mean = self.vla.action_model.predict_action(action_queries)
        return action_mean

    def forward_mean_and_features(
        self, batch_images: List, instructions: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Like forward_mean but also returns the (B, chunk_len, H)
        action_queries tensor so the value head can consume it without a
        second VLA forward."""
        action_queries = self.vla.get_action_queries(batch_images, instructions)
        with torch.autocast("cuda", dtype=torch.float32):
            action_mean = self.vla.action_model.predict_action(action_queries)
        return action_mean, action_queries

    # ── Sampling + log-prob ──────────────────────────────────────────

    def sample(self, action_mean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add fixed-std Gaussian noise. Returns (sampled_action, log_prob)."""
        noise = torch.randn_like(action_mean) * self.fixed_std
        sampled = action_mean + noise
        log_prob = _gaussian_log_prob(sampled, action_mean, self.fixed_std)
        return sampled, log_prob

    def log_prob_of_with_mean(
        self, action_mean: torch.Tensor, taken_action: torch.Tensor
    ) -> torch.Tensor:
        """Compute log π(taken_action | obs) given a pre-computed mean.
        Useful in PPO update where we already re-ran VLA forward."""
        return _gaussian_log_prob(taken_action, action_mean, self.fixed_std)


# ── Free helper (works on any policy that has .forward_mean) ─────────

def vla_log_prob_of(
    policy: VLAPolicy,
    batch_images: List,
    instructions: List[str],
    taken_action: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-shot: re-forward VLA, compute log-prob of taken_action.
    Returns (log_prob, action_mean)."""
    mean = policy.forward_mean(batch_images, instructions)
    lp = policy.log_prob_of_with_mean(mean, taken_action)
    return lp, mean


# ── Value head ────────────────────────────────────────────────────────

class VLAValueHead(nn.Module):
    """V(s) ≈ MLP(mean-pool over chunk_len of action_queries).

    Tiny — runs on the same GPU as VLA, adds ~half a MB of params.
    """

    def __init__(self, vla_hidden_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vla_hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, action_queries: torch.Tensor) -> torch.Tensor:
        """action_queries: (B, chunk_len, H) → (B,) scalar value."""
        pooled = action_queries.float().mean(dim=1)  # (B, H)
        return self.net(pooled).squeeze(-1)
