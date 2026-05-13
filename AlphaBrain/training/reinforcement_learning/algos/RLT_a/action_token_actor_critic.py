"""
ActionToken Actor-Critic, following the RL Token paper (Physical Intelligence)
closely on the actor/critic side (the deviations from the paper live in
action_token_encoder_decoder.py and action_token_trainer.py).

Key design choices from the paper:
  - Actor (Eq. 4): π_θ(a | x, ã) = N(μ_θ(x, ã), σ²I)
    The actor DIRECTLY outputs the action chunk, conditioned on (rl_token, vla_ref).
    VLA reference is an INPUT to the network, NOT a structural residual.
    BC regularization β‖a - ã‖² in the LOSS keeps actions close to VLA.
    Reference-action dropout (50%) prevents identity collapse.
  - Critic (Eq. 3): Q(s, a) — twin Q-networks (TD3-style).
"""

import copy

import torch
import torch.nn as nn


class ActionTokenActor(nn.Module):
    """
    ActionToken actor from paper (Eq. 4-5).

    π_θ(a_{1:C} | x, ã_{1:C}) = N(μ_θ(x, ã_{1:C}), σ²I)

    The network takes (rl_token, vla_reference_action) as input and
    DIRECTLY outputs the full action chunk. The VLA reference is just
    a conditioning signal — the BC regularization in the loss (not in
    the architecture) keeps the output close to VLA.
    """

    def __init__(
        self,
        bottleneck_dim: int = 256,
        action_dim: int = 7,
        chunk_len: int = 8,
        hidden_dim: int = 256,    # paper: 256 for most tasks, 512 for hard
        ref_dropout: float = 0.5,  # paper: 50%
        fixed_std: float = 0.1,    # paper: small fixed std
        prop_dim: int = 0,         # proprioceptive state dim (paper: eef_pos+axisangle+gripper=8)
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.ref_dropout = ref_dropout
        self.prop_dim = prop_dim

        flat_action_dim = action_dim * chunk_len
        input_dim = bottleneck_dim + prop_dim + flat_action_dim

        # Paper Appendix B: two-layer MLP (256 hidden) for most tasks,
        # three-layer MLP (512 hidden) for screw task
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_action_dim),
        )

        # Kaiming init for hidden layers (default), small normal for output layer.
        # Paper: actor directly outputs actions; BC regularization in the loss
        # (not architecture) keeps output close to VLA reference.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

        # Paper: small fixed standard deviation
        self.register_buffer("fixed_std", torch.tensor(fixed_std))

    def _get_mean(
        self,
        rl_token: torch.Tensor,
        vla_action: torch.Tensor,
        prop_state: torch.Tensor = None,
        apply_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Compute μ_θ(x, ã) — the mean action output (paper Eq. 4).

        The actor DIRECTLY outputs the full action chunk, conditioned on
        (z_rl, s_p, ã). NO residual connection — BC regularization β‖a - ã‖²
        in the loss (Eq. 5) keeps actions close to VLA reference.
        Reference-action dropout (50%) prevents identity collapse.
        """
        B = rl_token.size(0)
        rl_feat = rl_token.squeeze(1) if rl_token.dim() == 3 else rl_token  # (B, D)
        vla_flat = vla_action.reshape(B, -1)  # (B, C*A)

        # Reference-action dropout (paper Sec. IV-B):
        # zero out VLA ref input for a fraction of the batch
        if apply_dropout and self.ref_dropout > 0:
            mask = (torch.rand(B, 1, device=rl_feat.device) > self.ref_dropout).float()
            vla_flat_input = vla_flat * mask
        else:
            vla_flat_input = vla_flat

        if self.prop_dim > 0:
            if prop_state is None:
                prop_state = torch.zeros(B, self.prop_dim, device=rl_feat.device,
                                         dtype=rl_feat.dtype)
            x = torch.cat([rl_feat, prop_state, vla_flat_input], dim=-1)
        else:
            x = torch.cat([rl_feat, vla_flat_input], dim=-1)

        raw_output = self.net(x)  # (B, C*A)
        return raw_output.reshape(B, self.chunk_len, self.action_dim)

    def forward(
        self,
        rl_token: torch.Tensor,
        vla_action: torch.Tensor,
        prop_state: torch.Tensor = None,
        deterministic: bool = False,
    ):
        """
        Args:
            rl_token: (B, 1, D) or (B, D)
            vla_action: (B, chunk_len, action_dim) — VLA reference
            prop_state: (B, prop_dim) — proprioceptive state (eef_pos+axisangle+gripper)
        Returns:
            action: (B, chunk_len, action_dim)
            log_prob: (B,) or None if deterministic
        """
        mean = self._get_mean(rl_token, vla_action, prop_state,
                              apply_dropout=(self.training and not deterministic))

        if deterministic:
            return mean, None

        std = self.fixed_std.expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=(-2, -1))  # (B,)
        return action, log_prob

    def log_prob_of(
        self,
        rl_token: torch.Tensor,
        vla_action: torch.Tensor,
        taken_action: torch.Tensor,
        prop_state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute log_prob of a previously taken action under current policy."""
        mean = self._get_mean(rl_token, vla_action, prop_state, apply_dropout=False)
        std = self.fixed_std.expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(taken_action).sum(dim=(-2, -1))  # (B,)


def soft_update_target(source: nn.Module, target: nn.Module, tau: float = 0.005):
    """Polyak averaging: target = (1 - tau) * target + tau * source."""
    with torch.no_grad():
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


class ActionTokenQCritic(nn.Module):
    """
    Twin Q-critic from RL Token paper (Eq. 3, following TD3).

    Q_ψ(x, a_{1:C}) takes the RL token state AND the action chunk as input.
    Contains two independent Q-networks; use min(Q1, Q2) for target values.
    """

    def __init__(
        self,
        bottleneck_dim: int = 256,
        action_dim: int = 7,
        chunk_len: int = 8,
        hidden_dim: int = 256,
        prop_dim: int = 0,  # proprioceptive state dim
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.prop_dim = prop_dim

        flat_action_dim = action_dim * chunk_len
        input_dim = bottleneck_dim + prop_dim + flat_action_dim

        # Twin Q-networks (TD3 style)
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        rl_token: torch.Tensor,
        action: torch.Tensor,
        prop_state: torch.Tensor = None,
    ) -> tuple:
        """
        Returns: q1: (B,), q2: (B,)
        """
        if rl_token.dim() == 3:
            rl_token = rl_token.squeeze(1)
        B = rl_token.size(0)
        action_flat = action.reshape(B, -1)
        if self.prop_dim > 0:
            if prop_state is None:
                prop_state = torch.zeros(B, self.prop_dim, device=rl_token.device,
                                         dtype=rl_token.dtype)
            x = torch.cat([rl_token, prop_state, action_flat], dim=-1)
        else:
            x = torch.cat([rl_token, action_flat], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_forward(
        self,
        rl_token: torch.Tensor,
        action: torch.Tensor,
        prop_state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Single Q1 forward (used for actor loss to save compute)."""
        if rl_token.dim() == 3:
            rl_token = rl_token.squeeze(1)
        B = rl_token.size(0)
        action_flat = action.reshape(B, -1)
        if self.prop_dim > 0:
            if prop_state is None:
                prop_state = torch.zeros(B, self.prop_dim, device=rl_token.device,
                                         dtype=rl_token.dtype)
            x = torch.cat([rl_token, prop_state, action_flat], dim=-1)
        else:
            x = torch.cat([rl_token, action_flat], dim=-1)
        return self.q1(x).squeeze(-1)


# ── Keep old V(s) critic for backward compatibility with PPO path ──

class ActionTokenCritic(nn.Module):
    """State value estimator V(s) from rl_token. (Legacy, for PPO path only.)"""

    def __init__(
        self,
        bottleneck_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rl_token: torch.Tensor) -> torch.Tensor:
        if rl_token.dim() == 3:
            rl_token = rl_token.squeeze(1)
        return self.net(rl_token).squeeze(-1)
