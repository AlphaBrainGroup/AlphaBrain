"""Vanilla VLA + PPO — train the full VLA (including action head) directly
via clipped policy gradient. No encoder, no RLT_a tiny actor. Policy is
the VLA's own action head output + Gaussian noise; PPO log-prob is
recomputed by re-forwarding the VLA each update epoch (expensive)."""

from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_policy import (
    VLAPolicy, VLAValueHead, vla_log_prob_of,
)
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_ppo_rollout import (
    VLAPPOEpisode, VLAPPOStepRecord, vla_ppo_collect,
)
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_ppo_loss import (
    compute_vla_gae, vla_ppo_loss,
)

__all__ = [
    "VLAPolicy", "VLAValueHead", "vla_log_prob_of",
    "VLAPPOEpisode", "VLAPPOStepRecord", "vla_ppo_collect",
    "compute_vla_gae", "vla_ppo_loss",
]
