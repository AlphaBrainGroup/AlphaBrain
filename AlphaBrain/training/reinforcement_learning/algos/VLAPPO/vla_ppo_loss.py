"""GAE + PPO loss for vanilla VLA-PPO.

Key cost driver: each PPO epoch must re-forward the VLA over every
transition in the rollout (because the policy IS the VLA). We mini-batch
to keep peak memory bounded; the trainer chooses the mini-batch size.
"""
from typing import List, Tuple

import torch
import torch.nn.functional as F

from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_policy import (
    VLAPolicy, VLAValueHead,
)
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_ppo_rollout import (
    VLAPPOEpisode,
)


def compute_vla_gae(
    episode: VLAPPOEpisode,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """Per-step GAE advantage + return for one episode.

    Sparse-reward convention: episode reward is delivered at the last
    step (terminal). All earlier rewards are 0.
    """
    n = episode.finish_step
    if n == 0:
        return [], []
    values = [sr.value for sr in episode.step_records[:n]]
    rewards = [0.0] * n
    rewards[-1] = episode.reward

    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0
    next_value = 0.0  # bootstrap from 0 at terminal (sparse, no future reward)
    for t in range(n - 1, -1, -1):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]
    return advantages, returns


def vla_ppo_loss(
    *,
    policy: VLAPolicy,
    value_head: VLAValueHead,
    episodes: List[VLAPPOEpisode],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    micro_batch: int = 4,
    device: str = "cuda",
) -> Tuple[torch.Tensor, dict]:
    """Compute PPO loss summed over all transitions in the rollout.

    Each transition is re-forwarded through the VLA (in mini-batches of
    `micro_batch`) so new_log_prob and new_value have gradient. The
    summed loss is returned; the trainer does .backward() once.

    Returns: (loss, stats). loss is a scalar tensor with grad.
    """
    # ── 1. Flatten transitions + compute GAE per episode ─────────────
    flat_images: List = []
    flat_instrs: List[str] = []
    flat_props: List[torch.Tensor] = []
    flat_actions: List[torch.Tensor] = []
    flat_old_lp: List[float] = []
    flat_old_values: List[float] = []
    flat_advantages: List[float] = []
    flat_returns: List[float] = []

    for ep in episodes:
        adv, ret = compute_vla_gae(ep, gamma, gae_lambda)
        for t in range(ep.finish_step):
            sr = ep.step_records[t]
            flat_images.append(sr.images)
            flat_instrs.append(sr.instruction)
            flat_props.append(sr.prop_state)
            flat_actions.append(sr.action_taken)
            flat_old_lp.append(sr.old_log_prob)
            flat_old_values.append(sr.value)
            flat_advantages.append(adv[t])
            flat_returns.append(ret[t])

    N = len(flat_actions)
    if N == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {"n_steps": 0}

    actions_t = torch.stack(flat_actions).to(device)             # (N, C, A)
    old_lp_t = torch.tensor(flat_old_lp, device=device, dtype=torch.float32)
    old_values_t = torch.tensor(flat_old_values, device=device, dtype=torch.float32)
    adv_t = torch.tensor(flat_advantages, device=device, dtype=torch.float32)
    ret_t = torch.tensor(flat_returns, device=device, dtype=torch.float32)

    # Normalize advantages globally
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    # ── 2. Mini-batched VLA re-forward + per-batch loss accumulation ─
    total_pg = torch.tensor(0.0, device=device)
    total_vf = torch.tensor(0.0, device=device)
    n_batches = 0
    sum_ratio = 0.0
    sum_clip = 0.0

    for start in range(0, N, micro_batch):
        end = min(start + micro_batch, N)
        idx = list(range(start, end))

        batch_images = [flat_images[i] for i in idx]
        batch_instrs = [flat_instrs[i] for i in idx]
        batch_actions = actions_t[start:end]

        # VLA re-forward (with grad)
        new_mean, new_features = policy.forward_mean_and_features(batch_images, batch_instrs)
        new_lp = policy.log_prob_of_with_mean(new_mean, batch_actions)
        new_value = value_head(new_features)

        old_lp_b = old_lp_t[start:end]
        old_v_b = old_values_t[start:end]
        adv_b = adv_t[start:end]
        ret_b = ret_t[start:end]

        ratio = torch.exp(new_lp - old_lp_b)
        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
        pg = -torch.min(surr1, surr2).mean()

        # Value loss (clipped)
        v_clipped = old_v_b + torch.clamp(new_value - old_v_b, -10.0, 10.0)
        vf = torch.max((new_value - ret_b) ** 2, (v_clipped - ret_b) ** 2).mean()

        total_pg = total_pg + pg
        total_vf = total_vf + vf
        n_batches += 1
        sum_ratio += float(ratio.mean().detach().item())
        sum_clip += float(((ratio - 1.0).abs() > clip_eps).float().mean().detach().item())

    pg_loss = total_pg / max(n_batches, 1)
    vf_loss = total_vf / max(n_batches, 1)
    loss = pg_loss + vf_coef * vf_loss

    stats = {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "vf_loss": vf_loss.item(),
        "ratio_mean": sum_ratio / max(n_batches, 1),
        "clip_frac": sum_clip / max(n_batches, 1),
        "advantage_mean": float(adv_t.mean().item()),
        "return_mean": float(ret_t.mean().item()),
        "n_steps": N,
        "n_batches": n_batches,
    }
    return loss, stats
