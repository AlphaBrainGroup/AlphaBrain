"""
Off-policy replay buffer for RLT_a training (RL Token paper style).

Stores transitions as detached CPU tensors in a ring buffer.
Each transition stores:
  - rl_token (state), action_taken, reward, next_rl_token (next state), done
  - vla_ref_action: the VLA reference action chunk (for BC regularization)
  - next_vla_action: VLA reference at next state (for target policy sampling)
  - prop_state / next_prop_state: proprioceptive state (eef_pos+axisangle+gripper, 8-dim)
  - task_id: integer task index for per-task stratified sampling
"""

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity ring buffer for off-policy experience replay."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        # task_id index: task_id -> list of buffer positions (for stratified sampling)
        self._task_index: dict = defaultdict(list)

    def push(
        self,
        rl_token: torch.Tensor,        # (1, D) or (D,)
        vla_action: torch.Tensor,       # (chunk_len, action_dim) — VLA reference
        action_taken: torch.Tensor,     # (chunk_len, action_dim) — actual executed
        reward: float,
        next_rl_token: torch.Tensor,    # (1, D) or (D,)
        next_vla_action: torch.Tensor,  # (chunk_len, action_dim)
        done: bool,
        task_id: int = 0,
        prop_state: Optional[torch.Tensor] = None,       # (prop_dim,)
        next_prop_state: Optional[torch.Tensor] = None,  # (prop_dim,)
    ):
        """Store a single transition (all tensors detached to CPU)."""
        # Default zero prop states if not provided
        if prop_state is None:
            prop_state = torch.zeros(8, dtype=torch.float32)
        if next_prop_state is None:
            next_prop_state = torch.zeros(8, dtype=torch.float32)

        transition = (
            rl_token.detach().cpu(),
            vla_action.detach().cpu(),
            action_taken.detach().cpu(),
            torch.tensor(reward, dtype=torch.float32),
            next_rl_token.detach().cpu(),
            next_vla_action.detach().cpu(),
            torch.tensor(float(done), dtype=torch.float32),
            prop_state.detach().cpu(),
            next_prop_state.detach().cpu(),
            task_id,  # stored as plain int at index 9, stripped in _collect
        )
        if len(self.buffer) < self.capacity:
            idx = len(self.buffer)
            self.buffer.append(transition)
        else:
            idx = self.pos
            # Remove old task_id index entry for overwritten slot
            old_task_id = self.buffer[idx][9]
            old_list = self._task_index[old_task_id]
            # Efficiently remove the old index (swap with last element)
            try:
                pos_in_list = old_list.index(idx)
                old_list[pos_in_list] = old_list[-1]
                old_list.pop()
            except ValueError:
                pass
            self.buffer[idx] = transition

        self._task_index[task_id].append(idx)
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random mini-batch (uniform over all transitions).

        Returns:
            Tuple of (rl_tokens, vla_actions, actions_taken, rewards,
                       next_rl_tokens, next_vla_actions, dones,
                       prop_states, next_prop_states)
            Each tensor has batch dim prepended and is moved to `device`.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return self._collect(indices, device)

    def sample_balanced(
        self,
        batch_size: int,
        n_tasks: int,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, ...]:
        """
        Per-task stratified sampling: sample equal number of transitions from each task.

        Ensures each task contributes equally to each gradient update, matching
        GRPO's equal-frequency multi-task update property.

        Args:
            batch_size: total transitions to sample
            n_tasks: number of tasks (0..n_tasks-1)
            device: target device
        """
        per_task = max(1, batch_size // n_tasks)
        all_indices = []

        for tid in range(n_tasks):
            pool = self._task_index.get(tid, [])
            if not pool:
                continue
            n_sample = min(per_task, len(pool))
            chosen = np.random.choice(pool, n_sample, replace=(n_sample > len(pool)))
            all_indices.extend(chosen.tolist())

        if not all_indices:
            # Fallback to uniform if index is empty
            return self.sample(batch_size, device)

        # Trim or pad to batch_size
        if len(all_indices) > batch_size:
            all_indices = all_indices[:batch_size]

        return self._collect(all_indices, device)

    def _collect(self, indices, device: str) -> Tuple[torch.Tensor, ...]:
        """Collect transitions by index and move to device (strips task_id at index 9)."""
        batch = [self.buffer[i] for i in indices]
        # Each transition: (rl_tok, vla_act, act_taken, rew, next_rl_tok,
        #                    next_vla_act, done, prop, next_prop, task_id)
        # Strip task_id (index 9) before stacking tensors
        tensor_fields = [t[:9] for t in batch]
        return tuple(torch.stack(x).to(device) for x in zip(*tensor_fields))

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int = 256) -> bool:
        """Whether buffer has enough samples for at least one batch."""
        return len(self.buffer) >= min_size

    def task_counts(self) -> dict:
        """Return number of transitions per task (for diagnostics)."""
        return {tid: len(indices) for tid, indices in self._task_index.items()}
