"""Experience Replay (ER) — the canonical rehearsal-based CL baseline.

Stores samples from previously learned tasks and provides mixed batches
to mitigate catastrophic forgetting.

Supports:
- Reservoir sampling for memory-efficient storage
- Per-task buffer management
- Configurable replay ratio for batch mixing
- Conforms to the :class:`CLAlgorithm` interface.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from AlphaBrain.training.continual_learning.algorithms.base import (
    CLAlgorithm,
    CLContext,
)

logger = logging.getLogger(__name__)


class ER(CLAlgorithm):
    """Experience Replay — stores past-task samples and mixes them into training batches.

    Uses reservoir sampling to maintain a fixed-size buffer per task.
    At each training step the trainer invokes :meth:`modify_batch`, which
    replaces a configurable fraction of the current-task batch with samples
    drawn from the buffer.

    The actual memory update happens in :meth:`on_task_end` (populating the
    full buffer from the finished task's dataset via reservoir sampling), so
    the per-step :meth:`observe` hook is a no-op.

    Usage (direct):
        er = ER(
            buffer_size_per_task=500,
            replay_batch_ratio=0.3,
            balanced_sampling=False,
        )
        # Populate after finishing task 0:
        er.populate_from_dataset(task_id=0, dataset=task0_dataset)
        # Draw replay samples while training task 1:
        replay_samples = er.sample(batch_size=4)  # list[dict]

    Usage (via trainer hooks):
        er.modify_batch(current_batch, task_id)   # every step
        er.on_task_end(CLContext(task_id=0, task_dataset=task0_dataset))
    """

    def __init__(
        self,
        buffer_size_per_task: int = 500,
        replay_batch_ratio: float = 0.3,
        balanced_sampling: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            buffer_size_per_task: Maximum number of samples stored per task.
            replay_batch_ratio: Fraction of each training batch replaced by
                replay samples. The trainer used to hold this config —
                we pulled it into the algorithm so the trainer stays
                algorithm-agnostic.
            balanced_sampling: If True, draw equally from every stored task
                rather than uniformly across all stored samples.
            seed: Random seed for reproducibility.
        """
        self.buffer_size_per_task = buffer_size_per_task
        self.replay_batch_ratio = replay_batch_ratio
        self.balanced_sampling = balanced_sampling
        self.seed = seed
        self.rng = random.Random(seed)

        # task_id -> list of stored samples
        self._buffers: Dict[int, List[dict]] = {}
        # Total samples across all tasks
        self._total_samples = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_tasks(self) -> int:
        return len(self._buffers)

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def is_empty(self) -> bool:
        return self._total_samples == 0

    def get_task_ids(self) -> List[int]:
        return sorted(self._buffers.keys())

    def get_task_size(self, task_id: int) -> int:
        return len(self._buffers.get(task_id, []))

    def clear(self):
        self._buffers.clear()
        self._total_samples = 0

    # ------------------------------------------------------------------
    # Population / sampling (usable directly, also driven by hooks below)
    # ------------------------------------------------------------------
    def populate_from_dataset(
        self,
        task_id: int,
        dataset: Dataset,
        num_samples: Optional[int] = None,
    ):
        """Store samples from a dataset into the buffer using reservoir sampling.

        Args:
            task_id: Identifier for the task.
            dataset: Dataset to sample from (must support __len__ and __getitem__).
            num_samples: Number of samples to store. Defaults to buffer_size_per_task.
        """
        if num_samples is None:
            num_samples = self.buffer_size_per_task

        n = len(dataset)
        k = min(num_samples, n)

        indices = list(range(n))
        self.rng.shuffle(indices)
        selected_indices = sorted(indices[:k])

        samples = []
        for idx in selected_indices:
            try:
                sample = dataset[idx]
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to read sample {idx} for task {task_id}: {e}")
                continue

        if task_id in self._buffers:
            self._total_samples -= len(self._buffers[task_id])
        self._buffers[task_id] = samples
        self._total_samples += len(samples)

        logger.info(
            f"Replay buffer: stored {len(samples)} samples for task {task_id} "
            f"(total: {self._total_samples} across {self.num_tasks} tasks)"
        )

    def sample(self, batch_size: int) -> List[dict]:
        """Sample a batch uniformly from all stored tasks.

        Returns an empty list if the buffer is empty.
        """
        if self.is_empty():
            return []

        all_samples = []
        for task_samples in self._buffers.values():
            all_samples.extend(task_samples)

        if batch_size <= len(all_samples):
            return self.rng.sample(all_samples, k=batch_size)
        return self.rng.choices(all_samples, k=batch_size)

    def sample_balanced(self, batch_size: int) -> List[dict]:
        """Sample a batch with equal representation from each stored task."""
        if self.is_empty():
            return []

        samples_per_task = max(1, batch_size // self.num_tasks)
        result = []

        for task_samples in self._buffers.values():
            k = min(samples_per_task, len(task_samples))
            result.extend(self.rng.choices(task_samples, k=k))

        while len(result) < batch_size:
            task_id = self.rng.choice(list(self._buffers.keys()))
            result.append(self.rng.choice(self._buffers[task_id]))
        return result[:batch_size]

    # ------------------------------------------------------------------
    # CLAlgorithm hooks
    # ------------------------------------------------------------------
    def modify_batch(self, batch: List[dict], task_id: int) -> List[dict]:
        """Mix the current-task batch with replay samples.

        Uses `max(2, round(batch_size * replay_batch_ratio))` replay samples,
        clipped to leave at least one current-task sample. When the buffer is
        empty (first task) the batch is returned unchanged.

        The current batch is shuffled before slicing so we don't always drop
        the same elements when the DataLoader's intra-batch order is biased.
        """
        if self.is_empty():
            return batch

        batch_size = len(batch)
        num_replay = max(2, round(batch_size * self.replay_batch_ratio))
        num_replay = min(num_replay, batch_size - 1)  # keep ≥1 current sample
        num_current = batch_size - num_replay

        if self.balanced_sampling:
            replay_samples = self.sample_balanced(num_replay)
        else:
            replay_samples = self.sample(num_replay)

        current_shuffled = list(batch)
        self.rng.shuffle(current_shuffled)
        return current_shuffled[:num_current] + replay_samples

    def on_task_end(self, context: CLContext) -> None:
        """Populate the buffer from the finished task's dataset."""
        task_id = context.task_id
        dataset = context.task_dataset
        if task_id is None or dataset is None:
            logger.warning(
                "ER.on_task_end: task_id/dataset missing from context; "
                "skipping populate."
            )
            return
        self.populate_from_dataset(task_id=task_id, dataset=dataset)

    def describe(self) -> Dict[str, Any]:
        return {
            "algorithm": self.name,
            "buffer_size_per_task": self.buffer_size_per_task,
            "replay_batch_ratio": self.replay_batch_ratio,
            "balanced_sampling": self.balanced_sampling,
            "num_tasks_stored": self.num_tasks,
            "total_samples": self._total_samples,
        }

    def metrics(self) -> Dict[str, float]:
        return {"er_buffer_size": float(self._total_samples)}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Return serializable state for checkpointing.

        Only metadata is serialized — the actual sample tensors are not saved
        (they can be large).  On resume the trainer replays :meth:`on_task_end`
        for each completed task to rebuild the buffer from its dataset.
        """
        return {
            "algorithm": self.name,
            "buffer_size_per_task": self.buffer_size_per_task,
            "replay_batch_ratio": self.replay_batch_ratio,
            "balanced_sampling": self.balanced_sampling,
            "seed": self.seed,
            "num_tasks": self.num_tasks,
            "total_samples": self._total_samples,
            "task_sizes": {k: len(v) for k, v in self._buffers.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore metadata from a snapshot produced by :meth:`state_dict`.

        Hyperparameters are restored; samples must be re-populated by replaying
        each completed task's dataset through :meth:`on_task_end`.
        """
        self.buffer_size_per_task = state.get(
            "buffer_size_per_task", self.buffer_size_per_task
        )
        self.replay_batch_ratio = state.get(
            "replay_batch_ratio", self.replay_batch_ratio
        )
        self.balanced_sampling = state.get(
            "balanced_sampling", self.balanced_sampling
        )
        self.seed = state.get("seed", self.seed)
        self.rng = random.Random(self.seed)
        self._buffers = {}
        self._total_samples = 0

    def __repr__(self) -> str:
        task_info = ", ".join(
            f"task_{k}={len(v)}" for k, v in sorted(self._buffers.items())
        )
        return (
            f"ER(buffer_size_per_task={self.buffer_size_per_task}, "
            f"tasks={self.num_tasks}, total={self._total_samples}, [{task_info}])"
        )
