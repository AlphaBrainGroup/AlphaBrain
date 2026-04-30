"""Rehearsal-based continual learning algorithms.

These methods maintain a memory buffer of past-task samples and replay
them during training on new tasks to mitigate catastrophic forgetting.
The specific sampling / scoring policy differs across algorithms:

- :class:`ER` — Experience Replay: uniform (or per-task balanced) reservoir
  sampling over stored samples.  The canonical rehearsal baseline.
- :class:`MIR` — Maximally Interfered Retrieval (Aljundi et al. 2019):
  scores candidates by how much their loss would increase under a virtual
  SGD step and replays the top-k most-interfered samples.

Planned (not yet implemented):
- **DER / DER++** — Dark Experience Replay: stores logits alongside
  samples and uses them as distillation targets on the replay loss.
- **A-GEM** — Gradient Episodic Memory variant: gradient-space projection
  to avoid interfering update directions.
"""
from AlphaBrain.training.continual_learning.algorithms.rehearsal_based.er import ER
from AlphaBrain.training.continual_learning.algorithms.rehearsal_based.mir import MIR

__all__ = ["ER", "MIR"]
