"""Regularization-based continual learning algorithms.

These methods add a penalty term to the task loss that discourages moving
parameters away from values that mattered for previous tasks.  The penalty
is typically weighted by an importance estimate — Fisher information, path
integrals, etc.  No explicit memory of past-task samples is required at
training time (though Fisher estimation does re-read the task data once
at task-end).

- :class:`EWC` — Elastic Weight Consolidation (Kirkpatrick et al. 2017):
  λ · Σ F_i · (θ_i − θ*_i)² with diagonal Fisher computed at task end.

Planned (not yet implemented):
- **SI** — Synaptic Intelligence: path-integral importance accumulated
  online (no extra task-end pass).
- **LwF** — Learning without Forgetting: teacher-logit distillation on
  the current-task batch using a frozen snapshot of the previous model.
"""
from AlphaBrain.training.continual_learning.algorithms.regularization_based.ewc import EWC

__all__ = ["EWC"]
