"""Abstract base class for continual-learning algorithms.

All CL algorithms (Experience Replay / EWC / DER / RETAIN / DWE / ...) implement
this interface.  The continual trainer (`AlphaBrain.training.continual_learning.train`)
only talks to algorithms through this protocol, so new methods can be plugged
in without touching the training loop.

Implemented
-----------
- `ER`   (algorithms.rehearsal_based.er) — experience replay with reservoir
         sampling (uniform or per-task balanced).
- `MIR`  (algorithms.rehearsal_based.mir) — interference-aware replay.
- `EWC`  (algorithms.regularization_based.ewc) — Fisher-weighted L2 penalty.

Planned
-------
- `EWC`      Elastic Weight Consolidation   (Kirkpatrick et al. 2017)
- `DER`      Dark Experience Replay         (Buzzega et al. 2020)
- `RETAIN`   Weight Merging / model souping (Wortsman et al. 2022, variants)
- `DWE`      Dynamic Weight Expansion       (per-task adapters)
- `SLCA`     Slow Learner + Classifier Alignment — realized via the existing
             `build_param_lr_groups` (layered LR in YAML), no subclass needed.

Hook semantics
--------------
Every hook has a no-op default, so a new algorithm only overrides what it needs.

Per-step hooks (inner training loop):
    observe(batch, task_id)       — inspect current-task batch before forward
                                    (SI path integral, streaming reservoir).
    modify_batch(batch, task_id)  — return the batch the model will consume
                                    (ER / DER inject replay samples here).
    compute_penalty(model)        — return a scalar added to the task loss
                                    (EWC / SI regularizers).

Task-level hooks (bracket each CL task):
    on_task_start(context)        — before training starts on a new task
                                    (DWE expands the model here).
    on_task_end(context)          — after a task finishes training
                                    (ER populates buffer, EWC computes Fisher,
                                    RETAIN merges weights).

On resume, the trainer replays `on_task_end` for each completed task so
algorithms can rebuild state not captured in `state_dict`.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class CLContext:
    """Bundle of trainer handles passed to task-level hooks.

    Not every field is set at every hook point — algorithms should check
    for None and skip gracefully when a required handle is missing.

    Fields:
        task_id:         index of the current task in the CL sequence
        model:           the model being trained (possibly DeepSpeed-wrapped)
        task_dataset:    raw Dataset for the current task (pre-DataLoader)
        task_dataloader: prepared DataLoader for the current task
        accelerator:     HuggingFace Accelerator handle (for ZeRO gather, etc.)
    """
    task_id: Optional[int] = None
    model: Optional[Any] = None
    task_dataset: Optional[Dataset] = None
    task_dataloader: Optional[DataLoader] = None
    accelerator: Optional[Any] = None


class CLAlgorithm(ABC):
    """Interface every continual-learning algorithm must satisfy.

    All hooks default to no-ops; subclasses override only those they need.
    The trainer is oblivious to the concrete algorithm — it just invokes the
    hooks at the right points in the training loop.
    """

    # ------------------------------------------------------------------
    # Per-step hooks (invoked on every inner-loop iteration)
    # ------------------------------------------------------------------
    def observe(self, batch: Any, task_id: int) -> None:
        """Inspect the current-task batch before the model forward.

        Typical uses:
          * SI: accumulate path-integral contributions.
          * Streaming-ER: reservoir-sample into the buffer on the fly.

        Default: no-op.
        """
        return None

    def modify_batch(self, batch: Any, task_id: int) -> Any:
        """Return the (possibly modified) batch the model will consume.

        Typical uses:
          * ER / DER: mix replay samples into the current batch.

        Default: return the batch unchanged.
        """
        return batch

    def compute_penalty(self, model: Any) -> Optional[torch.Tensor]:
        """Return a scalar penalty to add to the task loss, or None.

        Typical uses:
          * EWC:  λ · Σ F_i · (θ_i − θ*_i)²
          * SI:   λ · Σ Ω_i · (θ_i − θ*_i)²

        Default: None (no extra penalty).
        """
        return None

    def after_backward(self, model: Any) -> None:
        """Called once per training step, *after* ``accelerator.backward()``
        populates parameter gradients but *before* ``optimizer.step()``
        consumes them.

        Typical uses:
          * MIR: snapshot the live gradient for the next step's virtual
                 SGD computation.  MIR cannot run its own extra backward
                 here (DeepSpeed ZeRO-2 forbids double reduction within a
                 micro-batch boundary), so it borrows the gradient that
                 the trainer has just computed.

        Default: no-op.
        """
        return None

    # ------------------------------------------------------------------
    # Task-level hooks
    # ------------------------------------------------------------------
    def on_task_start(self, context: CLContext) -> None:
        """Called once before a new task starts training.

        Typical uses:
          * DWE: expand the model with a new adapter / expert head.

        Default: no-op.
        """
        return None

    def on_task_end(self, context: CLContext) -> None:
        """Called once after a task finishes training.

        Typical uses:
          * ER:     populate the buffer from `context.task_dataset`.
          * EWC:    compute Fisher on `context.task_dataloader`, snapshot θ*.
          * RETAIN: merge the newly-trained weights into the running model.

        Default: no-op.
        """
        return None

    # ------------------------------------------------------------------
    # Reporting (logged by the trainer)
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """Return a dict of hyperparameters / state summary for logging."""
        return {"algorithm": self.name}

    def metrics(self) -> Dict[str, float]:
        """Return per-step metrics to merge into the trainer's log stream."""
        return {}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot saved alongside checkpoints."""
        return {"algorithm": self.name}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from a dict produced by `state_dict()`. Default: no-op."""
        return None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Short identifier used in logs / checkpoints (default = class name)."""
        return self.__class__.__name__
