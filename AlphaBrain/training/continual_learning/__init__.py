"""Continual Learning module.

Sub-packages:
    algorithms/   — CL algorithms (ER / MIR / EWC / …) grouped by mechanism
                    (rehearsal_based, regularization_based, dynamic_based)
                    plus the :class:`CLAlgorithm` base and :class:`CLContext`.
    datasets/     — Task sequences and per-task dataset filtering.

Top-level entry:
    train         — Continual training loop
                    (`AlphaBrain.training.continual_learning.train.main`).

Re-exports for convenience (fully-qualified paths also work):
    `ER`                 ← `algorithms.rehearsal_based.er.ER`
    `MIR`                ← `algorithms.rehearsal_based.mir.MIR`
    `EWC`                ← `algorithms.regularization_based.ewc.EWC`
    `CLAlgorithm`        ← `algorithms.base.CLAlgorithm`
    `CLContext`          ← `algorithms.base.CLContext`
    `build_cl_algorithm` ← `algorithms.build_cl_algorithm`
"""
from AlphaBrain.training.continual_learning.algorithms import (
    CLAlgorithm,
    CLContext,
    ER,
    EWC,
    MIR,
    build_cl_algorithm,
)

__all__ = [
    "CLAlgorithm",
    "CLContext",
    "ER",
    "EWC",
    "MIR",
    "build_cl_algorithm",
]
