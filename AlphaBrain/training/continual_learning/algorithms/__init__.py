"""Continual-learning algorithms.

Algorithms are grouped by their mechanism into three sub-packages:

- :mod:`.rehearsal_based`      — replay past-task samples
                                  (ER, MIR, planned DER / A-GEM)
- :mod:`.regularization_based` — penalise movement away from important
                                  parameters (EWC, planned SI / LwF)
- :mod:`.dynamic_based`        — adapt the model architecture across tasks
                                  (planned DWE / Weight Merge / PackNet)

Every algorithm implements the
:class:`AlphaBrain.training.continual_learning.algorithms.base.CLAlgorithm`
interface and is constructed from YAML via :func:`build_cl_algorithm`
(returns ``None`` if no CL method is enabled — the trainer then runs a
plain sequential baseline).

Public API
----------
The concrete classes are re-exported at this package level for convenience::

    from AlphaBrain.training.continual_learning.algorithms import EWC, MIR, ER

Fully-qualified paths also work, e.g.
``algorithms.regularization_based.ewc.EWC`` or
``algorithms.rehearsal_based.er.ER``.
"""
from typing import Optional

from AlphaBrain.training.continual_learning.algorithms.base import (
    CLAlgorithm,
    CLContext,
)
from AlphaBrain.training.continual_learning.algorithms.regularization_based import EWC
from AlphaBrain.training.continual_learning.algorithms.rehearsal_based import ER, MIR


def build_cl_algorithm(cfg, seed: int = 42) -> Optional[CLAlgorithm]:
    """Construct the CL algorithm specified by ``cfg.continual_learning``.

    Two config styles are supported:

    1. Replay-style (existing / back-compat)::

           continual_learning:
             replay:
               enabled: true
               method: experience_replay       # currently only ER
               buffer_size_per_task: 500
               replay_batch_ratio: 0.3
               balanced_sampling: false

    2. Generic algorithm (EWC / RETAIN / DER / DWE / …)::

           continual_learning:
             algorithm:
               name: ewc
               lambda: 1.0e4
               gamma: 1.0
               lora_only: true
               fisher_num_batches: 50
               fisher_clip: 1.0e4
               grad_clip_per_sample: 100.0

    Returns ``None`` when neither section is configured — the trainer then
    runs a plain sequential baseline without CL interventions.
    """
    cl_cfg = cfg.continual_learning

    # --- Style 1: replay-style config (back-compat with existing ER YAML) ---
    replay_cfg = cl_cfg.get("replay", None)
    if replay_cfg is not None and replay_cfg.get("enabled", False):
        method = replay_cfg.get("method", "experience_replay")
        if method == "experience_replay":
            return ER(
                buffer_size_per_task=replay_cfg.get("buffer_size_per_task", 500),
                replay_batch_ratio=replay_cfg.get("replay_batch_ratio", 0.3),
                balanced_sampling=replay_cfg.get("balanced_sampling", False),
                seed=seed,
            )
        raise ValueError(f"Unknown replay method: {method!r}")

    # --- Style 2: generic algorithm config ---
    algo_cfg = cl_cfg.get("algorithm", None)
    if algo_cfg is not None:
        name = algo_cfg.get("name", None)
        if name is None:
            return None
        key = str(name).lower()
        if key == "er":
            return ER(
                buffer_size_per_task=algo_cfg.get("buffer_size_per_task", 500),
                replay_batch_ratio=algo_cfg.get("replay_batch_ratio", 0.3),
                balanced_sampling=algo_cfg.get("balanced_sampling", False),
                seed=seed,
            )
        if key == "ewc":
            # `lambda` is a Python keyword; accept either `lambda` or
            # `ewc_lambda` (OmegaConf tolerates the former as a dict key).
            ewc_lambda = algo_cfg.get("lambda", None)
            if ewc_lambda is None:
                ewc_lambda = algo_cfg.get("ewc_lambda", 1.0e4)
            excl = algo_cfg.get("exclude_name_substrings", None)
            return EWC(
                ewc_lambda=ewc_lambda,
                gamma=algo_cfg.get("gamma", 1.0),
                lora_only=algo_cfg.get("lora_only", True),
                fisher_num_batches=algo_cfg.get("fisher_num_batches", 50),
                fisher_clip=algo_cfg.get("fisher_clip", 1.0e4),
                grad_clip_per_sample=algo_cfg.get("grad_clip_per_sample", 100.0),
                fisher_save_dir=algo_cfg.get("fisher_save_dir", None),
                exclude_name_substrings=list(excl) if excl is not None else None,
            )
        if key == "mir":
            return MIR(
                buffer_size_per_task=algo_cfg.get("buffer_size_per_task", 500),
                replay_batch_ratio=algo_cfg.get("replay_batch_ratio", 0.3),
                balanced_sampling=algo_cfg.get("balanced_sampling", False),
                seed=seed,
                mir_refresh_interval=algo_cfg.get("mir_refresh_interval", 200),
                mir_candidate_size=algo_cfg.get("mir_candidate_size", 16),
                mir_top_k=algo_cfg.get("mir_top_k", 8),
                mir_virtual_lr=algo_cfg.get("mir_virtual_lr", None),
                mir_lora_only=algo_cfg.get("mir_lora_only", True),
            )
        raise ValueError(
            f"CL algorithm {name!r} is declared in config but not yet implemented"
        )

    return None


__all__ = [
    "CLAlgorithm",
    "CLContext",
    "ER",
    "EWC",
    "MIR",
    "build_cl_algorithm",
]
