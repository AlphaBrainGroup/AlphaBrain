"""Elastic Weight Consolidation (EWC) for continual learning.

Reference
---------
Kirkpatrick et al. 2017, "Overcoming catastrophic forgetting in neural
networks" (https://arxiv.org/abs/1612.00796).

Idea
----
After each task ends we estimate the diagonal of the Fisher information
matrix on that task's data and snapshot the current parameters θ*.
During training on the next task the loss gains a regularizer

    L_EWC = λ · Σ_i F_i · (θ_i − θ*_i)²

that penalises moves away from θ* in directions the previous tasks were
"confident" about.  Diagonal Fisher is the standard approximation (full
Fisher is O(N²) in parameter count).

VLA / LoRA specialisation
-------------------------
For VLA models with LoRA adapters the default is to track Fisher only
over LoRA parameters (`'lora' in name.lower()`), which keeps memory
around 1–3 % of the full-model cost.  Mirrors the approach in
UT-Austin-RobIn/continual-vla-rl (rlinf/algorithms/ewc.py).

Numerical stability tricks borrowed from the same reference:
- optional per-sample gradient clip **before** squaring (reduces blow-ups
  when a rare batch produces a huge gradient);
- a ceiling on Fisher values (`fisher_clip`) post-aggregation;
- Fisher + θ* kept on CPU in fp32; a small per-device GPU cache is built
  lazily on first `compute_penalty` call to avoid H2D every step.

Multi-task accumulation
-----------------------
Across tasks we blend new and old Fisher with a decay factor γ:

    F ← γ · F_old + F_new

γ = 1.0 gives pure additive EWC (each task contributes equally).
γ < 1.0 gives "online EWC" — the exponentially-weighted variant.

Resume semantics
----------------
We do NOT serialize Fisher/θ* tensors (they can be hundreds of MB).  On
resume the trainer replays `on_task_end` for each completed task, and
each replay recomputes Fisher from the saved model state — this mirrors
how ER repopulates its samples on resume.  This is an
approximation (Fisher for early tasks gets recomputed against the final
saved model rather than the θ* that existed when that task finished) but
it keeps the trainer's resume path uniform across algorithms and is
standard practice in online-EWC implementations.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

from AlphaBrain.training.continual_learning.algorithms.base import (
    CLAlgorithm,
    CLContext,
)

logger = logging.getLogger(__name__)


def _unwrap(model: Any) -> nn.Module:
    """Strip DeepSpeed / DDP / Accelerate wrappers to expose the user module."""
    base = model
    while hasattr(base, "module"):
        base = base.module
    return base


def _zero_grad(model: Any) -> None:
    """Clear gradients portably across bare nn.Module / DDP / DeepSpeed.

    ``DeepSpeedEngine.zero_grad()`` doesn't accept ``set_to_none`` so we
    call it without kwargs.  On a bare ``nn.Module`` this is equivalent to
    ``set_to_none=True`` on torch ≥ 2.0 (the project's floor).
    """
    try:
        model.zero_grad()
    except TypeError:
        # Extremely old torch signatures — fall back to the explicit flag.
        model.zero_grad(set_to_none=True)


def _is_main_rank() -> bool:
    """Return True iff this process is rank 0 (or non-distributed)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


class EWC(CLAlgorithm):
    """Elastic Weight Consolidation with diagonal Fisher approximation.

    Hyperparameters
    ---------------
    ewc_lambda : regularization strength λ.  In continual-vla-rl the default is
        1e6 for full-param Fisher; for LoRA-only Fisher a smaller value
        (1e3–1e5) usually works.  Tune per-dataset.
    gamma : decay applied to the *old* Fisher when blending with a new one.
        1.0 = pure additive (standard EWC).  <1.0 = online EWC.
    lora_only : only compute Fisher over parameters whose name contains 'lora'.
        Essential for large VLA backbones; set to False for models without LoRA.
    fisher_num_batches : how many minibatches of the task dataloader to use
        for Fisher estimation.  Small values (20–100) are typical.
    fisher_clip : post-aggregation clamp on Fisher entries (protects against
        isolated gradient spikes).  Set to None to disable.
    grad_clip_per_sample : clip |grad| element-wise *before* squaring it into
        Fisher.  Set to None to disable.  Guards against bf16 overflow.
    """

    def __init__(
        self,
        ewc_lambda: float = 1e4,
        gamma: float = 1.0,
        lora_only: bool = True,
        fisher_num_batches: int = 50,
        fisher_clip: Optional[float] = 1e4,
        grad_clip_per_sample: Optional[float] = 100.0,
        fisher_save_dir: Optional[str] = None,
        exclude_name_substrings: Optional[List[str]] = None,
    ):
        self.ewc_lambda = float(ewc_lambda)
        self.gamma = float(gamma)
        self.lora_only = bool(lora_only)
        self.fisher_num_batches = int(fisher_num_batches)
        self.fisher_clip = None if fisher_clip is None else float(fisher_clip)
        self.grad_clip_per_sample = (
            None if grad_clip_per_sample is None else float(grad_clip_per_sample)
        )
        self.exclude_name_substrings = tuple(
            s.lower() for s in (exclude_name_substrings or [])
        )
        # When set, after every task we dump the accumulated Fisher + θ*
        # tensors to `<fisher_save_dir>/fisher_task_<k>.pt` for offline
        # analysis.  Rank 0 writes; other ranks are silent.
        self.fisher_save_dir = (
            None if fisher_save_dir is None else str(fisher_save_dir)
        )

        # CPU-resident master state (fp32, keyed by full parameter name)
        self._fisher: Optional[Dict[str, torch.Tensor]] = None
        self._old_params: Optional[Dict[str, torch.Tensor]] = None

        # Per-device GPU cache (lazy, invalidated on every on_task_end)
        self._device_cache: Optional[torch.device] = None
        self._fisher_gpu: Optional[Dict[str, torch.Tensor]] = None
        self._old_params_gpu: Optional[Dict[str, torch.Tensor]] = None

        # Bookkeeping
        self._completed_tasks: List[int] = []
        # Most recent raw (unmerged) per-task Fisher — for `metrics()`
        # and to help diagnose whether this task's estimate was sane.
        self._last_task_fisher_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Parameter iteration (respects lora_only filter)
    # ------------------------------------------------------------------
    def _iter_params(self, model: Any) -> Iterator[Tuple[str, torch.Tensor]]:
        base = _unwrap(model)
        for name, param in base.named_parameters():
            if not param.requires_grad:
                continue
            lower = name.lower()
            if self.lora_only and "lora" not in lower:
                continue
            if any(tok in lower for tok in self.exclude_name_substrings):
                continue
            yield name, param

    def _snapshot_params(self, model: Any) -> Dict[str, torch.Tensor]:
        snap: Dict[str, torch.Tensor] = {}
        for name, param in self._iter_params(model):
            snap[name] = (
                param.detach().to(dtype=torch.float32, device="cpu").clone()
            )
        return snap

    # ------------------------------------------------------------------
    # Fisher computation
    # ------------------------------------------------------------------
    def _compute_fisher(
        self,
        model: Any,
        dataloader: Any,
        accelerator: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Estimate diagonal Fisher by averaging squared per-batch gradients.

        The dataloader is iterated for up to ``fisher_num_batches`` batches;
        each batch forwards once, backprops once, and contributes its squared
        gradients (clipped if configured) to the running Fisher accumulator.

        DeepSpeed ZeRO-2 notes
        ----------------------
        Under DeepSpeed ZeRO-2 with ``contiguous_gradients: true``, ``param.grad``
        is cleared once autograd finishes (gradients live in DeepSpeed's
        contiguous bucket).  Reading ``param.grad`` directly after
        ``accelerator.backward`` therefore yields ``None`` and silently gives
        an all-zero Fisher.  To avoid this we register a per-parameter
        backward hook that fires *during* autograd — hooks run before
        DeepSpeed moves the gradient away, so we capture the live
        (pre-reduction, per-rank-local) gradient there.  For MIR-style
        heuristic use the per-rank local value is fine.
        """
        fisher: Dict[str, torch.Tensor] = {}
        for name, param in self._iter_params(model):
            fisher[name] = torch.zeros_like(
                param.data, dtype=torch.float32, device="cpu"
            )

        if not fisher:
            logger.warning(
                "EWC._compute_fisher: no trainable parameters matched "
                "(lora_only=%s). Returning empty Fisher.",
                self.lora_only,
            )
            return fisher

        base = _unwrap(model)
        was_training = base.training
        model.eval()

        # Install autograd hooks on every tracked LoRA param so we capture
        # gradients during backward (before DeepSpeed can clear `.grad`).
        captured: Dict[str, torch.Tensor] = {}
        hook_handles: List[Any] = []
        for name, param in self._iter_params(model):
            def _make_hook(pname: str):
                def _hook(grad: torch.Tensor) -> torch.Tensor:
                    # Clone so later autograd ops don't mutate our copy;
                    # cast to fp32 for numerically stable accumulation.
                    captured[pname] = grad.detach().to(torch.float32).clone()
                    return grad  # don't modify the gradient
                return _hook
            hook_handles.append(param.register_hook(_make_hook(name)))

        count = 0
        try:
            for batch in dataloader:
                if count >= self.fisher_num_batches:
                    break
                captured.clear()
                _zero_grad(model)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.forward(batch)
                    loss = output["action_loss"]

                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()

                # After backward, `captured` holds per-param gradients
                # regardless of whether DeepSpeed cleared .grad.
                for name, g in captured.items():
                    if self.grad_clip_per_sample is not None:
                        g = g.clamp(
                            -self.grad_clip_per_sample, self.grad_clip_per_sample
                        )
                    fisher[name] += g.pow(2).cpu()
                count += 1
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            _zero_grad(model)
            if was_training:
                base.train()

        if count > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(count)
        if self.fisher_clip is not None:
            for name in fisher:
                fisher[name] = fisher[name].clamp_(0.0, self.fisher_clip)

        # Cheap diagnostic summary — use WARNING level so it bypasses
        # accelerate's default INFO filter and actually shows up in logs.
        sum_all = float(sum(float(t.sum()) for t in fisher.values()))
        max_all = float(max(float(t.max()) for t in fisher.values()))
        nonzero_entries = int(
            sum(int((t > 0).sum()) for t in fisher.values())
        )
        total_entries = int(sum(t.numel() for t in fisher.values()))
        pct_nonzero = (
            100.0 * nonzero_entries / total_entries if total_entries else 0.0
        )
        self._last_task_fisher_stats = {
            "fisher_sum": sum_all,
            "fisher_max": max_all,
            "fisher_pct_nonzero": pct_nonzero,
            "fisher_num_params": float(len(fisher)),
            "fisher_num_batches_used": float(count),
        }
        logger.warning(
            "EWC: Fisher estimated over %d batches across %d params "
            "(lora_only=%s).  sum=%.4e  max=%.4e  nonzero=%.2f%%",
            count, len(fisher), self.lora_only,
            sum_all, max_all, pct_nonzero,
        )
        return fisher

    # ------------------------------------------------------------------
    # CLAlgorithm hooks
    # ------------------------------------------------------------------
    def on_task_end(self, context: CLContext) -> None:
        """Recompute Fisher on the finished task's data and snapshot θ*."""
        if context.model is None or context.task_dataloader is None:
            logger.warning(
                "EWC.on_task_end: model or task_dataloader missing from context; "
                "skipping Fisher computation."
            )
            return

        new_fisher = self._compute_fisher(
            model=context.model,
            dataloader=context.task_dataloader,
            accelerator=context.accelerator,
        )
        new_snapshot = self._snapshot_params(context.model)

        if self._fisher is None:
            self._fisher = new_fisher
        else:
            # Merge Fisher across tasks:  F ← γ · F_old + F_new
            merged: Dict[str, torch.Tensor] = {}
            for name in set(self._fisher.keys()) | set(new_fisher.keys()):
                old = self._fisher.get(name)
                incoming = new_fisher.get(name)
                if old is None:
                    merged[name] = incoming
                elif incoming is None:
                    merged[name] = self.gamma * old
                else:
                    merged[name] = self.gamma * old + incoming
            self._fisher = merged

        self._old_params = new_snapshot
        self._completed_tasks.append(int(context.task_id))

        # Invalidate the GPU cache — next penalty() call will rebuild it.
        self._device_cache = None
        self._fisher_gpu = None
        self._old_params_gpu = None

        # Use WARNING level so the message survives accelerate's INFO filter.
        logger.warning(
            "EWC.on_task_end: consolidated task %s "
            "(total tasks seen = %d, Fisher entries = %d)",
            context.task_id,
            len(self._completed_tasks),
            len(self._fisher or {}),
        )

        # Optionally persist the Fisher + θ* snapshot to disk for offline
        # analysis.  Only rank 0 writes (tensors are per-rank local but we
        # pick one deterministic view).
        if self.fisher_save_dir is not None and _is_main_rank():
            try:
                os.makedirs(self.fisher_save_dir, exist_ok=True)
                save_path = os.path.join(
                    self.fisher_save_dir,
                    f"fisher_task_{int(context.task_id)}.pt",
                )
                torch.save(
                    {
                        "task_id": int(context.task_id),
                        "completed_tasks": list(self._completed_tasks),
                        "fisher": self._fisher,
                        "old_params": self._old_params,
                        "last_task_fisher_stats": dict(self._last_task_fisher_stats),
                        "config": {
                            "ewc_lambda": self.ewc_lambda,
                            "gamma": self.gamma,
                            "lora_only": self.lora_only,
                            "fisher_num_batches": self.fisher_num_batches,
                            "fisher_clip": self.fisher_clip,
                            "grad_clip_per_sample": self.grad_clip_per_sample,
                        },
                    },
                    save_path,
                )
                logger.warning("EWC: saved Fisher snapshot -> %s", save_path)
            except Exception as e:
                logger.warning(
                    "EWC: failed to save Fisher snapshot for task %s: %s",
                    context.task_id, e,
                )

    def compute_penalty(self, model: Any) -> Optional[torch.Tensor]:
        """Return λ · Σ F · (θ − θ*)² or None if no tasks have been seen."""
        if (
            self._fisher is None
            or self._old_params is None
            or self.ewc_lambda == 0.0
        ):
            return None

        # Resolve the device of the live parameters (first trainable param).
        device: Optional[torch.device] = None
        for _, p in self._iter_params(model):
            device = p.device
            break
        if device is None:
            return None

        # Lazy per-device cache: only pay H2D once per (device, task_end) epoch.
        if self._device_cache != device:
            self._fisher_gpu = {n: f.to(device) for n, f in self._fisher.items()}
            self._old_params_gpu = {n: p.to(device) for n, p in self._old_params.items()}
            self._device_cache = device

        penalty: Optional[torch.Tensor] = None
        for name, param in self._iter_params(model):
            if name not in self._fisher_gpu:
                continue
            f = self._fisher_gpu[name]
            old = self._old_params_gpu[name]
            diff = param.to(torch.float32) - old
            term = (f * diff.pow(2)).sum()
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            return None
        return self.ewc_lambda * penalty

    # ------------------------------------------------------------------
    # Reporting / serialization
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "algorithm": self.name,
            "ewc_lambda": self.ewc_lambda,
            "gamma": self.gamma,
            "lora_only": self.lora_only,
            "fisher_num_batches": self.fisher_num_batches,
            "fisher_clip": self.fisher_clip,
            "grad_clip_per_sample": self.grad_clip_per_sample,
            "num_tasks_consolidated": len(self._completed_tasks),
            "fisher_entries": 0 if self._fisher is None else len(self._fisher),
        }

    def metrics(self) -> Dict[str, float]:
        m: Dict[str, float] = {
            "ewc_num_tasks_consolidated": float(len(self._completed_tasks)),
        }
        # Surface last task's Fisher stats so training logs show whether
        # Fisher estimation actually captured signal (vs silently zeroing).
        for k, v in self._last_task_fisher_stats.items():
            m[f"ewc_{k}"] = float(v)
        return m

    def state_dict(self) -> Dict[str, Any]:
        """Return only metadata — Fisher/θ* tensors are not serialized.

        On resume the trainer replays `on_task_end` for each completed task,
        which rebuilds Fisher from the saved model state.  This mirrors
        ER's behaviour and keeps the trainer's resume path uniform.
        """
        return {
            "algorithm": self.name,
            "ewc_lambda": self.ewc_lambda,
            "gamma": self.gamma,
            "lora_only": self.lora_only,
            "fisher_num_batches": self.fisher_num_batches,
            "fisher_clip": self.fisher_clip,
            "grad_clip_per_sample": self.grad_clip_per_sample,
            "completed_tasks": list(self._completed_tasks),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ewc_lambda = float(state.get("ewc_lambda", self.ewc_lambda))
        self.gamma = float(state.get("gamma", self.gamma))
        self.lora_only = bool(state.get("lora_only", self.lora_only))
        self.fisher_num_batches = int(
            state.get("fisher_num_batches", self.fisher_num_batches)
        )
        fc = state.get("fisher_clip", self.fisher_clip)
        self.fisher_clip = None if fc is None else float(fc)
        gc = state.get("grad_clip_per_sample", self.grad_clip_per_sample)
        self.grad_clip_per_sample = None if gc is None else float(gc)
        # Tensor state is reset; replay of on_task_end will rebuild it.
        self._fisher = None
        self._old_params = None
        self._device_cache = None
        self._fisher_gpu = None
        self._old_params_gpu = None
        self._completed_tasks = []
