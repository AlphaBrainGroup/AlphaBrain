"""Maximally Interfered Retrieval (MIR) for continual learning.

Reference
---------
Aljundi et al. 2019, "Online Continual Learning with Maximally Interfered
Retrieval" (NeurIPS 2019).  https://arxiv.org/abs/1908.04742

Idea
----
Instead of sampling the replay buffer uniformly (as in ER), MIR scores each
candidate by how much its loss would *increase* under a virtual SGD step
taken on the current batch:

    θ_v  = θ − η · ∇L(θ, B_curr)                       (virtual step)
    Δ_c  = L(θ_v, c) − L(θ, c)     for each candidate c
    replay ← top-k candidates by Δ_c

Intuitively, these are the samples most likely to be forgotten by the next
gradient step, so replaying them gives the most "bang for the buck".

Tractable-for-VLA variant
-------------------------
Vanilla MIR scores every step, which is ~50× slower on a 3B VLA (each step
needs |C|×2 extra forward passes).  We amortize cost via **periodic
refresh**: do the virtual-step scoring every ``mir_refresh_interval``
steps, cache the top-k as ``_interfered_cache``, and replay from that
cache in between refreshes.  At default settings (N=200, |C|=16, k=8)
overhead is roughly 10 % vs ER.

LoRA-only virtual step
----------------------
Only LoRA parameters (``'lora' in name.lower()``) participate in the
virtual step.  This matches EWC's filter and is essential on 3B-scale
backbones — otherwise the Δ direction is swamped by frozen backbone
noise.

Model access
------------
MIR needs the live model inside ``modify_batch`` to run forwards.  Rather
than changing ``CLAlgorithm.modify_batch``'s signature and dragging ER
along, MIR caches a reference to the model in ``on_task_start`` (via
``CLContext``) and uses the cached handle inside ``modify_batch``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

from AlphaBrain.training.continual_learning.algorithms.base import CLContext
from AlphaBrain.training.continual_learning.algorithms.rehearsal_based.er import ER

logger = logging.getLogger(__name__)


def _unwrap(model: Any) -> nn.Module:
    base = model
    while hasattr(base, "module"):
        base = base.module
    return base


class MIR(ER):
    """Maximally-Interfered Retrieval replay strategy.

    Extends :class:`ER` — the buffer population / storage logic
    is unchanged; only the sampling policy differs:

    - Every ``mir_refresh_interval`` steps, run a virtual SGD step on the
      current batch, score ``mir_candidate_size`` candidates by their
      loss increase, and cache the top ``mir_top_k``.
    - Other steps draw replay samples from that cache.
    - Before any refresh on a given task (or if the buffer is empty), we
      fall back to ``ER``'s uniform sampling — so the first few
      hundred steps of task N still get ER-quality replay.

    Hyperparameters beyond those of ``ER``:

    mir_refresh_interval : int
        Steps between virtual-step refreshes.  Lower = fresher cache, higher
        overhead.  Default 200.
    mir_candidate_size : int
        Number of buffer samples drawn per refresh for scoring (|C|).
        Default 16.
    mir_top_k : int
        Number of top-interfered candidates cached after each refresh.
        Must be ≤ ``mir_candidate_size``.  Default 8.
    mir_virtual_lr : float | None
        Learning rate used for the virtual step.  If None, uses
        ``DEFAULT_VIRTUAL_LR`` (1e-4) as a reasonable LoRA-scale guess.
        For best fidelity set this to the actual LoRA lr from your optimizer.
    mir_lora_only : bool
        Restrict the virtual step (and implicitly the grad capture) to
        parameters with 'lora' in the name.  Default True.
    """

    DEFAULT_VIRTUAL_LR: float = 1.0e-4

    def __init__(
        self,
        buffer_size_per_task: int = 500,
        replay_batch_ratio: float = 0.3,
        balanced_sampling: bool = False,
        seed: int = 42,
        mir_refresh_interval: int = 200,
        mir_candidate_size: int = 16,
        mir_top_k: int = 8,
        mir_virtual_lr: Optional[float] = None,
        mir_lora_only: bool = True,
    ):
        super().__init__(
            buffer_size_per_task=buffer_size_per_task,
            replay_batch_ratio=replay_batch_ratio,
            balanced_sampling=balanced_sampling,
            seed=seed,
        )
        if mir_top_k > mir_candidate_size:
            raise ValueError(
                f"mir_top_k ({mir_top_k}) must be ≤ mir_candidate_size "
                f"({mir_candidate_size})"
            )
        if mir_refresh_interval < 1:
            raise ValueError(f"mir_refresh_interval must be ≥ 1, got {mir_refresh_interval}")

        self.mir_refresh_interval = int(mir_refresh_interval)
        self.mir_candidate_size = int(mir_candidate_size)
        self.mir_top_k = int(mir_top_k)
        self.mir_virtual_lr = None if mir_virtual_lr is None else float(mir_virtual_lr)
        self.mir_lora_only = bool(mir_lora_only)

        # Runtime state (not serialized — rebuilt each task)
        self._model_ref: Any = None
        self._step_count: int = 0
        self._interfered_cache: List[dict] = []
        self._refresh_count: int = 0  # for describe/metrics
        # Most recent training-batch gradient, snapshot by autograd hooks
        # installed in on_task_start and committed in after_backward.
        #
        # We cannot read ``param.grad`` directly after ``accelerator.backward``
        # under DeepSpeed ZeRO-2 with ``contiguous_gradients: True`` (the
        # gradients live in DeepSpeed's internal buffer and .grad is None).
        # Instead we register a torch-native backward hook on each LoRA
        # parameter — the hook fires inside autograd when that param's
        # gradient is computed, *before* DeepSpeed moves it away.
        #
        # The gradient captured this way is per-rank local (pre-reduction).
        # For MIR scoring that's fine — it's a heuristic, and all ranks see
        # the same model so the direction is consistent.
        #
        # Tradeoff: g_curr is actually g_{t-1} (one-step stale) — this is the
        # standard "online" MIR variant and matches the literature.
        self._last_gradient: Dict[str, torch.Tensor] = {}
        # Staging dict — hooks deposit per-param grads here during backward;
        # after_backward commits to _last_gradient.
        self._pending_grad: Dict[str, torch.Tensor] = {}
        # Hook handles, so we can cleanly remove them on task end.
        self._hook_handles: List[Any] = []

    # ------------------------------------------------------------------
    # Parameter iteration (respects lora_only filter)
    # ------------------------------------------------------------------
    def _iter_params(self, model: Any) -> Iterator[Tuple[str, torch.Tensor]]:
        base = _unwrap(model)
        for name, param in base.named_parameters():
            if not param.requires_grad:
                continue
            if self.mir_lora_only and "lora" not in name.lower():
                continue
            yield name, param

    # ------------------------------------------------------------------
    # Virtual-step scoring
    # ------------------------------------------------------------------
    def _refresh_interfered_cache(self, current_batch: List[dict]) -> None:
        """Run MIR scoring: virtual step → rank candidates → cache top-k.

        Uses ``self._last_gradient`` (snapshot by :meth:`after_backward`
        on the previous training step) instead of running its own backward
        — computing a new backward here would double-reduce under DeepSpeed
        ZeRO-2.  The resulting g is one training step stale, which is the
        standard online-MIR compromise.
        """
        model = self._model_ref
        if model is None:
            logger.warning("MIR: model reference not set; skipping refresh.")
            return
        if self.is_empty():
            return
        if not self._last_gradient:
            logger.info(
                "MIR refresh skipped: no gradient snapshot yet "
                "(step %d; waiting for after_backward).",
                self._step_count,
            )
            return

        n_cands = min(self.mir_candidate_size, self._total_samples)
        k = min(self.mir_top_k, n_cands)
        if n_cands <= 0 or k <= 0:
            return

        candidates = self.sample(n_cands)   # uniform draw from union buffer
        if not candidates:
            return

        g_curr = self._last_gradient  # reference; we won't modify it

        base = _unwrap(model)
        was_training = base.training

        try:
            # ---------- 1) L(θ, c) for each candidate (eval, no grad) ----------
            model.eval()
            losses_theta: List[float] = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for c in candidates:
                    out = model.forward([c])
                    losses_theta.append(float(out["action_loss"].item()))

            # ---------- 2) Virtual step: θ ← θ − η · g_curr ----------
            virtual_lr = (
                self.mir_virtual_lr
                if self.mir_virtual_lr is not None
                else self.DEFAULT_VIRTUAL_LR
            )
            with torch.no_grad():
                for name, param in self._iter_params(model):
                    g = g_curr.get(name)
                    if g is not None:
                        # g may be fp32 (we cast in after_backward); param may be bf16
                        param.data.sub_(g, alpha=virtual_lr)

            # ---------- 3) L(θ_v, c) for each candidate ----------
            losses_theta_v: List[float] = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for c in candidates:
                    out = model.forward([c])
                    losses_theta_v.append(float(out["action_loss"].item()))

            # ---------- 4) Restore: θ ← θ + η · g_curr ----------
            with torch.no_grad():
                for name, param in self._iter_params(model):
                    g = g_curr.get(name)
                    if g is not None:
                        param.data.add_(g, alpha=virtual_lr)

            # ---------- 5) Rank by Δ and cache top-k ----------
            deltas = [
                l_v - l_t for l_t, l_v in zip(losses_theta, losses_theta_v)
            ]
            # Descending by Δ → largest loss-increase first
            top_idx = sorted(range(n_cands), key=lambda i: -deltas[i])[:k]
            self._interfered_cache = [candidates[i] for i in top_idx]
            self._refresh_count += 1

            logger.info(
                "MIR refresh #%d: scored %d candidates, cached top-%d. "
                "Δ-range=[%.4f, %.4f] (virtual_lr=%.2e)",
                self._refresh_count,
                n_cands,
                k,
                min(deltas),
                max(deltas),
                virtual_lr,
            )
        finally:
            if was_training:
                base.train()

    # ------------------------------------------------------------------
    # CLAlgorithm hooks
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Gradient capture via torch autograd hooks
    # ------------------------------------------------------------------
    def _install_grad_hooks(self, model: Any) -> None:
        """Attach a per-param ``register_hook`` on every tracked LoRA param.

        The hook fires inside autograd when that param's gradient is computed,
        which is *before* DeepSpeed moves it to its contiguous buffer and
        unsets ``param.grad``.  This is the only reliable way to capture
        gradients under DeepSpeed ZeRO-2 with ``contiguous_gradients: True``.

        Gradients captured by the hook are the **local per-rank** gradients
        (pre-reduction).  MIR scoring is a heuristic — per-rank is fine.
        """
        self._remove_grad_hooks()  # defensive — avoid double-registration
        for name, param in self._iter_params(model):
            # Closure with default-arg trick to bind `name` at hook-creation.
            def _hook(grad, n=name):
                self._pending_grad[n] = grad.detach().to(torch.float32).clone()
                return grad  # don't modify the gradient

            handle = param.register_hook(_hook)
            self._hook_handles.append(handle)

    def _remove_grad_hooks(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []

    # ------------------------------------------------------------------
    # CLAlgorithm hooks
    # ------------------------------------------------------------------
    def on_task_start(self, context: CLContext) -> None:
        """Cache the live model ref (for modify_batch), reset per-task state,
        and install autograd hooks on LoRA params for gradient capture."""
        self._model_ref = context.model
        self._step_count = 0
        self._interfered_cache = []
        self._last_gradient = {}
        self._pending_grad = {}
        if context.model is not None:
            self._install_grad_hooks(context.model)

    def on_task_end(self, context: CLContext) -> None:
        """Populate buffer (inherit ER behavior), remove hooks, drop stale cache."""
        super().on_task_end(context)
        self._remove_grad_hooks()
        self._interfered_cache = []
        self._last_gradient = {}
        self._pending_grad = {}

    def after_backward(self, model: Any) -> None:
        """Commit the per-step gradient snapshot.

        The autograd hooks registered in :meth:`on_task_start` populate
        ``self._pending_grad`` during backward.  Here, after the trainer's
        ``accelerator.backward()`` has completed, we atomically move the
        pending dict to ``self._last_gradient`` (so MIR's refresh in the
        next step sees a consistent full-snapshot) and clear the staging
        dict for the next step's fill.
        """
        if self._pending_grad:
            self._last_gradient = self._pending_grad
            self._pending_grad = {}

    def modify_batch(self, batch: List[dict], task_id: int) -> List[dict]:
        """Mix current batch with MIR-selected replay samples.

        Signature matches the base class and ER — no ``model`` arg.  MIR
        reaches the live model through the cached reference set in
        :meth:`on_task_start`.
        """
        self._step_count += 1

        # Empty buffer (first task) → return batch unchanged.
        if self.is_empty():
            return batch

        # No model ref yet → fall back to uniform (same as ER).
        if self._model_ref is None:
            return super().modify_batch(batch, task_id)

        # Periodic refresh
        if self._step_count % self.mir_refresh_interval == 0:
            self._refresh_interfered_cache(batch)

        # Cache warm-up: refresh hasn't fired yet for this task
        # → fall back to uniform instead of doing nothing useful.
        if not self._interfered_cache:
            return super().modify_batch(batch, task_id)

        batch_size = len(batch)
        n_replay = max(2, round(batch_size * self.replay_batch_ratio))
        n_replay = min(n_replay, batch_size - 1)
        n_current = batch_size - n_replay

        if n_replay <= len(self._interfered_cache):
            replay = self.rng.sample(self._interfered_cache, n_replay)
        else:
            # Cache smaller than replay slot — reuse with replacement.
            replay = self.rng.choices(self._interfered_cache, k=n_replay)

        current_shuffled = list(batch)
        self.rng.shuffle(current_shuffled)
        return current_shuffled[:n_current] + replay

    # ------------------------------------------------------------------
    # Reporting / serialization
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base.update(
            {
                "mir_refresh_interval": self.mir_refresh_interval,
                "mir_candidate_size": self.mir_candidate_size,
                "mir_top_k": self.mir_top_k,
                "mir_virtual_lr": self.mir_virtual_lr,
                "mir_lora_only": self.mir_lora_only,
                "mir_refresh_count": self._refresh_count,
                "mir_cache_size": len(self._interfered_cache),
            }
        )
        return base

    def metrics(self) -> Dict[str, float]:
        base = super().metrics()
        base.update(
            {
                "mir_refresh_count": float(self._refresh_count),
                "mir_cache_size": float(len(self._interfered_cache)),
            }
        )
        return base

    def state_dict(self) -> Dict[str, Any]:
        base = super().state_dict()
        base.update(
            {
                "mir_refresh_interval": self.mir_refresh_interval,
                "mir_candidate_size": self.mir_candidate_size,
                "mir_top_k": self.mir_top_k,
                "mir_virtual_lr": self.mir_virtual_lr,
                "mir_lora_only": self.mir_lora_only,
            }
        )
        return base

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.mir_refresh_interval = int(
            state.get("mir_refresh_interval", self.mir_refresh_interval)
        )
        self.mir_candidate_size = int(
            state.get("mir_candidate_size", self.mir_candidate_size)
        )
        self.mir_top_k = int(state.get("mir_top_k", self.mir_top_k))
        vlr = state.get("mir_virtual_lr", self.mir_virtual_lr)
        self.mir_virtual_lr = None if vlr is None else float(vlr)
        self.mir_lora_only = bool(state.get("mir_lora_only", self.mir_lora_only))
        self._model_ref = None
        self._step_count = 0
        self._interfered_cache = []
        self._refresh_count = 0
        self._last_gradient = {}


def _zero_grad(model: Any) -> None:
    """Clear gradients portably across bare nn.Module / DDP / DeepSpeed.

    DeepSpeedEngine.zero_grad() doesn't accept set_to_none kwargs; call it
    without args.  On a bare nn.Module this is equivalent to the torch-2.0+
    default of set_to_none=True.
    """
    try:
        model.zero_grad()
    except TypeError:
        model.zero_grad(set_to_none=True)
