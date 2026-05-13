#!/usr/bin/env python
"""Eval wrapper that forces single-step inference (re-query model every env
step, use only chunk[0] of returned action). Disables M1Inference's default
action chunking that caches a 10-action chunk between inferences.

Use to test if action chunking (vs single-step closed-loop) is the
bottleneck — probe-style inference inside the standard eval framework.

Run (libero env, server up):
    ${LIBERO_PYTHON} benchmarks/LIBERO/eval/eval_libero_singlestep.py \
        --args.pretrained-path <ckpt> \
        --args.host 127.0.0.1 --args.port 5795 \
        --args.task-suite-name libero_goal \
        --args.num-trials-per-task 50 \
        --args.video-out-path <out>/videos
"""
import sys, os
for _p in [p for p in os.environ.get("VLA_EXTRA_SYSPATH", "").split(":") if p]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tyro
from benchmarks.LIBERO.eval import eval_libero as base
from benchmarks.LIBERO.model2libero_interface import M1Inference

# Force single-step infer: every M1Inference.step() call re-queries the model
# and uses only chunk[0]. Achieved by clearing cached_raw_actions and resetting
# step_counter to 0 before each call.
_orig_step = M1Inference.step
def _singlestep_step(self, *args, **kwargs):
    self.cached_raw_actions = None
    self.step_counter = 0
    return _orig_step(self, *args, **kwargs)
M1Inference.step = _singlestep_step
print("[SINGLESTEP] M1Inference.step patched: re-infer every env step, always use chunk[0]")


if __name__ == "__main__":
    tyro.cli(base.eval_libero)
