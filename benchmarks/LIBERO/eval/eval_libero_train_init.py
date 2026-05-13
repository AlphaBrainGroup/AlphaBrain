#!/usr/bin/env python
"""Check 2 (train-init): eval only on the LIBERO init_states that match each
training demo's first-frame state.

Pairs with ``build_train_init_map.py`` which produces the
``{task_id: init_idx}`` JSON this script consumes. Runs exactly 1 trial
per task on the matched init_state, so SR near 100% means "model
memorized the training demo" and SR near 0 means the training pipeline
didn't produce actions that even reproduce the training trajectory.

Launch (libero python env, server already up):
    python benchmarks/LIBERO/eval/eval_libero_train_init.py \
        --args.pretrained-path ./results/training/Pi05-1traj-libero_goal/checkpoints/steps_30000 \
        --args.host 127.0.0.1 --args.port 5795 \
        --args.task-suite-name libero_goal \
        --args.video-out-path /tmp/check2_videos \
        --args.init-state-map-json /tmp/train_init_map.json
"""
import sys, os
for _p in [p for p in os.environ.get("VLA_EXTRA_SYSPATH", "").split(":") if p]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataclasses
import json

import tyro
from libero.libero import benchmark

from benchmarks.LIBERO.eval import eval_libero as base


@dataclasses.dataclass
class Args(base.Args):
    init_state_map_json: str = ""  # path to JSON {task_id: init_idx}


def main(args: Args) -> None:
    assert args.init_state_map_json, "--args.init-state-map-json required"
    with open(args.init_state_map_json) as f:
        init_map = {int(k): int(v) for k, v in json.load(f).items()}
    args.num_trials_per_task = 1  # single trial on the matched init

    # Monkey-patch get_task_init_states to return only the mapped init_state
    # for covered tasks. eval_libero's inner loop then uses initial_states[0]
    # which is the matched init.
    suite_cls = benchmark.get_benchmark_dict()[args.task_suite_name]
    _orig_get = suite_cls.get_task_init_states

    def _patched_get(self, task_id):
        full = _orig_get(self, task_id)
        if task_id in init_map:
            return [full[init_map[task_id]]]
        raise RuntimeError(
            f"task_id={task_id} not in init_state_map_json — "
            "discovery phase missed this task"
        )

    suite_cls.get_task_init_states = _patched_get
    print(f"[INIT OVERRIDE] {len(init_map)} task→init_idx remapped; num_trials=1 per task")
    print(f"[INIT OVERRIDE] map = {init_map}")

    base.eval_libero(args)


if __name__ == "__main__":
    tyro.cli(main)
