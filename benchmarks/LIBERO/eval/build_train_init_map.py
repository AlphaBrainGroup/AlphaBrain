#!/usr/bin/env python
"""Build a task_id -> init_state_idx JSON mapping for Check 2.

Two-phase because libero env lacks pyarrow and vla env lacks libero:

    # Phase A (vla env) — read parquet, dump {task: first-frame state}
    ${ALPHABRAIN_PYTHON} benchmarks/LIBERO/eval/build_train_init_map.py \
        --phase cache \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --num_traj_per_task 1 \
        --demo_states_cache /tmp/check2_demo_states.json

    # Phase B (libero env) — env-match against LIBERO init_states, write final map
    ${LIBERO_PYTHON} benchmarks/LIBERO/eval/build_train_init_map.py \
        --phase match \
        --demo_states_cache /tmp/check2_demo_states.json \
        --task_suite_name libero_goal \
        --output_json /tmp/check2_train_init_map.json
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def _filter_first_n_per_task(episodes, n):
    """Replicate AlphaBrain `_filter_trajectories_per_task` logic: keep first N per task."""
    seen = {}
    out = []
    for ep in episodes:
        task = ep["tasks"][0] if isinstance(ep["tasks"], list) else ep["tasks"]
        c = seen.get(task, 0)
        if c < n:
            out.append((task, ep["episode_index"]))
            seen[task] = c + 1
    return out


def _read_first_state(dataset_path: Path, episode_index: int):
    import pyarrow.parquet as pq
    chunk = episode_index // 1000
    pqf = dataset_path / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    tbl = pq.read_table(str(pqf), columns=["observation.state"])
    return [float(x) for x in tbl["observation.state"][0].as_py()]


def _quat2axisangle(quat):
    quat = np.asarray(quat)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if den < 1e-6:
        return np.zeros(3)
    angle = 2.0 * np.arccos(quat[3])
    return (quat[:3] / den) * angle


def _env_state_8dim(obs):
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"]),
        _quat2axisangle(obs["robot0_eef_quat"]),
        np.asarray(obs["robot0_gripper_qpos"]),
    ]).astype(np.float32)


def run_cache(args):
    ds_path = Path(args.lerobot_dataset_path)
    with (ds_path / "meta/episodes.jsonl").open() as f:
        all_eps = [json.loads(line) for line in f]
    filtered = _filter_first_n_per_task(all_eps, args.num_traj_per_task)

    cache = OrderedDict()
    for task, ep in filtered:
        cache[task] = {"episode_index": int(ep), "state": _read_first_state(ds_path, ep)}
        print(f"  {task!r:60s}  ep={ep:3d}  state0={cache[task]['state']}")

    Path(args.demo_states_cache).parent.mkdir(parents=True, exist_ok=True)
    with open(args.demo_states_cache, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\n[cache] wrote {args.demo_states_cache}  ({len(cache)} demos)")


def run_match(args):
    from libero.libero import benchmark
    from benchmarks.LIBERO.eval.eval_libero import _get_libero_env, LIBERO_ENV_RESOLUTION  # noqa

    with open(args.demo_states_cache) as f:
        cache = json.load(f)
    print(f"[match] loaded {len(cache)} demo states from {args.demo_states_cache}")

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    mapping = OrderedDict()

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        lang = task.language
        if lang not in cache:
            print(f"[task {task_id}] {lang!r} not in cache — skipping")
            continue
        target = np.asarray(cache[lang]["state"], dtype=np.float32)
        demo_ep = cache[lang]["episode_index"]

        init_states = suite.get_task_init_states(task_id)
        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=0)

        best_idx, best_dist = -1, float("inf")
        for i in range(len(init_states)):
            env.reset()
            obs = env.set_init_state(init_states[i])
            st = _env_state_8dim(obs)
            d = float(np.linalg.norm(st - target))
            if d < best_dist:
                best_dist, best_idx = d, i
        env.close()

        mapping[str(task_id)] = best_idx
        flag = "" if best_dist < args.match_threshold else f"  ⚠ L2>{args.match_threshold}"
        print(f"[task {task_id}] {lang!r:60s}  demo_ep={demo_ep:3d}  "
              f"init_idx={best_idx:3d}  L2={best_dist:.5f}{flag}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n[match] wrote {args.output_json}  ({len(mapping)} entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["cache", "match"], required=True)
    # Phase A (cache) args
    ap.add_argument("--lerobot_dataset_path", default="")
    ap.add_argument("--num_traj_per_task", type=int, default=1)
    # Phase B (match) args
    ap.add_argument("--task_suite_name", default="libero_goal")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--match_threshold", type=float, default=0.05)
    # Shared
    ap.add_argument("--demo_states_cache", required=True)

    args = ap.parse_args()

    if args.phase == "cache":
        if not args.lerobot_dataset_path:
            ap.error("--lerobot_dataset_path required for --phase cache")
        run_cache(args)
    else:
        if not args.output_json:
            ap.error("--output_json required for --phase match")
        run_match(args)


if __name__ == "__main__":
    main()
