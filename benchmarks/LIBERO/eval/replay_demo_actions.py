#!/usr/bin/env python
"""Replay a lerobot demo episode's recorded actions through LIBERO env.

Sanity check: does the dataset's action sequence, fed back into the env from
the matched init_state, actually reproduce a successful demo? If YES, the
dataset + env + action convention is internally consistent and any eval
failure is purely the model. If NO, there's a dataset/env mismatch (wrong
init_state, wrong gripper convention, delta-action scaling, etc.).

Two-phase because libero env lacks pyarrow:

    # Phase A (vla env) — read episode parquet, dump actions as npy
    ${ALPHABRAIN_PYTHON} benchmarks/LIBERO/eval/replay_demo_actions.py \
        --phase dump \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --episode_index 20 \
        --output_npz /tmp/replay_ep20_actions.npz

    # Phase B (libero env) — reset env to init_idx, step through actions
    ${LIBERO_PYTHON} benchmarks/LIBERO/eval/replay_demo_actions.py \
        --phase replay \
        --actions_npz /tmp/replay_ep20_actions.npz \
        --task_suite_name libero_goal \
        --task_id 0 \
        --init_state_idx 41 \
        --video_out /tmp/replay_ep20.mp4
"""
import argparse
import numpy as np
from pathlib import Path


def _binarize_gripper_dataset_to_robosuite(g_dataset: float) -> float:
    """dataset convention [0,1]: 0=close, 1=open → robosuite: +1=close, -1=open.

    Matches eval_libero.py's _binarize_gripper_open:
        > 0.5 (open in dataset)  → -1 (open in robosuite)
        <= 0.5 (close in dataset) → +1 (close in robosuite)
    """
    return -1.0 if float(g_dataset) > 0.5 else 1.0


def run_dump(args):
    import pyarrow.parquet as pq
    ds = Path(args.lerobot_dataset_path)
    chunk = args.episode_index // 1000
    pqf = ds / f"data/chunk-{chunk:03d}/episode_{args.episode_index:06d}.parquet"
    tbl = pq.read_table(str(pqf), columns=["action", "observation.state"])
    actions = np.array([tbl["action"][i].as_py() for i in range(len(tbl))], dtype=np.float32)
    state0 = np.array(tbl["observation.state"][0].as_py(), dtype=np.float32)
    print(f"episode {args.episode_index}: {len(actions)} frames, action shape {actions.shape}")
    print(f"  action[0]: {actions[0]}")
    print(f"  action[-1]: {actions[-1]}")
    print(f"  gripper (dim 6) range [{actions[:, 6].min():.3f}, {actions[:, 6].max():.3f}]")
    print(f"  state0: {state0}")
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, actions=actions, state0=state0, episode_index=args.episode_index)
    print(f"wrote {args.output_npz}")


def _load_actions(args):
    """Load (actions, state0) from either --actions_npz or --sweep_jsonl.

    sweep_jsonl uses `pred_action_dataset` (model output in dataset space:
    dims 0-5 = eef delta, dim 6 = gripper in [0,1]). state0 is absent in
    jsonl — fall back to reading the demo parquet for L2 sanity.
    """
    import json
    if args.actions_npz:
        data = np.load(args.actions_npz)
        return data["actions"], data["state0"], f"npz {args.actions_npz}"
    elif args.sweep_jsonl:
        rows = sorted([json.loads(l) for l in open(args.sweep_jsonl)], key=lambda r: r["step"])
        key = args.sweep_action_field
        actions = np.array([r[key] for r in rows], dtype=np.float32)
        state0 = np.zeros(8, dtype=np.float32)  # unused, only for reporting
        return actions, state0, f"sweep jsonl {args.sweep_jsonl} (field={key!r})"
    raise ValueError("Provide either --actions_npz or --sweep_jsonl")


def run_replay(args):
    import imageio
    from libero.libero import benchmark

    actions, state0_expected, src = _load_actions(args)
    print(f"loaded {len(actions)} actions from {src}")
    print(f"first action: {actions[0]}")
    print(f"gripper range: [{actions[:, 6].min():.3f}, {actions[:, 6].max():.3f}]")

    # LIBERO env setup — reuse eval_libero helpers
    from benchmarks.LIBERO.eval.eval_libero import _get_libero_env, LIBERO_ENV_RESOLUTION, _quat2axisangle, LIBERO_DUMMY_ACTION

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    print(f"task {args.task_id}: {task.language!r}")
    print(f"  init_states count: {len(init_states)}, using idx={args.init_state_idx}")

    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=args.seed)
    env.reset()
    obs = env.set_init_state(init_states[args.init_state_idx])

    # Report env's state0 vs dataset's
    state0_env = np.concatenate([
        np.asarray(obs["robot0_eef_pos"]),
        _quat2axisangle(obs["robot0_eef_quat"]),
        np.asarray(obs["robot0_gripper_qpos"]),
    ]).astype(np.float32)
    l2 = float(np.linalg.norm(state0_env - state0_expected))
    print(f"env state0 after set_init_state: {state0_env}")
    print(f"  L2 to demo state0: {l2:.5f}  (should be near 0 if init_state matches)")

    # Warmup (eval_libero.py does 10 dummy steps for objects to settle)
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    frames = []
    done = False
    for t, action in enumerate(actions):
        # Action format: first 6 = eef delta (pose), dim 6 = gripper in [0,1] dataset
        # LIBERO env needs 7-dim: 6 eef delta + 1 gripper in {-1, +1} robosuite
        env_action = np.concatenate([action[:6], [_binarize_gripper_dataset_to_robosuite(action[6])]]).astype(np.float32)
        obs, reward, done, info = env.step(env_action.tolist())
        # Record video frame (flip 180 to be upright, same as eval_libero.py does for inputs)
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        frames.append(img)
        if done:
            print(f"✓ SUCCESS at step {t}/{len(actions)}")
            break
    else:
        print(f"✗ FAILED — episode ended after all {len(actions)} actions, done={done}")

    Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.video_out, frames, fps=20)
    print(f"wrote video: {args.video_out}  ({len(frames)} frames)")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dump", "replay"], required=True)
    # dump args
    ap.add_argument("--lerobot_dataset_path", default="")
    ap.add_argument("--episode_index", type=int, default=-1)
    ap.add_argument("--output_npz", default="")
    # replay args
    ap.add_argument("--actions_npz", default="",
                    help="Load gt actions from npz (from --phase dump)")
    ap.add_argument("--sweep_jsonl", default="",
                    help="Alternative: load predicted actions from sweep jsonl")
    ap.add_argument("--sweep_action_field", default="pred_action_dataset",
                    choices=["pred_action_dataset", "gt_action"])
    ap.add_argument("--task_suite_name", default="libero_goal")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--init_state_idx", type=int, default=0)
    ap.add_argument("--video_out", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    args = ap.parse_args()

    if args.phase == "dump":
        assert args.lerobot_dataset_path and args.episode_index >= 0 and args.output_npz
        run_dump(args)
    else:
        assert (args.actions_npz or args.sweep_jsonl) and args.video_out, \
            "--phase replay requires (--actions_npz or --sweep_jsonl) and --video_out"
        run_replay(args)


if __name__ == "__main__":
    main()
