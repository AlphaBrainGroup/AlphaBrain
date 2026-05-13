#!/usr/bin/env python
"""Closed-loop eval probe: at each step, feed env render to model, record
model's predicted action AND step env with it. Compare to gt action stream
to see exactly how the model diverges during real eval.

Output JSONL per step:
    {"step": t, "env_state": [8], "gt_action": [7], "pred_action_dataset": [7]}

Run (libero env, server up):
    ${LIBERO_PYTHON} benchmarks/LIBERO/eval/probe_eval_trajectory.py \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --episode_index 20 \
        --task_description "open the middle drawer of the cabinet" \
        --task_suite_name libero_goal --task_id 0 --init_state_idx 41 \
        --port 5795 \
        --output_jsonl /tmp/eval_trajectory_ep20.jsonl
"""
import argparse
import json
import sys, os
for _p in [p for p in os.environ.get("VLA_EXTRA_SYSPATH", "").split(":") if p]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from pathlib import Path

from libero.libero import benchmark


def _binarize(g):
    return -1.0 if float(g) > 0.5 else 1.0


def _quat2axisangle(quat):
    quat = np.asarray(quat)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if den < 1e-6: return np.zeros(3)
    angle = 2.0 * np.arccos(quat[3])
    return (quat[:3] / den) * angle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_actions_npz", required=True,
                    help="npz with 'actions' key (precomputed in vla env from lerobot parquet)")
    ap.add_argument("--task_description", required=True)
    ap.add_argument("--task_suite_name", default="libero_goal")
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--init_state_idx", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--state_history_len", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=300, help="max env steps to record")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--video_out", default="", help="optional MP4 path to save env render frames")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from benchmarks.LIBERO.eval.eval_libero import _get_libero_env, LIBERO_ENV_RESOLUTION, LIBERO_DUMMY_ACTION
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    # gt actions for reference (precomputed in vla env to avoid pyarrow in libero env)
    gt_actions = np.load(args.gt_actions_npz)["actions"]
    print(f"loaded {len(gt_actions)} gt actions from {args.gt_actions_npz}")

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=args.seed)
    env.reset()
    obs = env.set_init_state(init_states[args.init_state_idx])
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    client = WebsocketClientPolicy(args.host, args.port)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.output_jsonl, "w")

    n_steps = min(args.max_steps, len(gt_actions))
    print(f"recording closed-loop eval trajectory for {n_steps} steps...")
    video_frames = []
    last_done = False
    for t in range(n_steps):
        # env-rendered image (same as what eval client sends)
        primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
        wrist   = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)
        st8 = np.concatenate([
            np.asarray(obs["robot0_eef_pos"]),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"]),
        ]).astype(np.float32)
        states = np.tile(st8, (args.state_history_len, 1)).astype(np.float32)

        payload = {
            "batch_images": [[primary, wrist]],
            "instructions": [args.task_description],
            "states": [states.tolist()],
            "do_sample": False,
            "use_ddim": False,
            "num_ddim_steps": 10,
        }
        resp = client.infer(payload)
        if resp.get("status") == "error" or "data" not in resp:
            raise RuntimeError(f"server error at t={t}: {resp}")
        a = np.asarray(resp["data"]["normalized_actions"], dtype=np.float32)
        if a.ndim == 3: a = a[0]
        first = a[0].copy()
        # Convert server-space gripper [+1,-1] back to dataset-space [0,1]
        grip_dataset = (1.0 - first[6]) / 2.0
        first[6] = grip_dataset

        out.write(json.dumps({
            "step": t,
            "env_state": st8.tolist(),
            "gt_action": gt_actions[t].tolist(),
            "pred_action_dataset": first.tolist(),
        }) + "\n")
        if t % 10 == 0:
            d_arm = float(np.linalg.norm(first[:6] - gt_actions[t][:6]))
            print(f"  t={t:3d}  arm L2={d_arm:.4f}  grip pred={first[6]:.3f} gt={gt_actions[t][6]:.3f}")

        # step env with model's action (closed loop), capture frame
        env_action = np.concatenate([first[:6], [_binarize(first[6])]]).astype(np.float32)
        obs, _, done, _ = env.step(env_action.tolist())
        last_done = done
        if args.video_out:
            video_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8))
        if done:
            print(f"\nenv done flag set at step {t}")
            break
    out.close()
    env.close()
    print(f"\nwrote {args.output_jsonl}")
    print(f"final done flag: {last_done}")
    if args.video_out and video_frames:
        import imageio
        Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video_out, video_frames, fps=20, codec="libx264")
        print(f"wrote {args.video_out}  ({len(video_frames)} frames @ 20fps)")


if __name__ == "__main__":
    main()
