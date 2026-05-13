#!/usr/bin/env python
"""Sweep all frames of one episode against a server. For each frame t,
send (primary_img, wrist_img, state) and record server output's first action.

Output JSONL format, one line per frame:
    {"step": t, "pred_action": [...7 dims...], "gt_action": [...7 dims...]}

Run from repo root, vla env (no proxy):
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
    export NO_PROXY=127.0.0.1,localhost
    python benchmarks/LIBERO/eval/probe_episode_sweep.py \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --episode_index 0 \
        --task_description "put the bowl on the plate" \
        --port 5798 \
        --output_jsonl /tmp/sweep_v3_step25k_ep0.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import imageio.v3 as iio

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot_dataset_path", required=True)
    ap.add_argument("--episode_index", type=int, required=True)
    ap.add_argument("--task_description", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--state_history_len", type=int, default=16)
    args = ap.parse_args()

    ds = Path(args.lerobot_dataset_path)
    chunk = args.episode_index // 1000
    pqf = ds / f"data/chunk-{chunk:03d}/episode_{args.episode_index:06d}.parquet"
    primary_video = ds / f"videos/chunk-{chunk:03d}/observation.images.image/episode_{args.episode_index:06d}.mp4"
    wrist_video   = ds / f"videos/chunk-{chunk:03d}/observation.images.wrist_image/episode_{args.episode_index:06d}.mp4"

    tbl = pq.read_table(str(pqf), columns=["observation.state", "action"])
    n_frames = len(tbl)
    print(f"episode {args.episode_index}: {n_frames} frames")

    # Read all frames at once (more efficient than indexing one by one)
    primary_frames = list(iio.imiter(str(primary_video)))
    wrist_frames   = list(iio.imiter(str(wrist_video)))
    assert len(primary_frames) == n_frames, f"primary {len(primary_frames)} != {n_frames}"
    assert len(wrist_frames)   == n_frames, f"wrist {len(wrist_frames)} != {n_frames}"

    client = WebsocketClientPolicy(args.host, args.port)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_jsonl, "w") as out:
        for t in range(n_frames):
            state = np.asarray(tbl["observation.state"][t].as_py(), dtype=np.float32)
            gt = np.asarray(tbl["action"][t].as_py(), dtype=np.float32)
            # Tile state to match eval's (n,8) shape
            states = np.tile(state, (args.state_history_len, 1)).astype(np.float32)
            payload = {
                "batch_images": [[primary_frames[t], wrist_frames[t]]],
                "instructions": [args.task_description],
                "states": [states.tolist()],
                "do_sample": False,
                "use_ddim": False,
                "num_ddim_steps": 10,
            }
            resp = client.infer(payload)
            if resp.get("status") == "error" or "data" not in resp:
                raise RuntimeError(f"server error at t={t}: {resp}")
            arr = np.asarray(resp["data"]["normalized_actions"], dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            pred = arr[0]  # first action of the predicted chunk
            # Convert server-space gripper [+1,-1] back to dataset-space [0,1]
            # (so curve is comparable to gt)
            pred_dataset = pred.copy()
            pred_dataset[6] = (1.0 - pred[6]) / 2.0
            out.write(json.dumps({
                "step": t,
                "gt_action": gt.tolist(),
                "pred_action_raw": pred.tolist(),
                "pred_action_dataset": pred_dataset.tolist(),
            }) + "\n")
            if t % 10 == 0:
                print(f"  t={t:3d}  gt[6]={gt[6]:.3f}  pred[6]={pred[6]:+.3f}→dataset {pred_dataset[6]:.3f}")

    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
