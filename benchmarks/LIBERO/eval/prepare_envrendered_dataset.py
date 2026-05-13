#!/usr/bin/env python
"""Re-render selected task's MP4 video files by replaying gt actions through
LIBERO env, so training-time images match eval-time env-render distribution.

Output dataset structure (mostly symlinks, only re-rendered MP4 files are new):
    <output_root>/<dataset_name>/
        meta/    → symlink to original
        data/    → symlink to original (parquets unchanged)
        videos/
            chunk-000/
                observation.images.image/
                    episode_XXXXXX.mp4   ← re-rendered for whitelisted task only
                observation.images.wrist_image/
                    episode_XXXXXX.mp4   ← re-rendered for whitelisted task only

Two phases (libero env lacks pyarrow):

    # Phase A (vla env): per task-X episode, dump (actions, state0) → npz
    ${ALPHABRAIN_PYTHON} benchmarks/LIBERO/eval/prepare_envrendered_dataset.py \
        --phase dump_actions \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --task_whitelist "open the middle drawer of the cabinet" \
        --output_npz /tmp/envrender_actions_taskdrawer.npz

    # Phase B (libero env): replay env per episode, write new MP4 files
    ${LIBERO_PYTHON} benchmarks/LIBERO/eval/prepare_envrendered_dataset.py \
        --phase render \
        --actions_npz /tmp/envrender_actions_taskdrawer.npz \
        --task_suite_name libero_goal --task_id 0 \
        --src_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --dst_dataset_path /share/zhanghe/Datasets_envrendered/libero_goal_no_noops_1.0.0_lerobot
"""
import argparse
import json
from pathlib import Path

import numpy as np


# ===== Phase A =====
def run_dump_actions(args):
    import pyarrow.parquet as pq
    ds = Path(args.lerobot_dataset_path)
    with (ds / "meta/episodes.jsonl").open() as f:
        eps = [json.loads(l) for l in f]
    keep = [e for e in eps if (e["tasks"][0] if isinstance(e["tasks"], list) else e["tasks"]) == args.task_whitelist]
    print(f"[dump] {len(keep)} episodes match task={args.task_whitelist!r}")

    ep_indices, all_actions, all_state0, lengths = [], [], [], []
    for ep in keep:
        ei = int(ep["episode_index"])
        chunk = ei // 1000
        pqf = ds / f"data/chunk-{chunk:03d}/episode_{ei:06d}.parquet"
        tbl = pq.read_table(str(pqf), columns=["action", "observation.state"])
        a = np.array([tbl["action"][i].as_py() for i in range(len(tbl))], dtype=np.float32)
        s0 = np.array(tbl["observation.state"][0].as_py(), dtype=np.float32)
        ep_indices.append(ei)
        all_actions.append(a)
        all_state0.append(s0)
        lengths.append(len(a))
    ep_indices = np.array(ep_indices, dtype=np.int64)
    all_state0 = np.stack(all_state0)
    lengths = np.array(lengths, dtype=np.int64)
    # actions vary in length → pad to max for storage, keep lengths separately
    max_len = int(lengths.max())
    padded = np.zeros((len(keep), max_len, all_actions[0].shape[-1]), dtype=np.float32)
    for i, a in enumerate(all_actions):
        padded[i, :len(a)] = a
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz,
                        episode_indices=ep_indices,
                        actions_padded=padded,
                        lengths=lengths,
                        state0=all_state0,
                        task_name=args.task_whitelist,
                        max_len=max_len)
    print(f"[dump] wrote {args.output_npz}  ({len(keep)} eps, max_len={max_len})")


# ===== Phase B =====
def _binarize(g):
    return -1.0 if float(g) > 0.5 else 1.0


def _quat2axisangle(quat):
    quat = np.asarray(quat)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if den < 1e-6: return np.zeros(3)
    angle = 2.0 * np.arccos(quat[3])
    return (quat[:3] / den) * angle


def _env_state_8d(obs):
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"]),
        _quat2axisangle(obs["robot0_eef_quat"]),
        np.asarray(obs["robot0_gripper_qpos"]),
    ]).astype(np.float32)


def _match_init_state(env, init_states, target_state8):
    """L2-match target 8-d state to init_states[i] (returns best idx, dist)."""
    best_i, best_d = -1, float("inf")
    for i in range(len(init_states)):
        env.reset()
        obs = env.set_init_state(init_states[i])
        st = _env_state_8d(obs)
        d = float(np.linalg.norm(st - target_state8))
        if d < best_d:
            best_d, best_i = d, i
    return best_i, best_d


def run_render(args):
    import imageio
    from libero.libero import benchmark
    from benchmarks.LIBERO.eval.eval_libero import _get_libero_env, LIBERO_ENV_RESOLUTION, LIBERO_DUMMY_ACTION

    data = np.load(args.actions_npz, allow_pickle=True)
    ep_indices = data["episode_indices"]
    actions_padded = data["actions_padded"]
    lengths = data["lengths"]
    state0_all = data["state0"]
    print(f"[render] loaded {len(ep_indices)} episodes from {args.actions_npz}")

    src = Path(args.src_dataset_path)
    dst = Path(args.dst_dataset_path)

    # Symlink meta + data dirs (parquets/metadata unchanged)
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("meta", "data"):
        target = dst / sub
        if not target.exists():
            target.symlink_to((src / sub).resolve())
            print(f"[render] symlink {target} → {(src/sub).resolve()}")
    (dst / "videos").mkdir(exist_ok=True)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    print(f"[render] task {args.task_id}: {task.language!r}, {len(init_states)} init_states")

    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=args.seed)

    summary = []
    for k in range(len(ep_indices)):
        ei = int(ep_indices[k])
        L = int(lengths[k])
        actions = actions_padded[k, :L]
        s0 = state0_all[k]

        # 1) Find best matching init_state for this episode's state0
        idx, dist = _match_init_state(env, init_states, s0)
        # 2) Reset env to that init, run num_steps_wait dummy steps
        env.reset()
        obs = env.set_init_state(init_states[idx])
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        # 3) Replay actions, capture frames at each step
        # We want exactly L frames (matching parquet length). Capture initial
        # frame (post-settle) + frames after each action[:-1], so total = L.
        primaries, wrists = [], []
        def snap():
            primaries.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8))
            wrists.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8))
        snap()  # frame 0
        for t in range(L - 1):
            ea = np.concatenate([actions[t][:6], [_binarize(actions[t][6])]]).astype(np.float32)
            obs, _, _, _ = env.step(ea.tolist())
            snap()

        # 4) Write MP4 files (h264 — pyav backend reads both h264 and av1 fine)
        chunk = ei // 1000
        for cam_name, frames in [("observation.images.image", primaries),
                                 ("observation.images.wrist_image", wrists)]:
            cam_dir = dst / f"videos/chunk-{chunk:03d}/{cam_name}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            mp4_path = cam_dir / f"episode_{ei:06d}.mp4"
            imageio.mimwrite(str(mp4_path), frames, fps=20, codec="libx264", quality=8)
        summary.append((ei, idx, dist, L))
        print(f"[render] ep{ei:3d} init_idx={idx:3d} L2={dist:.4f} L={L} → wrote 2 MP4s")

    env.close()
    print(f"\n[render] done. Summary:")
    for ei, idx, dist, L in summary:
        print(f"  ep{ei:3d} → init_idx={idx:3d} (L2={dist:.4f}) L={L}")
    print(f"[render] new dataset at {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dump_actions", "render"], required=True)
    # Phase A
    ap.add_argument("--lerobot_dataset_path", default="")
    ap.add_argument("--task_whitelist", default="")
    ap.add_argument("--output_npz", default="")
    # Phase B
    ap.add_argument("--actions_npz", default="")
    ap.add_argument("--task_suite_name", default="libero_goal")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--src_dataset_path", default="")
    ap.add_argument("--dst_dataset_path", default="")
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.phase == "dump_actions":
        assert args.lerobot_dataset_path and args.task_whitelist and args.output_npz
        run_dump_actions(args)
    else:
        assert args.actions_npz and args.src_dataset_path and args.dst_dataset_path
        run_render(args)


if __name__ == "__main__":
    main()
