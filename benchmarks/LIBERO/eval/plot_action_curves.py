#!/usr/bin/env python
"""Plot per-dim action curves from N sweep jsonls + ground truth.

Each jsonl row: {step, gt_action[7], pred_action_dataset[7], ...}
GT is read from the first jsonl (all jsonls should share the same gt).

Usage:
    python benchmarks/LIBERO/eval/plot_action_curves.py \
        --jsonl /tmp/sweep_v3_step25k_ep0.jsonl:v3_step25k \
        --jsonl /tmp/sweep_1traj_step30k_ep0.jsonl:1traj_step30k \
        --output_png /tmp/action_curves_ep0.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for l in f:
            rows.append(json.loads(l))
    rows.sort(key=lambda r: r["step"])
    steps = np.asarray([r["step"] for r in rows])
    gt = np.asarray([r["gt_action"] for r in rows])  # (T, 7)
    pred = np.asarray([r["pred_action_dataset"] for r in rows])
    return steps, gt, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", action="append", required=True,
                    help="Repeatable: <path>:<label>")
    ap.add_argument("--output_png", required=True)
    args = ap.parse_args()

    parsed = []
    for spec in args.jsonl:
        path, _, label = spec.partition(":")
        if not label:
            label = Path(path).stem
        parsed.append((path, label))

    # Load gt from first jsonl
    steps_ref, gt, _ = load_jsonl(parsed[0][0])

    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    axes = axes.flatten()

    # Use distinct, clearly visible colors
    pred_colors = ["tab:red", "tab:orange", "tab:purple", "tab:brown"]

    for d in range(7):
        ax = axes[d]
        ax.plot(steps_ref, gt[:, d], color="black", linewidth=2.0, label="ground truth", zorder=3)
        for j, (path, label) in enumerate(parsed):
            steps, _, pred = load_jsonl(path)
            ax.plot(steps, pred[:, d], color=pred_colors[j % len(pred_colors)],
                    linewidth=1.2, alpha=0.85, label=label)
        ax.set_title(f"action dim {d} ({DIM_NAMES[d]})", fontsize=11)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(loc="best", fontsize=9)
    axes[7].axis("off")

    fig.suptitle(
        "Action prediction along episode (ep 0, libero_goal: put the bowl on the plate)\n"
        "x=frame index | y=dataset-space action value | gripper: 0=close, 1=open",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=120, bbox_inches="tight")
    print(f"wrote {args.output_png}")


if __name__ == "__main__":
    main()
