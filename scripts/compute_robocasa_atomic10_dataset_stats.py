"""Compute dataset_statistics.json for the robocasa365_cl_atomic10 mixture.

Run this script ONCE to produce a fresh `dataset_statistics.json` derived
from the actual 10 atomic-task parquet files we train on, then drop the
output into the corresponding run directory under
`results/Checkpoints/<run_id>/dataset_statistics.json`, replacing any
borrowed-from-zhaoyiren stats.

Why we need this: the eval-side LoRA-merge + simulation_env.py loads
`dataset_statistics.json` from the run dir to denormalize policy actions
back into the simulator's action space.  If the stats came from a
different mixture (e.g. `qwen_gr00t_robocasa365_atomic_seen_wostate`),
the action scale can be off and the policy outputs garbage even with a
well-trained checkpoint.

Usage:
    python scripts/compute_robocasa_atomic10_dataset_stats.py \
        --config configs/continual_learning/qwengr00t_er_lora_robocasa_atomic10.yaml \
        --out results/Checkpoints/robocasa_atomic10_er_v2_nw0/dataset_statistics.json

The output schema matches what `LeRobotMixtureDataset.save_dataset_statistics`
emits — the same call the trainer makes at run start when starting a fresh
run.  This bypasses the trainer's heavy boot path (no model load, no
DeepSpeed init) so it works in a few minutes on CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a CL training yaml (e.g. qwengr00t_er_lora_robocasa_atomic10.yaml). "
        "Only `datasets.vla_data` is used.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the new dataset_statistics.json. "
        "If a file already exists at this path it WILL be overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the dataset and print mixture stats but skip the file write.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    from AlphaBrain.dataloader.lerobot_datasets import get_vla_dataset

    cfg = OmegaConf.load(args.config)
    vla_dataset_cfg = cfg.datasets.vla_data
    print(f"data_root_dir: {vla_dataset_cfg.data_root_dir}")
    print(f"dataset_mix : {vla_dataset_cfg.dataset_mix}")
    print(f"action_type : {vla_dataset_cfg.get('action_type', '<unset>')}")
    print()
    print("Building LeRobotMixtureDataset (this scans + reads parquet files)…")
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    n_subdatasets = len(getattr(dataset, "datasets", []))
    print(f"  → {n_subdatasets} sub-datasets in the mixture")

    if args.dry_run:
        print("--dry-run: skipping file write.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        backup = args.out.with_suffix(".json.borrowed_backup")
        print(f"Backing up existing stats → {backup}")
        args.out.replace(backup)

    print(f"Writing fresh dataset_statistics.json → {args.out}")
    dataset.save_dataset_statistics(args.out)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
