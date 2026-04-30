"""Compute `dataset_statistics.json` for any LeRobot-format CL mixture.

Run this once on the same yaml the trainer uses to produce a fresh
`dataset_statistics.json`, then drop it into the run directory under
`results/Checkpoints/<run_id>/dataset_statistics.json`.  The output schema
matches what `LeRobotMixtureDataset.save_dataset_statistics` emits — the
same call the trainer makes at run start.  This bypasses the trainer's
heavy boot path (no model load, no DeepSpeed init) so it finishes in a
few minutes on CPU.

Why we need this: the eval-side LoRA-merge + `eval_libero.py` /
`simulation_env.py` clients load `dataset_statistics.json` from the run
dir to denormalize policy actions back into the simulator's action
space.  If the file is missing (some legacy runs) or stats came from a
different mixture, the action scale is off and the policy outputs
garbage even with a well-trained checkpoint.

Auto-fix included: after computing stats, the script post-processes the
output to detect **binary-gripper axes** (q01==0 AND q99==1, the LIBERO
convention) and forces `mask=False` on those axes.  This matches the
QwenGR00T / LlamaOFT eval-side `unnormalize_actions` branch (see
`benchmarks/LIBERO/model2libero_interface.py`).  Without this fix the
gripper axis flips erratically near uncertainty, causing simulator
instability and 0% SR even for a perfectly-trained ckpt.

Usage:
    # LIBERO-Goal / Long (binary gripper auto-detected, mask[6] forced False)
    python scripts/compute_dataset_statistics.py \
        --config configs/continual_learning/qwengr00t_mir_lora_libero_refresh50.yaml \
        --out results/Checkpoints/<run_id>/dataset_statistics.json

    # Robocasa-atomic10 (no binary gripper axis, mask all True kept as-is)
    python scripts/compute_dataset_statistics.py \
        --config configs/continual_learning/qwengr00t_er_lora_robocasa_atomic10.yaml \
        --out results/Checkpoints/<run_id>/dataset_statistics.json
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
        help="Path to the same CL training yaml the trainer uses (any LIBERO / "
        "Robocasa-atomic10 / custom yaml).  Only `datasets.vla_data` is read.",
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

    # Post-process: detect binary-gripper axes (q01==0, q99==1) and set
    # mask[axis]=False so the eval-side `unnormalize_actions` takes the
    # binarize-then-passthrough branch (matches QwenGR00T/LlamaOFT
    # convention; current `generate_action_mask_for_used_keys` returns
    # all-True which causes gripper instability at eval, see
    # benchmarks/LIBERO/model2libero_interface.py:unnormalize_actions).
    import json
    with open(args.out) as _f:
        _stats = json.load(_f)
    _patched = []
    for _tag, _tag_stats in _stats.items():
        if not isinstance(_tag_stats, dict) or "action" not in _tag_stats:
            continue
        _act = _tag_stats["action"]
        _q01 = _act.get("q01", [])
        _q99 = _act.get("q99", [])
        _mask = _act.get("mask", [True] * len(_q01))
        for _i, (_lo, _hi) in enumerate(zip(_q01, _q99)):
            if _lo == 0.0 and _hi == 1.0 and _mask[_i] is True:
                _mask[_i] = False
                _patched.append(f"{_tag}.action.mask[{_i}]")
        _act["mask"] = _mask
    if _patched:
        print(f"Post-process: forced mask=False on binary-gripper axes: {_patched}")
        with open(args.out, "w") as _f:
            json.dump(_stats, _f, indent=2)
    else:
        print("Post-process: no binary-gripper axes detected, mask unchanged.")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
