#!/usr/bin/env python3
"""
merge_lora_checkpoint.py — Merge LoRA adapter + non-VLM weights into a full
checkpoint usable by the standard eval pipeline (server_policy.py +
BaseFramework.from_pretrained).

Thin CLI wrapper around the sibling `load_and_merge()` helper. Located inside
the peft module so it can be invoked via `python -m` without path hacks.

Usage (from repo root, starVLA env active):
    python -m AlphaBrain.training.trainer_utils.peft.merge_lora_checkpoint \\
        --model qwengr00t \\
        --lora_adapter_dir results/Checkpoints/.../task_4_id4_steps_50000_lora_adapter \\
        --action_model_pt  results/Checkpoints/.../task_4_id4_steps_50000_action_model.pt \\
        --output_path      results/Checkpoints/.../task_4_id4_steps_50000_pytorch_model.pt
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into full AlphaBrain checkpoint")
    cfg_group = parser.add_mutually_exclusive_group(required=True)
    cfg_group.add_argument("--model", type=str,
                           help=(
                               "Model name (qwengr00t | neurovla | llamaoft | paligemma). "
                               "Auto-expands to cl_base.yaml + models/<name>.yaml."
                           ))
    cfg_group.add_argument("--base_config", type=str, nargs="+",
                           help=(
                               "Explicit YAML paths merged left-to-right (advanced). "
                               "Prefer --model for standard runs."
                           ))
    parser.add_argument("--lora_adapter_dir", type=str, required=True,
                        help="Path to LoRA adapter directory (contains adapter_model.safetensors)")
    parser.add_argument("--action_model_pt", type=str, required=True,
                        help="Path to action_model.pt checkpoint (non-VLM weights)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for merged checkpoint (.pt file)")
    parser.add_argument("--vlm_module", type=str, default=None,
                        help="VLM interface attr name (default: auto-detect)")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from AlphaBrain.model.framework import build_framework
    from AlphaBrain.training.trainer_utils.config_tracker import wrap_config
    from AlphaBrain.training.trainer_utils.peft import load_and_merge

    if args.model is not None:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        # Walk up from peft/ → trainer_utils/ → training/ → AlphaBrain/ → repo root
        for _ in range(4):
            repo_root = os.path.dirname(repo_root)
        config_paths = [
            os.path.join(repo_root, "configs", "continual_learning", "cl_base.yaml"),
            os.path.join(repo_root, "configs", "continual_learning", "models", f"{args.model}.yaml"),
        ]
    else:
        config_paths = args.base_config

    print(f"[0/4] Loading config from {config_paths}")
    cfg = OmegaConf.load(config_paths[0])
    for _extra in config_paths[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(_extra))
    cfg = wrap_config(cfg)

    load_and_merge(
        base_model_factory=lambda: build_framework(cfg),
        lora_adapter_dir=args.lora_adapter_dir,
        action_model_pt=args.action_model_pt,
        output_path=args.output_path,
        vlm_module=args.vlm_module,
    )


if __name__ == "__main__":
    main()
