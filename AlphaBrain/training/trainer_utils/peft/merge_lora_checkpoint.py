#!/usr/bin/env python3
"""
merge_lora_checkpoint.py — Merge LoRA adapter + non-VLM weights into a full
checkpoint usable by the standard eval pipeline (server_policy.py +
BaseFramework.from_pretrained).

Thin CLI wrapper around the sibling `load_and_merge()` helper. Located inside
the peft module so it can be invoked via `python -m` without path hacks.

Usage (from repo root, starVLA env active):
    python -m AlphaBrain.training.trainer_utils.peft.merge_lora_checkpoint \\
        --base_config configs/continual_learning/cl_base.yaml \\
                      configs/continual_learning/models/qwengr00t.yaml \\
        --lora_adapter_dir results/Checkpoints/.../task_4_id4_steps_50000_lora_adapter \\
        --action_model_pt  results/Checkpoints/.../task_4_id4_steps_50000_action_model.pt \\
        --output_path      results/Checkpoints/.../task_4_id4_steps_50000_pytorch_model.pt
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into full AlphaBrain checkpoint")
    parser.add_argument("--base_config", type=str, nargs="+", required=True,
                        help=(
                            "One or more YAML config paths merged left-to-right "
                            "(same semantics as train.py --config_yaml). "
                            "E.g. --base_config cl_base.yaml models/qwengr00t.yaml"
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

    print(f"[0/4] Loading config from {args.base_config}")
    cfg = OmegaConf.load(args.base_config[0])
    for _extra in args.base_config[1:]:
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
