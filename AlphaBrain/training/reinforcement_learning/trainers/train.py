#!/usr/bin/env python3
"""ActionToken training entry for QwenOFT on LIBERO.

Phases:
  --phase pretrain          Encoder-decoder pretraining via reconstruction loss (RLActionToken track)
  --phase pretrain_rlt_ori  Phase-1 pretraining that follows the RL Token reference line-by-line
  --phase rl                On-policy multi-GPU PPO update (legacy; PPO/GRPO is a TODO)
  --phase rl_offpolicy      Off-policy TD3 with split rollout/training GPUs (production)

Usage:
    # Phase 1: Encoder pretraining (pragmatic track)
    python AlphaBrain/training/reinforcement_learning/trainers/train.py --phase pretrain \
        --ckpt_path results/training/my_sft/final_model \
        --suite libero_goal --task_id 0

    # Phase 1: Encoder pretraining (reference track; uses demo data)
    python AlphaBrain/training/reinforcement_learning/trainers/train.py --phase pretrain_rlt_ori \
        --ckpt_path results/training/my_sft/final_model \
        --demo_config configs/datasets/libero.yaml \
        --suite libero_goal --task_id 0

    # Phase 2 (production): off-policy TD3
    python AlphaBrain/training/reinforcement_learning/trainers/train.py --phase rl_offpolicy \
        --ckpt_path results/training/my_sft/final_model \
        --encoder_path results/action_token_training/pretrain/checkpoints/pretrain_best/encoder.pt \
        --suite libero_goal --task_id 0
"""
from AlphaBrain.training.reinforcement_learning._bootstrap import setup

setup()  # load .env, set TOKENIZERS_PARALLELISM, configure logging — before heavy imports

from AlphaBrain.training.reinforcement_learning.trainers.train_args import parse_args
from AlphaBrain.training.reinforcement_learning.trainers.train_pretrain import run_pretrain
from AlphaBrain.training.reinforcement_learning.trainers.train_rl_offpolicy import run_rl_offpolicy
from AlphaBrain.training.reinforcement_learning.trainers.train_rl_onpolicy import run_rl
from AlphaBrain.training.reinforcement_learning.trainers.train_rlt_ori_pretrain import run_rlt_ori_pretrain


def main():
    args = parse_args()
    if args.phase == "pretrain":
        run_pretrain(args)
    elif args.phase == "pretrain_rlt_ori":
        run_rlt_ori_pretrain(args)
    elif args.phase == "rl":
        run_rl(args)
    elif args.phase == "rl_offpolicy":
        run_rl_offpolicy(args)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
