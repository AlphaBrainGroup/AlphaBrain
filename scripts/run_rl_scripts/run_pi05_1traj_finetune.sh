#!/bin/bash
# π₀.₅ 1-traj LIBERO-goal finetune — feeds the RLT pipeline with a 1-traj
# pi05 VLA ckpt, analogous to the QwenOFT-1traj ckpt consumed by
# run_rlt_ori_pretrain.sh.
#
# Thin wrapper around scripts/run_finetune.sh; the `pi05_1traj` mode in
# configs/finetune_config.yaml owns all hyperparams (PaliGemmaPi05 framework,
# num_traj_per_task=1, 80k steps, LR 2.5e-5, 5 GPU,
# pretrained_checkpoint=$PI05_PRETRAINED_PATH).
#
# Produces: results/training/Pi05-1traj-libero_goal/final_model
#   → consume via  CKPT_PATH=results/training/Pi05-1traj-libero_goal/final_model
#                  bash scripts/run_rl_scripts/run_rlt_ori_pretrain.sh 0
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

# Point PaliGemma's AutoTokenizer at the local model dir (has tokenizer.json
# + tokenizer.model). Otherwise _init_tokenizer tries to hit the HF hub for
# `google/paligemma-3b-pt-224`, fails offline, and falls back to sentencepiece
# which may not be installed.
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/datasets/peligemma}"

# Extra CLI overrides pass straight through to run_finetune.sh / accelerate.
# Example:  bash scripts/run_rl_scripts/run_pi05_1traj_finetune.sh \
#              --training.max_train_steps 40000
exec bash scripts/run_finetune.sh pi05_1traj "$@"
