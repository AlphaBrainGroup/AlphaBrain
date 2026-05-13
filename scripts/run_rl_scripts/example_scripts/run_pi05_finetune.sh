#!/bin/bash
# Pi05 finetune launcher — thin wrapper around scripts/run_finetune.sh.
# Pick the variant via VARIANT={1traj,5traj,task0}. The actual hyperparams
# (LR, batch, save_interval, num_gpus, dataset_mix) live in the corresponding
# mode block of configs/finetune_config.yaml.
#
# Usage:
#   VARIANT=1traj  bash scripts/run_rl_scripts/run_pi05_finetune.sh
#   VARIANT=5traj  bash scripts/run_rl_scripts/run_pi05_finetune.sh
#   VARIANT=task0  bash scripts/run_rl_scripts/run_pi05_finetune.sh
#
# Extra CLI overrides pass through to run_finetune.sh / accelerate, e.g.:
#   VARIANT=1traj bash scripts/run_rl_scripts/run_pi05_finetune.sh \
#       --training.max_train_steps 40000
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
# Avoid HF hub fetch / sentencepiece fallback for the PaliGemma tokenizer.
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/datasets/peligemma}"

VARIANT="${VARIANT:-1traj}"
case "${VARIANT}" in
    1traj|5traj)
        MODE="pi05_goal_${VARIANT}_openpi"
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
        ;;
    task0)
        MODE="pi05_goal_task0"
        # task0 mode config sets num_gpus=5; pin to 4 GPUs that exclude the
        # user's commonly-busy 0/5/6.
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
        ;;
    *)
        echo "ERROR: unknown VARIANT='${VARIANT}'. Must be one of: 1traj, 5traj, task0." >&2
        exit 1
        ;;
esac

exec bash scripts/run_finetune.sh "${MODE}" "$@"
