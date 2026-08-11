#!/bin/bash
# Phase-1 encoder pretrain. Two tracks share this launcher:
#
#   TRACK=rlt    (default)  paper-faithful RL Token encoder/decoder over the
#                           full VLM token stream; reconstruction loss against
#                           stop-gradient VLA hidden states. Uses --phase pretrain_rlt.
#
#   TRACK=rlt_a             action-token variant — encoder consumes the
#                           action-query slice + Linear(H→D=256) bottleneck.
#                           Uses --phase pretrain (the original recipe).
#
# Usage:
#   bash scripts/run_rl_scripts/run_rlt_pretrain.sh [GPU_ID]                 # RLT, GPU 0
#   TRACK=rlt_a bash scripts/run_rl_scripts/run_rlt_pretrain.sh 0            # RLT_a, GPU 0
#
# Common env overrides:
#   CKPT_PATH    VLA finetune checkpoint to encode (default: QwenOFT-1traj)
#   RUN_TAG     subdir tag under results/rlt_training/
#   MAX_STEPS    pretrain step budget (default 30000)
#   BATCH_SIZE   pretrain batch size (default 8 for rlt, 32 for rlt_a)
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

if [ "${ALPHABRAIN_UI_LAUNCH:-0}" != "1" ] && [ -f .env ]; then
    set -a; source .env; set +a
fi
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GPU_ID=${1:-0}
TRACK=${TRACK:-rlt}
SEED=${SEED:-42}
CKPT_PATH="${CKPT_PATH:-results/training/QwenOFT-5traj-libero_goal/final_model}"
PYTHON_CMD=(python)
if [ "${ALPHABRAIN_UI_LAUNCH:-0}" != "1" ]; then
    PYTHON_CMD=(env "CUDA_VISIBLE_DEVICES=${GPU_ID}" python)
fi

TIMESTAMP=$(date +%m%d_%H%M)

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2
    exit 1
fi

case "${TRACK}" in
    rlt)
        # Paper-faithful track: pretrain_rlt, full VLM tokens, encoder-decoder
        # cross-attention. Demo-driven via --demo_config (from the SFT ckpt's
        # own framework_config.yaml) if present; otherwise random rollouts.
        RUN_TAG="${RUN_TAG:-rlt_$(basename ${CKPT_PATH%/*})}"
        OUTPUT_DIR="${OUTPUT_DIR:-results/rlt_training/${RUN_TAG}_${TIMESTAMP}/pretrain}"
        MAX_STEPS=${MAX_STEPS:-30000}
        EPOCHS=${EPOCHS:-10000}
        BATCH_SIZE=${BATCH_SIZE:-8}
        LR=${LR:-1e-4}
        ALPHA_VLA=${ALPHA_VLA:-0.0}
        DEMO_CONFIG="${DEMO_CONFIG:-${CKPT_PATH}/framework_config.yaml}"
        if [ -f "${DEMO_CONFIG}" ]; then DEMO_FLAG="--demo_config ${DEMO_CONFIG}"; else DEMO_FLAG=""; fi

        echo "============================================================"
        echo " RLT Phase-1 pretrain  (TRACK=rlt, GPU ${GPU_ID})"
        echo "   ckpt:    ${CKPT_PATH}"
        echo "   demo:    ${DEMO_CONFIG:-<none, falling back to random rollouts>}"
        echo "   budget:  max_steps=${MAX_STEPS}  batch=${BATCH_SIZE}  lr=${LR}"
        echo "   output:  ${OUTPUT_DIR}"
        echo "============================================================"

        "${PYTHON_CMD[@]}" AlphaBrain/training/reinforcement_learning/trainers/train.py \
            --phase pretrain_rlt \
            --ckpt_path "${CKPT_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            ${DEMO_FLAG} \
            --suite libero_goal \
            --all_tasks \
            --image_only \
            --encoder_layers 2 \
            --decoder_layers 2 \
            --encoder_heads 8 \
            --max_len 4096 \
            --pretrain_epochs ${EPOCHS} \
            --pretrain_max_steps ${MAX_STEPS} \
            --pretrain_lr ${LR} \
            --pretrain_batch_size ${BATCH_SIZE} \
            --alpha_vla ${ALPHA_VLA} \
            --seed ${SEED} \
            --use_wandb \
            --wandb_project AlphaBrain_RLT \
            --run_name rlt_pretrain_${RUN_TAG}
        ;;

    rlt_a)
        # Action-token track: pretrain on action-query slice, D=256 bottleneck,
        # encoder_heads=4. Observations from random rollout (the original recipe).
        RUN_TAG="${RUN_TAG:-rlt_a_$(basename ${CKPT_PATH%/*})}"
        OUTPUT_DIR="${OUTPUT_DIR:-results/rlt_training/${RUN_TAG}_${TIMESTAMP}/pretrain}"
        EPOCHS=${EPOCHS:-500}
        BATCH_SIZE=${BATCH_SIZE:-32}
        LR=${LR:-1e-4}

        echo "============================================================"
        echo " RLT_a Phase-1 pretrain  (TRACK=rlt_a, GPU ${GPU_ID})"
        echo "   ckpt:    ${CKPT_PATH}"
        echo "   recipe:  3000 rollout obs × 20 steps/reset, epochs=${EPOCHS}"
        echo "   output:  ${OUTPUT_DIR}"
        echo "============================================================"

        "${PYTHON_CMD[@]}" AlphaBrain/training/reinforcement_learning/trainers/train.py \
            --phase pretrain \
            --ckpt_path "${CKPT_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --suite libero_goal \
            --all_tasks \
            --bottleneck_dim 256 \
            --encoder_layers 2 \
            --encoder_heads 4 \
            --pretrain_n_obs 3000 \
            --pretrain_steps_per_reset 20 \
            --pretrain_epochs ${EPOCHS} \
            --pretrain_lr ${LR} \
            --pretrain_batch_size ${BATCH_SIZE} \
            --vla_extract_batch_size 16 \
            --num_envs_per_task 8 \
            --seed ${SEED} \
            --use_wandb \
            --wandb_project AlphaBrain_RLT \
            --run_name rlt_a_pretrain_${RUN_TAG}
        ;;

    *)
        echo "ERROR: TRACK must be 'rlt' or 'rlt_a' (got '${TRACK}')" >&2
        exit 1
        ;;
esac
