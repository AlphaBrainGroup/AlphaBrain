#!/bin/bash
# RLT Phase-2 off-policy TD3 launcher.
#
# Two backbones supported, identical hyperparams + steplock — only the
# VLA checkpoint and Phase-1 encoder path differ:
#
#   BACKBONE=qwen                 (default) Qwen2.5-VL-3B + MLP action head
#   BACKBONE=pi05  VARIANT=1traj            PaliGemmaPi05 + flow matching (10-demo)
#   BACKBONE=pi05  VARIANT=5traj            PaliGemmaPi05 + flow matching (50-demo)
#
# Phase-2 RL dispatches on framework type (is_pi05) inside the trainer;
# the Pi05 path uses pi05_inference_zhanghe.py to fuse PaliGemma prefix
# Gemma forward + flow-matching diffusion per rollout step.
#
# Phase-1 (encoder pretrain) must already have produced encoder.pt — run
# scripts/run_rl_scripts/run_rlt_pretrain.sh first.
#
# Usage:
#   bash scripts/run_rl_scripts/run_rlt_rl.sh [GPU_ID]                              # Qwen, task 0
#   BACKBONE=pi05 VARIANT=1traj TASK_ID=3 bash scripts/run_rl_scripts/run_rlt_rl.sh # Pi05 1traj, task 3
#
# Env overrides:
#   BACKBONE     qwen|pi05     (default qwen)
#   VARIANT      1traj|5traj   (pi05 only; default 1traj)
#   TASK_ID      libero_goal task index (default 0)
#   CKPT_PATH    override VLA ckpt directory
#   ENCODER_PATH override Phase-1 encoder.pt
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GPU_ID=${1:-0}
TASK_ID=${TASK_ID:-0}
BACKBONE=${BACKBONE:-qwen}

case "${BACKBONE}" in
    qwen)
        DEFAULT_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
        DEFAULT_ENCODER="results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt"
        RUN_TAG="release"
        ;;
    pi05)
        VARIANT=${VARIANT:-1traj}
        if [[ "${VARIANT}" != "1traj" && "${VARIANT}" != "5traj" ]]; then
            echo "ERROR: VARIANT must be '1traj' or '5traj' for BACKBONE=pi05 (got '${VARIANT}')"
            exit 1
        fi
        # Pi05 needs the PaliGemma tokenizer accessible offline; otherwise
        # _init_tokenizer falls through to HF hub fetch and then sentencepiece.
        export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/datasets/peligemma}"
        DEFAULT_CKPT="results/training/Pi05-goal-${VARIANT}-openpi/checkpoints/steps_30000"
        # Auto-discover latest matching pretrain dir; caller can still override.
        _PRETRAIN_DIR=$(ls -td results/rlt_training/pi05_${VARIANT}_openpi_strict_*/pretrain 2>/dev/null | head -1 || true)
        DEFAULT_ENCODER="${_PRETRAIN_DIR:+${_PRETRAIN_DIR}/checkpoints/pretrain_best/encoder.pt}"
        RUN_TAG="release_pi05_${VARIANT}"
        ;;
    *)
        echo "ERROR: BACKBONE must be 'qwen' or 'pi05' (got '${BACKBONE}')" >&2
        exit 1
        ;;
esac

CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
ENCODER_PATH="${ENCODER_PATH:-${DEFAULT_ENCODER}}"

RUN_NAME="rlt_rl_t${TASK_ID}_${RUN_TAG}"
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="results/rlt_training/${RUN_NAME}_${TIMESTAMP}/rl_offpolicy"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2
    exit 1
fi
if [ -z "${ENCODER_PATH}" ] || [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: RLT encoder not found: ${ENCODER_PATH:-<unset>}" >&2
    echo "       Run scripts/run_rl_scripts/run_rlt_pretrain.sh first." >&2
    exit 1
fi

echo "============================================================"
echo " RLT Phase-2 TD3 (backbone=${BACKBONE}${VARIANT:+, variant=${VARIANT}}, libero_goal task ${TASK_ID})"
echo "   GPU:        ${GPU_ID}"
echo "   ckpt:       ${CKPT_PATH}"
echo "   encoder:    ${ENCODER_PATH}"
echo "   output:     ${OUTPUT_DIR}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --encoder_mode rlt \
    --ckpt_path ${CKPT_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal \
    --task_id ${TASK_ID} \
    --rollout_gpus 0 \
    --train_gpu 0 \
    --bottleneck_dim 2048 \
    --encoder_layers 2 \
    --encoder_heads 8 \
    --actor_hidden_dim 512 \
    --critic_hidden_dim 512 \
    --ref_dropout 0.5 \
    --fixed_std 0.1 \
    --G 64 \
    --group_size 8 \
    --num_envs 64 \
    --reward_coef 5.0 \
    --lr_actor 1e-3 \
    --lr_critic 1e-3 \
    --gamma 0.99 \
    --max_grad_norm 1.0 \
    --buffer_capacity 1000000 \
    --buffer_warmup 256 \
    --warmup_iters 5 \
    --td_updates_per_iter 10000 \
    --utd_ratio 10.0 \
    --td_batch_size 1024 \
    --tau 0.005 \
    --beta 1.0 \
    --actor_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --max_iter 300 \
    --eval_interval 10 \
    --eval_n_episodes 20 \
    --save_interval 25 \
    --save_video_interval 999 \
    --seed 42 \
    --use_wandb \
    --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_NAME}" \
    --log_interval 1 \
    --use_steplock \
    2>&1 | tee "${TRAIN_LOG}"
