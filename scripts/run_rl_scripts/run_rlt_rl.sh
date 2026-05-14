#!/bin/bash
# Phase-2 off-policy TD3 RL. Two tracks + two backbones share this launcher.
#
# TRACK selects the encoder family:
#
#   TRACK=rlt   (default)  RL Token encoder over full VLM tokens (paper-faithful).
#                          `--encoder_mode rlt`. Bottleneck = VLA hidden dim H.
#   TRACK=rlt_a            action-token encoder over the action-query slice.
#                          `--encoder_mode action_token`. Bottleneck D=256.
#
# BACKBONE selects the VLA family:
#
#   BACKBONE=qwen (default) Qwen2.5-VL-3B + MLP action head
#   BACKBONE=pi05           PaliGemmaPi05 + flow matching (RLT only — RLT_a
#                           × Pi05 is roadmapped, not wired)
#
# For BACKBONE=pi05 also pick VARIANT={1traj,5traj}.
#
# Usage:
#   bash scripts/run_rl_scripts/run_rlt_rl.sh [GPU_ID]                              # RLT × Qwen, task 0
#   BACKBONE=pi05 VARIANT=5traj TASK_ID=3 bash scripts/run_rl_scripts/run_rlt_rl.sh # RLT × Pi05-5traj, task 3
#   TRACK=rlt_a bash scripts/run_rl_scripts/run_rlt_rl.sh                           # RLT_a × Qwen, all tasks
#
# Env overrides:
#   TRACK        rlt | rlt_a     (default rlt)
#   BACKBONE     qwen | pi05     (default qwen; pi05 valid for rlt only)
#   VARIANT      1traj | 5traj   (pi05 only; default 1traj)
#   TASK_ID      libero_goal task index (rlt only; rlt_a always runs all tasks)
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
TRACK=${TRACK:-rlt}
BACKBONE=${BACKBONE:-qwen}
TASK_ID=${TASK_ID:-0}

# ── Backbone-specific defaults (ckpt + encoder dir prefix) ──
# Encoder.pt format depends on TRACK (RLT's encoder-decoder cross-attention
# layout vs RLT_a's CLS+bottleneck layout), so DEFAULT_ENCODER is per-track.
case "${BACKBONE}" in
    qwen)
        DEFAULT_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
        if [ "${TRACK}" = "rlt_a" ]; then
            # RLT_a (action_token) format encoder; latest rlt_a_* pretrain dir.
            _PRETRAIN_DIR=$(ls -td results/rlt_training/{rlt_a,smoke_rlt_a,5traj_alltasks}_*/pretrain 2>/dev/null | head -1 || true)
            DEFAULT_ENCODER="${_PRETRAIN_DIR:+${_PRETRAIN_DIR}/checkpoints/pretrain_best/encoder.pt}"
        else
            # RLT (paper-faithful) format encoder.
            DEFAULT_ENCODER="results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt"
        fi
        BACKBONE_TAG="qwen"
        ;;
    pi05)
        if [ "${TRACK}" = "rlt_a" ]; then
            echo "ERROR: RLT_a × Pi05 isn't wired yet (PaliGemmaPi05 has no get_vla_action equivalent). Use TRACK=rlt for Pi05." >&2
            exit 1
        fi
        VARIANT=${VARIANT:-1traj}
        if [[ "${VARIANT}" != "1traj" && "${VARIANT}" != "5traj" ]]; then
            echo "ERROR: VARIANT must be '1traj' or '5traj' for BACKBONE=pi05 (got '${VARIANT}')" >&2
            exit 1
        fi
        export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/datasets/peligemma}"
        DEFAULT_CKPT="results/training/Pi05-goal-${VARIANT}-openpi/checkpoints/steps_30000"
        _PRETRAIN_DIR=$(ls -td results/rlt_training/pi05_${VARIANT}_openpi_strict_*/pretrain 2>/dev/null | head -1 || true)
        DEFAULT_ENCODER="${_PRETRAIN_DIR:+${_PRETRAIN_DIR}/checkpoints/pretrain_best/encoder.pt}"
        BACKBONE_TAG="pi05_${VARIANT}"
        ;;
    *)
        echo "ERROR: BACKBONE must be 'qwen' or 'pi05' (got '${BACKBONE}')" >&2
        exit 1
        ;;
esac

CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
ENCODER_PATH="${ENCODER_PATH:-${DEFAULT_ENCODER}}"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2
    exit 1
fi
if [ -z "${ENCODER_PATH}" ] || [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: encoder.pt not found: ${ENCODER_PATH:-<unset>}" >&2
    echo "       Run scripts/run_rl_scripts/run_rlt_pretrain.sh first." >&2
    exit 1
fi

TIMESTAMP=$(date +%m%d_%H%M)
RUN_NAME="${TRACK}_rl_${BACKBONE_TAG}_t${TASK_ID}"
OUTPUT_DIR="results/rlt_training/${RUN_NAME}_${TIMESTAMP}/rl_offpolicy"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

echo "============================================================"
echo " ${TRACK^^} Phase-2 TD3 (backbone=${BACKBONE_TAG}, task ${TASK_ID})"
echo "   GPU:        ${GPU_ID}"
echo "   ckpt:       ${CKPT_PATH}"
echo "   encoder:    ${ENCODER_PATH}"
echo "   output:     ${OUTPUT_DIR}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# ── Track-specific RL hyperparams ──
case "${TRACK}" in
    rlt)
        # RLT: single-task, bottleneck=H, 8 heads, 1e-3 lr, 300 iters
        python AlphaBrain/training/reinforcement_learning/trainers/train.py \
            --phase rl_offpolicy --encoder_mode rlt \
            --ckpt_path "${CKPT_PATH}" --encoder_path "${ENCODER_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --suite libero_goal --task_id ${TASK_ID} \
            --rollout_gpus 0 --train_gpu 0 \
            --bottleneck_dim 2048 --encoder_layers 2 --encoder_heads 8 \
            --actor_hidden_dim 512 --critic_hidden_dim 512 \
            --ref_dropout 0.5 --fixed_std 0.1 \
            --G 64 --group_size 8 --num_envs 64 \
            --reward_coef 5.0 \
            --lr_actor 1e-3 --lr_critic 1e-3 --gamma 0.99 --max_grad_norm 1.0 \
            --buffer_capacity 1000000 --buffer_warmup 256 --warmup_iters 5 \
            --td_updates_per_iter 10000 --utd_ratio 10.0 --td_batch_size 1024 \
            --tau 0.005 --beta 1.0 \
            --actor_update_freq 2 --target_noise_std 0.2 --target_noise_clip 0.5 \
            --max_iter 300 --eval_interval 10 --eval_n_episodes 20 \
            --save_interval 25 --save_video_interval 999 \
            --seed 42 --use_wandb --wandb_project AlphaBrain_RLT \
            --run_name "${RUN_NAME}" --log_interval 1 --use_steplock \
            2>&1 | tee "${TRAIN_LOG}"
        ;;

    rlt_a)
        # RLT_a: multi-task default, bottleneck=256, 4 heads, 3e-4 lr, 400 iters
        # G_per_task/num_envs_per_task instead of G/num_envs (multi-task scaling).
        python AlphaBrain/training/reinforcement_learning/trainers/train.py \
            --phase rl_offpolicy --encoder_mode action_token \
            --ckpt_path "${CKPT_PATH}" --encoder_path "${ENCODER_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --suite libero_goal --all_tasks \
            --rollout_gpus 0 --train_gpu 0 \
            --bottleneck_dim 256 --encoder_layers 2 --encoder_heads 4 \
            --actor_hidden_dim 512 --critic_hidden_dim 512 \
            --ref_dropout 0.5 --fixed_std 0.1 \
            --G_per_task 30 --group_size 1 --num_envs_per_task 10 \
            --reward_coef 5.0 \
            --lr_actor 3e-4 --lr_critic 3e-4 --gamma 0.99 --max_grad_norm 1.0 \
            --buffer_capacity 1000000 --buffer_warmup 1024 --warmup_iters 5 \
            --td_updates_per_iter 10000 --utd_ratio 10.0 --td_batch_size 1024 \
            --tau 0.005 --beta 1.0 \
            --actor_update_freq 2 --target_noise_std 0.2 --target_noise_clip 0.5 \
            --max_iter 400 --eval_interval 20 --eval_n_episodes 20 \
            --save_interval 50 --save_video_interval 100 \
            --seed 42 --use_wandb --wandb_project AlphaBrain_RLT \
            --run_name "${RUN_NAME}" --log_interval 1 --use_steplock \
            2>&1 | tee "${TRAIN_LOG}"
        ;;

    *)
        echo "ERROR: TRACK must be 'rlt' or 'rlt_a' (got '${TRACK}')" >&2
        exit 1
        ;;
esac
