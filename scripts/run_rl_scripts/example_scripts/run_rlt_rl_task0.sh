#!/bin/bash
# RLT Phase-2 TD3 on libero_goal task 0 (single task).
# Reuses the RLT_a TD3 pipeline, swaps the encoder family to
# RLT via --encoder_mode rlt. Async BatchInferenceServer path
# (no --use_steplock yet on the RLT track).
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GPU_IDS=${1:-"0,1"}                  # two GPUs: rollout + train
ROLLOUT_GPUS=${ROLLOUT_GPUS:-"0"}
TRAIN_GPU=${TRAIN_GPU:-1}

CKPT_PATH="results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model"
ENCODER_PATH="results/rlt_training/1traj_libero_goal_step30k_0423_0545/pretrain/checkpoints/pretrain_best/encoder.pt"

RUN_NAME="rlt_rl_task0"
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="results/rlt_training/${RUN_NAME}_${TIMESTAMP}/rl_offpolicy"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}"
    exit 1
fi
if [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: RLT encoder not found: ${ENCODER_PATH}"
    exit 1
fi

echo "============================================================"
echo " RLT Phase-2 TD3  (libero_goal task 0)"
echo "   rollout GPUs: ${ROLLOUT_GPUS}"
echo "   train GPU:    ${TRAIN_GPU}"
echo "   ckpt:         ${CKPT_PATH}"
echo "   encoder:      ${ENCODER_PATH}"
echo "   output:       ${OUTPUT_DIR}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${GPU_IDS}

python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --encoder_mode rlt \
    --ckpt_path ${CKPT_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal \
    --task_id 0 \
    --rollout_gpus ${ROLLOUT_GPUS} \
    --train_gpu ${TRAIN_GPU} \
    --bottleneck_dim 2048 \
    --encoder_layers 2 \
    --encoder_heads 8 \
    --actor_hidden_dim 512 \
    --critic_hidden_dim 512 \
    --ref_dropout 0.5 \
    --fixed_std 0.1 \
    --G_per_task 10 \
    --group_size 1 \
    --num_envs_per_task 8 \
    --reward_coef 5.0 \
    --lr_actor 3e-4 \
    --lr_critic 3e-4 \
    --gamma 0.99 \
    --max_grad_norm 1.0 \
    --buffer_capacity 500000 \
    --buffer_warmup 512 \
    --warmup_iters 3 \
    --td_updates_per_iter 5000 \
    --utd_ratio 10.0 \
    --td_batch_size 512 \
    --tau 0.005 \
    --beta 1.0 \
    --actor_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --max_iter 200 \
    --eval_interval 20 \
    --eval_n_episodes 10 \
    --save_interval 50 \
    --save_video_interval 50 \
    --seed 42 \
    --use_wandb \
    --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_NAME}" \
    --log_interval 1
