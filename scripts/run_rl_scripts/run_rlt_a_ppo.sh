#!/bin/bash
# RLT_a Phase-2 PPO (on-policy) launcher — Qwen backbone, single-task or
# multi-task LIBERO. Sibling of run_rlt_rl.sh (which is TD3 off-policy);
# this script wires the on-policy PPO path through the same trainer.
#
# PPO vs the TD3 launcher:
#   - No replay buffer / no buffer warmup
#   - Stochastic actor (Gaussian, fixed_std) instead of deterministic + noise
#   - Value critic V(s), trained jointly via clipped value loss
#   - GAE(λ) advantage from collected rollouts
#   - N PPO epochs per iteration over the just-collected batch
#
# Usage:
#   bash scripts/run_rl_scripts/run_rlt_a_ppo.sh [GPU_ID]                 # single task
#   TASK_ID=3 bash scripts/run_rl_scripts/run_rlt_a_ppo.sh 0              # different task
#   MULTI_TASK=1 bash scripts/run_rl_scripts/run_rlt_a_ppo.sh 0           # all 10 libero_goal tasks
#
# Env overrides:
#   TASK_ID         libero_goal task index (default 0)
#   MULTI_TASK      set to 1 for --all_tasks (default 0 = single task)
#   CKPT_PATH       VLA ckpt (default Qwen 5traj)
#   ENCODER_PATH    Phase-1 encoder.pt (auto-discovered if unset)
#   PPO_EPOCHS      PPO epochs per iter (default 10)
#   G_PER_TASK      episodes per iter per task (default 16)
#   NUM_ENVS_PER_TASK  parallel envs (default 8)
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
TASK_ID=${TASK_ID:-0}
MULTI_TASK=${MULTI_TASK:-0}
PPO_EPOCHS=${PPO_EPOCHS:-10}
G_PER_TASK=${G_PER_TASK:-16}
NUM_ENVS_PER_TASK=${NUM_ENVS_PER_TASK:-8}
MAX_ITER=${MAX_ITER:-300}
EVAL_INTERVAL=${EVAL_INTERVAL:-20}

DEFAULT_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
# Auto-discover latest RLT_a-format encoder (rlt_a / smoke_rlt_a / 5traj_alltasks_*).
_PRETRAIN_DIR=$(ls -td results/rlt_training/{rlt_a,smoke_rlt_a,5traj_alltasks}_*/pretrain 2>/dev/null | head -1 || true)
DEFAULT_ENCODER="${_PRETRAIN_DIR:+${_PRETRAIN_DIR}/checkpoints/pretrain_best/encoder.pt}"

CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
ENCODER_PATH="${ENCODER_PATH:-${DEFAULT_ENCODER}}"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2
    exit 1
fi
if [ -z "${ENCODER_PATH}" ] || [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: RLT_a encoder not found: ${ENCODER_PATH:-<unset>}" >&2
    echo "       Run TRACK=rlt_a bash scripts/run_rl_scripts/run_rlt_pretrain.sh first." >&2
    exit 1
fi

if [ "${MULTI_TASK}" = "1" ]; then
    TASK_FLAG="--all_tasks"
    RUN_TAG="rlt_a_ppo_qwen_alltasks"
else
    TASK_FLAG="--task_id ${TASK_ID}"
    RUN_TAG="rlt_a_ppo_qwen_t${TASK_ID}"
fi
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="${OUTPUT_DIR:-results/rlt_training/${RUN_TAG}_${TIMESTAMP}/rl_onpolicy}"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

echo "============================================================"
echo " RLT_a Phase-2 PPO (Qwen, ${TASK_FLAG})"
echo "   GPU:          ${GPU_ID}"
echo "   ckpt:         ${CKPT_PATH}"
echo "   encoder:      ${ENCODER_PATH}"
echo "   PPO epochs:   ${PPO_EPOCHS}"
echo "   G/task:       ${G_PER_TASK}    envs/task: ${NUM_ENVS_PER_TASK}"
echo "   output:       ${OUTPUT_DIR}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python -u AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl --encoder_mode action_token \
    --ckpt_path ${CKPT_PATH} --encoder_path ${ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal ${TASK_FLAG} \
    --bottleneck_dim 256 --encoder_layers 2 --encoder_heads 4 \
    --actor_hidden_dim 512 --critic_hidden_dim 512 \
    --ref_dropout 0.5 --fixed_std 0.1 \
    --G_per_task ${G_PER_TASK} --num_envs_per_task ${NUM_ENVS_PER_TASK} \
    --reward_coef 5.0 \
    --lr_actor 3e-4 --lr_critic 3e-4 --gamma 0.99 --max_grad_norm 1.0 \
    --ppo_epochs ${PPO_EPOCHS} --gae_lambda 0.95 \
    --max_iter ${MAX_ITER} --eval_interval ${EVAL_INTERVAL} --eval_n_episodes 20 \
    --save_interval 50 --save_video_interval 100 \
    --seed 42 \
    --use_wandb --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_TAG}" --log_interval 1 \
    2>&1 | tee "${TRAIN_LOG}"
