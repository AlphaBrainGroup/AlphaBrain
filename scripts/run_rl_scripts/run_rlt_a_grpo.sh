#!/bin/bash
# RLT_a Phase-2 GRPO (group-relative policy optimization) launcher.
# DeepSeek-style: no value critic, per-episode advantage is
# (R - μ_group) / σ_group where group = same initial state (state_idx).
# KL penalty to a frozen reference actor (snapshot at iter 0).
#
# Sibling of run_rlt_a_ppo.sh (PPO). Same actor / encoder / rollout
# infrastructure; only the loss + trainer differ.
#
# Usage:
#   bash scripts/run_rl_scripts/run_rlt_a_grpo.sh [GPU_ID]
#   TASK_ID=3 bash scripts/run_rl_scripts/run_rlt_a_grpo.sh 0
#   MULTI_TASK=1 bash scripts/run_rl_scripts/run_rlt_a_grpo.sh 0
#
# Env overrides:
#   TASK_ID            libero_goal task index (default 0)
#   MULTI_TASK         1 = --all_tasks (default 0 = single task)
#   CKPT_PATH          VLA ckpt (default Qwen 5traj)
#   ENCODER_PATH       Phase-1 encoder.pt (auto-discovered if unset)
#   GRPO_EPOCHS        epochs per iter (default 4; reuses --ppo_epochs flag)
#   GRPO_KL_COEF       KL-to-ref coefficient (default 0.04)
#   REF_UPDATE_INTERVAL  refresh reference actor every N iters (default 0 = never)
#   G_PER_TASK         episodes per iter per task (default 16; ≥ 8 recommended for group signal)
#   GROUP_SIZE         episodes per initial state (default 4; needs ≥ 2 for advantage signal)
#   NUM_ENVS_PER_TASK  parallel envs (default 8)
#   MAX_ITER           total iterations (default 300)
#   EVAL_INTERVAL      eval cadence (default 20)
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
TASK_SCOPE=${TASK_SCOPE:-}
TASK_IDS=${TASK_IDS:-}
GRPO_EPOCHS=${GRPO_EPOCHS:-4}
GRPO_KL_COEF=${GRPO_KL_COEF:-0.04}
REF_UPDATE_INTERVAL=${REF_UPDATE_INTERVAL:-0}
G_PER_TASK=${G_PER_TASK:-16}
GROUP_SIZE=${GROUP_SIZE:-4}
NUM_ENVS_PER_TASK=${NUM_ENVS_PER_TASK:-8}
MAX_ITER=${MAX_ITER:-300}
EVAL_INTERVAL=${EVAL_INTERVAL:-20}

DEFAULT_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
_PRETRAIN_DIR=$(ls -td results/rlt_training/{rlt_a,smoke_rlt_a,5traj_alltasks}_*/pretrain 2>/dev/null | head -1 || true)
DEFAULT_ENCODER="${_PRETRAIN_DIR:+${_PRETRAIN_DIR}/checkpoints/pretrain_best/encoder.pt}"

CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
ENCODER_PATH="${ENCODER_PATH:-${DEFAULT_ENCODER}}"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2; exit 1
fi
if [ -z "${ENCODER_PATH}" ] || [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: RLT_a encoder not found: ${ENCODER_PATH:-<unset>}" >&2
    echo "       Run TRACK=rlt_a bash scripts/run_rl_scripts/run_rlt_pretrain.sh first." >&2
    exit 1
fi

if [ -z "${TASK_SCOPE}" ]; then [ "${MULTI_TASK}" = "1" ] && TASK_SCOPE=all || TASK_SCOPE=single; fi
case "${TASK_SCOPE}" in
    all) TASK_ARGS=(--all_tasks); TASK_LABEL="all tasks"; RUN_TAG="rlt_a_grpo_qwen_alltasks" ;;
    subset)
        [ -n "${TASK_IDS}" ] || { echo "ERROR: TASK_IDS is required when TASK_SCOPE=subset" >&2; exit 1; }
        TASK_ARGS=(--task_ids "${TASK_IDS}"); TASK_LABEL="tasks ${TASK_IDS}"; RUN_TAG="rlt_a_grpo_qwen_subset"
        ;;
    single) TASK_ARGS=(--task_id "${TASK_ID}"); TASK_LABEL="task ${TASK_ID}"; RUN_TAG="rlt_a_grpo_qwen_t${TASK_ID}" ;;
    *) echo "ERROR: TASK_SCOPE must be single, subset, or all" >&2; exit 1 ;;
esac
PYTHON_CMD=(python)
if [ "${ALPHABRAIN_UI_LAUNCH:-0}" != "1" ]; then PYTHON_CMD=(env "CUDA_VISIBLE_DEVICES=${GPU_ID}" python); fi
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="${OUTPUT_DIR:-results/rlt_training/${RUN_TAG}_${TIMESTAMP}/rl_grpo}"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

echo "============================================================"
echo " RLT_a Phase-2 GRPO (Qwen, ${TASK_LABEL})"
echo "   GPU:          ${GPU_ID}"
echo "   ckpt:         ${CKPT_PATH}"
echo "   encoder:      ${ENCODER_PATH}"
echo "   epochs/iter:  ${GRPO_EPOCHS}    kl_coef: ${GRPO_KL_COEF}"
echo "   G/task:       ${G_PER_TASK}    group_size: ${GROUP_SIZE}    envs/task: ${NUM_ENVS_PER_TASK}"
echo "   ref_update:   every ${REF_UPDATE_INTERVAL} iter (0=never)"
echo "   output:       ${OUTPUT_DIR}"
echo "============================================================"

"${PYTHON_CMD[@]}" -u AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase grpo --encoder_mode action_token \
    --ckpt_path ${CKPT_PATH} --encoder_path ${ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite "${SUITE:-libero_goal}" "${TASK_ARGS[@]}" \
    --bottleneck_dim 256 --encoder_layers 2 --encoder_heads 4 \
    --actor_hidden_dim 512 --critic_hidden_dim 512 \
    --ref_dropout 0.5 --fixed_std 0.1 \
    --G_per_task ${G_PER_TASK} --group_size ${GROUP_SIZE} --num_envs_per_task ${NUM_ENVS_PER_TASK} \
    --reward_coef 5.0 \
    --lr_actor 3e-4 --lr_critic 3e-4 --gamma 0.99 --max_grad_norm 1.0 \
    --ppo_epochs ${GRPO_EPOCHS} --clip_eps 0.2 \
    --grpo_kl_coef ${GRPO_KL_COEF} --ref_update_interval ${REF_UPDATE_INTERVAL} \
    --max_iter ${MAX_ITER} --eval_interval ${EVAL_INTERVAL} --eval_n_episodes 20 \
    --save_interval 50 --save_video_interval 100 \
    --seed 42 \
    --use_wandb --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_TAG}" --log_interval 1 \
    2>&1 | tee "${TRAIN_LOG}"
