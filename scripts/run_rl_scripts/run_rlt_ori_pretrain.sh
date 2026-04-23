#!/bin/bash
# RLT_ori Phase-1 pretraining (encoder-decoder reconstruction over
# frozen VLA hidden states). Strict-reference mode: --image_only
# keeps only image-token positions as z_{1:M} (Fig. 2, footnote 1).
#
# Duration model — matches the sibling RLActionToken pretrain:
#   - `--pretrain_max_steps N`: stop after N optimizer.step() calls.
#     When >0 this is the hard budget; --pretrain_epochs becomes an
#     upper cap that's usually never hit.
#   - `--pretrain_epochs E`: max # of dataset passes. If you want a
#     pure epoch-driven run, set MAX_STEPS=0 and dial EPOCHS.
# The sibling RLActionToken pretrain used 500 epochs × 3000 obs / bs 32
# ≈ 46k steps. We default here to MAX_STEPS=30000 to match that order
# (≈ 30 min on an A800 for the 1-traj LIBERO-goal demo set at batch 8).
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ── knobs ─────────────────────────────────────────────────────────────────
GPU_ID=${1:-0}
MAX_STEPS=${MAX_STEPS:-30000}           # hard gradient-step budget (0 = off)
EPOCHS=${EPOCHS:-10000}                 # epoch cap (mostly irrelevant when MAX_STEPS>0)
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-4}
ALPHA_VLA=${ALPHA_VLA:-0.0}             # >0 → joint VLA SFT (Alg. 1, line 3)
ENCODER_LAYERS=${ENCODER_LAYERS:-2}
DECODER_LAYERS=${DECODER_LAYERS:-2}
ENCODER_HEADS=${ENCODER_HEADS:-8}
MAX_LEN=${MAX_LEN:-4096}
SEED=${SEED:-42}
RUN_TAG=${RUN_TAG:-1traj_libero_goal}

CKPT_PATH="${CKPT_PATH:-results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model}"
# Demo config shipped with the 1-traj SFT ckpt — has `datasets.vla_data.data_root_dir`
# and `dataset_mix: libero_goal` already set; the DataLoader pulls LeRobot
# LIBERO data from $LIBERO_DATA_ROOT/libero_goal_no_noops_1.0.0_lerobot.
DEMO_CONFIG="${DEMO_CONFIG:-${CKPT_PATH}/framework_config.yaml}"

TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="results/rlt_ori_training/${RUN_TAG}_${TIMESTAMP}/pretrain"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}"
    exit 1
fi
if [ ! -f "${DEMO_CONFIG}" ]; then
    echo "WARNING: demo config not found: ${DEMO_CONFIG}"
    echo "         Will fall back to random-rollout observations (deviates from reference)."
    DEMO_FLAG=""
else
    DEMO_FLAG="--demo_config ${DEMO_CONFIG}"
fi

echo "============================================================"
echo " RLT_ori Phase-1 pretrain  (GPU ${GPU_ID})"
echo "   ckpt:        ${CKPT_PATH}"
echo "   demo cfg:    ${DEMO_CONFIG:-<none>}"
echo "   budget:      max_steps=${MAX_STEPS}  epochs_cap=${EPOCHS}  batch=${BATCH_SIZE}"
echo "   alpha_vla:   ${ALPHA_VLA}"
echo "   output:      ${OUTPUT_DIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU_ID} python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase pretrain_rlt_ori \
    --ckpt_path "${CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    ${DEMO_FLAG} \
    --suite libero_goal \
    --all_tasks \
    --image_only \
    --encoder_layers ${ENCODER_LAYERS} \
    --decoder_layers ${DECODER_LAYERS} \
    --encoder_heads ${ENCODER_HEADS} \
    --max_len ${MAX_LEN} \
    --pretrain_epochs ${EPOCHS} \
    --pretrain_max_steps ${MAX_STEPS} \
    --pretrain_lr ${LR} \
    --pretrain_batch_size ${BATCH_SIZE} \
    --alpha_vla ${ALPHA_VLA} \
    --seed ${SEED} \
    --use_wandb \
    --wandb_project AlphaBrain_RLT \
    --run_name rlt_ori_pretrain_${RUN_TAG}

# ── how to change duration ───────────────────────────────────────────────
# step-budget (paper-like):    MAX_STEPS=30000  bash scripts/run_rl_scripts/run_rlt_ori_pretrain.sh 0
# quick smoke:                 MAX_STEPS=500    bash scripts/run_rl_scripts/run_rlt_ori_pretrain.sh 0
# pure-epoch mode:             MAX_STEPS=0 EPOCHS=50  bash scripts/run_rl_scripts/run_rlt_ori_pretrain.sh 0
# joint VLA SFT:               ALPHA_VLA=1.0 MAX_STEPS=20000 bash ...
# 5-traj ckpt:                 CKPT_PATH=results/training/QwenOFT-5traj-libero_goal/final_model \
#                              RUN_TAG=5traj_libero_goal MAX_STEPS=30000 \
#                              bash scripts/run_rl_scripts/run_rlt_ori_pretrain.sh 0
