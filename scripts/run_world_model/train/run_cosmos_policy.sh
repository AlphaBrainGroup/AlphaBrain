#!/usr/bin/env bash
# =============================================================================
# Cosmos Policy Training: Full-DiT 2B finetune on LIBERO
#
# Mirrors the production recipe that produced run_id=cosmos-8gpu-bs20-fix2-0408:
#   8 x 80GB GPUs, DeepSpeed ZeRO-2, per_device_batch_size=20,
#   grad_accum=12, effective batch=1920, 40K steps, warmup=1000, save every 5K.
#
# Usage:
#   bash scripts/run_world_model/train/run_cosmos_policy.sh
#
#   # Resume
#   RESUME=true bash scripts/run_world_model/train/run_cosmos_policy.sh
#
#   # Smaller GPU count (effective batch stays 1920 by auto-scaling grad_accum)
#   NUM_GPUS=4 PER_DEVICE_BATCH=20 GRAD_ACCUM=24 bash scripts/run_world_model/train/run_cosmos_policy.sh
#
# Environment variables:
#   GPU_IDS          comma-separated GPU list        (default: 0,1,2,3,4,5,6,7)
#   NUM_GPUS         process count                   (default: 8)
#   PER_DEVICE_BATCH per-GPU batch size              (default: 20)
#   GRAD_ACCUM       gradient accumulation steps     (default: 12)
#   MAX_STEPS        total training steps            (default: 40000)
#   SAVE_INTERVAL    ckpt save every N steps         (default: 5000)
#   WARMUP_STEPS     LR warmup steps                 (default: 1000)
#   LOG_FREQ         console/wandb log every N steps (default: 100)
#   EVAL_INTERVAL    eval every N steps              (default: 40001, i.e. off)
#   MASTER_PORT      distributed port                (default: 29600)
#   RESUME           true to resume from latest      (default: false)
#   RUN_ID           wandb + output dir name         (default: cosmos-<N>gpu-bs<BS>-<date>)
#   DEEPSPEED_CONFIG accelerate config               (default: configs/deepspeed/accelerate_zero2.yaml)
# =============================================================================
set -euo pipefail

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-12}"
MAX_STEPS="${MAX_STEPS:-40000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-40001}"
MASTER_PORT="${MASTER_PORT:-29600}"
RESUME="${RESUME:-false}"
RUN_ID="${RUN_ID:-cosmos-${NUM_GPUS}gpu-bs${PER_DEVICE_BATCH}-$(date +%m%d)}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/accelerate_zero2.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ "${ALPHABRAIN_UI_LAUNCH:-0}" != "1" ] && [ -f .env ]; then
    set -a; source .env; set +a
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

EFFECTIVE_BATCH=$((NUM_GPUS * PER_DEVICE_BATCH * GRAD_ACCUM))
OUTPUT_ROOT="${PROJECT_ROOT}/results/training"
mkdir -p "${OUTPUT_ROOT}/${RUN_ID}"

echo "============================================================"
echo "  Cosmos Policy Training (full-DiT 2B finetune)"
echo "  GPUs          : ${NUM_GPUS} (${GPU_IDS})"
echo "  Batch/GPU     : ${PER_DEVICE_BATCH}"
echo "  Grad accum    : ${GRAD_ACCUM}"
echo "  Effective BS  : ${EFFECTIVE_BATCH}"
echo "  Max steps     : ${MAX_STEPS} (save every ${SAVE_INTERVAL}, warmup ${WARMUP_STEPS})"
echo "  Resume        : ${RESUME}"
echo "  Run ID        : ${RUN_ID}"
echo "  Output        : ${OUTPUT_ROOT}/${RUN_ID}"
echo "============================================================"

python -m accelerate.commands.launch \
    --config_file "${DEEPSPEED_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    --main_process_port "${MASTER_PORT}" \
    AlphaBrain/training/train_alphabrain.py \
    --config_yaml configs/finetune_config.yaml \
    --mode cosmos_policy \
    "trainer.max_train_steps=${MAX_STEPS}" \
    "trainer.save_interval=${SAVE_INTERVAL}" \
    "trainer.eval_interval=${EVAL_INTERVAL}" \
    "trainer.num_warmup_steps=${WARMUP_STEPS}" \
    "trainer.logging_frequency=${LOG_FREQ}" \
    "trainer.gradient_accumulation_steps=${GRAD_ACCUM}" \
    "trainer.is_resume=${RESUME}" \
    "datasets.vla_data.per_device_batch_size=${PER_DEVICE_BATCH}" \
    "common.num_gpus=${NUM_GPUS}" \
    "common.output_root_dir=${OUTPUT_ROOT}" \
    "run_id=${RUN_ID}" \
    2>&1 | tee "${OUTPUT_ROOT}/${RUN_ID}/train_$(date +%Y%m%d_%H%M%S).log"

echo "[ok] Cosmos Policy training complete: ${OUTPUT_ROOT}/${RUN_ID}"
