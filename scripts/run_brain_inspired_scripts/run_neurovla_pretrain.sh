#!/bin/bash
# =============================================================================
# NeuroVLA Pre-training Launch Script
#
# Trains NeuroVLA (Qwen-VL + QFormer + SNN action head) from scratch on a LIBERO
# subset (or the full 4-suite mix) using standard backprop.
#
# Default settings (RECOMMENDED — reproduces our reported results):
#   per_device_batch_size : 16       (override with --batch-size)
#   attn_implementation   : sdpa     (override with --attn)
#   gradient_accumulation : 1
#   effective batch       : 16 x num_gpus x 1
#
# !!! IMPORTANT !!!
# We strongly recommend keeping attn_implementation = sdpa.
# In our experiments, swapping to flash_attention_2 led to unstable
# training dynamics on NeuroVLA (numerical differences propagate through
# the QFormer + SNN stack). Only change if you have a specific reason
# and are prepared to re-tune.
#
# Default training steps:
#   single suite (libero_goal / spatial / object / 10)  -> 30000 steps
#   multi-suite  (libero_all)                           -> 50000 steps
# Override with --steps <N>.
#
# Usage:
#   bash run_neurovla_pretrain.sh                                  # libero_goal, 30k, bs=16, sdpa
#   bash run_neurovla_pretrain.sh --dataset libero_all             # all 4 suites, 50k, bs=16, sdpa
#   bash run_neurovla_pretrain.sh --batch-size 8                   # smaller batch
#   bash run_neurovla_pretrain.sh --attn flash_attention_2         # enable flash attention
#   bash run_neurovla_pretrain.sh --steps 50000 --run-id my_pretrain
#
# Supported --dataset values:
#   libero_goal      (default)  - 10 goal-directed tasks
#   libero_spatial              - 10 spatial reasoning tasks
#   libero_object               - 10 object manipulation tasks
#   libero_10                   - 10 long-horizon tasks (aka libero_long)
#   libero_all                  - combined mixture of all 4 suites
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

if [ "${ALPHABRAIN_UI_LAUNCH:-0}" != "1" ] && [ -f .env ]; then
    set -a; source .env; set +a
fi

# ---------- defaults ----------
CONFIG_YAML="${CONFIG_YAML:-configs/finetune_config.yaml}"
MODE="${MODE:-neuro_vla}"
DATASET="libero_goal"
BATCH_SIZE=16
ATTN_IMPL="sdpa"
MAX_STEPS=""
RUN_ID=""
NUM_GPUS="${NUM_GPUS:-4}"
MAIN_PORT="${MAIN_PROCESS_PORT:-29500}"
DS_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/accelerate_zero2.yaml}"
CONDA_ENV="${CONDA_ENV:-alphabrain}"

# ---------- parse CLI ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)     MODE="$2"; shift 2 ;;
        --dataset)  DATASET="$2"; shift 2 ;;
        --batch-size|--bs) BATCH_SIZE="$2"; shift 2 ;;
        --attn|--attn-impl) ATTN_IMPL="$2"; shift 2 ;;
        --steps|--max-steps) MAX_STEPS="$2"; shift 2 ;;
        --run-id)   RUN_ID="$2"; shift 2 ;;
        --gpus)     NUM_GPUS="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------- dataset-dependent default step count ----------
if [ -z "$MAX_STEPS" ]; then
    if [ "$DATASET" = "libero_all" ]; then
        MAX_STEPS=50000
    else
        MAX_STEPS=30000
    fi
fi

# ---------- build OmegaConf overrides ----------
OVERRIDES=(
    --datasets.vla_data.dataset_mix "$DATASET"
    --datasets.vla_data.per_device_batch_size "$BATCH_SIZE"
    --framework.qwenvl.attn_implementation "$ATTN_IMPL"
    --trainer.max_train_steps "$MAX_STEPS"
)
[ -n "$RUN_ID" ] && OVERRIDES+=(--run_id "$RUN_ID")

echo "=============================================="
echo "  NeuroVLA Pre-training"
echo "  Mode:         $MODE"
echo "  Dataset:      $DATASET"
echo "  Batch size:   $BATCH_SIZE  (per device)"
echo "  Attention:    $ATTN_IMPL"
echo "  Max steps:    $MAX_STEPS"
echo "  GPUs:         $NUM_GPUS"
[ -n "$RUN_ID" ] && echo "  Run ID:       $RUN_ID"
echo "=============================================="

# ---------- resolve accelerate binary (prefer conda env, fall back to PATH) ----------
ENV_ACC=""
for __p in /root/miniconda3/envs/${CONDA_ENV}/bin/accelerate /opt/conda/envs/${CONDA_ENV}/bin/accelerate; do
    if [ -x "$__p" ]; then ENV_ACC="$__p"; break; fi
done
if [ -n "$ENV_ACC" ] && [ -x "$ENV_ACC" ]; then
    ACC="$ENV_ACC"
elif command -v accelerate >/dev/null 2>&1; then
    ACC=accelerate
else
    echo "ERROR: accelerate binary not found in conda env ${CONDA_ENV} or PATH" >&2
    exit 1
fi

# ---------- launch ----------
"$ACC" launch \
    --config_file "$DS_CONFIG" \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PORT" \
    AlphaBrain/training/train_alphabrain.py \
    --config_yaml "$CONFIG_YAML" \
    --mode "$MODE" \
    "${OVERRIDES[@]}"
