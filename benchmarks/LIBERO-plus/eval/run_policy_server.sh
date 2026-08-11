#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-${EVAL_CHECKPOINT:-}}"
: "${checkpoint:?Pass a checkpoint as argument 1 or set EVAL_CHECKPOINT}"
port="${EVAL_PORT:-9883}"
python_executable="${ALPHABRAIN_PYTHON:-python}"

export DEBUG="${DEBUG:-1}"
CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID:-0}" "${python_executable}" \
    deployment/model_server/server_policy.py \
    --ckpt_path "${checkpoint}" \
    --port "${port}" \
    --use_bf16
