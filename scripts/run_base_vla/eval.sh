#!/usr/bin/env bash
# =============================================================================
# Base VLA 统一评估脚本
#
# 自动启 server → 等就绪 → 跑 eval → 清理
#
# 用法:
#   bash scripts/run_base_vla/eval.sh <eval_mode> [config_file]
#
# ─── PaliGemmaPi05 eval ─────────────────────────────────────────
#   bash scripts/run_base_vla/eval.sh paligemma_pi05_eval
#
# ─── PaliGemmaOFT eval ──────────────────────────────────────────
#   bash scripts/run_base_vla/eval.sh paligemma_oft_eval
#   bash scripts/run_base_vla/eval.sh paligemma_oft_mlp_goal30k_eval
#   bash scripts/run_base_vla/eval.sh paligemma_oft_bs128_goal_eval
#   bash scripts/run_base_vla/eval.sh paligemma_oft_bs128_spatial_eval
#   bash scripts/run_base_vla/eval.sh paligemma_oft_bs128_object_eval
#   bash scripts/run_base_vla/eval.sh paligemma_oft_bs128_long_eval
#
# ─── LlamaOFT eval ──────────────────────────────────────────────
#   bash scripts/run_base_vla/eval.sh llama_oft_eval
#
# Eval modes 在 configs/finetune_config.yaml 中定义, 需包含:
#   type: "eval", checkpoint, task_suite, num_trials, host, port, gpu_id
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:?Usage: $0 <eval_mode> [config_file]}"
CONFIG="${2:-}"

exec bash "${SCRIPT_DIR}/../run_eval.sh" "${MODE}" ${CONFIG:+"${CONFIG}"}
