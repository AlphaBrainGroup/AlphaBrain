#!/usr/bin/env bash
# =============================================================================
# Base VLA 统一训练脚本
#
# 支持三种框架 × 两种 VLM backbone 的组合:
#   - PaliGemmaOFT   (PaliGemma 3B + DiT action head)
#   - PaliGemmaPi05   (PaliGemma 3B + Pi0.5 flow matching action expert)
#   - LlamaOFT       (Llama 3.2 Vision 11B + DiT action head)
#
# 支持 libero_all (multi-task) 和单任务 (goal/spatial/object/long)
#
# 用法:
#   bash scripts/run_base_vla/train.sh <mode> [config_file]
#
# ─── Multi-task (libero_all) ────────────────────────────────────
#   bash scripts/run_base_vla/train.sh paligemma_pi05_openpi_aligned_v3
#   bash scripts/run_base_vla/train.sh paligemma_oft_all_150k
#   bash scripts/run_base_vla/train.sh llama_oft_all_150k
#
# ─── Single-task ────────────────────────────────────────────────
#   bash scripts/run_base_vla/train.sh paligemma_oft_goal
#   bash scripts/run_base_vla/train.sh paligemma_oft_spatial
#   bash scripts/run_base_vla/train.sh paligemma_oft_object
#   bash scripts/run_base_vla/train.sh paligemma_oft_long
#   bash scripts/run_base_vla/train.sh llama_oft_goal
#   bash scripts/run_base_vla/train.sh llama_oft_spatial
#   bash scripts/run_base_vla/train.sh llama_oft_object
#   bash scripts/run_base_vla/train.sh llama_oft_long
#
# 所有超参在 configs/finetune_config.yaml 中的 mode 下定义
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:?Usage: $0 <mode> [config_file]}"
CONFIG="${2:-}"

exec bash "${SCRIPT_DIR}/../run_finetune.sh" "${MODE}" ${CONFIG:+"${CONFIG}"}
