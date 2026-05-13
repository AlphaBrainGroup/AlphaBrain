#!/bin/bash
# Auto-eval the latest steps_XXXXX ckpt of a given Pi05 training run.
#
# Usage:
#   bash scripts/run_rl_scripts/run_eval_pi05_latest_zhanghe.sh
#       → defaults to RUN_DIR=results/training/Pi05-goal-task0,
#         YAML=scripts/run_rl_scripts/pi05_eval.yaml,
#         MODE=pi05_goal_task0_eval
#
#   RUN_DIR=results/training/Pi05-something \
#   YAML=scripts/run_rl_scripts/pi05_something_eval.yaml \
#   MODE=pi05_something_eval \
#       bash scripts/run_rl_scripts/run_eval_pi05_latest_zhanghe.sh
#
# Side effect: rewrites the `checkpoint:` line of the eval YAML in place.
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

RUN_DIR="${RUN_DIR:-results/training/Pi05-goal-task0}"
YAML="${YAML:-scripts/run_rl_scripts/pi05_eval.yaml}"
MODE="${MODE:-pi05_goal_task0_eval}"

# Find latest steps_X dir by numeric suffix
LATEST=$(ls -d "${RUN_DIR}/checkpoints/steps_"* 2>/dev/null \
    | sed -E 's|.*/steps_||' | sort -n | tail -1)
[ -z "${LATEST}" ] && { echo "ERROR: no steps_X ckpt under ${RUN_DIR}/checkpoints/" >&2; exit 1; }
CKPT="${RUN_DIR}/checkpoints/steps_${LATEST}"
[ -d "${CKPT}" ] || { echo "ERROR: ${CKPT} missing" >&2; exit 1; }

echo "Latest ckpt: ${CKPT}"
# Rewrite the checkpoint line of YAML in place (sed -i, only the checkpoint:
# line — leave port/gpu_id/etc untouched).
sed -i -E "s|^(\s*checkpoint:\s*).*$|\1\"./${CKPT}\"|" "${YAML}"
grep -E "checkpoint:" "${YAML}"

# Disable proxy so localhost websocket survives (we hit this bug repeatedly).
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost"

mkdir -p logs
LOG="logs/$(basename "${YAML}" .yaml)_$(basename "${CKPT}").log"
echo "Launching eval → log: ${LOG}"
nohup bash scripts/run_base_vla/eval.sh "${MODE}" "${YAML}" > "${LOG}" 2>&1 &
echo "PID=$!"
