#!/usr/bin/env bash
# =============================================================================
# MIR hyperparameter sweep on LIBERO-Goal.
#
# Default launches 4 configurations covering refresh_interval / candidate-set
# size / virtual_lr variations so we can identify which dimensions of MIR
# actually matter on our setup.
#
# Configurations (friendly_name | refresh | candidates | top_k | virtual_lr):
#
#   mir_ri200_vlrdefault   200   16   8   null   (baseline — mir_virtual_lr=null → 1e-4)
#   mir_ri50_vlrmatched     50   16   8   2.5e-5 (aggressive refresh + matched lr)
#   mir_ri200_wide         200   32  16   2.5e-5 (larger candidate pool, matched lr)
#   mir_ri500_slow         500   16   8   2.5e-5 (amortize more, matched lr)
#
# Usage:
#   # Sequential on GPUs 1,2 (single pair, default sweep):
#   bash scripts/run_continual_learning_scripts/run_mir_sweep.sh
#
#   # 3-pair parallel on GPUs 1,2 + 3,4 + 5,6:
#   GPU_PAIRS="1,2 3,4 5,6" bash .../run_mir_sweep.sh
#
#   # Smoke-sized for pipeline check (2000 → 200 steps/task):
#   STEPS_PER_TASK=200 bash .../run_mir_sweep.sh
#
#   # Cherry-pick a subset:
#   MIR_CONFIGS="mir_ri50_vlrmatched mir_ri200_wide" bash .../run_mir_sweep.sh
#
# Environment variables:
#   MIR_CONFIGS       Space-separated names to run (default: all 4 MIR-specific
#                     knob configs).  Extended replay-ratio / buffer-size
#                     configs also available — see cfg_args() below.
#   STEPS_PER_TASK    Default: 2000 (first-pass sweep; full production 10000).
#   GPU_PAIRS         Space-separated list of GPU pairs, e.g. "1,2 3,4 5,6".
#                     Parallelism = number of pairs.  Default: "1,2".
#   PARALLEL          [Deprecated.] If =1 and GPU_PAIRS unset, falls back to
#                     GPUS_A + GPUS_B.
#   GPUS_A, GPUS_B    [Deprecated — use GPU_PAIRS instead.]
#   LOG_DIR           Per-run stdout logs. Default: /tmp/alphabrain_mir_sweep.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

MIR_CONFIGS="${MIR_CONFIGS:-mir_ri200_vlrdefault mir_ri50_vlrmatched mir_ri200_wide mir_ri500_slow}"
STEPS_PER_TASK="${STEPS_PER_TASK:-2000}"
LOG_DIR="${LOG_DIR:-/tmp/alphabrain_mir_sweep}"

if [[ -z "${GPU_PAIRS:-}" ]]; then
    PARALLEL="${PARALLEL:-0}"
    GPUS_A="${GPUS_A:-1,2}"
    GPUS_B="${GPUS_B:-3,4}"
    if [[ "$PARALLEL" = "1" ]]; then
        GPU_PAIRS="${GPUS_A} ${GPUS_B}"
    else
        GPU_PAIRS="${GPUS_A}"
    fi
fi

YAML="configs/continual_learning/qwengr00t_cl_lora_mir_libero.yaml"

mkdir -p "$LOG_DIR"

# ---------- config table ----------
# Each entry → space-separated: refresh candidates topk vlr ratio buffer_size
# `null` for vlr means use MIR.DEFAULT_VIRTUAL_LR (1e-4).
# ratio and buffer_size `default` → keep the YAML default (0.3 / 500).
cfg_args() {
    case "$1" in
        # --- MIR-specific knob sweep (refresh / candidates / vlr) ---
        mir_ri200_vlrdefault) echo "200 16 8 null 0.3 500" ;;
        mir_ri50_vlrmatched)  echo "50 16 8 2.5e-5 0.3 500" ;;
        mir_ri200_wide)       echo "200 32 16 2.5e-5 0.3 500" ;;
        mir_ri500_slow)       echo "500 16 8 2.5e-5 0.3 500" ;;
        # --- replay_batch_ratio sweep (shared ER/MIR knob) ---
        mir_ratio010)         echo "200 16 8 2.5e-5 0.1 500" ;;
        mir_ratio050)         echo "200 16 8 2.5e-5 0.5 500" ;;
        mir_ratio070)         echo "200 16 8 2.5e-5 0.7 500" ;;
        # --- buffer_size sweep ---
        mir_buf200)           echo "200 16 8 2.5e-5 0.3 200" ;;
        mir_buf1000)          echo "200 16 8 2.5e-5 0.3 1000" ;;
        *) echo "UNKNOWN_CONFIG"; return 1 ;;
    esac
}

launch_one() {
    local NAME="$1"
    local GPUS="$2"
    local LOGFILE="$3"
    local PARTS; PARTS=($(cfg_args "$NAME"))
    if [[ "${PARTS[0]}" = "UNKNOWN_CONFIG" ]]; then
        echo "[error] Unknown MIR config: $NAME"; return 1
    fi
    local REFRESH="${PARTS[0]}"
    local CANDS="${PARTS[1]}"
    local TOPK="${PARTS[2]}"
    local VLR="${PARTS[3]}"
    local RATIO="${PARTS[4]}"
    local BUFSIZE="${PARTS[5]}"

    local RUN_ID="${NAME}_stepsPerTask_${STEPS_PER_TASK}"
    echo "  [launch] ${NAME}  refresh=${REFRESH}  cands=${CANDS}  topk=${TOPK}  vlr=${VLR}  ratio=${RATIO}  buf=${BUFSIZE}"
    echo "           gpus=${GPUS}  run_id=${RUN_ID}"
    echo "           → log=${LOGFILE}"

    local VLR_ARG=()
    [[ "$VLR" != "null" ]] && VLR_ARG=(--continual_learning.algorithm.mir_virtual_lr=${VLR})

    bash scripts/run_continual_learning_scripts/run_cl_train.sh \
        --yaml "$YAML" \
        --run-id "$RUN_ID" \
        --gpus "$GPUS" -- \
        --continual_learning.steps_per_task=${STEPS_PER_TASK} \
        --continual_learning.algorithm.mir_refresh_interval=${REFRESH} \
        --continual_learning.algorithm.mir_candidate_size=${CANDS} \
        --continual_learning.algorithm.mir_top_k=${TOPK} \
        --continual_learning.algorithm.replay_batch_ratio=${RATIO} \
        --continual_learning.algorithm.buffer_size_per_task=${BUFSIZE} \
        "${VLR_ARG[@]}" \
        > "$LOGFILE" 2>&1
}

banner() {
    printf '\n%s\n' "================================================================"
    printf '%s\n' "  MIR hyperparameter sweep"
    printf '%s\n' "================================================================"
    printf '  %-16s %s\n' "Configs"    "${MIR_CONFIGS}"
    printf '  %-16s %s\n' "Steps/task" "${STEPS_PER_TASK}"
    printf '  %-16s %s (%d workers)\n' "GPU pairs" "${GPU_PAIRS}" "$(echo $GPU_PAIRS | wc -w)"
    printf '  %-16s %s\n' "Logs"       "${LOG_DIR}"
    printf '%s\n\n' "================================================================"
}

banner

# ---------- N-pair parallel sweep ----------
CONFIG_ARR=($MIR_CONFIGS)
PAIRS_ARR=($GPU_PAIRS)
NUM_WORKERS=${#PAIRS_ARR[@]}
N=${#CONFIG_ARR[@]}

i=0
round=1
while (( i < N )); do
    echo ">>> Round ${round}: launching up to ${NUM_WORKERS} jobs in parallel"
    PIDS=()
    for (( w=0; w < NUM_WORKERS && i < N; w++, i++ )); do
        NAME="${CONFIG_ARR[$i]}"
        GPUS="${PAIRS_ARR[$w]}"
        LOGFILE="${LOG_DIR}/${NAME}.log"
        launch_one "$NAME" "$GPUS" "$LOGFILE" &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
    round=$((round+1))
done

echo
echo "================================================================"
echo "  MIR sweep — all runs complete"
echo "================================================================"
echo "Checkpoints : results/Checkpoints/<name>_stepsPerTask_${STEPS_PER_TASK}/"
echo "Logs        : ${LOG_DIR}/"
