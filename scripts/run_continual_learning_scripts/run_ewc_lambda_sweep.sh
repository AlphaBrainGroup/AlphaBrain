#!/usr/bin/env bash
# =============================================================================
# EWC hyperparameter sweep on LIBERO-Goal.
#
# Launches one training run per hyperparameter combination using the production
# EWC YAML (qwengr00t_cl_lora_ewc_libero.yaml).  Supports an arbitrary number
# of parallel worker pairs via the `GPU_PAIRS` environment variable.
#
# Usage:
#   # Sequential on GPUs 3,4 (default λ sweep):
#   bash scripts/run_continual_learning_scripts/run_ewc_lambda_sweep.sh
#
#   # 2-pair parallel:
#   GPU_PAIRS="3,4 5,6" bash .../run_ewc_lambda_sweep.sh
#
#   # 3-pair parallel (good when 6 GPUs free):
#   GPU_PAIRS="1,2 3,4 5,6" bash .../run_ewc_lambda_sweep.sh
#
#   # Custom sweep values:
#   LAMBDAS="3e3 1e4 3e4 1e5" STEPS_PER_TASK=2000 bash .../run_ewc_lambda_sweep.sh
#
#   # Fine sweep with per-config γ (online-EWC test alongside λ):
#   LAMBDAS="1e4 1e4 1e4" GAMMAS="1.0 0.9 0.5" bash .../run_ewc_lambda_sweep.sh
#
# Environment variables:
#   LAMBDAS           Space-separated λ values.  Default: "1e3 1e4 1e5 1e6".
#   GAMMAS            Optional same-length list of γ per λ (aligned by index).
#                     Omit for default γ=1.0 (pure additive EWC).
#   STEPS_PER_TASK    Steps per task.  Default: 10000.
#   FISHER_BATCHES    Minibatches for Fisher estimation.  Default: 100.
#   GPU_PAIRS         Space-separated list of GPU pairs, e.g. "3,4 5,6".
#                     One worker per pair → parallelism = number of pairs.
#                     Default: "3,4" (single pair, sequential).
#   PARALLEL          [Deprecated, kept for back-compat.] If =1 and GPU_PAIRS
#                     is unset, falls back to the old GPUS_A + GPUS_B pair.
#   GPUS_A, GPUS_B    [Deprecated — use GPU_PAIRS instead.]
#   LOG_DIR           Per-run stdout logs. Default: /tmp/alphabrain_ewc_sweep.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

LAMBDAS="${LAMBDAS:-1e3 1e4 1e5 1e6}"
GAMMAS="${GAMMAS:-}"
STEPS_PER_TASK="${STEPS_PER_TASK:-10000}"
FISHER_BATCHES="${FISHER_BATCHES:-100}"
LOG_DIR="${LOG_DIR:-/tmp/alphabrain_ewc_sweep}"

# ---------- resolve GPU_PAIRS (new API) with back-compat shims --------------
if [[ -z "${GPU_PAIRS:-}" ]]; then
    PARALLEL="${PARALLEL:-0}"
    GPUS_A="${GPUS_A:-3,4}"
    GPUS_B="${GPUS_B:-5,6}"
    if [[ "$PARALLEL" = "1" ]]; then
        GPU_PAIRS="${GPUS_A} ${GPUS_B}"
    else
        GPU_PAIRS="${GPUS_A}"
    fi
fi

YAML="configs/continual_learning/qwengr00t_cl_lora_ewc_libero.yaml"
mkdir -p "$LOG_DIR"

# ---------- validate GAMMAS alignment (if provided) -------------------------
LAM_ARR=($LAMBDAS)
GAMMA_ARR=()
if [[ -n "$GAMMAS" ]]; then
    GAMMA_ARR=($GAMMAS)
    if [[ ${#GAMMA_ARR[@]} -ne ${#LAM_ARR[@]} ]]; then
        echo "[error] GAMMAS has ${#GAMMA_ARR[@]} entries but LAMBDAS has ${#LAM_ARR[@]}"
        exit 1
    fi
fi

launch_one() {
    # $1=lambda  $2=gamma  $3=gpus  $4=logfile
    local LAM="$1"
    local GAM="$2"
    local GPUS="$3"
    local LOGFILE="$4"

    # Normalise run_id: include gamma only if it differs from default 1.0
    local GAM_TAG=""
    [[ -n "$GAM" && "$GAM" != "1.0" && "$GAM" != "1" ]] && GAM_TAG="_gamma${GAM}"
    local RUN_ID="ewc_lambda_${LAM}${GAM_TAG}_stepsPerTask_${STEPS_PER_TASK}"

    local GAMMA_ARG=()
    [[ -n "$GAM" ]] && GAMMA_ARG=(--continual_learning.algorithm.gamma=${GAM})

    echo "  [launch] λ=${LAM} γ=${GAM:-default} gpus=${GPUS} run_id=${RUN_ID}"
    echo "           → log=${LOGFILE}"
    bash scripts/run_continual_learning_scripts/run_cl_train.sh \
        --yaml "$YAML" \
        --run-id "$RUN_ID" \
        --gpus "$GPUS" -- \
        --continual_learning.algorithm.lambda=${LAM} \
        --continual_learning.steps_per_task=${STEPS_PER_TASK} \
        --continual_learning.algorithm.fisher_num_batches=${FISHER_BATCHES} \
        "${GAMMA_ARG[@]}" \
        > "$LOGFILE" 2>&1
}

banner() {
    printf '\n%s\n' "================================================================"
    printf '%s\n' "  EWC hyperparameter sweep"
    printf '%s\n' "================================================================"
    printf '  %-16s %s\n' "Lambdas"        "${LAMBDAS}"
    [[ -n "$GAMMAS" ]] && printf '  %-16s %s\n' "Gammas" "${GAMMAS}"
    printf '  %-16s %s\n' "Steps/task"     "${STEPS_PER_TASK}"
    printf '  %-16s %s\n' "Fisher batches" "${FISHER_BATCHES}"
    printf '  %-16s %s (%d workers)\n' "GPU pairs" "${GPU_PAIRS}" "$(echo $GPU_PAIRS | wc -w)"
    printf '  %-16s %s\n' "Logs"           "${LOG_DIR}"
    printf '%s\n\n' "================================================================"
}

banner

# ---------- N-pair parallel sweep -------------------------------------------
PAIRS_ARR=($GPU_PAIRS)
NUM_WORKERS=${#PAIRS_ARR[@]}
N=${#LAM_ARR[@]}

i=0
round=1
while (( i < N )); do
    echo ">>> Round ${round}: launching up to ${NUM_WORKERS} jobs in parallel"
    PIDS=()
    for (( w=0; w < NUM_WORKERS && i < N; w++, i++ )); do
        LAM="${LAM_ARR[$i]}"
        GAM=""
        [[ ${#GAMMA_ARR[@]} -gt 0 ]] && GAM="${GAMMA_ARR[$i]}"
        GPUS="${PAIRS_ARR[$w]}"
        GAM_TAG=""
        [[ -n "$GAM" && "$GAM" != "1.0" && "$GAM" != "1" ]] && GAM_TAG="_gamma${GAM}"
        LOGFILE="${LOG_DIR}/ewc_lambda_${LAM}${GAM_TAG}.log"
        launch_one "$LAM" "$GAM" "$GPUS" "$LOGFILE" &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
    round=$((round+1))
done

echo
echo "================================================================"
echo "  EWC sweep — all runs complete"
echo "================================================================"
echo "Checkpoints : results/Checkpoints/ewc_lambda_*_stepsPerTask_${STEPS_PER_TASK}/"
echo "Logs        : ${LOG_DIR}/"
echo
echo "Next: evaluate each run via run_cl_eval.sh, compare Avg SR / NBT."
