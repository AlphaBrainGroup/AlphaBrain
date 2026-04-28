#!/usr/bin/env bash
# =============================================================================
# Continual Learning · Robocasa-atomic10 Matrix Evaluation
#
# Per-checkpoint × per-task SR matrix for the 10-task Robocasa-atomic10 stream
# defined in `AlphaBrain/dataloader/gr00t_lerobot/mixtures.py:170`.  Auto-detects
# LoRA checkpoints (runs adapter+action_model merge first), launches one policy
# server (AlphaBrain env, GPU `--gpus`) + one MuJoCo simulator client (separate
# `ROBOCASA365_PYTHON` env, CPU-only), aggregates per-task SR.
#
# Mirrors the LIBERO `run_cl_eval.sh` UX/flags, but the eval client is
# `benchmarks/Robocasa365/eval/simulation_env.py` and the task list is the
# 10 atomic env names in CL training order.
#
# Usage (from repo root):
#   bash scripts/run_continual_learning_scripts/run_cl_eval_robocasa.sh \
#       --run-id alphabrain_er_lora_robocasa_atomic10_v1 \
#       --base-config configs/continual_learning/qwengr00t_er_lora_robocasa_atomic10.yaml \
#       --gpus 0
#
# Prerequisites:
#   - `.env` with `ROBOCASA365_PYTHON=<robocasa env>/bin/python` set
#     (the simulator import imports `robocasa` + `robosuite` + `mujoco`).
#   - `.env` with `ROBOCASA365_DATA_DIR` set if your training used it.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

# ---------- 10-task CL stream (matches mixtures.py:170 + train order) ----------
# These are Robocasa env names (i.e. the strings simulation_env.py expects).
TASK_LIST=(
    "NavigateKitchen"
    "OpenDrawer"
    "OpenCabinet"
    "CloseFridge"
    "CloseBlenderLid"
    "CoffeeSetupMug"
    "PickPlaceCounterToCabinet"
    "PickPlaceSinkToCounter"
    "TurnOnMicrowave"
    "TurnOffStove"
)
TASK_LIST_CSV="$(IFS=, ; echo "${TASK_LIST[*]}")"

# ---------- defaults ----------
RUN_ID=""
BASE_CONFIG=""
GPUS="0"
N_EPISODES=20
N_ENVS=1
N_ACTION_STEPS=16
PORT_BASE=5680
OUTPUT_BASE=""
LAST_ONLY=0
EXTRA=()

usage() {
    cat <<EOF
Usage: bash $0 --run-id <RUN_ID> [options]

Required:
  --run-id ID           Run directory name under results/Checkpoints/

Common:
  --base-config PATH    LoRA merge base config (required when ckpts contain *_lora_adapter)
                        e.g. configs/continual_learning/qwengr00t_er_lora_robocasa_atomic10.yaml
                             configs/continual_learning/qwengr00t_mir_lora_robocasa_atomic10.yaml
  --gpus N              Single CUDA device for the policy server (default 0)
  --n-episodes N        Rollouts per task (default 20)
  --n-envs N            Vectorized envs per task (default 1; >1 needs more sim memory)
  --n-action-steps N    Action chunk size returned by the policy server (default 16)
  --port-base N         Starting port (default 5680)
  --output-base PATH    Eval results root (default results/eval_cl/<RUN_ID>_robocasa)
  --last-only           Only evaluate the final task_* checkpoint
  --                    Pass-through args (appended to server_policy.py)
  -h, --help            Show this help

Notes:
  - The eval iterates the 10 atomic tasks in CL training order — same as
    \`AlphaBrain/dataloader/gr00t_lerobot/mixtures.py:170\`.  Eval pos N is
    the (N-1)-th trained task; eval pos 1 = NavigateKitchen (first trained),
    eval pos 10 = TurnOffStove (last trained).
  - Per-task results land in \`<output-base>/<ckpt_name>/<env_name>/stats.json\`,
    aggregate stats in \`<output-base>/<ckpt_name>/aggregate_stats.json\`.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)        RUN_ID="$2"; shift 2 ;;
        --base-config)   BASE_CONFIG="$2"; shift 2 ;;
        --gpus)          GPUS="$2"; shift 2 ;;
        --n-episodes)    N_EPISODES="$2"; shift 2 ;;
        --n-envs)        N_ENVS="$2"; shift 2 ;;
        --n-action-steps) N_ACTION_STEPS="$2"; shift 2 ;;
        --port-base)     PORT_BASE="$2"; shift 2 ;;
        --output-base)   OUTPUT_BASE="$2"; shift 2 ;;
        --last-only)     LAST_ONLY=1; shift ;;
        --)              shift; EXTRA=("$@"); break ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "[error] unknown arg: $1"; usage; exit 1 ;;
    esac
done

if [ -z "$RUN_ID" ]; then echo "[error] --run-id required"; usage; exit 1; fi

CKPT_SUB="results/Checkpoints/$RUN_ID/checkpoints"
[ -z "$OUTPUT_BASE" ] && OUTPUT_BASE="results/eval_cl/${RUN_ID}_robocasa"
mkdir -p "$OUTPUT_BASE"

if [ ! -d "$CKPT_SUB" ]; then
    echo "[error] $CKPT_SUB does not exist"; exit 1
fi

# ---------- load .env ----------
if [ -f "$REPO_ROOT/.env" ]; then
    set -a; # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"; set +a
fi

# Server (and merge) need to import the in-tree `AlphaBrain` package and the
# `benchmarks/` namespace.  Mirror run_cl_eval.sh (line 213).
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SERVER_PYTHON="${ALPHABRAIN_PYTHON:-python}"
ROBOCASA_PYTHON="${ROBOCASA365_PYTHON:-}"

if [ -z "$ROBOCASA_PYTHON" ] || [ ! -x "$ROBOCASA_PYTHON" ]; then
    echo "[error] ROBOCASA365_PYTHON unset or not executable in .env"
    echo "        Set ROBOCASA365_PYTHON=<robocasa conda env>/bin/python in .env"
    echo "        before running this script.  See benchmarks/Robocasa365/README.md §1."
    exit 1
fi
if ! "$ROBOCASA_PYTHON" -c "import robocasa, robosuite" 2>/dev/null; then
    echo "[error] ROBOCASA365_PYTHON cannot import robocasa + robosuite"
    echo "        Re-install the robocasa365 conda env per benchmarks/Robocasa365/README.md."
    exit 1
fi

# ---------- collect ckpts ----------
mapfile -t LORA_CKPTS < <(ls -d "$CKPT_SUB"/task_*_lora_adapter 2>/dev/null | sort)
mapfile -t FULL_CKPTS < <(ls -d "$CKPT_SUB"/task_*_pytorch_model.pt 2>/dev/null | sort)

LORA_MODE=0
if [ "${#LORA_CKPTS[@]}" -gt 0 ]; then
    LORA_MODE=1
    if [ -z "$BASE_CONFIG" ]; then
        echo "[error] LoRA ckpts found but --base-config missing"
        echo "        Pass --base-config configs/continual_learning/qwengr00t_(er|mir)_lora_robocasa_atomic10.yaml"
        exit 1
    fi
    CKPT_NAMES=()
    for adapter in "${LORA_CKPTS[@]}"; do
        CKPT_NAMES+=("$(basename "${adapter%_lora_adapter}")")
    done
else
    CKPT_NAMES=()
    for ckpt in "${FULL_CKPTS[@]}"; do
        CKPT_NAMES+=("$(basename "${ckpt%_pytorch_model.pt}")")
    done
fi

if [ "${#CKPT_NAMES[@]}" -eq 0 ]; then
    echo "[error] no checkpoints found under $CKPT_SUB"; exit 1
fi
[ "$LAST_ONLY" -eq 1 ] && CKPT_NAMES=("${CKPT_NAMES[-1]}")

cat <<EOF

────────────────────────────────────────────────────────────────────────────────
  ▶  Continual Learning · Robocasa-atomic10 Matrix Evaluation
────────────────────────────────────────────────────────────────────────────────
  RunID         │  $RUN_ID
  LoRA          │  $([ "$LORA_MODE" -eq 1 ] && echo "true (will merge first)" || echo "false (full-param)")
  Checkpoints   │  ${#CKPT_NAMES[@]}  ($([ "$LAST_ONLY" -eq 1 ] && echo "last-only" || echo "all"))
  GPU           │  $GPUS  (port $PORT_BASE)
  Tasks         │  ${#TASK_LIST[@]}  (atomic10 stream)
  Episodes/task │  $N_EPISODES
  Server py     │  $SERVER_PYTHON
  Sim    py     │  $ROBOCASA_PYTHON
  Output        │  $OUTPUT_BASE
  Start         │  $(date)
────────────────────────────────────────────────────────────────────────────────

EOF

# ---------- LoRA merge step (if needed) ----------
if [ "$LORA_MODE" -eq 1 ]; then
    echo ""
    echo "[Step 1] Merge LoRA checkpoints"
    for name in "${CKPT_NAMES[@]}"; do
        adapter="$CKPT_SUB/${name}_lora_adapter"
        action_model="$CKPT_SUB/${name}_action_model.pt"
        merged="$CKPT_SUB/${name}_merged.pt"
        if [ -f "$merged" ]; then
            printf "[%s] SKIP merge (exists): %s\n" "$(date '+%H:%M:%S')" "$name"
            continue
        fi
        printf "[%s] merging: %s\n" "$(date '+%H:%M:%S')" "$name"
        CUDA_VISIBLE_DEVICES="$GPUS" "$SERVER_PYTHON" -m \
            AlphaBrain.training.trainer_utils.peft.merge_lora_checkpoint \
            --base_config "$BASE_CONFIG" \
            --lora_adapter_dir "$adapter" \
            --action_model_pt  "$action_model" \
            --output_path      "$merged"
    done
    echo "[$(date '+%H:%M:%S')] ✓ All merges complete."
    echo ""
fi

# ---------- per-checkpoint loop ----------
echo "[Step 2] Matrix evaluation (${#CKPT_NAMES[@]} checkpoints × ${#TASK_LIST[@]} tasks)"

run_one_ckpt() {
    local name="$1"
    local port="$2"
    local out_dir="$OUTPUT_BASE/$name"
    mkdir -p "$out_dir"

    local ckpt
    if [ "$LORA_MODE" -eq 1 ]; then
        ckpt="$CKPT_SUB/${name}_merged.pt"
    else
        ckpt="$CKPT_SUB/${name}_pytorch_model.pt"
    fi

    local ts; ts="[$(date '+%H:%M:%S')]"
    printf "%s [1/3] launching server: %s (port=%s)\n" "$ts" "$name" "$port"
    CUDA_VISIBLE_DEVICES="$GPUS" "$SERVER_PYTHON" \
        "$REPO_ROOT/deployment/model_server/server_policy.py" \
        --ckpt_path "$ckpt" --port "$port" --use_bf16 "${EXTRA[@]}" \
        > "$out_dir/server.log" 2>&1 &
    local server_pid=$!

    local waited=0
    while ! "$SERVER_PYTHON" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $port)); s.close()" 2>/dev/null; do
        if ! kill -0 $server_pid 2>/dev/null; then
            printf "[%s] [error] server died: %s\n" "$(date '+%H:%M:%S')" "$name"
            tail -30 "$out_dir/server.log"; return 1
        fi
        if [ $waited -ge 900 ]; then
            printf "[%s] [error] server timeout (900s): %s\n" "$(date '+%H:%M:%S')" "$name"
            kill $server_pid 2>/dev/null; wait $server_pid 2>/dev/null || true; return 1
        fi
        sleep 10; waited=$((waited+10))
        if [ $((waited % 30)) -eq 0 ]; then
            printf "[%s] ⋯ still loading server (%ss elapsed): %s\n" "$(date '+%H:%M:%S')" "$waited" "$name"
        fi
    done
    printf "[%s] ✓ server ready in %ss: %s\n" "$(date '+%H:%M:%S')" "$waited" "$name"

    printf "[%s] [2/3] running client (%d episodes × %d tasks): %s\n" \
        "$(date '+%H:%M:%S')" "$N_EPISODES" "${#TASK_LIST[@]}" "$name"

    "$ROBOCASA_PYTHON" "$REPO_ROOT/benchmarks/Robocasa365/eval/simulation_env.py" \
        --args.pretrained-path "$ckpt" \
        --args.host 127.0.0.1 --args.port "$port" \
        --args.task-list "$TASK_LIST_CSV" \
        --args.no-sort-tasks \
        --args.n-episodes "$N_EPISODES" \
        --args.n-envs "$N_ENVS" \
        --args.n-action-steps "$N_ACTION_STEPS" \
        --args.video-out-path "$out_dir" \
        --args.split "${ROBOCASA_EVAL_SPLIT:-pretrain}" \
        2>&1 \
        | tee "$out_dir/eval.log" \
        | grep --line-buffered -E "Running simulation|Results for|EP [0-9]+ success|Saved aggregate|cumulative=" \
        || true

    printf "[%s] [3/3] killing server: %s\n" "$(date '+%H:%M:%S')" "$name"
    kill $server_pid 2>/dev/null; wait $server_pid 2>/dev/null || true

    # Pull aggregate stats (split-prefixed by simulation_env.py at line 240)
    local split="${ROBOCASA_EVAL_SPLIT:-pretrain}"
    local agg="$out_dir/$split/aggregate_stats.json"
    if [ -f "$agg" ]; then
        local mean
        mean=$("$SERVER_PYTHON" -c "import json; d=json.load(open('$agg')); print(f\"{d['mean_success_rate']*100:.1f}\")" 2>/dev/null || echo "?")
        printf "[%s] ✓ done: %s  →  Avg SR: %s%%\n" "$(date '+%H:%M:%S')" "$name" "$mean"
    else
        printf "[%s] [warn] no aggregate_stats.json at %s\n" "$(date '+%H:%M:%S')" "$agg"
    fi
}

port="$PORT_BASE"
for name in "${CKPT_NAMES[@]}"; do
    run_one_ckpt "$name" "$port"
    port=$((port+1))
done

cat <<EOF

────────────────────────────────────────────────────────────────────────────────
  ✓  All evaluations complete  $(date)
────────────────────────────────────────────────────────────────────────────────
  Results       │  $OUTPUT_BASE/
────────────────────────────────────────────────────────────────────────────────
EOF
