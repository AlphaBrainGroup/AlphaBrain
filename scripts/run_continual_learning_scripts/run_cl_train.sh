#!/usr/bin/env bash
# =============================================================================
# Continual Learning Training — one-command, self-contained.
#
# Three-axis interface: pick model × algo × dataset; shared config is handled
# automatically by composing cl_base.yaml with the model overlay.
#
# Usage (from repo root):
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh
#       # default: QwenGR00T + ER + LIBERO-Goal
#
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh --smoke
#       # 5 steps × 10 tasks pipeline check
#
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh \
#       --model qwengr00t --algo mir --dataset libero_goal --gpus 0,1,2,3
#       # MIR 77 % recipe on LIBERO-Goal
#
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh \
#       --model qwengr00t --algo er --dataset libero_long
#       # ER on LIBERO-Long
#
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh \
#       --model qwengr00t --algo er -- --lora.enabled=false
#       # full-parameter ER (disable LoRA via passthrough override)
#
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh \
#       --yaml configs/continual_learning/cl_base.yaml \
#              configs/continual_learning/models/qwengr00t.yaml
#       # legacy / advanced: explicit yaml composition
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

# ---------- defaults ----------
# Three-axis defaults (used when --yaml is not given)
MODEL="qwengr00t"   # qwengr00t | neurovla | llamaoft | paligemma
ALGO="er"           # er | mir
DATASET="libero_goal"  # libero_goal | libero_long

# Legacy / advanced: explicit yaml path(s) — skip three-axis composition
YAML_PATHS=()  # populated by --yaml; each --yaml value is one path

GPUS=""
RUN_ID=""
# Default port: Python socket-bind to get a guaranteed-free port from the OS.
# We tried passing 0 to accelerate (per its docs), but it hangs in NCCL init
# under DeepSpeed plugin (futex_wait_queue_me, no I/O, no CUDA progress).
# This Python one-liner is reliable: bind on ephemeral port, read assigned
# port, close (brief race window but >99% safe in practice).
PORT=""
SMOKE=0
EXTRA=()

# ---------- parse CLI ----------
usage() {
    cat <<EOF
Usage: bash $0 [options] [-- OmegaConf overrides]

Three-axis mode (default):
  --model MODEL      VLA backbone. Default: ${MODEL}
                     Choices: qwengr00t | neurovla | llamaoft | paligemma
  --algo  ALGO       CL algorithm. Default: ${ALGO}
                     Choices: er | mir
                       er  — Experience Replay (buf=1000, ratio=0.5, balanced)
                       mir — MIR refresh=50 frozen recipe (77 % on LIBERO-Goal)
  --dataset DS       Task stream. Default: ${DATASET}
                     Choices: libero_goal | libero_long

Advanced / legacy:
  --yaml PATH ...    One or more explicit config yaml paths merged left-to-right.
                     When given, --model/--algo/--dataset are ignored.
                     Single full config:
                       --yaml configs/continual_learning/cl_base.yaml
                              configs/continual_learning/models/qwengr00t.yaml
                     Custom config:
                       --yaml /path/to/my_full_config.yaml

Common:
  --run-id ID        Override auto-generated run_id (checkpoint dir name).
  --gpus SPEC        Count ("2") or id-list ("1,2,3"). List pins CUDA_VISIBLE_DEVICES.
                     Default: auto-detect all visible GPUs.
  --port N           accelerate main_process_port. Default: auto-pick free port.
  --smoke            5 steps/task × batch 4 — pipeline check, not convergence.
  --                 Pass-through OmegaConf overrides for train.py.
  -h, --help         Show this help.

Examples:
  # default: QwenGR00T LoRA + ER on LIBERO-Goal (~15 h on 2× A800)
  bash $0

  # pipeline smoke test
  bash $0 --smoke

  # MIR 77 % LIBERO-Goal recipe
  bash $0 --model qwengr00t --algo mir --dataset libero_goal --gpus 0,1,2,3

  # MIR on LIBERO-Long
  bash $0 --model qwengr00t --algo mir --dataset libero_long --gpus 0,1,2,3

  # ER on LIBERO-Long
  bash $0 --model qwengr00t --algo er --dataset libero_long

  # NeuroVLA + ER
  bash $0 --model neurovla --algo er --dataset libero_goal

  # QwenGR00T full-param ER (no LoRA)
  bash $0 --model qwengr00t --algo er -- --lora.enabled=false

  # custom OmegaConf override (e.g. bump LoRA rank)
  bash $0 --model qwengr00t --algo er -- --lora.rank=64
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2"; shift 2 ;;
        --algo)     ALGO="$2"; shift 2 ;;
        --dataset)  DATASET="$2"; shift 2 ;;
        --yaml)     YAML_PATHS+=("$2"); shift 2 ;;
        --gpus)     GPUS="$2"; shift 2 ;;
        --run-id)   RUN_ID="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --smoke)    SMOKE=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        --)         shift; EXTRA=("$@"); break ;;
        *)          echo "[error] Unknown arg: $1"; usage; exit 1 ;;
    esac
done

# ---------- validate three-axis values ----------
_VALID_MODELS="qwengr00t neurovla llamaoft paligemma"
_VALID_ALGOS="er mir"
_VALID_DATASETS="libero_goal libero_long"

if [ "${#YAML_PATHS[@]}" -eq 0 ]; then
    # Three-axis validation
    if ! echo "$_VALID_MODELS" | grep -qw "$MODEL"; then
        echo "[error] --model must be one of: $_VALID_MODELS (got: $MODEL)"; exit 1
    fi
    if ! echo "$_VALID_ALGOS" | grep -qw "$ALGO"; then
        echo "[error] --algo must be one of: $_VALID_ALGOS (got: $ALGO)"; exit 1
    fi
    if ! echo "$_VALID_DATASETS" | grep -qw "$DATASET"; then
        echo "[error] --dataset must be one of: $_VALID_DATASETS (got: $DATASET)"; exit 1
    fi
fi

# ---------- resolve port ----------
if [ -z "$PORT" ]; then
    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); p=s.getsockname()[1]; s.close(); print(p)" 2>/dev/null)
    if [ -z "$PORT" ]; then
        PORT=$((29500 + RANDOM % 1000))
    fi
fi

# ---------- load .env ----------
if [ -f "$REPO_ROOT/.env" ]; then
    set -a; # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"; set +a
fi

: "${PRETRAINED_MODELS_DIR:?need PRETRAINED_MODELS_DIR in .env}"
: "${LEROBOT_LIBERO_DATA_DIR:?need LEROBOT_LIBERO_DATA_DIR in .env}"

# ---------- build config yaml list + overrides ----------
BASE_CL="$REPO_ROOT/configs/continual_learning/cl_base.yaml"
ALGO_OVERRIDES=()
DATASET_OVERRIDES=()

if [ "${#YAML_PATHS[@]}" -gt 0 ]; then
    # Legacy / advanced mode: user supplies explicit yaml path(s)
    CONFIG_YAML_ARGS=()
    for p in "${YAML_PATHS[@]}"; do
        if [[ "$p" = /* ]]; then
            CONFIG_YAML_ARGS+=("$p")
        else
            CONFIG_YAML_ARGS+=("$REPO_ROOT/$p")
        fi
    done
    for f in "${CONFIG_YAML_ARGS[@]}"; do
        [ -f "$f" ] || { echo "[error] Config not found: $f"; exit 1; }
    done
    # Banner labels from first yaml (best-effort probe)
    _PROBE_CONFIG="${CONFIG_YAML_ARGS[0]}"
    DISPLAY_MODEL=$(python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$_PROBE_CONFIG')
for f in $(printf "'%s' " "${CONFIG_YAML_ARGS[@]:1}" | sed 's/ *$//'):
    cfg = OmegaConf.merge(cfg, OmegaConf.load(f))
print(cfg.get('framework', {}).get('name', '<unknown>'))
" 2>/dev/null || echo '<unknown>')
    DISPLAY_ALGO=$(python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$_PROBE_CONFIG')
cl = cfg.get('continual_learning', {})
r = cl.get('replay', None)
if r and r.get('enabled', False):
    print({'experience_replay':'ER'}.get(r.get('method',''), r.get('method','ER')))
else:
    a = cl.get('algorithm', None)
    print(str(a.get('name', 'none')).upper() if a else 'none')
" 2>/dev/null || echo '<unknown>')
    DISPLAY_DATASET="<from yaml>"
    [ -z "$RUN_ID" ] && RUN_ID_DISPLAY="<from yaml>" || RUN_ID_DISPLAY="$RUN_ID"
else
    # Three-axis mode: compose cl_base.yaml + model overlay + overrides
    [ -f "$BASE_CL" ] || { echo "[error] Base config not found: $BASE_CL"; exit 1; }
    MODEL_YAML="$REPO_ROOT/configs/continual_learning/models/${MODEL}.yaml"
    [ -f "$MODEL_YAML" ] || { echo "[error] Model config not found: $MODEL_YAML"; exit 1; }
    CONFIG_YAML_ARGS=("$BASE_CL" "$MODEL_YAML")

    # Dataset overrides (libero_goal is the base default — no overrides needed)
    if [ "$DATASET" = "libero_long" ]; then
        DATASET_OVERRIDES=(
            --datasets.vla_data.dataset_mix=libero_long
            --continual_learning.task_sequence=libero_long
        )
    fi

    # Algo overrides — ER is the base default (replay block already enabled)
    if [ "$ALGO" = "mir" ]; then
        # Frozen refresh=50 recipe: disable replay block, activate algorithm block
        ALGO_OVERRIDES=(
            --continual_learning.replay.enabled=false
            --continual_learning.algorithm.name=mir
            --continual_learning.algorithm.buffer_size_per_task=1000
            --continual_learning.algorithm.replay_batch_ratio=0.5
            --continual_learning.algorithm.balanced_sampling=true
            --continual_learning.algorithm.mir_refresh_interval=50
            --continual_learning.algorithm.mir_candidate_size=16
            --continual_learning.algorithm.mir_top_k=8
            --continual_learning.algorithm.mir_lora_only=true
            --datasets.vla_data.per_device_batch_size=4
        )
    fi

    # Auto run_id: {model}_{algo}_{dataset} (overridable via --run-id)
    if [ -z "$RUN_ID" ]; then
        RUN_ID="${MODEL}_${ALGO}_${DATASET}"
    fi
    DISPLAY_MODEL="$MODEL"
    DISPLAY_ALGO=$(echo "$ALGO" | tr '[:lower:]' '[:upper:]')
    DISPLAY_DATASET="$DATASET"
    RUN_ID_DISPLAY="$RUN_ID"
fi

# ---------- resolve GPUs ----------
# Accept either a count ("--gpus 2") or a comma-separated ID list ("--gpus 1,2").
# For a list we pin CUDA_VISIBLE_DEVICES and derive the count for accelerate.
if [ -z "$GPUS" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NUM_PROCESSES=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    elif command -v nvidia-smi >/dev/null 2>&1; then
        # `head -n1` closes the pipe early → nvidia-smi dies with SIGPIPE → under
        # `set -o pipefail` the whole assignment returns 141 and `set -e` kills
        # the script silently. Count via `-L` + `wc -l` instead; wc reads all.
        NUM_PROCESSES=$(nvidia-smi -L 2>/dev/null | wc -l)
        NUM_PROCESSES="${NUM_PROCESSES:-2}"
    else
        NUM_PROCESSES=2
    fi
    GPUS="$NUM_PROCESSES"    # display-only
elif [[ "$GPUS" == *,* ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    NUM_PROCESSES=$(echo "$GPUS" | awk -F',' '{print NF}')
else
    NUM_PROCESSES="$GPUS"
fi

# ---------- NCCL defaults ----------
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

# ---------- assemble python args ----------
TRAIN_ENTRY="$REPO_ROOT/AlphaBrain/training/continual_learning/train.py"
DS_CONFIG="$REPO_ROOT/configs/deepspeed/accelerate_zero2.yaml"
[ -f "$TRAIN_ENTRY" ] || { echo "[error] train entry missing: $TRAIN_ENTRY"; exit 1; }
[ -f "$DS_CONFIG" ]   || { echo "[error] deepspeed config missing: $DS_CONFIG"; exit 1; }

PY_ARGS=(--config_yaml "${CONFIG_YAML_ARGS[@]}")
[ -n "$RUN_ID" ] && PY_ARGS+=(--run_id "$RUN_ID")

# System overrides (algo + dataset) — applied before smoke and user extras so
# that smoke can override batch size / steps regardless of algo.
PY_ARGS+=("${ALGO_OVERRIDES[@]}" "${DATASET_OVERRIDES[@]}")

if [ "$SMOKE" = "1" ]; then
    # Override both replay and algorithm buffer keys so smoke works with any algo
    PY_ARGS+=(
        --continual_learning.steps_per_task=5
        --continual_learning.replay.buffer_size_per_task=10
        --continual_learning.algorithm.buffer_size_per_task=10
        --datasets.vla_data.per_device_batch_size=4
    )
fi

# User pass-through extras (after -- ; always appended last so they win)
PY_ARGS+=("${EXTRA[@]}")

# ---------- launch ----------
if [ -t 1 ]; then
    C0=$'\033[0m'
    CC=$'\033[38;5;51m'   # bright cyan  (rule)
    CT=$'\033[1;38;5;45m' # bold sky     (title)
    CK=$'\033[38;5;244m'  # gray         (label)
    CH=$'\033[1;38;5;214m'  # bold orange  (highlight)
    CG=$'\033[38;5;120m'  # green        (GPUs)
    CM=$'\033[38;5;213m'  # magenta      (model)
    CD=$'\033[2m'         # dim
    CS=$'\033[1;38;5;196m'; CSE=$'\033[0m'  # smoke warning
else
    C0=""; CC=""; CT=""; CK=""; CH=""; CG=""; CM=""; CD=""; CS=""; CSE=""
fi
_rule() { printf "${CC}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C0}\n"; }
_kv()   { printf "  ${CK}%-10s${C0} ${CD}│${C0}  %s\n" "$1" "$2"; }

echo
_rule
printf "  ${CT}▶  Continual Learning Training${C0}\n"
_rule
_kv "Model"      "${CM}${DISPLAY_MODEL}${C0}"
_kv "CL Algo"    "${CH}${DISPLAY_ALGO}${C0}"
_kv "Dataset"    "${CH}${DISPLAY_DATASET}${C0}"
if [[ "$GPUS" == *,* ]]; then
    _kv "GPUs"   "${CG}${GPUS}${C0}  ${CD}(${NUM_PROCESSES} procs, port ${PORT})${C0}"
else
    _kv "GPUs"   "${CG}${GPUS}${C0}  ${CD}(port ${PORT})${C0}"
fi
_kv "RunID"      "${CH}${RUN_ID_DISPLAY}${C0}"
[ "$SMOKE" = "1" ] && _kv "Smoke" "${CS}5 steps × 10 tasks × batch 4${CSE}"
_rule
echo

exec accelerate launch --config_file "$DS_CONFIG" \
    --num_processes "$NUM_PROCESSES" --main_process_port "$PORT" \
    "$TRAIN_ENTRY" "${PY_ARGS[@]}"
