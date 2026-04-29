#!/usr/bin/env bash
# =============================================================================
# Continual Learning Training — one-command, self-contained.
#
# Takes a yaml config path directly via --yaml.  Default: QwenGR00T LoRA +
# ER on LIBERO-Goal.  One yaml → one run; switch CL algorithm / backbone /
# benchmark by pointing --yaml at a different preset.
#
# Usage (from repo root):
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh                    # default (QwenGR00T LoRA + ER)
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh --smoke            # 5 step × 10 task smoke
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh \
#       --yaml configs/continual_learning/qwengr00t_mir_lora_libero_refresh50.yaml # MIR 77% recipe
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh --run-id exp_v2 --gpus 4
#   bash scripts/run_continual_learning_scripts/run_cl_train.sh -- \
#       --continual_learning.steps_per_task=2000                                    # pass-through OmegaConf override
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

# ---------- defaults ----------
YAML="configs/continual_learning/qwengr00t_er_lora_libero.yaml"  # QwenGR00T LoRA + ER on LIBERO-Goal
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

Common:
  --yaml PATH        CL config yaml.  Default: ${YAML}
                     Available in configs/continual_learning/ (see scripts/run_continual_learning_scripts/README.md
                     for the full table).  Highlights:
                       LIBERO-Goal:
                         qwengr00t_er_lora_libero.yaml             (LoRA + ER, default)
                         qwengr00t_mir_lora_libero.yaml            (LoRA + MIR, default knobs)
                         qwengr00t_mir_lora_libero_refresh50.yaml  (LoRA + MIR — 77% recipe)
                         qwengr00t_er_libero.yaml                  (Full-param + ER)
                         qwengr00t_ewc_lora_libero.yaml            (LoRA + EWC)
                       LIBERO-Long (libero_10):
                         qwengr00t_er_lora_libero_long.yaml         (LoRA + ER)
                         qwengr00t_mir_lora_libero_long_refresh50.yaml  (LoRA + MIR — 77% recipe ported to long)
                       Other backbones (LIBERO-Goal):
                         neurovla_er_lora_libero.yaml              (NeuroVLA  + LoRA + ER)
                         llamaoft_er_lora_libero.yaml              (LlamaOFT  + LoRA + ER)
                         paligemma_oft_er_libero.yaml              (PaliGemmaOFT Full-param + ER)
                       Smoke configs (pipeline check, not convergence):
                         qwengr00t_er_lora_test.yaml / qwengr00t_mir_lora_test.yaml / qwengr00t_ewc_lora_test.yaml
  --run-id ID        Override run_id in yaml (checkpoint dir name)
  --gpus SPEC        Either a count ("2") or a GPU-id list ("1,2,3"). A list
                     pins CUDA_VISIBLE_DEVICES to those IDs. (default: auto-detect)
  --port N           accelerate main_process_port (default: auto-pick free port)
  --smoke            Run smoke test (5 steps/task, buffer=10, batch=4). ~3 min on 2× A800.
  --                 Pass-through OmegaConf overrides for train.py (e.g. --lora.rank=16)
  -h, --help         Show this help

Examples:
  # default training (~15 h on 2× A800)
  bash $0

  # smoke test to verify pipeline
  bash $0 --smoke

  # MIR 77% LIBERO-Goal recipe
  bash $0 --yaml configs/continual_learning/qwengr00t_mir_lora_libero_refresh50.yaml --gpus 0,1,2,3

  # MIR on LIBERO-Long (same recipe ported to libero_long stream)
  bash $0 --yaml configs/continual_learning/qwengr00t_mir_lora_libero_long_refresh50.yaml --gpus 0,1,2,3

  # custom override passed to OmegaConf
  bash $0 --yaml configs/continual_learning/qwengr00t_er_lora_libero.yaml -- \\
      --lora.rank=16 --trainer.max_train_steps=50000
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yaml)     YAML="$2"; shift 2 ;;
        --gpus)     GPUS="$2"; shift 2 ;;
        --run-id)   RUN_ID="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --smoke)    SMOKE=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        --)         shift; EXTRA=("$@"); break ;;
        *)          echo "[error] Unknown arg: $1"; usage; exit 1 ;;
    esac
done

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

# ---------- resolve yaml path (relative or absolute) ----------
if [[ "$YAML" = /* ]]; then
    CONFIG="$YAML"
else
    CONFIG="$REPO_ROOT/$YAML"
fi
[ -f "$CONFIG" ] || { echo "[error] Config not found: $CONFIG"; exit 1; }

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

# ---------- probe framework + base VLM + CL method (best-effort) ----------
FRAMEWORK=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').framework.name)" 2>/dev/null || echo '<unknown>')
BASE_VLM_PATH=$(python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CONFIG').framework
for k in ('qwenvl', 'llamavl', 'paligemma'):
    if k in cfg and 'base_vlm' in cfg[k]:
        print(cfg[k].base_vlm); break
" 2>/dev/null || echo '<unknown>')
BASE_VLM_NAME=$(basename "$BASE_VLM_PATH" 2>/dev/null || echo "$BASE_VLM_PATH")

# Detect active CL method: mirrors build_cl_algorithm's dispatch order
#   replay.enabled=True               → use replay.method (ER / ...)
#   algorithm.name=<name>             → use that name (EWC / DER / ...)
#   neither                           → "none" (plain sequential baseline)
CL_METHOD=$(python -c "
from omegaconf import OmegaConf
cl = OmegaConf.load('$CONFIG').get('continual_learning', None) or {}
replay = cl.get('replay', None)
if replay is not None and replay.get('enabled', False):
    m = replay.get('method', 'experience_replay')
    print({'experience_replay': 'ER'}.get(m, m.upper()))
else:
    algo = cl.get('algorithm', None)
    name = algo.get('name', None) if algo is not None else None
    print(str(name).upper() if name else 'none')
" 2>/dev/null || echo '<unknown>')

# ---------- assemble python args ----------
PY_ARGS=(--config_yaml "$CONFIG")
[ -n "$RUN_ID" ] && PY_ARGS+=(--run_id "$RUN_ID")
if [ "$SMOKE" = "1" ]; then
    PY_ARGS+=(--continual_learning.steps_per_task=5
              --continual_learning.replay.buffer_size_per_task=10
              --datasets.vla_data.per_device_batch_size=4)
elif [ "${#EXTRA[@]}" -gt 0 ]; then
    PY_ARGS+=("${EXTRA[@]}")
fi

TRAIN_ENTRY="$REPO_ROOT/AlphaBrain/training/continual_learning/train.py"
DS_CONFIG="$REPO_ROOT/configs/deepspeed/accelerate_zero2.yaml"
[ -f "$TRAIN_ENTRY" ] || { echo "[error] train entry missing: $TRAIN_ENTRY"; exit 1; }
[ -f "$DS_CONFIG" ]   || { echo "[error] deepspeed config missing: $DS_CONFIG"; exit 1; }

# ---------- launch ----------
if [ -t 1 ]; then
    C0=$'\033[0m'        # reset
    CB=$'\033[1m'        # bold
    CD=$'\033[2m'        # dim
    CC=$'\033[38;5;51m'  # bright cyan  (rule)
    CT=$'\033[1;38;5;45m'  # bold sky     (title)
    CK=$'\033[38;5;244m' # gray         (label)
    CV=$'\033[38;5;228m' # light yellow (value)
    CH=$'\033[1;38;5;214m' # bold orange (highlight — mode / runid)
    CG=$'\033[38;5;120m' # green        (GPUs)
    CM=$'\033[38;5;213m' # magenta      (framework / model)
    CS=$'\033[1;38;5;196m'; CSE=$'\033[0m'  # smoke warning
else
    C0=""; CB=""; CD=""; CC=""; CT=""; CK=""; CV=""; CH=""; CG=""; CM=""; CS=""; CSE=""
fi
_rule() { printf "${CC}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C0}\n"; }
_kv()   { printf "  ${CK}%-10s${C0} ${CD}│${C0}  %s\n" "$1" "$2"; }

echo
_rule
printf "  ${CT}▶  Continual Learning Training${C0}\n"
_rule
_kv "Config"     "${CH}${CONFIG#$REPO_ROOT/}${C0}"
_kv "Framework"  "${CM}${FRAMEWORK}${C0}"
_kv "Base VLM"   "${CM}${BASE_VLM_NAME}${C0}"
_kv "CL Method"  "${CH}${CL_METHOD}${C0}"
if [[ "$GPUS" == *,* ]]; then
    _kv "GPUs"   "${CG}${GPUS}${C0}  ${CD}(${NUM_PROCESSES} procs, port ${PORT})${C0}"
else
    _kv "GPUs"   "${CG}${GPUS}${C0}  ${CD}(port ${PORT})${C0}"
fi
_kv "RunID"      "${CH}${RUN_ID:-<from yaml>}${C0}"
[ "$SMOKE" = "1" ] && _kv "Smoke" "${CS}5 steps × 10 tasks × batch 4${CSE}"
_rule
echo

exec accelerate launch --config_file "$DS_CONFIG" \
    --num_processes "$NUM_PROCESSES" --main_process_port "$PORT" \
    "$TRAIN_ENTRY" "${PY_ARGS[@]}"
