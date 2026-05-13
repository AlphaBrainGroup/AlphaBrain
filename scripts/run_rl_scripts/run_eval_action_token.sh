#!/bin/bash
# Eval an RLT_a run: 10 libero_goal tasks split across 3 GPUs, then
# aggregated into <RUN_DIR>/eval_<ITER>/summary.json.
# Edit VLA_CKPT / RUN_DIR / ITER / GPU_IDS below, or override via argv.
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"  # e.g. /path/to/AlphaBrain

[ -f .env ] && { set -a; source .env; set +a; }

# AlphaBrain isn't pip-installed in the vla env; when launching a script by path
# (not `-m`), sys.path[0] is the script's dir, so the AlphaBrain package isn't
# importable. Prepend repo root to PYTHONPATH to fix.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ── Edit these before running ────────────────────────────────
VLA_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
RUN_DIR=${1:-"results/rlt_training_TD3/rlt_5traj_alltasks_v3_release_0414_1727/rl_offpolicy"}

ITER="iter_00400"
GPU_IDS=${2:-"0,1,2"}
# ─────────────────────────────────────────────────────────────

ACTION_TOKEN_CKPT="${RUN_DIR}/checkpoints/rl_offpolicy_${ITER}"
OUT_DIR="${RUN_DIR}/eval_${ITER}"
IFS=',' read -r GPU_A GPU_B GPU_C <<< "${GPU_IDS}"
mkdir -p "${OUT_DIR}"

# Remove stale shard JSONs from prior runs so aggregate never silently reads
# pre-refactor results if the fresh eval fails.
rm -f "${OUT_DIR}"/shard_*.json "${OUT_DIR}"/summary.json

# Round-robin split of 10 libero_goal tasks across 3 GPUs:
TASKS_A="0,1,2,3"   # 4 tasks
TASKS_B="4,5,6"     # 3 tasks
TASKS_C="7,8,9"     # 3 tasks

N_EPS=50
NUM_WORKERS=4
SUITE=libero_goal
ARCH_ARGS="--bottleneck_dim 256 --encoder_layers 2 --encoder_heads 4 --actor_hidden_dim 512 --ref_dropout 0.5 --fixed_std 0.1 --prop_dim 8"

if [ ! -d "${ACTION_TOKEN_CKPT}" ]; then
    echo "ERROR: RLT_a ckpt not found: ${ACTION_TOKEN_CKPT}" >&2
    exit 1
fi
if [ ! -d "${VLA_CKPT}" ]; then
    echo "ERROR: VLA ckpt not found: ${VLA_CKPT}" >&2
    exit 1
fi

echo "============================================================"
echo " Eval RLT_a (${ITER}) | ${N_EPS} eps/task, ${NUM_WORKERS} workers/shard"
echo "   ckpt: ${ACTION_TOKEN_CKPT}"
echo "   vla:  ${VLA_CKPT}"
echo "   out:  ${OUT_DIR}"
echo "   GPU ${GPU_A}: tasks [${TASKS_A}]"
echo "   GPU ${GPU_B}: tasks [${TASKS_B}]"
echo "   GPU ${GPU_C}: tasks [${TASKS_C}]"
echo "============================================================"

# Cleanup on any exit (Ctrl-C, error, normal). Registered BEFORE launching
# anything so partial state is still cleaned up on early failures.
TAIL_PIDS=()
SHARD_PIDS=()
_cleanup () {
    trap - EXIT INT TERM   # avoid re-entry
    # Kill each shard's ENTIRE process group (python + spawned libero_env_worker
    # subprocesses). `setsid` below gives each shard PGID == shard's PID, so
    # `kill -- -PID` signals the whole group.
    for pid in "${SHARD_PIDS[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    # Kill log-mirror tails (pipeline subshells).
    [ "${#TAIL_PIDS[@]}" -gt 0 ] && kill "${TAIL_PIDS[@]}" 2>/dev/null || true
    # Belt-and-suspenders: kill any orphaned libero_env_worker owned by us.
    pkill -TERM -u "$(id -u)" -f "AlphaBrain/training/reinforcement_learning/envs/libero_env_worker" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

run_shard () {
    local gpu=$1 tasks=$2 tag=$3
    local out="${OUT_DIR}/shard_${tag}.json"
    local log="${OUT_DIR}/shard_${tag}.log"
    # setsid → fresh session/process group. Lets _cleanup take down the whole
    # subtree (python + its libero_env_worker children) via `kill -- -PGID`.
    CUDA_VISIBLE_DEVICES=${gpu} setsid python AlphaBrain/training/reinforcement_learning/eval/eval_libero.py \
        --vla_ckpt "${VLA_CKPT}" \
        --action_token_ckpt "${ACTION_TOKEN_CKPT}" \
        --suite ${SUITE} \
        --n_eps_per_task ${N_EPS} \
        --gpu 0 \
        --task_ids "${tasks}" \
        --results_json "${out}" \
        --num_workers ${NUM_WORKERS} \
        ${ARCH_ARGS} \
        > "${log}" 2>&1 &
    local pid=$!
    eval "PID_${tag}=${pid}"
    SHARD_PIDS+=(${pid})
}

run_shard "${GPU_A}" "${TASKS_A}" "a"
run_shard "${GPU_B}" "${TASKS_B}" "b"
run_shard "${GPU_C}" "${TASKS_C}" "c"

# Mirror each shard log to main stdout with a `[tag] ` prefix so progress is
# visible live. Tails are killed by _cleanup on any exit.
for tag in a b c; do
    _log="${OUT_DIR}/shard_${tag}.log"
    : > "${_log}"   # create empty file so tail -F starts following immediately
    (tail -F -q -n +1 "${_log}" 2>/dev/null | sed -u "s/^/[${tag}] /") &
    TAIL_PIDS+=($!)
done

# Wait on each PID individually and abort if any shard failed — unlike
# bare `wait`, this propagates non-zero exit codes from background children.
fail=0
for tag in a b c; do
    pid_var="PID_${tag}"
    pid=${!pid_var}
    if ! wait "${pid}"; then
        echo "ERROR: shard ${tag} (pid ${pid}) FAILED — see ${OUT_DIR}/shard_${tag}.log" >&2
        fail=1
    fi
done
# Let tails flush last lines, then stop them.
sleep 1
kill "${TAIL_PIDS[@]}" 2>/dev/null || true
[ "${fail}" -eq 0 ] || { echo "Aborting before aggregate (some shards failed)." >&2; exit 1; }
echo "[shards] all done."

echo "============================================================"
echo " Aggregating per-task SR"
echo "============================================================"
python AlphaBrain/training/reinforcement_learning/eval/aggregate_shards.py \
    --out_dir  "${OUT_DIR}" \
    --action_token_ckpt "${ACTION_TOKEN_CKPT}" \
    --vla_ckpt "${VLA_CKPT}" \
    --suite    "${SUITE}" \
    --n_eps    "${N_EPS}"
