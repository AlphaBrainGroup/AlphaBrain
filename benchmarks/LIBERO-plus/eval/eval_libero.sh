#!/usr/bin/env bash
set -euo pipefail

# Legacy standalone LIBERO-plus client launcher.  The UI uses scripts/run_eval.sh,
# but this entrypoint remains useful when a policy server is already running.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export LIBERO_HOME="${LIBERO_PLUS_HOME:-${LIBERO_HOME:-}}"
: "${LIBERO_HOME:?Set LIBERO_PLUS_HOME (or LIBERO_HOME) to the LIBERO-plus checkout}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
LIBERO_PYTHON="${LIBERO_PLUS_PYTHON:-${LIBERO_PYTHON:-python}}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_HOME}${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint="${1:-${EVAL_CHECKPOINT:-}}"
: "${checkpoint:?Pass a checkpoint as argument 1 or set EVAL_CHECKPOINT}"
output_dir="${2:-${EVAL_OUTPUT_DIR:-results/evaluation/libero_plus}}"
host="${EVAL_HOST:-127.0.0.1}"
port="${EVAL_PORT:-9883}"
gpu_id="${EVAL_GPU_ID:-0}"
num_trials="${NUM_TRIALS:-1}"
folder_name=$(echo "${checkpoint}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

log_dir="${output_dir}/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${log_dir}"

result_files=()
for task_suite_name in libero_goal libero_spatial libero_object libero_10; do
    video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
    result_path="${video_out_path}/evaluation-result-v1.json"
    result_files+=("${result_path}")
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${LIBERO_PYTHON}" \
        ./benchmarks/LIBERO-plus/eval/eval_libero.py \
        --args.pretrained-path "${checkpoint}" \
        --args.host "${host}" \
        --args.port "${port}" \
        --args.task-suite-name "${task_suite_name}" \
        --args.task-ids "${EVAL_TASK_IDS:-}" \
        --args.task-limit "${EVAL_TASK_LIMIT:-0}" \
        --args.num-trials-per-task "${num_trials}" \
        --args.num-views "${EVAL_NUM_VIEWS:-2}" \
        --args.seed "${EVAL_SEED:-7}" \
        --args.video-out-path "${video_out_path}" \
        --args.result-out-path "${result_path}" \
        --args.log-path "${log_dir}" \
        2>&1 | tee "${log_dir}/${task_suite_name}.log" &
done

echo "Waiting for all evaluation tasks to finish..."
wait

"${ALPHABRAIN_PYTHON:-python}" -m alphabrain_ui.evaluation_result aggregate \
    --output "${output_dir}/evaluation-result-v1.json" \
    --suite-name libero_all \
    "${result_files[@]}"
