#!/usr/bin/env bash
# =========================================================================================
# 统一评估启动脚本
# 自动完成两步流程：1) 后台启动推理服务  2) 运行评估客户端  3) 结束后清理服务
#
# 使用方法: bash scripts/run_eval.sh [mode] [config_file]
# 示例: bash scripts/run_eval.sh libero_eval
#       bash scripts/run_eval.sh libero_eval configs/finetune_config.yaml
# UI 托管模式:
#       ALPHABRAIN_UI_MANAGED=1 EVAL_OUTPUT_DIR=/exact/run/dir \
#         EVAL_PROGRESS_PATH=/exact/run/dir/progress.jsonl \
#         bash scripts/run_eval.sh ui_eval /read-only/generated-config.yaml
# EVAL_OUTPUT_DIR 是本次运行的精确目录；最终结果固定为 evaluation-result-v1.json。
# =========================================================================================
set -euo pipefail

# ── 颜色定义（非 tty 时自动关闭）─────────────────────────────────────────
if [ -t 1 ]; then
    C_RESET="\033[0m"
    C_BOLD="\033[1m"
    C_DIM="\033[2m"
    C_RED="\033[91m"
    C_GREEN="\033[92m"
    C_YELLOW="\033[93m"
    C_CYAN="\033[96m"
    C_BOLD_CYAN="\033[1;96m"
    C_BOLD_YELLOW="\033[1;93m"
    C_BOLD_GREEN="\033[1;92m"
    C_BOLD_RED="\033[1;91m"
else
    C_RESET="" C_BOLD="" C_DIM="" C_RED="" C_GREEN=""
    C_YELLOW="" C_CYAN="" C_BOLD_CYAN="" C_BOLD_YELLOW=""
    C_BOLD_GREEN="" C_BOLD_RED=""
fi

info()    { echo -e "${C_CYAN}[info]${C_RESET} $*"; }
warn()    { echo -e "${C_BOLD_YELLOW}[warn]${C_RESET} $*"; }
error()   { echo -e "${C_BOLD_RED}[error]${C_RESET} $*" >&2; }
success() { echo -e "${C_BOLD_GREEN}[ok]${C_RESET} $*"; }

# 默认参数
MODE="${1:-libero_eval}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
CONFIG_FILE="${EVAL_CONFIG_FILE:-${2:-configs/finetune_config.yaml}}"

# UI-managed launches already receive a curated environment.  Loading .env in
# that mode could silently replace the selected checkpoint, GPU, or simulator.
case "${ALPHABRAIN_UI_MANAGED:-0}" in
    1|true|TRUE|yes|YES) ;;
    *)
        if [ -f "${PROJECT_ROOT}/.env" ]; then
            set -a; source "${PROJECT_ROOT}/.env"; set +a
        fi
        ;;
esac

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    error "Config file '${C_YELLOW}${CONFIG_FILE}${C_RESET}' not found!"
    exit 1
fi

# 使用 Python 脚本解析配置文件并加载到当前 shell 环境
info "Loading configuration from ${C_YELLOW}${CONFIG_FILE}${C_RESET}  mode: ${C_BOLD}${MODE}${C_RESET}"
eval "$(python scripts/parse_config.py --config "$CONFIG_FILE" --mode "$MODE")"

# 检查是否为 eval 模式
if [ "$IS_EVAL" != "true" ]; then
    error "Mode '${C_YELLOW}${MODE}${C_RESET}' is not an eval mode. Use ${C_CYAN}scripts/run_finetune.sh${C_RESET} for training."
    exit 1
fi

# 配置 LIBERO 环境
# todo: [zhanghe] 现在这里只支持libero，这是肯定不行的；后续需要同时支持更多的环境；
LIBERO_HOME="${LIBERO_HOME:-../LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
LIBERO_PLUS_PYTHON="${EVAL_CLIENT_PYTHON:-${LIBERO_PLUS_PYTHON:-${LIBERO_PYTHON}}}"
ROBOCASA_TABLETOP_PYTHON="${EVAL_CLIENT_PYTHON:-${ROBOCASA_TABLETOP_PYTHON:-${LIBERO_PYTHON}}}"
ROBOCASA365_PYTHON="${EVAL_CLIENT_PYTHON:-${ROBOCASA365_PYTHON:-${LIBERO_PYTHON}}}"
SERVER_PYTHON="${EVAL_SERVER_PYTHON:-${ALPHABRAIN_PYTHON:-python}}"
EVAL_BENCHMARK="${EVAL_BENCHMARK:-libero}"
EVAL_RESULT_PATH=""

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${LIBERO_HOME}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 准备输出目录：日志和视频统一放在结果目录下
folder_name=$(echo "$EVAL_CHECKPOINT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
RESULT_GROUP="${TASK_SUITE}"
if [ "${EVAL_BENCHMARK}" = "robocasa_tabletop" ]; then
    ROBOCASA_ENV_SLUG=$(echo "${EVAL_ENV_NAME}" | tr '/ ' '__')
    RESULT_GROUP="robocasa_tabletop/${ROBOCASA_ENV_SLUG}"
elif [ "${EVAL_BENCHMARK}" = "robocasa365" ]; then
    ROBOCASA365_TASK_SLUG=$(echo "${EVAL_TASK_SET}" | tr ',/ ' '___')
    RESULT_GROUP="robocasa365/${EVAL_SPLIT}/${ROBOCASA365_TASK_SLUG}"
fi
DEFAULT_EVAL_OUT_DIR="results/evaluation/${RESULT_GROUP}/${folder_name}"
# EVAL_OUTPUT_DIR is the exact directory for one UI run.  The legacy nested
# layout is retained when it is absent.
EVAL_OUT_DIR="${EVAL_OUTPUT_DIR:-${EVAL_CONFIG_OUTPUT_DIR:-${DEFAULT_EVAL_OUT_DIR}}}"
mkdir -p "${EVAL_OUT_DIR}"

# 视频放入 videos/ 子目录
video_out_path="${EVAL_OUT_DIR}/videos"
mkdir -p "${video_out_path}"

# 日志直接放在结果目录下
SERVER_LOG="${EVAL_OUT_DIR}/server.log"
EVAL_LOG="${EVAL_OUT_DIR}/eval.log"
EVAL_RESULT_PATH="${EVAL_OUT_DIR}/evaluation-result-v1.json"
SUITE_RESULTS=()

# Resolve and validate model-server code only for temporary evaluations.  A
# managed deployment is already running and must remain independent of any
# later checkout or Python-environment changes.
EVAL_REUSE_SERVER="${EVAL_REUSE_SERVER:-false}"
EVAL_SERVER_ENTRYPOINT="${EVAL_SERVER_ENTRYPOINT:-deployment/model_server/server_policy.py}"
SERVER_ENTRYPOINT_PATH=""
if [ "${EVAL_REUSE_SERVER}" != "true" ]; then
    if [[ "${EVAL_SERVER_ENTRYPOINT}" = /* ]]; then
        SERVER_ENTRYPOINT_PATH="${EVAL_SERVER_ENTRYPOINT}"
    else
        SERVER_ENTRYPOINT_PATH="${PROJECT_ROOT}/${EVAL_SERVER_ENTRYPOINT}"
    fi
    SERVER_ENTRYPOINT_PATH="$(realpath -m "${SERVER_ENTRYPOINT_PATH}")"
    case "${SERVER_ENTRYPOINT_PATH}" in
        "${PROJECT_ROOT}"/*) ;;
        *)
            error "Server entrypoint must be inside the AlphaBrain checkout: ${SERVER_ENTRYPOINT_PATH}"
            exit 1
            ;;
    esac
    if [ ! -f "${SERVER_ENTRYPOINT_PATH}" ] || [ "${SERVER_ENTRYPOINT_PATH##*.}" != "py" ]; then
        error "Configured server entrypoint is not a Python file: ${SERVER_ENTRYPOINT_PATH}"
        exit 1
    fi
fi

EVAL_CLIENT_ENTRYPOINT="${EVAL_CLIENT_ENTRYPOINT:-}"
if [ "${EVAL_REUSE_SERVER}" != "true" ] && [ -z "${EVAL_CLIENT_ENTRYPOINT}" ] && \
   [ "${SERVER_ENTRYPOINT_PATH}" = "${PROJECT_ROOT}/deployment/model_server/server_policy_cosmos.py" ]; then
    EVAL_CLIENT_ENTRYPOINT="benchmarks/LIBERO/eval/eval_libero_cosmos.py"
fi
CLIENT_ENTRYPOINT_PATH=""
if [ -n "${EVAL_CLIENT_ENTRYPOINT}" ]; then
    if [[ "${EVAL_CLIENT_ENTRYPOINT}" = /* ]]; then
        CLIENT_ENTRYPOINT_PATH="${EVAL_CLIENT_ENTRYPOINT}"
    else
        CLIENT_ENTRYPOINT_PATH="${PROJECT_ROOT}/${EVAL_CLIENT_ENTRYPOINT}"
    fi
    CLIENT_ENTRYPOINT_PATH="$(realpath -m "${CLIENT_ENTRYPOINT_PATH}")"
    case "${CLIENT_ENTRYPOINT_PATH}" in
        "${PROJECT_ROOT}"/*) ;;
        *)
            error "Client entrypoint must be inside the AlphaBrain checkout: ${CLIENT_ENTRYPOINT_PATH}"
            exit 1
            ;;
    esac
    if [ ! -f "${CLIENT_ENTRYPOINT_PATH}" ] || [ "${CLIENT_ENTRYPOINT_PATH##*.}" != "py" ]; then
        error "Configured client entrypoint is not a Python file: ${CLIENT_ENTRYPOINT_PATH}"
        exit 1
    fi
fi
COSMOS_SERVER_PATH="${PROJECT_ROOT}/deployment/model_server/server_policy_cosmos.py"
COSMOS_CLIENT_PATH="${PROJECT_ROOT}/benchmarks/LIBERO/eval/eval_libero_cosmos.py"
if [ "${EVAL_REUSE_SERVER}" != "true" ] && { \
   { [ "${SERVER_ENTRYPOINT_PATH}" = "${COSMOS_SERVER_PATH}" ] && \
     [ "${CLIENT_ENTRYPOINT_PATH}" != "${COSMOS_CLIENT_PATH}" ]; } || \
   { [ "${SERVER_ENTRYPOINT_PATH}" != "${COSMOS_SERVER_PATH}" ] && \
     [ "${CLIENT_ENTRYPOINT_PATH}" = "${COSMOS_CLIENT_PATH}" ]; }; }; then
    error "Cosmos Policy server and LIBERO client must be selected together."
    exit 1
fi

SERVER_ARGS=()
if [ "${EVAL_REUSE_SERVER}" = "true" ]; then
    :
elif [ -n "${EVAL_SERVER_ARGS_JSON:-}" ] && [ "${EVAL_SERVER_ARGS_JSON}" != "[]" ]; then
    if ! SERVER_ARGS_TEXT="$(
        "${SERVER_PYTHON}" -c \
            'import json,sys; value=json.loads(sys.argv[1]); assert isinstance(value,list); print("\n".join(str(x) for x in value))' \
            "${EVAL_SERVER_ARGS_JSON}"
    )"; then
        error "Invalid EVAL_SERVER_ARGS_JSON"
        exit 1
    fi
    if [ -n "${SERVER_ARGS_TEXT}" ]; then
        mapfile -t SERVER_ARGS <<< "${SERVER_ARGS_TEXT}"
    fi
elif [ "${SERVER_ENTRYPOINT_PATH}" = "${PROJECT_ROOT}/deployment/model_server/server_policy.py" ]; then
    SERVER_ARGS=(--ckpt_path "${EVAL_CHECKPOINT}" --port "${EVAL_PORT}")
    if [ "${EVAL_USE_BF16}" = "true" ]; then
        SERVER_ARGS+=(--use_bf16)
    fi
else
    error "server_args is required for non-default server adapter: ${EVAL_SERVER_ENTRYPOINT}"
    error "For Cosmos Policy pass --ckpt_dir, --pretrained_dir, --port and any adapter-specific options."
    exit 1
fi

progress() {
    local stage="$1"
    local state="$2"
    local message="${3:-}"
    echo "[progress] stage=${stage} status=${state} message=${message}"
    if [ -n "${EVAL_PROGRESS_PATH:-}" ]; then
        "${SERVER_PYTHON}" -m alphabrain_ui.evaluation_result progress \
            --path "${EVAL_PROGRESS_PATH}" \
            --stage "${stage}" \
            --status "${state}" \
            --message "${message}" || warn "Unable to append progress event"
    fi
}

echo ""
echo -e "${C_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo -e "  ${C_BOLD_CYAN}▶  Starting Evaluation${C_RESET}"
echo -e "  ${C_DIM}Mode:${C_RESET}       ${C_BOLD}${MODE}${C_RESET}"
echo -e "  ${C_DIM}Benchmark:${C_RESET}  ${C_BOLD_YELLOW}${EVAL_BENCHMARK}${C_RESET}"
echo -e "  ${C_DIM}Checkpoint:${C_RESET} ${C_YELLOW}${EVAL_CHECKPOINT}${C_RESET}"
echo -e "  ${C_DIM}Task Suite:${C_RESET} ${C_BOLD_YELLOW}${TASK_SUITE}${C_RESET}"
echo -e "  ${C_DIM}Num Trials:${C_RESET} ${C_YELLOW}${NUM_TRIALS}${C_RESET}"
echo -e "  ${C_DIM}Server:${C_RESET}     GPU ${C_YELLOW}${EVAL_GPU_ID}${C_RESET}, ${C_CYAN}${EVAL_HOST}:${EVAL_PORT}${C_RESET}"
echo -e "  ${C_DIM}BF16:${C_RESET}       ${C_YELLOW}${EVAL_USE_BF16}${C_RESET}"
if [ "${EVAL_REUSE_SERVER}" = "true" ]; then
    echo -e "  ${C_DIM}Server mode:${C_RESET}  ${C_DIM}reuse managed deployment${C_RESET}"
else
    echo -e "  ${C_DIM}Server entry:${C_RESET} ${C_DIM}${SERVER_ENTRYPOINT_PATH}${C_RESET}"
fi
if [ -n "${CLIENT_ENTRYPOINT_PATH}" ]; then
    echo -e "  ${C_DIM}Client entry:${C_RESET} ${C_DIM}${CLIENT_ENTRYPOINT_PATH}${C_RESET}"
fi
echo -e "  ${C_DIM}Output Dir:${C_RESET} ${C_DIM}${EVAL_OUT_DIR}${C_RESET}"
echo -e "${C_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo ""

# ── Step 1: 启动临时服务，或连接现有托管服务 ─────────────────────────────
SERVER_PID=""
if [ "${EVAL_REUSE_SERVER}" = "true" ]; then
    info "${C_BOLD}[Step 1/2]${C_RESET} Reusing managed policy server at ${C_CYAN}${EVAL_HOST}:${EVAL_PORT}${C_RESET} ..."
    progress "server_reuse" "running" "Connecting to managed policy server"
else
    info "${C_BOLD}[Step 1/2]${C_RESET} Starting policy server on GPU ${C_YELLOW}${EVAL_GPU_ID}${C_RESET}, port ${C_CYAN}${EVAL_PORT}${C_RESET} ..."
    progress "server_start" "running" "Starting policy server"

    CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" "${SERVER_PYTHON}" "${SERVER_ENTRYPOINT_PATH}" \
        "${SERVER_ARGS[@]}" \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
fi

# 确保脚本退出时清理 server 进程
cleanup() {
    local exit_status=$?
    if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo ""
        warn "Shutting down policy server (PID: ${C_YELLOW}${SERVER_PID}${C_RESET}) ..."
        kill "${SERVER_PID}" 2>/dev/null
        wait "${SERVER_PID}" 2>/dev/null || true
        success "Server stopped."
    fi
    if [ "${exit_status}" -ne 0 ]; then
        progress "evaluation" "failed" "Evaluation exited with code ${exit_status}"
        EXISTING_SUITE_RESULTS=()
        for suite_result in "${SUITE_RESULTS[@]}"; do
            if [ -f "${suite_result}" ]; then
                EXISTING_SUITE_RESULTS+=("${suite_result}")
            fi
        done
        if [ "${#EXISTING_SUITE_RESULTS[@]}" -gt 0 ]; then
            "${SERVER_PYTHON}" -m alphabrain_ui.evaluation_result aggregate \
                --output "${EVAL_RESULT_PATH}" \
                --suite-name libero_all \
                "${EXISTING_SUITE_RESULTS[@]}" || true
        fi
        "${SERVER_PYTHON}" -m alphabrain_ui.evaluation_result fail \
            --output "${EVAL_RESULT_PATH}" \
            --benchmark "${EVAL_BENCHMARK}" \
            --checkpoint "${EVAL_CHECKPOINT}" \
            --suite-name "${TASK_SUITE}" \
            --error "Evaluation launcher exited with code ${exit_status}" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 等待 server 就绪（轮询端口）
info "Waiting for server to be ready on port ${C_CYAN}${EVAL_PORT}${C_RESET} ..."
MAX_WAIT=900   # 最多等 15 分钟（PaliGemmaPi0 cold-read from /datasets/pi05 + peligemma 可达 ~5min）
WAITED=0
while ! "${SERVER_PYTHON}" -c \
    'import socket,sys; s=socket.socket(); s.settimeout(1); s.connect((sys.argv[1], int(sys.argv[2]))); s.close()' \
    "${EVAL_HOST}" "${EVAL_PORT}" 2>/dev/null; do
    if [ "${EVAL_REUSE_SERVER}" != "true" ] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        error "Server process exited unexpectedly. Check log: ${C_DIM}${SERVER_LOG}${C_RESET}"
        tail -20 "${SERVER_LOG}"
        exit 1
    fi
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        if [ "${EVAL_REUSE_SERVER}" = "true" ]; then
            error "Managed server did not become reachable within ${MAX_WAIT}s."
        else
            error "Server did not become ready within ${MAX_WAIT}s. Check log: ${C_DIM}${SERVER_LOG}${C_RESET}"
            tail -20 "${SERVER_LOG}"
        fi
        exit 1
    fi
    sleep 3
    WAITED=$((WAITED + 3))
    echo -e "  ${C_DIM}... waited ${WAITED}s${C_RESET}"
done
success "Server is ready! (waited ${C_YELLOW}${WAITED}s${C_RESET})"
progress "server_ready" "completed" "Policy server is accepting connections"

# ── Step 2: 运行评估客户端 ───────────────────────────────────────────────
echo ""
info "${C_BOLD}[Step 2/2]${C_RESET} Running evaluation client ..."
progress "simulation" "running" "Starting benchmark client"

if [ "${EVAL_BENCHMARK}" = "robocasa_tabletop" ]; then
    info "Using RoboCasa tabletop client: ${C_YELLOW}${EVAL_ENV_NAME}${C_RESET}"
    NUMBA_DISABLE_JIT="${EVAL_NUMBA_DISABLE_JIT}" \
    MUJOCO_GL="${EVAL_MUJOCO_GL}" \
    PYOPENGL_PLATFORM="${EVAL_PYOPENGL_PLATFORM}" \
    CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" \
    "${ROBOCASA_TABLETOP_PYTHON}" ./benchmarks/Robocasa_tabletop/eval/simulation_env.py \
        --args.pretrained-path "${EVAL_CHECKPOINT}" \
        --args.host "${EVAL_HOST}" \
        --args.port "${EVAL_PORT}" \
        --args.env-name "${EVAL_ENV_NAME}" \
        --args.n-episodes "${EVAL_NUM_EPISODES}" \
        --args.n-envs "${EVAL_NUM_ENVS}" \
        --args.max-episode-steps "${EVAL_MAX_EPISODE_STEPS}" \
        --args.n-action-steps "${EVAL_N_ACTION_STEPS}" \
        --args.seed "${EVAL_SEED:-7}" \
        --args.video-out-path "${video_out_path}" \
        --args.result-out-path "${EVAL_RESULT_PATH}" \
        2>&1 | tee "${EVAL_LOG}"
elif [ "${EVAL_BENCHMARK}" = "robocasa365" ]; then
    info "Using RoboCasa365 client: task_set=${C_YELLOW}${EVAL_TASK_SET}${C_RESET} split=${C_YELLOW}${EVAL_SPLIT}${C_RESET}"
    ROBOCASA365_SORT_ARGS=()
    if [ "${EVAL_SORT_TASKS:-true}" = "false" ]; then
        # Tyro exposes a default-true bool as the negated --no-* flag.
        ROBOCASA365_SORT_ARGS+=(--args.no-sort-tasks)
    fi
    NUMBA_DISABLE_JIT="${EVAL_NUMBA_DISABLE_JIT}" \
    MUJOCO_GL="${EVAL_MUJOCO_GL}" \
    PYOPENGL_PLATFORM="${EVAL_PYOPENGL_PLATFORM}" \
    CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" \
    PYTHONPATH="$(pwd)${EVAL_CLIENT_PYTHONPATH:+:${EVAL_CLIENT_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${ROBOCASA365_PYTHON}" ./benchmarks/Robocasa365/eval/simulation_env.py \
        --args.pretrained-path "${EVAL_CHECKPOINT}" \
        --args.host "${EVAL_HOST}" \
        --args.port "${EVAL_PORT}" \
        --args.task-set "${EVAL_TASK_SET}" \
        --args.task-list "${EVAL_TASK_LIST:-}" \
        --args.task-limit "${EVAL_TASK_LIMIT:-0}" \
        --args.split "${EVAL_SPLIT}" \
        --args.n-episodes "${EVAL_NUM_EPISODES}" \
        --args.n-envs "${EVAL_NUM_ENVS}" \
        --args.max-episode-steps "${EVAL_MAX_EPISODE_STEPS}" \
        --args.n-action-steps "${EVAL_N_ACTION_STEPS}" \
        --args.seed "${EVAL_SEED:-7}" \
        "${ROBOCASA365_SORT_ARGS[@]}" \
        --args.video-out-path "${video_out_path}" \
        --args.result-out-path "${EVAL_RESULT_PATH}" \
        2>&1 | tee "${EVAL_LOG}"
elif [ "${EVAL_BENCHMARK}" = "libero" ] || [ "${EVAL_BENCHMARK}" = "libero_plus" ]; then
    run_libero_suite() {
        local suite="$1"
        local suite_video="$2"
        local suite_result="$3"
        local suite_log="$4"
        local client_script="./benchmarks/LIBERO/eval/eval_libero.py"
        local client_python="${LIBERO_PYTHON}"
        local predict_video_args=()
        if [ "${EVAL_BENCHMARK}" = "libero" ] && [ "${EVAL_PREDICT_VIDEO:-false}" = "true" ]; then
            predict_video_args+=(--args.predict-video)
        fi
        if [ -n "${CLIENT_ENTRYPOINT_PATH}" ]; then
            if [ "${CLIENT_ENTRYPOINT_PATH}" = "${PROJECT_ROOT}/benchmarks/LIBERO/eval/eval_libero_cosmos.py" ]; then
                if [ "${EVAL_BENCHMARK}" != "libero" ]; then
                    error "The Cosmos Policy client only supports the LIBERO benchmark"
                    return 1
                fi
                local cosmos_seed=0
                if [ "${EVAL_SEED_CONFIGURED:-false}" = "true" ]; then
                    cosmos_seed="${EVAL_SEED}"
                fi
                "${client_python}" "${CLIENT_ENTRYPOINT_PATH}" \
                    --ckpt-dir "${EVAL_CHECKPOINT}" \
                    --host "${EVAL_HOST}" \
                    --port "${EVAL_PORT}" \
                    --task-suite-name "${suite}" \
                    --task-ids "${EVAL_TASK_IDS:-}" \
                    --task-limit "${EVAL_TASK_LIMIT:-0}" \
                    --num-trials-per-task "${NUM_TRIALS}" \
                    --seed "${cosmos_seed}" \
                    --video-out-path "${suite_video}" \
                    --result-out-path "${suite_result}" \
                    2>&1 | tee "${suite_log}"
                return
            elif [ "${CLIENT_ENTRYPOINT_PATH}" = "${PROJECT_ROOT}/benchmarks/LIBERO/eval/eval_libero.py" ]; then
                client_script="${CLIENT_ENTRYPOINT_PATH}"
            else
                error "Configured LIBERO client entrypoint is not wired: ${CLIENT_ENTRYPOINT_PATH}"
                return 1
            fi
        fi
        if [ "${EVAL_BENCHMARK}" = "libero_plus" ]; then
            client_script="./benchmarks/LIBERO-plus/eval/eval_libero.py"
            client_python="${LIBERO_PLUS_PYTHON}"
            local libero_plus_root="${LIBERO_PLUS_HOME:-${LIBERO_HOME}}"
            LIBERO_PLUS_HOME="${libero_plus_root}" \
            LIBERO_HOME="${libero_plus_root}" \
            LIBERO_CONFIG_PATH="${libero_plus_root}/libero" \
            PYTHONPATH="${PROJECT_ROOT}:${libero_plus_root}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${client_python}" "${client_script}" \
                --args.pretrained-path "${EVAL_CHECKPOINT}" \
                --args.host "${EVAL_HOST}" \
                --args.port "${EVAL_PORT}" \
                --args.task-suite-name "${suite}" \
                --args.task-ids "${EVAL_TASK_IDS:-}" \
                --args.task-limit "${EVAL_TASK_LIMIT:-0}" \
                --args.num-trials-per-task "${NUM_TRIALS}" \
                --args.num-views "${EVAL_NUM_VIEWS:-2}" \
                --args.seed "${EVAL_SEED:-7}" \
                --args.video-out-path "${suite_video}" \
                --args.log-path "$(dirname "${suite_log}")" \
                --args.result-out-path "${suite_result}" \
                "${predict_video_args[@]}" \
                2>&1 | tee "${suite_log}"
        else
            "${client_python}" "${client_script}" \
                --args.pretrained-path "${EVAL_CHECKPOINT}" \
                --args.host "${EVAL_HOST}" \
                --args.port "${EVAL_PORT}" \
                --args.task-suite-name "${suite}" \
                --args.task-ids "${EVAL_TASK_IDS:-}" \
                --args.task-limit "${EVAL_TASK_LIMIT:-0}" \
                --args.num-trials-per-task "${NUM_TRIALS}" \
                --args.num-views "${EVAL_NUM_VIEWS:-2}" \
                --args.seed "${EVAL_SEED:-7}" \
                --args.video-out-path "${suite_video}" \
                --args.result-out-path "${suite_result}" \
                "${predict_video_args[@]}" \
                2>&1 | tee "${suite_log}"
        fi
    }

    if [ "${TASK_SUITE}" = "libero_all" ]; then
        SUITE_RESULTS=()
        for _suite in libero_goal libero_spatial libero_object libero_10; do
            if [ -n "${EVAL_OUTPUT_DIR:-${EVAL_CONFIG_OUTPUT_DIR:-}}" ]; then
                _suite_dir="${EVAL_OUT_DIR}/suites/${_suite}"
            else
                _suite_dir="${EVAL_OUT_DIR%/*}/${_suite}/${folder_name}"
            fi
            mkdir -p "${_suite_dir}/videos"
            _suite_log="${_suite_dir}/eval.log"
            _suite_video="${_suite_dir}/videos"
            _suite_result="${_suite_dir}/evaluation-result-v1.json"
            SUITE_RESULTS+=("${_suite_result}")
            echo ""
            info "Running ${_suite} ..."
            run_libero_suite \
                "${_suite}" "${_suite_video}" "${_suite_result}" "${_suite_log}"
        done
        progress "aggregation" "running" "Combining LIBERO suite results"
        "${SERVER_PYTHON}" -m alphabrain_ui.evaluation_result aggregate \
            --output "${EVAL_RESULT_PATH}" \
            --suite-name libero_all \
            "${SUITE_RESULTS[@]}"
        progress "aggregation" "completed" "Combined LIBERO suite results"
    else
        run_libero_suite \
            "${TASK_SUITE}" "${video_out_path}" "${EVAL_RESULT_PATH}" "${EVAL_LOG}"
    fi
else
    error "Unsupported evaluation benchmark: ${EVAL_BENCHMARK}"
    exit 1
fi

if [ ! -f "${EVAL_RESULT_PATH}" ]; then
    error "Evaluation client completed without writing ${EVAL_RESULT_PATH}"
    exit 1
fi
progress "simulation" "completed" "Benchmark client completed"
progress "evaluation" "completed" "Evaluation result is ready"
echo ""
success "Evaluation complete!"
echo -e "  ${C_DIM}Results dir:${C_RESET} ${C_YELLOW}${EVAL_OUT_DIR}${C_RESET}"
echo -e "  ${C_DIM}Videos:${C_RESET}      ${C_DIM}${video_out_path}${C_RESET}"
echo -e "  ${C_DIM}Eval log:${C_RESET}    ${C_DIM}${EVAL_LOG}${C_RESET}"
if [ "${EVAL_REUSE_SERVER}" != "true" ]; then
    echo -e "  ${C_DIM}Server log:${C_RESET}  ${C_DIM}${SERVER_LOG}${C_RESET}"
fi
echo -e "  ${C_DIM}Result JSON:${C_RESET} ${C_DIM}${EVAL_RESULT_PATH}${C_RESET}"
# cleanup 会被 trap EXIT 自动调用
