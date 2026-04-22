#!/usr/bin/env bash
# =========================================================================================
# 统一评估启动脚本
# 自动完成两步流程：1) 后台启动推理服务  2) 运行评估客户端  3) 结束后清理服务
#
# 使用方法: bash scripts/run_eval.sh [mode] [config_file]
# 示例: bash scripts/run_eval.sh libero_eval
#       bash scripts/run_eval.sh libero_eval configs/finetune_config.yaml
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

# 加载环境变量（如果存在）
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# 默认参数
MODE="${1:-libero_eval}"
CONFIG_FILE="${2:-configs/finetune_config.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

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
ROBOCASA_TABLETOP_PYTHON="${EVAL_CLIENT_PYTHON:-${ROBOCASA_TABLETOP_PYTHON:-${LIBERO_PYTHON}}}"
ROBOCASA365_PYTHON="${EVAL_CLIENT_PYTHON:-${ROBOCASA365_PYTHON:-${LIBERO_PYTHON}}}"
SERVER_PYTHON="${EVAL_SERVER_PYTHON:-${ALPHABRAIN_PYTHON:-python}}"
EVAL_BENCHMARK="${EVAL_BENCHMARK:-libero}"

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
EVAL_OUT_DIR="results/evaluation/${RESULT_GROUP}/${folder_name}"
mkdir -p "${EVAL_OUT_DIR}"

# 视频放入 videos/ 子目录
video_out_path="${EVAL_OUT_DIR}/videos"
mkdir -p "${video_out_path}"

# 日志直接放在结果目录下
SERVER_LOG="${EVAL_OUT_DIR}/server.log"
EVAL_LOG="${EVAL_OUT_DIR}/eval.log"

# BF16 参数
BF16_FLAG=""
if [ "${EVAL_USE_BF16}" = "true" ]; then
    BF16_FLAG="--use_bf16"
fi

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
echo -e "  ${C_DIM}Output Dir:${C_RESET} ${C_DIM}${EVAL_OUT_DIR}${C_RESET}"
echo -e "${C_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo ""

# ── Step 1: 后台启动推理服务 ─────────────────────────────────────────────
info "${C_BOLD}[Step 1/2]${C_RESET} Starting policy server on GPU ${C_YELLOW}${EVAL_GPU_ID}${C_RESET}, port ${C_CYAN}${EVAL_PORT}${C_RESET} ..."

CUDA_VISIBLE_DEVICES=${EVAL_GPU_ID} "${SERVER_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${EVAL_CHECKPOINT}" \
    --port "${EVAL_PORT}" \
    ${BF16_FLAG} \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# 确保脚本退出时清理 server 进程
cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo ""
        warn "Shutting down policy server (PID: ${C_YELLOW}${SERVER_PID}${C_RESET}) ..."
        kill "${SERVER_PID}" 2>/dev/null
        wait "${SERVER_PID}" 2>/dev/null || true
        success "Server stopped."
    fi
}
trap cleanup EXIT INT TERM

# 等待 server 就绪（轮询端口）
info "Waiting for server to be ready on port ${C_CYAN}${EVAL_PORT}${C_RESET} ..."
MAX_WAIT=300   # 最多等 5 分钟（模型加载可能较慢）
WAITED=0
while ! "${SERVER_PYTHON}" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('${EVAL_HOST}', ${EVAL_PORT})); s.close()" 2>/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        error "Server process exited unexpectedly. Check log: ${C_DIM}${SERVER_LOG}${C_RESET}"
        tail -20 "${SERVER_LOG}"
        exit 1
    fi
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        error "Server did not become ready within ${MAX_WAIT}s. Check log: ${C_DIM}${SERVER_LOG}${C_RESET}"
        tail -20 "${SERVER_LOG}"
        exit 1
    fi
    sleep 3
    WAITED=$((WAITED + 3))
    echo -e "  ${C_DIM}... waited ${WAITED}s${C_RESET}"
done
success "Server is ready! (waited ${C_YELLOW}${WAITED}s${C_RESET})"

# ── Step 2: 运行评估客户端 ───────────────────────────────────────────────
echo ""
info "${C_BOLD}[Step 2/2]${C_RESET} Running evaluation client ..."

if [ "${EVAL_BENCHMARK}" = "robocasa_tabletop" ]; then
    info "Using RoboCasa tabletop client: ${C_YELLOW}${EVAL_ENV_NAME}${C_RESET}"
    NUMBA_DISABLE_JIT="${EVAL_NUMBA_DISABLE_JIT}" \
    MUJOCO_GL="${EVAL_MUJOCO_GL}" \
    PYOPENGL_PLATFORM="${EVAL_PYOPENGL_PLATFORM}" \
    CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" \
    "${ROBOCASA_TABLETOP_PYTHON}" ./benchmarks/Robocasa_tabletop/eval/simulation_env.py \
        --args.pretrained_path "${EVAL_CHECKPOINT}" \
        --args.host "${EVAL_HOST}" \
        --args.port "${EVAL_PORT}" \
        --args.env_name "${EVAL_ENV_NAME}" \
        --args.n_episodes "${EVAL_NUM_EPISODES}" \
        --args.n_envs "${EVAL_NUM_ENVS}" \
        --args.max_episode_steps "${EVAL_MAX_EPISODE_STEPS}" \
        --args.n_action_steps "${EVAL_N_ACTION_STEPS}" \
        --args.video_out_path "${video_out_path}" \
        2>&1 | tee "${EVAL_LOG}"
elif [ "${EVAL_BENCHMARK}" = "robocasa365" ]; then
    info "Using RoboCasa365 client: task_set=${C_YELLOW}${EVAL_TASK_SET}${C_RESET} split=${C_YELLOW}${EVAL_SPLIT}${C_RESET}"
    NUMBA_DISABLE_JIT="${EVAL_NUMBA_DISABLE_JIT}" \
    MUJOCO_GL="${EVAL_MUJOCO_GL}" \
    PYOPENGL_PLATFORM="${EVAL_PYOPENGL_PLATFORM}" \
    CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" \
    PYTHONPATH="$(pwd)${EVAL_CLIENT_PYTHONPATH:+:${EVAL_CLIENT_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${ROBOCASA365_PYTHON}" ./benchmarks/Robocasa365/eval/simulation_env.py \
        --args.pretrained_path "${EVAL_CHECKPOINT}" \
        --args.host "${EVAL_HOST}" \
        --args.port "${EVAL_PORT}" \
        --args.task_set "${EVAL_TASK_SET}" \
        --args.split "${EVAL_SPLIT}" \
        --args.n_episodes "${EVAL_NUM_EPISODES}" \
        --args.n_envs "${EVAL_NUM_ENVS}" \
        --args.max_episode_steps "${EVAL_MAX_EPISODE_STEPS}" \
        --args.n_action_steps "${EVAL_N_ACTION_STEPS}" \
        --args.video_out_path "${video_out_path}" \
        2>&1 | tee "${EVAL_LOG}"
else
    if [ "${TASK_SUITE}" = "libero_all" ]; then
        # Run all 4 LIBERO suites sequentially, reusing the same server
        for _suite in libero_goal libero_spatial libero_object libero_10; do
            _suite_dir="${EVAL_OUT_DIR%/*}/${_suite}/${folder_name}"
            mkdir -p "${_suite_dir}/videos"
            _suite_log="${_suite_dir}/eval.log"
            _suite_video="${_suite_dir}/videos"
            echo ""
            info "Running ${_suite} ..."
            "${LIBERO_PYTHON}" ./benchmarks/LIBERO/eval/eval_libero.py \
                --args.pretrained-path "${EVAL_CHECKPOINT}" \
                --args.host "${EVAL_HOST}" \
                --args.port "${EVAL_PORT}" \
                --args.task-suite-name "${_suite}" \
                --args.num-trials-per-task "${NUM_TRIALS}" \
                --args.num-views "${EVAL_NUM_VIEWS:-2}" \
                --args.video-out-path "${_suite_video}" \
                2>&1 | tee "${_suite_log}"
        done
    else
        "${LIBERO_PYTHON}" ./benchmarks/LIBERO/eval/eval_libero.py \
            --args.pretrained-path "${EVAL_CHECKPOINT}" \
            --args.host "${EVAL_HOST}" \
            --args.port "${EVAL_PORT}" \
            --args.task-suite-name "${TASK_SUITE}" \
            --args.num-trials-per-task "${NUM_TRIALS}" \
            --args.num-views "${EVAL_NUM_VIEWS:-2}" \
            --args.video-out-path "${video_out_path}" \
            2>&1 | tee "${EVAL_LOG}"
    fi
fi






echo ""
success "Evaluation complete!"
echo -e "  ${C_DIM}Results dir:${C_RESET} ${C_YELLOW}${EVAL_OUT_DIR}${C_RESET}"
echo -e "  ${C_DIM}Videos:${C_RESET}      ${C_DIM}${video_out_path}${C_RESET}"
echo -e "  ${C_DIM}Eval log:${C_RESET}    ${C_DIM}${EVAL_LOG}${C_RESET}"
echo -e "  ${C_DIM}Server log:${C_RESET}  ${C_DIM}${SERVER_LOG}${C_RESET}"
# cleanup 会被 trap EXIT 自动调用
