#!/usr/bin/env bash
# =========================================================================================
# 统一训练启动脚本 — 唯一入口
# 使用方法: bash scripts/run_finetune.sh [config_file]
# 示例: bash scripts/run_finetune.sh
#       bash scripts/run_finetune.sh configs/finetune_config.yaml
# =========================================================================================

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

# ── 清理训练日志噪声 ─────────────────────────────────────────
# 关掉 albumentations 的版本检查弹窗
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
# torchrun 默认会警告 OMP_NUM_THREADS 未设置，给个合理默认
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# 注：DeepSpeed import 时会用 distutils.has_function 探测 libaio / libcufile，
# 顺手把一坨 `gcc -pthread … compiler_compat …` 命令打到 stdout。DS_BUILD_AIO /
# DS_BUILD_GDS 只能跳过 JIT 构建，并不会关掉这个 is_compatible 探测；这里用
# 文末的 grep 过滤（在 tee 之前）一次性吃掉，pattern 同时含 gcc -pthread 和
# conda 的 compiler_compat 目录，不会误伤用户输出。

# 解析参数：
#   bash scripts/run_finetune.sh qwen_oft                  → mode=qwen_oft, config=默认
#   bash scripts/run_finetune.sh --mode qwen_oft            → 同上
#   bash scripts/run_finetune.sh my_config.yaml             → config=my_config.yaml
#   bash scripts/run_finetune.sh my_config.yaml --mode xxx  → config=my_config.yaml, mode=xxx
CONFIG_FILE=""
MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --mode=*)
            MODE="${1#--mode=}"; shift ;;
        *)
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$1"
            fi
            shift ;;
    esac
done
# 如果第一个参数不是文件路径（不含 / 且不以 .yaml/.yml 结尾），视为 mode 名
if [ -n "$CONFIG_FILE" ] && [ -z "$MODE" ] && [[ "$CONFIG_FILE" != */* ]] && [[ "$CONFIG_FILE" != *.yaml ]] && [[ "$CONFIG_FILE" != *.yml ]]; then
    MODE="$CONFIG_FILE"
    CONFIG_FILE=""
fi
CONFIG_FILE="${CONFIG_FILE:-configs/finetune_config.yaml}"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    error "Config file '${C_YELLOW}${CONFIG_FILE}${C_RESET}' not found!"
    exit 1
fi

# 预训练权重根目录必填（参与训练/评估的所有 ${PRETRAINED_MODELS_DIR}/<name> 路径都依赖它）
: "${PRETRAINED_MODELS_DIR:?need PRETRAINED_MODELS_DIR in .env (parent dir for paligemma-3b-pt-224, Llama-3.2-11B-Vision-Instruct, pi05_base, etc.)}"

# 缺失的预训练权重自动从 HuggingFace 拉取（设 ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1 关闭）
if [ -n "$MODE" ]; then
    python scripts/download_pretrained.py --config "$CONFIG_FILE" --mode "$MODE" || exit 1
fi

# 使用 Python 脚本解析配置文件并加载到当前 shell 环境
info "Loading configuration from ${C_YELLOW}${CONFIG_FILE}${C_RESET}  mode: ${C_BOLD}${MODE:-<default>}${C_RESET}"
eval "$(python scripts/parse_config.py --config "$CONFIG_FILE" ${MODE:+--mode "$MODE"})"

# 检查是否成功加载配置
if [ -z "$RUN_ID" ]; then
    error "Failed to load configuration"
    exit 1
fi

# =====================================================================
# 检查数据集和基础模型是否存在，缺失则提示用户下载
# =====================================================================

# --- 如果 DATA_ROOT 为空，先询问用户 ---
if [ -z "${DATA_ROOT}" ]; then
    echo ""
    warn "DATA_ROOT is not set (data_root_dir or env var missing)."
    _default="./data"
    read -rp "$(echo -e "${C_BOLD}Please enter dataset root path${C_RESET} [${C_DIM}${_default}${C_RESET}]: ")" DATA_ROOT
    DATA_ROOT="${DATA_ROOT:-${_default}}"
    success "DATA_ROOT set to: ${C_YELLOW}${DATA_ROOT}${C_RESET}"
fi


# --- 基础模型检查 ---
if [ "$MISSING_MODEL" = "1" ]; then
    echo ""
    warn "Base model not found: ${C_YELLOW}${BASE_VLM}${C_RESET}"
    if [ -n "$MODEL_HF_REPO" ]; then
        info "Can download from HuggingFace: ${C_CYAN}${MODEL_HF_REPO}${C_RESET}"
        read -rp "$(echo -e "Download model to '${C_YELLOW}${BASE_VLM}${C_RESET}'? [Y/n] ")" _ans
        _ans="${_ans:-Y}"
        if [[ "$_ans" =~ ^[Yy]$ ]]; then
            info "Downloading model ${C_CYAN}${MODEL_HF_REPO}${C_RESET} ..."
            MODEL_DIR="$(dirname "${BASE_VLM}")"
            mkdir -p "${MODEL_DIR}"
            huggingface-cli download "${MODEL_HF_REPO}" \
                --local-dir "${BASE_VLM}"
            success "Model downloaded to ${C_YELLOW}${BASE_VLM}${C_RESET}"
        else
            error "Aborted. Please download the model manually and retry."
            exit 1
        fi
    else
        error "Unknown model '${MODEL_NAME}', cannot auto-download."
        error "Please download it manually to: ${C_YELLOW}${BASE_VLM}${C_RESET}"
        exit 1
    fi
fi

# --- 数据集检查 ---
if [ "$MISSING_DATA" = "1" ]; then
    echo ""
    warn "Dataset not found under DATA_ROOT: ${C_YELLOW}${DATA_ROOT}${C_RESET}"
    if [ ${#DATA_HF_REPOS[@]} -gt 0 ]; then
        info "Missing datasets: ${C_YELLOW}${DATA_MISSING_NAMES[*]}${C_RESET}"
        info "Will download from HuggingFace: ${C_CYAN}${DATA_HF_REPOS[*]}${C_RESET}"

        _default_dest="${LIBERO_DATA_ROOT:-${DATA_ROOT}}"
        read -rp "$(echo -e "Download destination [${C_DIM}${_default_dest}${C_RESET}]: ")" _dest
        _dest="${_dest:-${_default_dest}}"
        mkdir -p "${_dest}"

        for i in "${!DATA_HF_REPOS[@]}"; do
            _repo="${DATA_HF_REPOS[$i]}"
            _subdir="${DATA_SUBDIRS[$i]}"
            info "Downloading dataset ${C_CYAN}${_repo}${C_RESET} ..."
            huggingface-cli download "${_repo}" \
                --repo-type dataset \
                --local-dir "${_dest}/${_subdir}"
        done

        # 如果下载路径与 DATA_ROOT 不同，更新 DATA_ROOT
        if [ "${_dest}" != "${DATA_ROOT}" ]; then
            DATA_ROOT="${_dest}"
            info "DATA_ROOT updated to: ${C_YELLOW}${DATA_ROOT}${C_RESET}"
        fi

        # 拷贝 modality.json（LIBERO 数据集需要）
        MODALITY_SRC="examples/LIBERO/train_files/modality.json"
        if [ -f "${MODALITY_SRC}" ]; then
            for i in "${!DATA_SUBDIRS[@]}"; do
                _meta_dir="${_dest}/${DATA_SUBDIRS[$i]}/meta"
                if [ -d "${_meta_dir}" ] && [ ! -f "${_meta_dir}/modality.json" ]; then
                    cp "${MODALITY_SRC}" "${_meta_dir}/modality.json"
                    success "Copied modality.json to ${C_DIM}${_meta_dir}${C_RESET}"
                fi
            done
        fi

        success "Dataset download complete."
    else
        error "Unknown datasets, cannot auto-download."
        error "Please prepare data manually under: ${C_YELLOW}${DATA_ROOT}${C_RESET}"
        exit 1
    fi
fi

success "Dependencies check passed."

# 准备输出目录
OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"  # 备份当前脚本
cp "$CONFIG_FILE" "${OUTPUT_DIR}/"  # 备份配置文件

# 训练日志保存到输出目录
TRAIN_LOG="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# 启动训练
echo ""
echo -e "${C_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo -e "  ${C_BOLD_CYAN}▶  Starting Finetune${C_RESET}"
echo -e "  ${C_DIM}Run ID:${C_RESET}    ${C_BOLD_YELLOW}${RUN_ID}${C_RESET}"
echo -e "  ${C_DIM}Base VLM:${C_RESET}  ${C_YELLOW}${BASE_VLM}${C_RESET}"
echo -e "  ${C_DIM}GPUs:${C_RESET}      ${C_YELLOW}${NUM_GPUS}${C_RESET}"
echo -e "  ${C_DIM}Output:${C_RESET}    ${C_DIM}${OUTPUT_DIR}${C_RESET}"
echo -e "  ${C_DIM}Log:${C_RESET}       ${C_DIM}${TRAIN_LOG}${C_RESET}"
echo -e "${C_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo ""

export ACCELERATE_CONFIG_FILE="${DEEPSPEED_CONFIG}"
python -m accelerate.commands.launch \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  AlphaBrain/training/train_alphabrain.py \
  --config_yaml "${CONFIG_FILE}" \
  --mode "${MODE}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 \
  | grep --line-buffered -vE 'gcc -pthread .*compiler_compat' \
  | tee "${TRAIN_LOG}"
