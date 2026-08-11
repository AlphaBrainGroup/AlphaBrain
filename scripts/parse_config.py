"""
配置文件解析工具
用于从 YAML 配置文件中读取训练/评估参数并输出为 shell 变量

支持功能:
  - 训练模式: 输出 accelerate launch 所需的全部变量
  - 评估模式: 输出 eval 脚本所需的全部变量
  - ${ENV_VAR:-default} 语法的环境变量展开
  - paths.pretrained_models_dir 与 base_vlm 的自动拼接
"""
import argparse
import json
import os
import re
import shlex
import sys

import yaml


def expand_env_vars(value: str) -> str:
    """展开 ${ENV_VAR:-default} 语法的环境变量"""
    if not isinstance(value, str):
        return value

    def _replace(match):
        var_name = match.group(1)
        default = match.group(3)  # group(3) is the default after :-
        return os.environ.get(var_name, default if default is not None else "")

    # Match ${VAR} or ${VAR:-default}
    return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(:-(.*?))?\}', _replace, value)


def parse_config(config_path: str, mode: str):
    """解析配置文件并输出 shell 变量"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 检查模式是否存在
    if mode not in config['modes']:
        print(f"Error: Mode '{mode}' not found in config", file=sys.stderr)
        print(f"Available modes: {', '.join(config['modes'].keys())}", file=sys.stderr)
        sys.exit(1)

    mode_config = config['modes'][mode]
    common_config = config.get('common', {})
    env_config = config.get('environment', {})
    paths_config = config.get('paths', {})

    # 展开 paths 段中的环境变量
    pretrained_models_dir = expand_env_vars(paths_config.get('pretrained_models_dir', 'data/pretrained_models'))
    # Mode-level deepspeed_config overrides global defaults
    deepspeed_config = mode_config.get('deepspeed_config',
                        paths_config.get('deepspeed_config',
                        common_config.get('deepspeed_config', '')))

    # 判断是 eval 模式还是 train 模式
    is_eval = mode_config.get('type') == 'eval'

    # 输出环境变量
    print(
        "export WANDB_MODE="
        + shlex.quote(str(mode_config.get('wandb_mode', env_config.get('wandb_mode', 'disabled'))))
    )

    # NCCL 配置
    nccl = env_config.get('nccl', {})
    if 'ib_hca' in nccl:
        print(f"export NCCL_IB_HCA={shlex.quote(str(nccl['ib_hca']))}")
    if 'blocking_wait' in nccl:
        print(f"export TORCH_NCCL_BLOCKING_WAIT={shlex.quote(str(nccl['blocking_wait']))}")
    if 'async_error_handling' in nccl:
        print(f"export TORCH_NCCL_ASYNC_ERROR_HANDLING={shlex.quote(str(nccl['async_error_handling']))}")
    if 'timeout' in nccl:
        print(f"export NCCL_TIMEOUT={shlex.quote(str(nccl['timeout']))}")
    if 'socket_timeout_ms' in nccl:
        print(f"export NCCL_SOCKET_TIMEOUT_MS={shlex.quote(str(nccl['socket_timeout_ms']))}")

    if is_eval:
        # === 评估模式输出 ===
        def eval_int(name, default):
            value = mode_config.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Eval mode {name} must be an integer")
            return value

        checkpoint = expand_env_vars(str(mode_config.get('checkpoint', '')))
        benchmark = expand_env_vars(str(mode_config.get('benchmark', 'libero')))
        task_suite = expand_env_vars(
            str(mode_config.get('task_suite', mode_config.get('task_set', 'libero_goal')))
        )
        use_bf16 = mode_config.get('use_bf16', True)
        if not isinstance(use_bf16, bool):
            raise ValueError("Eval mode use_bf16 must be a boolean")
        sort_tasks = mode_config.get('sort_tasks', True)
        if not isinstance(sort_tasks, bool):
            raise ValueError("Eval mode sort_tasks must be a boolean")
        reuse_server = mode_config.get('reuse_server', False)
        if not isinstance(reuse_server, bool):
            raise ValueError("Eval mode reuse_server must be a boolean")
        print(f"EVAL_CHECKPOINT={shlex.quote(checkpoint)}")
        print(f"EVAL_BENCHMARK={shlex.quote(benchmark)}")
        print(f"TASK_SUITE={shlex.quote(task_suite)}")
        num_trials = eval_int('num_trials', 50)
        print(f"NUM_TRIALS={num_trials}")
        print(f"EVAL_HOST={shlex.quote(expand_env_vars(str(mode_config.get('host', '127.0.0.1'))))}")
        print(f"EVAL_PORT={eval_int('port', 5694)}")
        print(f"EVAL_GPU_ID={eval_int('gpu_id', 0)}")
        print(f"EVAL_USE_BF16={'true' if use_bf16 else 'false'}")
        print(f"EVAL_REUSE_SERVER={'true' if reuse_server else 'false'}")
        print(f"EVAL_SERVER_PYTHON={shlex.quote(expand_env_vars(str(mode_config.get('server_python', ''))))}")
        server_entrypoint = expand_env_vars(
            str(mode_config.get(
                'server_entrypoint',
                'deployment/model_server/server_policy.py',
            ))
        )
        server_args = mode_config.get('server_args', [])
        if not isinstance(server_args, list) or not all(
            isinstance(item, (str, int, float)) and not isinstance(item, bool)
            for item in server_args
        ):
            raise ValueError("Eval mode server_args must be a list of scalar values")
        server_args = [expand_env_vars(str(item)) for item in server_args]
        if any('\n' in item or '\r' in item for item in server_args):
            raise ValueError("Eval mode server_args may not contain newlines")
        print(f"EVAL_SERVER_ENTRYPOINT={shlex.quote(server_entrypoint)}")
        print(f"EVAL_SERVER_ARGS_JSON={shlex.quote(json.dumps(server_args))}")
        print(f"EVAL_CLIENT_PYTHON={shlex.quote(expand_env_vars(str(mode_config.get('client_python', ''))))}")
        print(f"EVAL_CLIENT_PYTHONPATH={shlex.quote(expand_env_vars(str(mode_config.get('client_pythonpath', ''))))}")
        print(
            f"EVAL_CLIENT_ENTRYPOINT={shlex.quote(expand_env_vars(str(mode_config.get('client_entrypoint', ''))))}"
        )
        print(f"EVAL_ENV_NAME={shlex.quote(expand_env_vars(str(mode_config.get('env_name', ''))))}")
        print(f"EVAL_TASK_SET={shlex.quote(expand_env_vars(str(mode_config.get('task_set', ''))))}")
        print(f"EVAL_SPLIT={shlex.quote(expand_env_vars(str(mode_config.get('split', ''))))}")
        print(f"EVAL_NUM_EPISODES={eval_int('n_episodes', num_trials)}")
        print(f"EVAL_NUM_ENVS={eval_int('n_envs', 1)}")
        print(f"EVAL_MAX_EPISODE_STEPS={eval_int('max_episode_steps', 720)}")
        print(f"EVAL_N_ACTION_STEPS={eval_int('n_action_steps', 12)}")
        print(f"EVAL_TASK_IDS={shlex.quote(expand_env_vars(str(mode_config.get('task_ids', ''))))}")
        print(f"EVAL_TASK_LIST={shlex.quote(expand_env_vars(str(mode_config.get('task_list', ''))))}")
        # Keep an explicit UI-managed EVAL_TASK_LIMIT when an older generated
        # config omits the field; run_eval.sh defaults it to zero otherwise.
        if 'task_limit' in mode_config:
            print(f"EVAL_TASK_LIMIT={eval_int('task_limit', 0)}")
        print(f"EVAL_SEED={eval_int('seed', 7)}")
        print(f"EVAL_SEED_CONFIGURED={'true' if 'seed' in mode_config else 'false'}")
        print(f"EVAL_NUM_VIEWS={eval_int('num_views', 2)}")
        print(f"EVAL_PREDICT_VIDEO={'true' if bool(mode_config.get('predict_video', False)) else 'false'}")
        print(f"EVAL_SORT_TASKS={'true' if sort_tasks else 'false'}")
        print(
            f"EVAL_CONFIG_OUTPUT_DIR={shlex.quote(expand_env_vars(str(mode_config.get('output_dir', ''))))}"
        )
        print(f"EVAL_NUMBA_DISABLE_JIT={eval_int('numba_disable_jit', 1)}")
        print(f"EVAL_MUJOCO_GL={shlex.quote(expand_env_vars(str(mode_config.get('mujoco_gl', 'egl'))))}")
        print(f"EVAL_PYOPENGL_PLATFORM={shlex.quote(expand_env_vars(str(mode_config.get('pyopengl_platform', 'egl'))))}")
        print("IS_EVAL=true")
    else:
        # === 训练模式输出 ===
        # base_vlm: 如果是相对路径，拼接 pretrained_models_dir
        base_vlm_raw = mode_config.get('base_vlm', '')
        if not os.path.isabs(base_vlm_raw) and not base_vlm_raw.startswith('./') and not base_vlm_raw.startswith('data/'):
            base_vlm = os.path.join(pretrained_models_dir, base_vlm_raw)
        else:
            base_vlm = base_vlm_raw

        # 展开 data_root 中的环境变量
        data_root = expand_env_vars(mode_config.get('data_root', ''))

        print(f"RUN_ID='{mode_config['run_id']}'")
        print(f"FRAMEWORK_NAME='{mode_config.get('framework_name', '')}'")
        print(f"MODE='{mode}'")
        print(f"BASE_VLM='{base_vlm}'")
        # config_yaml is optional: not needed when all config is inlined in finetune_config.yaml
        print(f"CONFIG_YAML='{mode_config.get('config_yaml', '')}'")
        print(f"DATA_ROOT='{data_root}'")
        print(f"DATASET_MIX='{mode_config.get('dataset_mix', '')}'")
        print(f"OUTPUT_ROOT_DIR='{common_config.get('output_root_dir', './results/training')}'")
        print(f"NUM_GPUS={mode_config.get('num_gpus', common_config.get('num_gpus', 2))}")
        print(f"MAIN_PROCESS_PORT={os.environ.get('MASTER_PORT', mode_config.get('main_process_port', common_config.get('main_process_port', 29500)))}")
        print(f"DEEPSPEED_CONFIG='{deepspeed_config}'")
        print(f"TRAIN_PYTHON='{expand_env_vars(mode_config.get('python_env', ''))}'")

        # 训练参数
        training = mode_config.get('training', {})
        print(f"PER_DEVICE_BATCH_SIZE={training.get('per_device_batch_size', 2)}")
        print(f"GRAD_ACC_STEPS={training.get('gradient_accumulation_steps', 1)}")
        print(f"MAX_TRAIN_STEPS={training.get('max_train_steps', 80000)}")
        print(f"SAVE_INTERVAL={training.get('save_interval', 10000)}")
        print(f"EVAL_INTERVAL={training.get('eval_interval', 100)}")
        print(f"FREEZE_MODULES='{training.get('freeze_modules', '')}'")

        # 额外参数（转换为数组）
        extra_args = mode_config.get('extra_args', [])
        if extra_args:
            args_str = ' '.join(f'"{arg}"' for arg in extra_args)
            print(f"EXTRA_ARGS=({args_str})")
        else:
            print("EXTRA_ARGS=()")

        print("IS_EVAL=false")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse finetune config')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--mode', required=False, default=None,
                        help='Mode name (e.g., qwen_oft, neuro_vla, libero_eval). '
                             'Defaults to defaults.model in the config file.')

    args = parser.parse_args()

    # 未指定 mode 时，依次尝试: defaults.mode → defaults.model → modes 中第一个非 eval 的 mode
    if args.mode is None:
        with open(args.config, 'r') as f:
            _cfg = yaml.safe_load(f)
        defaults = _cfg.get('defaults', {})
        args.mode = defaults.get('mode') or defaults.get('model')
        if not args.mode:
            # 取 modes 中第一个非 eval 类型的 mode
            for name, m in _cfg.get('modes', {}).items():
                if m.get('type') != 'eval':
                    args.mode = name
                    break
        if not args.mode:
            print("Error: --mode not specified and no default mode found in config", file=sys.stderr)
            sys.exit(1)
        print(f"# Using default mode from config: {args.mode}", file=sys.stderr)

    parse_config(args.config, args.mode)
    
    
    
