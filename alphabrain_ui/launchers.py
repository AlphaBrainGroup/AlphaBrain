from __future__ import annotations

import copy
import json
import os
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass
class LaunchStage:
    name: str
    phase: str
    command: list[str]
    environment: dict[str, str]
    cwd: str
    output_dir: str
    metrics_path: str
    requested_gpu_count: int
    requested_gpu_ids: list[int] = field(default_factory=list)
    dependency_position: int | None = None
    config_snapshot_path: str = ""

    def redacted_preview(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "command": self.command,
            "environment": {
                key: ("[REDACTED]" if any(x in key.upper() for x in ("TOKEN", "KEY", "SECRET", "PASSWORD")) else val)
                for key, val in self.environment.items()
            },
            "cwd": self.cwd,
            "output_dir": self.output_dir,
            "requested_gpu_count": self.requested_gpu_count,
            "requested_gpu_ids": self.requested_gpu_ids,
        }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _value(spec: dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        current: Any = spec
        found = True
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found and current is not None:
            return current
    return default


def normalize_family(spec: dict[str, Any]) -> str:
    raw = str(
        _value(
            spec,
            "training.family",
            "training.method",
            "method",
            "family",
            default="imitation_learning",
        )
    ).lower()
    aliases = {
        "imitation": "imitation_learning",
        "supervised": "imitation_learning",
        "finetune": "imitation_learning",
        "neurovla": "neurovla_pretrain",
        "stdp": "r_stdp",
        "continual": "continual_learning",
        "cl": "continual_learning",
        "world": "world_model",
        "rlt": "rl_token",
        "rlt_a": "rl_token",
        "rl": "rl_token",
    }
    return aliases.get(raw, raw)


def sanitize_run_id(value: str) -> str:
    value = value.strip().replace(" ", "-")
    if not RUN_ID_RE.fullmatch(value):
        raise ValueError("run_id must use only letters, numbers, dot, underscore, and dash")
    return value


def find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resources(spec: dict[str, Any]) -> tuple[int, list[int]]:
    count = int(_value(spec, "resources.gpu_count", "resources.num_gpus", "gpu_count", default=1))
    ids = [int(x) for x in _value(spec, "resources.gpu_ids", "gpu_ids", default=[]) or []]
    if ids:
        count = len(ids)
    if count < 1:
        raise ValueError("gpu_count must be at least 1")
    return count, ids


def _base_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "PYTHONUNBUFFERED": "1",
        "NO_COLOR": "1",
        "ALPHABRAIN_DISABLE_AUTO_DOWNLOAD": "1",
        "ALPHABRAIN_UI_LAUNCH": "1",
    }
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def _expert_dict(spec: dict[str, Any]) -> dict[str, Any]:
    raw = spec.get("expert_overrides", {})
    if isinstance(raw, str):
        loaded = yaml.safe_load(raw) or {}
        if not isinstance(loaded, dict):
            raise ValueError("expert_overrides YAML must be a mapping")
        return loaded
    if not isinstance(raw, dict):
        raise ValueError("expert_overrides must be a mapping or YAML mapping")
    return raw


def _write_finetune_snapshot(
    repo_root: Path,
    spec: dict[str, Any],
    run_id: str,
    gpu_count: int,
    target: Path,
) -> tuple[str, Path]:
    source = repo_root / str(_value(spec, "config_file", "training.config_file", default="configs/finetune_config.yaml"))
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    mode = str(_value(spec, "training.mode", "training.template_mode", "mode", "base_mode", default="qwen_oft"))
    if mode not in config.get("modes", {}):
        raise ValueError(f"training mode '{mode}' is not present in {source}")
    mode_cfg = config["modes"][mode]
    mode_cfg["run_id"] = run_id
    mode_cfg["num_gpus"] = gpu_count
    mode_cfg["main_process_port"] = int(_value(spec, "resources.port", default=find_free_port()))

    params = spec.get("parameters", {}) if isinstance(spec.get("parameters", {}), dict) else {}
    training = params.get("training", {}) if isinstance(params.get("training", {}), dict) else {}
    for key in (
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "max_train_steps",
        "save_interval",
        "eval_interval",
        "freeze_modules",
        "pretrained_checkpoint",
    ):
        value = training.get(key, params.get(key))
        if value is not None:
            mode_cfg.setdefault("training", {})[key] = value

    dataset = spec.get("dataset", {}) if isinstance(spec.get("dataset", {}), dict) else {}
    dataset_mix = dataset.get("mix") or dataset.get("id") or params.get("dataset_mix")
    if dataset_mix:
        mode_cfg["dataset_mix"] = dataset_mix
        mode_cfg.setdefault("datasets", {}).setdefault("vla_data", {})["dataset_mix"] = dataset_mix
    if dataset.get("root"):
        mode_cfg["data_root"] = dataset["root"]
        mode_cfg.setdefault("datasets", {}).setdefault("vla_data", {})["data_root_dir"] = dataset["root"]

    architecture = spec.get("architecture", {}) if isinstance(spec.get("architecture", {}), dict) else {}
    if isinstance(architecture.get("framework"), dict):
        _deep_merge(mode_cfg.setdefault("framework", {}), architecture["framework"])
    if isinstance(params.get("framework"), dict):
        _deep_merge(mode_cfg.setdefault("framework", {}), params["framework"])
    if isinstance(params.get("trainer"), dict):
        _deep_merge(mode_cfg.setdefault("trainer", {}), params["trainer"])
    _deep_merge(mode_cfg, _expert_dict(spec))

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return mode, target


def _resolved_snapshot_plan(
    repo_root: Path,
    config_dir: Path,
    spec: dict[str, Any],
    *,
    write_snapshots: bool,
    common_env: dict[str, str],
) -> list[LaunchStage] | None:
    """Use the capability resolver's canonical config and command contract."""
    resolved_config = spec.get("resolved_config")
    preview = spec.get("command_preview")
    if not isinstance(resolved_config, dict) or not isinstance(preview, dict):
        return None

    target = config_dir / "resolved_config.yaml"
    if write_snapshots:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (target.parent / "experiment_snapshot.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    else:
        # Preview paths are placeholders and are never executed.
        target = Path("${RESOLVED_CONFIG_PATH}")

    gpu_count, gpu_ids = _resources(spec)
    run_id = sanitize_run_id(
        str(resolved_config.get("run_id") or _value(spec, "parameters.run_id", default="experiment"))
    )
    output_root = Path(str(resolved_config.get("output_root_dir", "results/training"))).expanduser()
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    base_output = (output_root / run_id).resolve()

    def render(value: str, config_target: Path = target) -> str:
        return str(value).replace("${RESOLVED_CONFIG_PATH}", str(config_target)).replace("${ALLOCATED_GPU_IDS}", "{gpu_ids}")

    def stage_from(
        stage_preview: dict[str, Any],
        position: int,
        id_to_position: dict[str, int],
    ) -> LaunchStage:
        stage_id = str(stage_preview.get("id") or ("train" if position == 0 else f"stage_{position}"))
        phase = stage_id
        stage_target = target
        stage_config = stage_preview.get("resolved_config")
        if isinstance(stage_config, dict):
            stage_target = config_dir / f"resolved_config.{stage_id}.yaml" if write_snapshots else Path("${RESOLVED_CONFIG_PATH}")
            if write_snapshots:
                stage_target.parent.mkdir(parents=True, exist_ok=True)
                stage_target.write_text(yaml.safe_dump(stage_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        stage_env = dict(common_env)
        stage_env.update({str(key): render(str(value), stage_target) for key, value in preview.get("environment", {}).items()})
        stage_env.update({str(key): render(str(value), stage_target) for key, value in stage_preview.get("environment", {}).items()})
        raw_stage_output = stage_preview.get("output_dir")
        if raw_stage_output:
            stage_output = Path(str(raw_stage_output)).expanduser()
            if not stage_output.is_absolute():
                stage_output = repo_root / stage_output
            stage_output = stage_output.resolve()
        else:
            stage_output = base_output if len(preview.get("stages", [])) <= 1 else base_output / stage_id
        stage_env.setdefault("RUN_TAG", run_id)
        if stage_id == "encoder_pretrain":
            stage_output = base_output / "pretrain"
        if stage_id.startswith(("rl_", "encoder_")) or stage_id == "vla_ppo":
            stage_env["OUTPUT_DIR"] = str(stage_output)
            dependencies = stage_preview.get("depends_on", [])
            if dependencies and not stage_env.get("ENCODER_PATH"):
                stage_env["ENCODER_PATH"] = str(base_output / "pretrain/checkpoints/pretrain_best/encoder.pt")
        argv = [render(str(item), stage_target) for item in stage_preview.get("argv", preview.get("argv", []))]
        dependencies = stage_preview.get("depends_on", [])
        dependency_position = id_to_position.get(str(dependencies[0])) if dependencies else None
        default_single_gpu = stage_id.startswith(("rl_", "encoder_")) or stage_id == "vla_ppo"
        requested_gpu_count = int(stage_preview.get("requested_gpu_count", 1 if default_single_gpu else gpu_count))
        return LaunchStage(
            name=stage_id.replace("_", " ").title(),
            phase=phase,
            command=argv,
            environment=stage_env,
            cwd=str(preview.get("working_directory", repo_root)),
            output_dir=str(stage_output),
            metrics_path=str(stage_output / "metrics.jsonl"),
            requested_gpu_count=requested_gpu_count,
            requested_gpu_ids=gpu_ids[:requested_gpu_count] if gpu_ids else [],
            dependency_position=dependency_position,
            config_snapshot_path=str(stage_target),
        )

    stage_previews = preview.get("stages")
    if isinstance(stage_previews, list) and stage_previews:
        positions = {str(item.get("id")): index for index, item in enumerate(stage_previews)}
        return [stage_from(item, index, positions) for index, item in enumerate(stage_previews)]

    launcher = str(preview.get("launcher_id", normalize_family(spec)))
    phase = {
        "neuro_stdp": "stdp",
        "neuro_pretrain": "pretrain",
        "continual_learning": "continual_learning",
        "world_model": "world_model",
        "cosmos_policy": "world_model",
        "vanilla_vla_ppo": "vla_ppo",
    }.get(launcher, "train")
    stage_preview = {"id": phase, "argv": preview.get("argv", []), "environment": preview.get("environment", {})}
    stage = stage_from(stage_preview, 0, {})
    if launcher == "vanilla_vla_ppo":
        stage.output_dir = str(base_output / "vla_ppo")
        stage.metrics_path = str(Path(stage.output_dir) / "metrics.jsonl")
        stage.environment["OUTPUT_DIR"] = stage.output_dir
    return [stage]


def _write_recipe_snapshot(repo_root: Path, source: str, overrides: dict[str, Any], target: Path) -> Path:
    path = repo_root / source
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _deep_merge(config, overrides)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


def build_launch_plan(
    repo_root: Path,
    state_dir: Path,
    name: str,
    spec: dict[str, Any],
    *,
    snapshot_key: str = "preview",
    environment: dict[str, str] | None = None,
    write_snapshots: bool = False,
) -> list[LaunchStage]:
    """Turn the UI experiment spec into existing, CLI-compatible launch stages."""
    family = normalize_family(spec)
    run_id = sanitize_run_id(str(_value(spec, "run_id", "parameters.run_id", default=name)))
    gpu_count, gpu_ids = _resources(spec)
    config_dir = state_dir / "configs" / snapshot_key
    common_env = _base_environment(environment)
    preferred_port = int(_value(spec, "resources.port", "resources.main_process_port", default=find_free_port()))
    common_env["MASTER_PORT"] = "{main_process_port}"
    common_env["ALPHABRAIN_UI_PREFERRED_PORT"] = str(preferred_port)

    canonical = _resolved_snapshot_plan(
        repo_root,
        config_dir,
        spec,
        write_snapshots=write_snapshots,
        common_env=common_env,
    )
    if canonical is not None:
        return canonical

    if family in {"imitation_learning", "finetune"}:
        target = config_dir / "resolved_config.yaml"
        if write_snapshots:
            mode, snapshot = _write_finetune_snapshot(repo_root, spec, run_id, gpu_count, target)
        else:
            mode = str(_value(spec, "training.mode", "mode", "base_mode", default="qwen_oft"))
            snapshot = repo_root / str(_value(spec, "config_file", default="configs/finetune_config.yaml"))
        out = (repo_root / str(_value(spec, "output_root", default="results/training")) / run_id).resolve()
        return [
            LaunchStage(
                name="Training",
                phase="train",
                command=["bash", "-o", "pipefail", "scripts/run_finetune.sh", str(snapshot), "--mode", mode],
                environment=common_env,
                cwd=str(repo_root),
                output_dir=str(out),
                metrics_path=str(out / "metrics.jsonl"),
                requested_gpu_count=gpu_count,
                requested_gpu_ids=gpu_ids,
                config_snapshot_path=str(snapshot),
            )
        ]

    if family in {"neurovla_pretrain", "r_stdp"}:
        target = config_dir / "resolved_config.yaml"
        adjusted = copy.deepcopy(spec)
        adjusted.setdefault("training", {})["mode"] = "neuro_vla_stdp" if family == "r_stdp" else "neuro_vla"
        if write_snapshots:
            mode, snapshot = _write_finetune_snapshot(repo_root, adjusted, run_id, gpu_count, target)
        else:
            mode = adjusted["training"]["mode"]
            snapshot = repo_root / "configs/finetune_config.yaml"
        script = (
            "scripts/run_brain_inspired_scripts/run_stdp_finetune.sh"
            if family == "r_stdp"
            else "scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh"
        )
        params = spec.get("parameters", {})
        dataset = str(_value(spec, "dataset.id", "dataset.mix", default="libero_goal"))
        command = [
            "bash",
            "-o",
            "pipefail",
            script,
            "--mode",
            mode,
            "--dataset",
            dataset,
            "--gpus",
            str(gpu_count),
            "--run-id",
            run_id,
        ]
        for flag, key in (
            ("--batch-size", "batch_size"),
            ("--steps", "max_train_steps"),
            ("--attn", "attn_implementation"),
        ):
            value = params.get(key)
            if value is not None:
                command.extend([flag, str(value)])
        if family == "r_stdp":
            checkpoint = _value(spec, "checkpoint", "training.checkpoint", "parameters.pretrained_checkpoint")
            if checkpoint:
                command.extend(["--pretrained", str(checkpoint)])
        env = dict(common_env)
        env["CONFIG_YAML"] = str(snapshot)
        env["CONDA_ENV"] = os.environ.get("CONDA_DEFAULT_ENV", "")
        out = (repo_root / str(_value(spec, "output_root", default="results/training")) / run_id).resolve()
        return [
            LaunchStage(
                name="R-STDP Fine-tuning" if family == "r_stdp" else "NeuroVLA Pre-training",
                phase="stdp" if family == "r_stdp" else "pretrain",
                command=command,
                environment=env,
                cwd=str(repo_root),
                output_dir=str(out),
                metrics_path=str(out / "metrics.jsonl"),
                requested_gpu_count=gpu_count,
                requested_gpu_ids=gpu_ids,
                config_snapshot_path=str(snapshot),
            )
        ]

    if family == "continual_learning":
        training = spec.get("training", {})
        model = str(
            training.get("model")
            or _value(
                spec,
                "architecture.framework",
                "architecture.backbone",
                default="qwengr00t",
            )
        )
        algo = str(training.get("algorithm", "er"))
        dataset = str(_value(spec, "dataset.id", "dataset.mix", default="libero_goal"))
        command = [
            "bash",
            "-o",
            "pipefail",
            "scripts/run_continual_learning_scripts/run_cl_train.sh",
            "--model",
            model,
            "--algo",
            algo,
            "--dataset",
            dataset,
            "--gpus",
            str(gpu_count),
            "--run-id",
            run_id,
            "--port",
            "{main_process_port}",
        ]
        out = (repo_root / str(_value(spec, "output_root", default="results/Checkpoints")) / run_id).resolve()
        return [
            LaunchStage(
                name="Continual Learning",
                phase="continual_learning",
                command=command,
                environment=common_env,
                cwd=str(repo_root),
                output_dir=str(out),
                metrics_path=str(out / "metrics.jsonl"),
                requested_gpu_count=gpu_count,
                requested_gpu_ids=gpu_ids,
            )
        ]

    if family in {"world_model", "cosmos_policy"}:
        if family == "cosmos_policy":
            script = "scripts/run_world_model/train/run_cosmos_policy.sh"
            env = dict(common_env)
            env.update(
                {
                    "GPU_IDS": "{gpu_ids}",
                    "NUM_GPUS": str(gpu_count),
                    "RUN_ID": run_id,
                    "MASTER_PORT": "{main_process_port}",
                }
            )
            params = spec.get("parameters", {})
            for ui_key, env_key in (
                ("batch_size", "PER_DEVICE_BATCH"),
                ("gradient_accumulation_steps", "GRAD_ACCUM"),
                ("max_train_steps", "MAX_STEPS"),
                ("save_interval", "SAVE_INTERVAL"),
            ):
                if params.get(ui_key) is not None:
                    env[env_key] = str(params[ui_key])
            out = (repo_root / "results/training" / run_id).resolve()
            return [
                LaunchStage(
                    name="Cosmos Policy",
                    phase="world_model",
                    command=["bash", "-o", "pipefail", script],
                    environment=env,
                    cwd=str(repo_root),
                    output_dir=str(out),
                    metrics_path=str(out / "metrics.jsonl"),
                    requested_gpu_count=gpu_count,
                    requested_gpu_ids=gpu_ids,
                )
            ]
        model = str(_value(spec, "training.variant", "architecture.backbone", "world_model", default="cos2"))
        aliases = {"cosmos2": "cos2", "cosmos25": "cos25_4gpu", "v-jepa": "vjepa", "wan2.2": "wan22"}
        model = aliases.get(model.lower(), model)
        env = dict(common_env)
        env.update({"MODEL": model, "NUM_GPUS": str(gpu_count), "MASTER_PORT": "{main_process_port}"})
        out = (repo_root / "results/training" / run_id).resolve()
        # World recipes carry their own run_id. A UI snapshot makes that identity immutable.
        recipe_source = str(_value(spec, "training.config_file", default=f"configs/models/config_{model}.yaml"))
        target = config_dir / "resolved_config.yaml"
        if write_snapshots:
            overrides = {"run_id": run_id, "output_root_dir": str(repo_root / "results/training")}
            _deep_merge(overrides, _expert_dict(spec))
            snapshot = _write_recipe_snapshot(repo_root, recipe_source, overrides, target)
        else:
            snapshot = repo_root / recipe_source
        env["CONFIG_YAML"] = str(snapshot)
        return [
            LaunchStage(
                name="World Model Training",
                phase="world_model",
                command=["bash", "-o", "pipefail", "scripts/run_world_model/train/run_world_model.sh"],
                environment=env,
                cwd=str(repo_root),
                output_dir=str(out),
                metrics_path=str(out / "metrics.jsonl"),
                requested_gpu_count=gpu_count,
                requested_gpu_ids=gpu_ids,
                config_snapshot_path=str(snapshot),
            )
        ]

    if family in {"rl_token", "vla_ppo"}:
        training = spec.get("training", {})
        track = str(training.get("track", "rlt"))
        algorithm = str(training.get("algorithm", "offpolicy"))
        checkpoint = str(_value(spec, "checkpoint", "training.checkpoint", default=""))
        encoder = str(_value(spec, "training.encoder_checkpoint", "encoder_checkpoint", default=""))
        full_pipeline = bool(training.get("full_pipeline", not encoder and family != "vla_ppo"))
        stages: list[LaunchStage] = []
        pretrain_out = (repo_root / "results/rlt_training" / run_id / "pretrain").resolve()
        if full_pipeline:
            env = dict(common_env)
            env.update(
                {
                    "TRACK": track,
                    "CKPT_PATH": checkpoint,
                    "RUN_TAG": run_id,
                    "OUTPUT_DIR": str(pretrain_out),
                }
            )
            stages.append(
                LaunchStage(
                    name="RL Encoder Pre-training",
                    phase="rl_pretrain",
                    command=["bash", "-o", "pipefail", "scripts/run_rl_scripts/run_rlt_pretrain.sh", "{first_gpu}"],
                    environment=env,
                    cwd=str(repo_root),
                    output_dir=str(pretrain_out),
                    metrics_path=str(pretrain_out / "metrics.jsonl"),
                    requested_gpu_count=1,
                    requested_gpu_ids=gpu_ids[:1],
                )
            )
            encoder = str(pretrain_out / "checkpoints/pretrain_best/encoder.pt")

        if family == "vla_ppo" or algorithm == "vla_ppo":
            script = "scripts/run_rl_scripts/run_qwen_vla_ppo.sh"
            phase = "vla_ppo"
        elif algorithm == "ppo":
            script, phase = "scripts/run_rl_scripts/run_rlt_a_ppo.sh", "rl_ppo"
        elif algorithm == "grpo":
            script, phase = "scripts/run_rl_scripts/run_rlt_a_grpo.sh", "rl_grpo"
        else:
            script, phase = "scripts/run_rl_scripts/run_rlt_rl.sh", "rl_offpolicy"
        rl_out = (repo_root / "results/rlt_training" / run_id / phase).resolve()
        env = dict(common_env)
        env.update(
            {
                "TRACK": track,
                "CKPT_PATH": checkpoint,
                "ENCODER_PATH": encoder,
                "RUN_TAG": run_id,
                "OUTPUT_DIR": str(rl_out),
                "BACKBONE": str(training.get("backbone", "qwen")),
                "TASK_ID": str(training.get("task_id", 0)),
            }
        )
        stages.append(
            LaunchStage(
                name="RL Training",
                phase=phase,
                command=["bash", "-o", "pipefail", script, "{first_gpu}"],
                environment=env,
                cwd=str(repo_root),
                output_dir=str(rl_out),
                metrics_path=str(rl_out / "metrics.jsonl"),
                requested_gpu_count=1,
                requested_gpu_ids=gpu_ids[:1],
                dependency_position=0 if full_pipeline else None,
            )
        )
        return stages

    raise ValueError(f"No launcher adapter for training family '{family}'")
