"""Safe discovery and command construction for repository checkpoint tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

LORA_MODELS = {"qwengr00t", "neurovla", "llamaoft", "paligemma"}
_ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def discover_lora_bundle(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser()
    try:
        if path.is_symlink():
            return {"available": False, "reason": "checkpoint_symlink_not_allowed"}
        path = path.resolve(strict=True)
    except OSError:
        return {"available": False, "reason": "checkpoint_not_found"}

    adapter: Path | None = None
    action: Path | None = None
    prefix: Path | None = None
    if path.is_dir() and (path / "adapter_config.json").is_file() and any(
        (path / name).is_file() for name in _ADAPTER_WEIGHTS
    ):
        adapter = path
        name = path.name
        if name.endswith("_lora_adapter"):
            prefix = path.with_name(name[: -len("_lora_adapter")])
    elif path.is_file() and path.name.endswith("_action_model.pt"):
        prefix = path.with_name(path.name[: -len("_action_model.pt")])
        action = path

    if prefix is not None:
        adapter = adapter or prefix.with_name(prefix.name + "_lora_adapter")
        action = action or prefix.with_name(prefix.name + "_action_model.pt")
    if adapter is None or action is None:
        return {"available": False, "reason": "lora_bundle_not_detected"}
    if adapter.is_symlink() or action.is_symlink():
        return {"available": False, "reason": "checkpoint_symlink_not_allowed"}
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
        return {"available": False, "reason": "lora_adapter_incomplete"}
    if not any((adapter / name).is_file() for name in _ADAPTER_WEIGHTS):
        return {"available": False, "reason": "lora_adapter_weights_missing"}
    if not action.is_file():
        return {"available": False, "reason": "lora_action_model_missing"}
    output = prefix.with_name(prefix.name + "_merged.pt")
    return {
        "available": True,
        "adapter_path": str(adapter.resolve()),
        "action_model_path": str(action.resolve()),
        "suggested_output_path": str(output.resolve(strict=False)),
        "output_exists": output.exists(),
        "models": sorted(LORA_MODELS),
    }


def safe_output_name(value: str) -> str:
    value = value.strip()
    if not value.endswith(".pt") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", value):
        raise ValueError("invalid_checkpoint_output_name")
    return value


def build_lora_merge_command(
    *,
    python_executable: str,
    model: str,
    adapter_path: Path,
    action_model_path: Path,
    output_path: Path,
) -> list[str]:
    if model not in LORA_MODELS:
        raise ValueError("unsupported_lora_model")
    return [
        python_executable,
        "-m",
        "AlphaBrain.training.trainer_utils.peft.merge_lora_checkpoint",
        "--model",
        model,
        "--lora_adapter_dir",
        str(adapter_path),
        "--action_model_pt",
        str(action_model_path),
        "--output_path",
        str(output_path),
    ]
