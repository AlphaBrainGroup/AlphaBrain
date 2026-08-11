"""Truthful resource inventory and safe command construction."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PRETRAINED_SCRIPT = "scripts/download_pretrained.py"

LIBERO_REPOS = (
    "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
    "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
    "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
    "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
)

WORLD_MODEL_RESOURCES: dict[str, dict[str, Any]] = {
    "world_model.cosmos_predict2": {"name": "Cosmos-Predict2-2B-Video2World", "preprocess": ["t5"]},
    "world_model.cosmos_reason1": {"name": "Cosmos-Reason1-7B", "preprocess": ["reason1", "reason1_projection"]},
    "world_model.vjepa2": {"name": "V-JEPA 2.1", "preprocess": ["t5"]},
    "world_model.wan22": {"name": "Wan2.2-TI2V-5B", "preprocess": ["umt5"]},
}

PREPROCESS_SCRIPTS = {
    "t5": "scripts/run_world_model/preprocess/precompute_text_embeddings/precompute_t5.py",
    "reason1": "scripts/run_world_model/preprocess/precompute_text_embeddings/precompute_reason1.py",
    "umt5": "scripts/run_world_model/preprocess/precompute_text_embeddings/precompute_umt5.py",
    "reason1_projection": "scripts/run_world_model/preprocess/extract_nvidia_reason1_proj.py",
}


def _pretrained_registry(repo_root: Path) -> dict[str, dict[str, str]]:
    path = repo_root / PRETRAINED_SCRIPT
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "REGISTRY":
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return value
    raise RuntimeError("pretrained_registry_unavailable")


def _present_model(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in ("config.json", "model.safetensors"))


def resource_catalog(repo_root: Path, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    environment = settings.get("environment", {}) if isinstance(settings.get("environment"), dict) else {}
    root_value = str(environment.get("PRETRAINED_MODELS_DIR") or settings.get("pretrained_root") or "")
    pretrained_root = Path(root_value).expanduser().resolve(strict=False) if root_value else None
    records: list[dict[str, Any]] = []
    for name, spec in _pretrained_registry(repo_root).items():
        target = pretrained_root / name if pretrained_root else None
        repo_id = spec["hf_repo"]
        records.append(
            {
                "id": f"pretrained.{name}", "kind": "pretrained_model", "name": name,
                "source": repo_id, "installable": True, "requires_admin": True,
                "requires_hf_token": repo_id.startswith(("google/paligemma", "meta-llama/")),
                "install_root": str(pretrained_root) if pretrained_root else None,
                "target_path": str(target) if target else None,
                "status": "installed" if target and _present_model(target) else ("missing" if target else "unconfigured"),
                "dependencies": [],
            }
        )

    libero_root_value = str(environment.get("LEROBOT_LIBERO_DATA_DIR") or environment.get("LIBERO_DATA_ROOT") or "")
    libero_root = Path(libero_root_value).expanduser().resolve(strict=False) if libero_root_value else None
    expected = [libero_root / repo.rsplit("/", 1)[-1] for repo in LIBERO_REPOS] if libero_root else []
    records.append(
        {
            "id": "dataset.libero", "kind": "dataset", "name": "LIBERO LeRobot (4 suites)",
            "source": list(LIBERO_REPOS), "installable": True, "requires_admin": True,
            "install_root": str(libero_root) if libero_root else None,
            "target_path": str(libero_root) if libero_root else None,
            "status": "installed" if expected and all((item / "meta/info.json").is_file() for item in expected) else ("missing" if libero_root else "unconfigured"),
            "dependencies": [],
        }
    )

    paths = settings.get("resource_paths", {}) if isinstance(settings.get("resource_paths"), dict) else {}
    for resource_id, definition in WORLD_MODEL_RESOURCES.items():
        raw_path = str(paths.get(resource_id) or "")
        path = Path(raw_path).expanduser().resolve(strict=False) if raw_path else None
        records.append(
            {
                "id": resource_id, "kind": "world_model", "name": definition["name"],
                "source": None, "installable": False, "registerable": True, "requires_admin": True,
                "target_path": str(path) if path else None,
                "status": "installed" if path and path.is_dir() and any(path.iterdir()) else ("missing" if path else "unconfigured"),
                "preprocess": definition["preprocess"], "dependencies": [],
            }
        )
    return records


def install_command(resource_id: str, *, repo_root: Path, target_root: Path) -> tuple[list[str], dict[str, str], str]:
    if resource_id.startswith("pretrained."):
        name = resource_id.removeprefix("pretrained.")
        definition = _pretrained_registry(repo_root).get(name)
        if definition is None:
            raise ValueError("unknown_resource")
        target = target_root / name
        return (
            [
                sys.executable,
                "-m",
                "alphabrain_ui.utility_worker",
                "download-pretrained",
                "--repo-id",
                definition["hf_repo"],
                "--target",
                str(target),
                "--run-id",
                "{run_id}",
                "--progress",
                "{progress_path}",
            ],
            {},
            str(target),
        )
    if resource_id == "dataset.libero":
        return (
            [
                sys.executable,
                "-m",
                "alphabrain_ui.utility_worker",
                "download-libero",
                "--target",
                str(target_root),
                "--repo-root",
                str(repo_root),
                "--run-id",
                "{run_id}",
                "--progress",
                "{progress_path}",
            ],
            {}, str(target_root),
        )
    raise ValueError("resource_not_installable")


def preprocess_command(kind: str, *, repo_root: Path, inputs: Mapping[str, str]) -> tuple[list[str], dict[str, str], str]:
    script = PREPROCESS_SCRIPTS.get(kind)
    if not script or not (repo_root / script).is_file():
        raise ValueError("unknown_preprocess_kind")
    if kind == "reason1_projection":
        source = str(inputs.get("src") or "")
        destination = str(inputs.get("dst") or "")
        if not source or not destination:
            raise ValueError("projection_source_and_destination_required")
        return [sys.executable, str(repo_root / script), "--src", source, "--dst", destination], {}, destination
    environment = {str(key).upper(): str(value) for key, value in inputs.items() if value}
    output = environment.get("OUTPUT_PATH") or environment.get("OUTPUT_DIR") or ""
    return [sys.executable, str(repo_root / script)], environment, output
