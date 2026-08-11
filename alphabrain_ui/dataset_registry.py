"""Dataset registration helpers that avoid importing the heavy training stack."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .datasets import _validate_lerobot_dataset


_BUILDER_SUPPORT = {
    "lerobot": "direct",
    "lerobot_collection": "mixture_only",
    "cosmos": "inventory_only",
    "vlm_json": "inventory_only",
    "unknown": "unsupported",
}


def _with_builder_support(report: dict[str, Any]) -> dict[str, Any]:
    support = _BUILDER_SUPPORT.get(str(report.get("format_family", "unknown")), "unsupported")
    return {**report, "builder_support": support, "builder_ready": support == "direct"}


def resolve_local_directory(value: str, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve(strict=False)
    if not path.exists() or not path.is_dir():
        raise ValueError("dataset_directory_not_found")
    if not os.access(path, os.R_OK | os.X_OK):
        raise ValueError("dataset_directory_not_readable")
    return path


def is_within_roots(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    found = False
    files = [
        path / "meta/info.json",
        path / "meta/modality.json",
        path / "meta/tasks.jsonl",
        path / "meta/episodes.jsonl",
        path / "success_only/dataset_statistics.json",
    ]
    for relative in (
        "*/meta/info.json",
        "*/meta/modality.json",
        "*/meta/tasks.jsonl",
        "*/meta/episodes.jsonl",
    ):
        files.extend(sorted(path.glob(relative)))
    for item in files:
        if not item.is_file():
            continue
        found = True
        stat_result = item.stat()
        digest.update(str(item.relative_to(path)).encode())
        digest.update(str(stat_result.st_size).encode())
        digest.update(str(stat_result.st_mtime_ns).encode())
        if stat_result.st_size <= 2 * 1024 * 1024:
            digest.update(item.read_bytes())
    if not found:
        digest.update(str(path).encode())
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    total = 0
    for root, _directories, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _json_preview(path: Path, limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as stream:
                for raw in stream:
                    if raw.strip():
                        value = json.loads(raw)
                        if isinstance(value, dict):
                            result.append(value)
                        if len(result) >= limit:
                            break
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            rows = value if isinstance(value, list) else [value]
            result = [row for row in rows if isinstance(row, dict)][:limit]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    # Avoid returning arbitrary prompts or large embedded values.  Dataset
    # preview exposes field names and small scalar metadata only.
    return [
        {str(key): value for key, value in row.items() if isinstance(value, (str, int, float, bool)) and len(str(value)) <= 256}
        for row in result
    ]


def inspect_dataset(path: Path) -> dict[str, Any]:
    if (path / "meta/info.json").is_file():
        report = _validate_lerobot_dataset(path)
        report.update(
            {
                "format_family": "lerobot",
                "compatible_loaders": ["lerobot", "gr00t_lerobot", "paligemma"],
                "fingerprint": dataset_fingerprint(path),
                "size_bytes": directory_size(path),
            }
        )
        try:
            info = json.loads((path / "meta/info.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            info = {}
        report["step_count"] = int(info.get("total_frames", info.get("total_steps", 0)) or 0)
        return _with_builder_support(report)

    lerobot_children = sorted(item for item in path.iterdir() if item.is_dir() and (item / "meta/info.json").is_file())
    if lerobot_children:
        reports = [inspect_dataset(item) for item in lerobot_children]
        issues = [
            {**issue, "dataset_path": report["path"]}
            for report in reports
            for issue in report.get("issues", [])
        ]
        return _with_builder_support({
            "path": str(path), "format": "LeRobot collection", "format_family": "lerobot_collection",
            "compatible_loaders": ["lerobot", "gr00t_lerobot", "paligemma"],
            "dataset_count": len(reports),
            "datasets": [{"path": report["path"], "format": report["format"], "valid": report["valid"]} for report in reports],
            "episode_count": sum(int(report.get("episode_count", 0)) for report in reports),
            "step_count": sum(int(report.get("step_count", 0)) for report in reports),
            "parquet_count": sum(int(report.get("parquet_count", 0)) for report in reports),
            "issues": issues, "valid": all(report["valid"] for report in reports),
            "fingerprint": dataset_fingerprint(path), "size_bytes": directory_size(path),
        })

    cosmos_stats = path / "success_only/dataset_statistics.json"
    cosmos_hdf5 = list(path.glob("**/*.hdf5")) + list(path.glob("**/*.h5"))
    if cosmos_stats.is_file() and cosmos_hdf5:
        return _with_builder_support({
            "path": str(path), "format": "Cosmos LIBERO", "format_family": "cosmos",
            "compatible_loaders": ["cosmos"], "episode_count": len(cosmos_hdf5),
            "parquet_count": 0, "issues": [], "valid": True,
            "fingerprint": dataset_fingerprint(path), "size_bytes": directory_size(path),
        })

    annotations = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))
    image_dirs = [item for item in (path / "images", path / "image", path / "data") if item.is_dir()]
    if annotations and image_dirs:
        valid_annotations = any(_json_preview(item, 1) for item in annotations)
        issues = [] if valid_annotations else [{"level": "error", "code": "dataset_invalid_annotations", "message": "No valid JSON annotation records were found."}]
        return _with_builder_support({
            "path": str(path), "format": "VLM / LLaVA JSON", "format_family": "vlm_json",
            "compatible_loaders": ["vlm", "llava_json"], "episode_count": 0,
            "parquet_count": 0, "issues": issues, "valid": not issues,
            "fingerprint": dataset_fingerprint(path), "size_bytes": directory_size(path),
        })

    return _with_builder_support({
        "path": str(path), "format": "unknown", "format_family": "unknown",
        "compatible_loaders": [], "episode_count": 0, "parquet_count": 0,
        "issues": [{"level": "error", "code": "dataset_format_unknown", "message": "The directory does not match a supported AlphaBrain dataset format."}],
        "valid": False, "fingerprint": dataset_fingerprint(path), "size_bytes": directory_size(path),
    })


def preview_dataset(path: Path, *, limit: int = 20) -> dict[str, Any]:
    report = inspect_dataset(path)
    samples: list[dict[str, Any]] = []
    if report["format_family"] == "lerobot":
        episodes = path / "meta/episodes.jsonl"
        tasks = path / "meta/tasks.jsonl"
        samples = _json_preview(episodes, limit)
        task_rows = _json_preview(tasks, limit)
    elif report["format_family"] == "lerobot_collection":
        samples = []
        task_rows = []
        for child in sorted(item for item in path.iterdir() if item.is_dir() and (item / "meta/info.json").is_file()):
            for row in _json_preview(child / "meta/episodes.jsonl", limit - len(samples)):
                samples.append({"dataset": child.name, **row})
            for row in _json_preview(child / "meta/tasks.jsonl", limit - len(task_rows)):
                task_rows.append({"dataset": child.name, **row})
            if len(samples) >= limit and len(task_rows) >= limit:
                break
    elif report["format_family"] == "vlm_json":
        annotation = next(iter(sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))), None)
        samples = _json_preview(annotation, limit) if annotation else []
        task_rows = []
    else:
        task_rows = []
    return {"summary": report, "episodes": samples, "tasks": task_rows}
