from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

from .registry import load_catalog


def _issue(level: str, code: str, zh: str, en: str, **detail: Any) -> dict[str, Any]:
    return {
        "level": level,
        "code": code,
        "message": en,
        "message_i18n": {"zh-CN": zh, "en-US": en},
        "detail": detail,
    }


def _resolve_path(value: str, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return path.resolve(strict=False)


def list_directories(path_value: str | None = None) -> dict[str, Any]:
    """List server-local directories without reading or returning file contents."""

    path = _resolve_path(path_value or str(Path.home()))
    if not path.exists():
        raise ValueError("directory_not_found")
    if not path.is_dir():
        raise ValueError("not_a_directory")
    if not os.access(path, os.R_OK | os.X_OK):
        raise ValueError("directory_not_readable")

    directories: list[dict[str, str]] = []
    try:
        children = sorted(path.iterdir(), key=lambda item: item.name.casefold())
    except OSError as exc:
        raise ValueError("directory_not_readable") from exc
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir():
                directories.append({"name": child.name, "path": str(child.resolve(strict=False))})
        except OSError:
            continue
        if len(directories) >= 1000:
            break
    parent = path.parent if path.parent != path else None
    return {
        "path": str(path),
        "parent": str(parent) if parent is not None else None,
        "directories": directories,
    }


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _validate_jsonl(path: Path, required: set[str]) -> tuple[int, str | None]:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict) or not required.issubset(value):
                    missing = sorted(required - set(value if isinstance(value, dict) else {}))
                    return count, f"line {line_number} is missing fields: {missing}"
                count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return count, str(exc)
    if count == 0:
        return 0, "file is empty"
    return count, None


def _looks_like_parquet(path: Path) -> bool:
    try:
        if path.stat().st_size < 12:
            return False
        with path.open("rb") as handle:
            header = handle.read(4)
            handle.seek(-4, 2)
            footer = handle.read(4)
            return header == b"PAR1" and footer == b"PAR1"
    except OSError:
        return False


def _validate_lerobot_dataset(path: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    info_path = path / "meta/info.json"
    modality_path = path / "meta/modality.json"
    for required in (info_path, modality_path):
        if not required.is_file():
            issues.append(
                _issue(
                    "error",
                    "dataset_required_file_missing",
                    f"缺少 AlphaBrain LeRobot 必需文件：{required.relative_to(path)}",
                    f"Missing required AlphaBrain LeRobot file: {required.relative_to(path)}",
                    path=str(required),
                )
            )

    info: dict[str, Any] = {}
    if info_path.is_file():
        raw_info, error = _read_json(info_path)
        if error or not isinstance(raw_info, dict):
            issues.append(
                _issue(
                    "error",
                    "dataset_invalid_info",
                    f"meta/info.json 不是有效的 JSON 对象：{error or 'invalid object'}",
                    f"meta/info.json is not a valid JSON object: {error or 'invalid object'}",
                    path=str(info_path),
                )
            )
        else:
            info = raw_info
            missing = [key for key in ("features", "data_path", "video_path", "chunks_size") if key not in info]
            if missing:
                issues.append(
                    _issue(
                        "error",
                        "dataset_info_fields_missing",
                        f"meta/info.json 缺少 AlphaBrain dataloader 所需字段：{', '.join(missing)}",
                        f"meta/info.json is missing AlphaBrain dataloader fields: {', '.join(missing)}",
                        path=str(info_path),
                        missing=missing,
                    )
                )

    modality: dict[str, Any] = {}
    if modality_path.is_file():
        raw_modality, error = _read_json(modality_path)
        if error or not isinstance(raw_modality, dict):
            issues.append(
                _issue(
                    "error",
                    "dataset_invalid_modality",
                    f"meta/modality.json 不是有效的 JSON 对象：{error or 'invalid object'}",
                    f"meta/modality.json is not a valid JSON object: {error or 'invalid object'}",
                    path=str(modality_path),
                )
            )
        else:
            modality = raw_modality
            for group in ("state", "action", "video"):
                values = modality.get(group)
                if not isinstance(values, dict) or not values:
                    issues.append(
                        _issue(
                            "error",
                            "dataset_modality_missing",
                            f"meta/modality.json 缺少非空的 {group} 映射。",
                            f"meta/modality.json needs a non-empty {group} mapping.",
                            path=str(modality_path),
                            modality=group,
                        )
                    )
            for group in ("state", "action"):
                values = modality.get(group, {})
                if not isinstance(values, dict):
                    continue
                for name, field in values.items():
                    if not isinstance(field, dict) or not isinstance(field.get("start"), int) or not isinstance(field.get("end"), int) or field["end"] <= field["start"]:
                        issues.append(
                            _issue(
                                "error",
                                "dataset_modality_invalid",
                                f"{group}.{name} 必须包含有效的 start/end 索引。",
                                f"{group}.{name} must contain valid start/end indices.",
                                path=str(modality_path),
                            )
                        )

    version = ""
    episode_count = 0
    tasks_v2 = path / "meta/tasks.jsonl"
    episodes_v2 = path / "meta/episodes.jsonl"
    tasks_v3 = path / "meta/tasks.parquet"
    episodes_v3 = sorted(path.glob("meta/episodes/*/*.parquet"))
    if tasks_v2.is_file() and episodes_v2.is_file():
        version = "LeRobot v2.0"
        _, tasks_error = _validate_jsonl(tasks_v2, {"task_index"})
        episode_count, episodes_error = _validate_jsonl(episodes_v2, {"episode_index", "length"})
        if tasks_error:
            issues.append(_issue("error", "dataset_invalid_tasks", f"任务元数据无效：{tasks_error}", f"Invalid task metadata: {tasks_error}", path=str(tasks_v2)))
        if episodes_error:
            issues.append(_issue("error", "dataset_invalid_episodes", f"Episode 元数据无效：{episodes_error}", f"Invalid episode metadata: {episodes_error}", path=str(episodes_v2)))
    elif tasks_v3.is_file() and episodes_v3:
        version = "LeRobot v3.0"
        parquet_meta = [tasks_v3, *episodes_v3]
        invalid_meta = next((item for item in parquet_meta if not _looks_like_parquet(item)), None)
        if invalid_meta:
            issues.append(_issue("error", "dataset_invalid_parquet", f"元数据 Parquet 文件无效：{invalid_meta}", f"Invalid metadata Parquet file: {invalid_meta}", path=str(invalid_meta)))
        episode_count = int(info.get("total_episodes", 0) or 0)
    else:
        issues.append(
            _issue(
                "error",
                "dataset_episode_metadata_missing",
                "缺少 LeRobot v2 的 tasks.jsonl/episodes.jsonl，或 v3 的 tasks.parquet/episodes Parquet。",
                "Missing LeRobot v2 tasks.jsonl/episodes.jsonl or v3 tasks.parquet/episode Parquet metadata.",
                path=str(path / "meta"),
            )
        )

    parquet_files = sorted(path.glob("data/*/*.parquet"))
    if not parquet_files:
        issues.append(_issue("error", "dataset_data_missing", "data 目录中没有训练 Parquet 文件。", "No training Parquet files were found under data.", path=str(path / "data")))
    else:
        invalid_data = next((item for item in parquet_files if not _looks_like_parquet(item)), None)
        if invalid_data:
            issues.append(_issue("error", "dataset_invalid_parquet", f"训练 Parquet 文件无效：{invalid_data}", f"Invalid training Parquet file: {invalid_data}", path=str(invalid_data)))

    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    video_fields = modality.get("video", {}) if isinstance(modality.get("video"), dict) else {}
    external_video_required = False
    for name, field in video_fields.items():
        original_key = field.get("original_key") if isinstance(field, dict) else None
        feature = features.get(original_key or name)
        if not isinstance(feature, dict):
            issues.append(_issue("error", "dataset_video_feature_missing", f"info.json 的 features 中缺少视频字段：{original_key or name}", f"Video feature is missing from info.json features: {original_key or name}", path=str(info_path)))
        elif feature.get("dtype", "video") != "image":
            external_video_required = True
    if external_video_required:
        videos = [item for item in path.glob("videos/**/*") if item.is_file() and item.suffix.lower() in {".mp4", ".avi", ".mkv", ".webm"}]
        if not videos:
            issues.append(_issue("error", "dataset_video_missing", "视频模态已声明，但 videos 目录中没有视频文件。", "Video modalities are declared, but no video files were found under videos.", path=str(path / "videos")))

    if not (path / "meta/stats_gr00t.json").is_file():
        issues.append(
            _issue(
                "warning",
                "dataset_stats_missing",
                "未找到 meta/stats_gr00t.json；首次训练会计算并写入统计信息。",
                "meta/stats_gr00t.json was not found; the first training run will compute and write statistics.",
                path=str(path / "meta/stats_gr00t.json"),
            )
        )

    return {
        "path": str(path),
        "format": version or "unknown",
        "episode_count": episode_count,
        "parquet_count": len(parquet_files),
        "issues": issues,
        "valid": not any(item["level"] == "error" for item in issues),
    }


def _mixture_patterns(dataset_id: str, dataset_mix: str | None) -> tuple[str, list[str]]:
    catalog = load_catalog()
    datasets = {item["id"]: item for item in catalog.get("components", {}).get("datasets", [])}
    dataset = datasets.get(dataset_id)
    if dataset is None:
        raise ValueError("unknown_dataset")
    mix = dataset_mix or str(dataset.get("default_mix") or "")
    # AlphaBrain.dataloader imports the training stack eagerly (torch,
    # accelerate, ...), which the lightweight UI environment intentionally
    # does not require. Read this data-only registry through the Python AST so
    # validation stays aligned with the trainer without importing it.
    registry_path = Path(__file__).resolve().parents[1] / "AlphaBrain/dataloader/gr00t_lerobot/mixtures.py"
    try:
        tree = ast.parse(registry_path.read_text(encoding="utf-8"), filename=str(registry_path))
        mixtures: dict[str, Any] | None = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "DATASET_NAMED_MIXTURES" for target in node.targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, dict):
                    mixtures = value
                break
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError("dataset_registry_unavailable") from exc
    if mixtures is None:
        raise ValueError("dataset_registry_unavailable")
    entries = mixtures.get(mix)
    if not entries:
        raise ValueError("unknown_dataset_mix")
    return mix, list(dict.fromkeys(str(entry[0]) for entry in entries))


def _matches_for_root(root: Path, patterns: list[str]) -> list[Path] | None:
    matched: list[Path] = []
    for pattern in patterns:
        values = sorted(item.resolve(strict=False) for item in root.glob(pattern) if item.is_dir())
        if not values:
            return None
        matched.extend(values)
    return list(dict.fromkeys(matched))


def validate_dataset_directory(
    path_value: str,
    dataset_id: str,
    dataset_mix: str | None = None,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    selected = _resolve_path(path_value, base_dir)
    issues: list[dict[str, Any]] = []
    if not selected.exists():
        issues.append(_issue("error", "dataset_directory_not_found", f"数据集目录不存在：{selected}", f"Dataset directory does not exist: {selected}", path=str(selected)))
    elif not selected.is_dir():
        issues.append(_issue("error", "dataset_not_a_directory", f"所选路径不是文件夹：{selected}", f"Selected path is not a directory: {selected}", path=str(selected)))
    elif not os.access(selected, os.R_OK | os.X_OK):
        issues.append(_issue("error", "dataset_directory_not_readable", f"数据集目录不可读：{selected}", f"Dataset directory is not readable: {selected}", path=str(selected)))
    if issues:
        return {"valid": False, "path": str(selected), "normalized_root": str(selected), "dataset_id": dataset_id, "dataset_mix": dataset_mix, "format": "unknown", "dataset_count": 0, "episode_count": 0, "parquet_count": 0, "issues": issues}

    try:
        mix, patterns = _mixture_patterns(dataset_id, dataset_mix)
    except ValueError as exc:
        code = str(exc)
        issues.append(_issue("error", code, f"当前数据集配方无法解析：{code}", f"The selected dataset recipe cannot be resolved: {code}"))
        return {"valid": False, "path": str(selected), "normalized_root": str(selected), "dataset_id": dataset_id, "dataset_mix": dataset_mix, "format": "unknown", "dataset_count": 0, "episode_count": 0, "parquet_count": 0, "issues": issues}

    root = selected
    matches = _matches_for_root(root, patterns)
    if matches is None and len(patterns) == 1 and not any(char in patterns[0] for char in "*?["):
        parts = Path(patterns[0]).parts
        if len(selected.parts) >= len(parts) and tuple(selected.parts[-len(parts):]) == parts:
            root = selected
            for _ in parts:
                root = root.parent
            matches = _matches_for_root(root, patterns)

    if matches is None:
        issues.append(
            _issue(
                "error",
                "dataset_mixture_not_found",
                f"目录不符合 {mix} 的布局，未找到：{', '.join(patterns)}",
                f"Directory does not match the {mix} layout; expected: {', '.join(patterns)}",
                path=str(selected),
                expected=patterns,
            )
        )
        return {"valid": False, "path": str(selected), "normalized_root": str(selected), "dataset_id": dataset_id, "dataset_mix": mix, "format": "unknown", "dataset_count": 0, "episode_count": 0, "parquet_count": 0, "issues": issues}

    reports = [_validate_lerobot_dataset(path) for path in matches]
    for report in reports:
        issues.extend(report["issues"])
    formats = sorted({report["format"] for report in reports if report["format"] != "unknown"})
    return {
        "valid": not any(item["level"] == "error" for item in issues),
        "path": str(selected),
        "normalized_root": str(root),
        "dataset_id": dataset_id,
        "dataset_mix": mix,
        "format": ", ".join(formats) if formats else "unknown",
        "dataset_count": len(reports),
        "episode_count": sum(int(report["episode_count"]) for report in reports),
        "parquet_count": sum(int(report["parquet_count"]) for report in reports),
        "issues": issues,
    }
