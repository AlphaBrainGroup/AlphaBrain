"""Constrained worker commands used by UtilityManager.

This module intentionally does not expose a generic shell-command escape hatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .resource_catalog import LIBERO_REPOS


class ProgressWriter:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._last_percent = -1.0
        self._last_write = 0.0

    @property
    def percent(self) -> float:
        return max(0, self._last_percent)

    def update(
        self,
        percent: float,
        *,
        phase: str,
        current: int | float | None = None,
        total: int | float | None = None,
        unit: str = "",
        message: str = "",
        force: bool = False,
    ) -> None:
        percent = max(0.0, min(100.0, float(percent)))
        now = time.monotonic()
        with self._lock:
            if not force and percent < self._last_percent + 0.1 and now < self._last_write + 0.25:
                return
            payload: dict[str, Any] = {
                "percent": round(percent, 2),
                "phase": phase,
                "unit": unit,
            }
            if current is not None:
                payload["current"] = current
            if total is not None:
                payload["total"] = total
            if message:
                payload["message"] = message
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, self.path)
            self._last_percent = percent
            self._last_write = now


def _safe_hf_etag(etag: str | None) -> str | None:
    """Return an ETag that is safe to use as part of a Windows filename."""

    if etag is None:
        return None
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", etag)


@contextmanager
def _windows_safe_hf_etags():  # type: ignore[no-untyped-def]
    """Work around Hub/proxy ETags containing Windows-reserved characters."""

    if os.name != "nt":
        yield
        return
    from huggingface_hub import file_download

    original = file_download._normalize_etag

    def normalize(etag: str | None) -> str | None:
        return _safe_hf_etag(original(etag))

    file_download._normalize_etag = normalize
    try:
        yield
    finally:
        file_download._normalize_etag = original


def _download_error_message(exc: Exception) -> str:
    message = str(exc).splitlines()[0].strip()
    return f"{type(exc).__name__}: {message}"[:500]


def _download_progress_class(
    writer: ProgressWriter,
    *,
    group_index: int,
    group_count: int,
    message: str,
):  # type: ignore[no-untyped-def]
    from huggingface_hub.utils.tqdm import tqdm as huggingface_tqdm

    class DownloadProgress(huggingface_tqdm):
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self._alphabrain_track = str(kwargs.get("desc") or "").startswith("Fetching ")
            super().__init__(*args, **kwargs)
            self._alphabrain_report()

        def _alphabrain_report(self) -> None:
            if not self._alphabrain_track:
                return
            total = int(self.total or 0)
            current = min(int(self.n), total) if total else int(self.n)
            fraction = current / total if total else 0.0
            percent = 5.0 + 90.0 * (group_index + fraction) / max(1, group_count)
            writer.update(
                percent,
                phase="downloading",
                current=current,
                total=total,
                unit="files",
                message=message,
            )

        def update(self, n=1):  # type: ignore[no-untyped-def]
            result = super().update(n)
            self._alphabrain_report()
            return result

        def close(self) -> None:
            self._alphabrain_report()
            super().close()

    return DownloadProgress


def _remove_incomplete_destination(destination: Path, sentinels: tuple[str, ...]) -> bool:
    if not destination.exists():
        return False
    if destination.is_dir() and any((destination / name).exists() for name in sentinels):
        return True
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    else:
        shutil.rmtree(destination)
    return False


def download_pretrained(
    repo_id: str,
    target: Path,
    run_id: str,
    progress_path: Path,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required") from exc
    writer = ProgressWriter(progress_path)
    writer.update(1, phase="preparing", message=repo_id, force=True)
    if _remove_incomplete_destination(target, ("config.json", "model.safetensors")):
        writer.update(100, phase="completed", message=repo_id, force=True)
        return
    temporary = target.parent / f".{target.name}.partial-{run_id}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _windows_safe_hf_etags():
            snapshot_download(
                repo_id=repo_id,
                local_dir=temporary,
                token=os.environ.get("HF_TOKEN"),
                tqdm_class=_download_progress_class(
                    writer,
                    group_index=0,
                    group_count=1,
                    message=repo_id,
                ),
            )
        writer.update(98, phase="finalizing", message=repo_id, force=True)
        os.replace(temporary, target)
        writer.update(100, phase="completed", message=repo_id, force=True)
    except Exception as exc:
        writer.update(
            max(1, writer.percent),
            phase="failed",
            message=_download_error_message(exc),
            force=True,
        )
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def copy_tree(source: Path, target: Path, run_id: str) -> None:
    if not source.is_dir():
        raise SystemExit("source directory does not exist")
    if target.exists():
        raise SystemExit("target directory already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.partial-{run_id}"
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(source, temporary, symlinks=True)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def dataset_stats(source: Path, output: Path) -> None:
    files = 0
    size = 0
    digest = hashlib.sha256()
    for root, _directories, names in os.walk(source):
        for name in sorted(names):
            path = Path(root) / name
            try:
                info = path.stat()
            except OSError:
                continue
            relative = str(path.relative_to(source))
            files += 1
            size += info.st_size
            digest.update(relative.encode())
            digest.update(str(info.st_size).encode())
            digest.update(str(info.st_mtime_ns).encode())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps({"file_count": files, "size_bytes": size, "fingerprint": digest.hexdigest()}), encoding="utf-8")
    os.replace(temporary, output)


def download_libero(target: Path, repo_root: Path, run_id: str, progress_path: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required") from exc
    writer = ProgressWriter(progress_path)
    writer.update(1, phase="preparing", message="LIBERO", force=True)
    target.mkdir(parents=True, exist_ok=True)
    temporary_root = target / f".libero.partial-{run_id}"
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir(parents=True)
    modality = repo_root / "benchmarks/LIBERO/train/modality.json"
    token = os.environ.get("HF_TOKEN")
    pending: list[tuple[str, Path, Path]] = []
    try:
        for index, repo in enumerate(LIBERO_REPOS):
            name = repo.rsplit("/", 1)[-1]
            destination = target / name
            if _remove_incomplete_destination(destination, ("meta/info.json",)):
                continue
            staged = temporary_root / name
            with _windows_safe_hf_etags():
                snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    local_dir=staged,
                    token=token,
                    tqdm_class=_download_progress_class(
                        writer,
                        group_index=index,
                        group_count=len(LIBERO_REPOS),
                        message=repo,
                    ),
                )
            metadata = staged / "meta"
            metadata.mkdir(parents=True, exist_ok=True)
            if modality.is_file():
                shutil.copy2(modality, metadata / "modality.json")
            pending.append((repo, staged, destination))
        writer.update(98, phase="finalizing", message="LIBERO", force=True)
        for _repo, staged, destination in pending:
            os.replace(staged, destination)
        writer.update(100, phase="completed", message="LIBERO", force=True)
    except Exception as exc:
        writer.update(
            max(1, writer.percent),
            phase="failed",
            message=_download_error_message(exc),
            force=True,
        )
        raise
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def archive_path(source: Path, output: Path, run_id: str, progress_path: Path) -> None:
    if source.is_symlink() or not (source.is_file() or source.is_dir()):
        raise SystemExit("archive source is invalid")
    writer = ProgressWriter(progress_path)
    writer.update(1, phase="scanning", message=source.name, force=True)
    files = [source] if source.is_file() else [
        path
        for path in source.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    total = sum(path.stat().st_size for path in files)
    completed = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.partial-{run_id}"
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for path in files:
                relative = Path(source.name) if source.is_file() else Path(source.name) / path.relative_to(source)
                info = zipfile.ZipInfo.from_file(path, arcname=relative.as_posix())
                info.compress_type = zipfile.ZIP_DEFLATED
                with path.open("rb") as source_stream, archive.open(info, "w", force_zip64=True) as target_stream:
                    while chunk := source_stream.read(1024 * 1024):
                        target_stream.write(chunk)
                        completed += len(chunk)
                        fraction = completed / total if total else 1.0
                        writer.update(
                            2 + 96 * fraction,
                            phase="archiving",
                            current=completed,
                            total=total,
                            unit="bytes",
                            message=source.name,
                        )
        os.replace(temporary, output)
        writer.update(
            100,
            phase="completed",
            current=total,
            total=total,
            unit="bytes",
            message=source.name,
            force=True,
        )
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    copy_parser = commands.add_parser("copy-tree")
    copy_parser.add_argument("--source", required=True)
    copy_parser.add_argument("--target", required=True)
    copy_parser.add_argument("--run-id", required=True)
    stats_parser = commands.add_parser("dataset-stats")
    stats_parser.add_argument("--source", required=True)
    stats_parser.add_argument("--output", required=True)
    libero_parser = commands.add_parser("download-libero")
    libero_parser.add_argument("--target", required=True)
    libero_parser.add_argument("--repo-root", required=True)
    libero_parser.add_argument("--run-id", required=True)
    libero_parser.add_argument("--progress", required=True)
    pretrained_parser = commands.add_parser("download-pretrained")
    pretrained_parser.add_argument("--repo-id", required=True)
    pretrained_parser.add_argument("--target", required=True)
    pretrained_parser.add_argument("--run-id", required=True)
    pretrained_parser.add_argument("--progress", required=True)
    archive_parser = commands.add_parser("archive")
    archive_parser.add_argument("--source", required=True)
    archive_parser.add_argument("--output", required=True)
    archive_parser.add_argument("--run-id", required=True)
    archive_parser.add_argument("--progress", required=True)
    args = parser.parse_args()
    if args.command == "copy-tree":
        copy_tree(Path(args.source), Path(args.target), args.run_id)
    elif args.command == "dataset-stats":
        dataset_stats(Path(args.source), Path(args.output))
    elif args.command == "download-libero":
        download_libero(
            Path(args.target),
            Path(args.repo_root),
            args.run_id,
            Path(args.progress),
        )
    elif args.command == "download-pretrained":
        download_pretrained(
            args.repo_id,
            Path(args.target),
            args.run_id,
            Path(args.progress),
        )
    else:
        archive_path(
            Path(args.source),
            Path(args.output),
            args.run_id,
            Path(args.progress),
        )


if __name__ == "__main__":
    main()
