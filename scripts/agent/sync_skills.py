#!/usr/bin/env python3
"""Synchronize canonical .agents skills into the generated .claude mirror."""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / ".agents" / "skills"
TARGET_ROOT = REPO_ROOT / ".claude" / "skills"


def skill_names() -> list[str]:
    if not SOURCE_ROOT.is_dir():
        raise RuntimeError(f"canonical skill directory does not exist: {SOURCE_ROOT}")
    return sorted(path.name for path in SOURCE_ROOT.iterdir() if path.is_dir() and (path / "SKILL.md").is_file())


def compare_tree(source: Path, target: Path, prefix: str = "") -> list[str]:
    if target.is_symlink():
        return [f"mirror must be a copy, not a symlink: {prefix or source.name}"]
    if not target.is_dir():
        return [f"missing mirror directory: {prefix or source.name}"]
    comparison = filecmp.dircmp(source, target, ignore=["__pycache__", ".DS_Store"])
    differences: list[str] = []
    differences.extend(f"missing in mirror: {prefix}{name}" for name in comparison.left_only)
    differences.extend(f"unexpected in mirror: {prefix}{name}" for name in comparison.right_only)
    differences.extend(f"type mismatch: {prefix}{name}" for name in comparison.common_funny)
    for name in comparison.common_files:
        if (target / name).is_symlink():
            differences.append(f"mirror must be a copy, not a symlink: {prefix}{name}")
        elif not filecmp.cmp(source / name, target / name, shallow=False):
            differences.append(f"content differs: {prefix}{name}")
    for name in comparison.common_dirs:
        differences.extend(compare_tree(source / name, target / name, f"{prefix}{name}/"))
    return differences


def check() -> list[str]:
    differences: list[str] = []
    for name in skill_names():
        differences.extend(compare_tree(SOURCE_ROOT / name, TARGET_ROOT / name, f"{name}/"))
    return differences


def sync() -> None:
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    for name in skill_names():
        source = SOURCE_ROOT / name
        target = TARGET_ROOT / name
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report drift without modifying the mirror")
    args = parser.parse_args()
    if args.check:
        differences = check()
        if differences:
            print("\n".join(differences), file=sys.stderr)
            return 1
        print(f"Skill mirror is current ({len(skill_names())} skills).")
        return 0
    sync()
    differences = check()
    if differences:
        print("\n".join(differences), file=sys.stderr)
        return 1
    print(f"Synchronized {len(skill_names())} skills into {TARGET_ROOT.relative_to(REPO_ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
