"""
Ensure pretrained backbones are present under $PRETRAINED_MODELS_DIR.

Missing model directories are downloaded from HuggingFace via
``huggingface_hub.snapshot_download``.

Usage
-----
Auto-detect from a finetune_config mode (used by ``scripts/run_finetune.sh``)::

    python scripts/download_pretrained.py --config configs/finetune_config.yaml \
        --mode paligemma_pi05_openpi_aligned_v3

Explicit list (manual prefetch)::

    python scripts/download_pretrained.py --names paligemma-3b-pt-224 pi05_base

Set ``ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1`` to skip downloads — the script then
reports what *would* have been fetched and exits 0, letting the trainer surface
the missing-weight error instead.

Gated repos (PaliGemma, Llama) require ``HF_TOKEN`` in the environment.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable

import yaml


REGISTRY: dict[str, dict[str, str]] = {
    "paligemma-3b-pt-224":          {"hf_repo": "google/paligemma-3b-pt-224"},
    "Llama-3.2-11B-Vision-Instruct": {"hf_repo": "meta-llama/Llama-3.2-11B-Vision-Instruct"},
    "Qwen2.5-VL-3B-Instruct":       {"hf_repo": "Qwen/Qwen2.5-VL-3B-Instruct"},
    "Qwen3-VL-4B-Instruct":         {"hf_repo": "Qwen/Qwen3-VL-4B-Instruct"},
    "pi05_base":                    {"hf_repo": "lerobot/pi05_base"},
}

_NAME_RE = re.compile(
    r"\$\{(?:oc\.env:)?PRETRAINED_MODELS_DIR(?::-[^}]*)?\}/([A-Za-z0-9_.\-]+)"
)


def _is_present(target_dir: str) -> bool:
    """True iff ``target_dir`` looks like a complete download.

    Heuristic: directory exists and contains either ``config.json`` (HF
    standard) or ``model.safetensors`` (pi05_base layout). A bare empty dir
    or one with only partial files counts as absent.
    """
    if not os.path.isdir(target_dir):
        return False
    sentinels = ("config.json", "model.safetensors")
    return any(os.path.exists(os.path.join(target_dir, s)) for s in sentinels)


def _names_from_mode(config_path: str, mode: str) -> list[str]:
    """Walk the chosen mode and its base model config; collect required registry names.

    Two patterns are recognised:
      1. ``${PRETRAINED_MODELS_DIR}/<name>`` (or the ``${oc.env:...}`` form) anywhere
         in the mode/model YAML.
      2. A top-level ``base_vlm: <bare-name>`` — ``scripts/parse_config.py`` resolves
         these by joining with ``PRETRAINED_MODELS_DIR`` (see parse_config.py:102).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    modes = cfg.get("modes", {})
    if mode not in modes:
        raise SystemExit(
            f"[download] mode {mode!r} not found in {config_path}. "
            f"Available: {sorted(modes)}"
        )

    mode_cfg = modes[mode]
    targets: list[dict | list | str] = [mode_cfg]

    model_key = mode_cfg.get("model") or cfg.get("defaults", {}).get("model")
    if model_key:
        model_yaml = os.path.join("configs", "models", f"{model_key}.yaml")
        if os.path.exists(model_yaml):
            with open(model_yaml) as f:
                targets.append(yaml.safe_load(f) or {})

    found: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name in REGISTRY and name not in seen:
            seen.add(name)
            found.append(name)

    def _walk(node):
        if isinstance(node, str):
            for m in _NAME_RE.finditer(node):
                _add(m.group(1))
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    for t in targets:
        _walk(t)

    # Mirror parse_config.py:102 — top-level bare `base_vlm` is implicitly
    # joined with PRETRAINED_MODELS_DIR.
    bare = mode_cfg.get("base_vlm", "")
    if (
        isinstance(bare, str)
        and bare
        and not os.path.isabs(bare)
        and not bare.startswith("./")
        and not bare.startswith("data/")
        and "${" not in bare
    ):
        _add(bare)

    return found


def _download(name: str, root: str) -> None:
    spec = REGISTRY[name]
    target = os.path.join(root, name)

    if _is_present(target):
        print(f"[skip] {name} already present at {target}")
        return

    if os.environ.get("ALPHABRAIN_DISABLE_AUTO_DOWNLOAD") == "1":
        print(
            f"[disabled] would download {name} ← {spec['hf_repo']} "
            f"(ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1)"
        )
        return

    repo = spec["hf_repo"]
    print(f"[download] {name} ← {repo} → {target}", flush=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "[download] huggingface_hub is required for auto-download. "
            "Install with `pip install huggingface_hub`."
        ) from e

    os.makedirs(target, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=target,
        local_dir_use_symlinks=False,
    )
    print(f"[ok] {name} ready at {target}")


def ensure(names: Iterable[str]) -> None:
    root = os.environ.get("PRETRAINED_MODELS_DIR")
    if not root:
        raise SystemExit(
            "[download] PRETRAINED_MODELS_DIR is not set. "
            "Export it (or put it in .env) before calling this script."
        )
    for name in names:
        if name not in REGISTRY:
            print(f"[warn] no registry entry for {name!r}, skipping", file=sys.stderr)
            continue
        _download(name, root)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--names", nargs="+", help="Explicit model dir names to ensure.")
    g.add_argument("--config", help="finetune_config YAML to scan.")
    p.add_argument("--mode", help="Mode name (required with --config).")
    args = p.parse_args()

    if args.config:
        if not args.mode:
            p.error("--mode is required with --config")
        names = _names_from_mode(args.config, args.mode)
        if not names:
            print(f"[download] no registered models referenced by mode {args.mode!r}; nothing to do")
            return
    else:
        names = args.names

    ensure(names)


if __name__ == "__main__":
    main()
