"""Small subprocess entry points for managed UI utility runs.

Credentials are accepted only through the child environment and are never
placed in argv, SQLite, or logs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def publish_huggingface(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Hugging Face publish token is not configured")
    source = Path(args.source).expanduser().resolve(strict=True)
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=bool(args.private),
        exist_ok=True,
    )
    if source.is_dir():
        api.upload_large_folder(
            folder_path=str(source),
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
        )
    elif source.is_file():
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=source.name,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
        )
    else:
        raise RuntimeError("Publication source is not a file or directory")
    print(f"Published model artifact to https://huggingface.co/{args.repo_id}/tree/{args.revision}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaBrain managed utility tasks")
    subparsers = parser.add_subparsers(dest="task", required=True)

    publish = subparsers.add_parser("publish-huggingface")
    publish.add_argument("--source", required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--revision", default="main")
    publish.add_argument("--private", action="store_true")
    publish.set_defaults(func=publish_huggingface)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
