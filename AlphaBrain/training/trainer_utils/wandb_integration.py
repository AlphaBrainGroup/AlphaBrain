"""Small W&B adapter shared by AlphaBrain's heterogeneous trainers.

The UI passes only non-secret preferences through environment variables.  This
adapter makes those preferences win over older hard-coded ``wandb.init``
arguments and filters ``wandb.log`` records by the selected content categories.
When the UI marker is absent, behavior is unchanged for command-line users.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping
from typing import Any

_CATEGORIES_ENV = "ALPHABRAIN_WANDB_CATEGORIES"
_WRAPPED_MARKER = "_alphabrain_content_filter_installed"


def selected_categories() -> set[str] | None:
    raw = os.environ.get(_CATEGORIES_ENV)
    if raw is None:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def category_enabled(category: str) -> bool:
    selected = selected_categories()
    return selected is None or category in selected


def _looks_like_video(key: Any, value: Any) -> bool:
    normalized = str(key).lower()
    if normalized.startswith(("video/", "videos/")) or normalized.endswith(("/video", "/videos")):
        return True
    return value.__class__.__name__ == "Video" and value.__class__.__module__.startswith("wandb")


def _override_init_metadata(kwargs: dict[str, Any]) -> None:
    mapping = {
        "WANDB_PROJECT": "project",
        "WANDB_ENTITY": "entity",
        "WANDB_NAME": "name",
        "WANDB_RUN_GROUP": "group",
        "WANDB_JOB_TYPE": "job_type",
        "WANDB_NOTES": "notes",
    }
    for environment_name, argument_name in mapping.items():
        if environment_name not in os.environ:
            continue
        value = os.environ[environment_name]
        kwargs[argument_name] = value or None
    if "WANDB_TAGS" in os.environ:
        kwargs["tags"] = [item for item in os.environ["WANDB_TAGS"].split(",") if item]


def configure_wandb_module(wandb_module: Any) -> None:
    """Install an idempotent metadata/content filter on an imported W&B SDK."""

    if getattr(wandb_module, _WRAPPED_MARKER, False):
        return
    original_init = wandb_module.init
    original_log = wandb_module.log

    @functools.wraps(original_init)
    def init(*args: Any, **kwargs: Any) -> Any:
        _override_init_metadata(kwargs)
        if not category_enabled("config"):
            kwargs.pop("config", None)
        return original_init(*args, **kwargs)

    @functools.wraps(original_log)
    def log(data: Any = None, *args: Any, **kwargs: Any) -> Any:
        metrics_enabled = category_enabled("metrics")
        videos_enabled = category_enabled("videos")
        if isinstance(data, Mapping):
            filtered = {
                key: value
                for key, value in data.items()
                if (videos_enabled if _looks_like_video(key, value) else metrics_enabled)
            }
            if not filtered:
                return None
            data = filtered
        elif not metrics_enabled:
            return None
        return original_log(data, *args, **kwargs)

    wandb_module.init = init
    wandb_module.log = log
    setattr(wandb_module, _WRAPPED_MARKER, True)
