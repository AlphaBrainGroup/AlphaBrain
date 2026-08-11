"""W&B configuration and credential handling for the research console.

The API key deliberately lives outside SQLite and experiment snapshots.  The
only consumer of :meth:`WandbSecretStore.read_api_key` is the process launcher,
which adds it to the child environment after every persistable value has been
resolved and redacted.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import WandbRunConfig

WANDB_API_KEY_ENV = "WANDB_API_KEY"
WANDB_CATEGORIES_ENV = "ALPHABRAIN_WANDB_CATEGORIES"
WANDB_CATEGORY_IDS = ("metrics", "config", "system", "checkpoints", "videos")

_CATEGORY_TEXT: dict[str, dict[str, Any]] = {
    "metrics": {
        "title": {"zh-CN": "训练指标", "en-US": "Training metrics"},
        "description": {
            "zh-CN": "上传 loss、学习率、成功率等训练器已经记录的标量。",
            "en-US": (
                "Upload scalar values already emitted by the trainer, such as loss, learning rate, and success rate."
            ),
        },
    },
    "config": {
        "title": {"zh-CN": "实验配置", "en-US": "Experiment config"},
        "description": {
            "zh-CN": "上传不含密钥的 resolved_config.yaml 快照。",
            "en-US": "Upload the credential-free resolved_config.yaml snapshot.",
        },
    },
    "system": {
        "title": {"zh-CN": "系统指标", "en-US": "System metrics"},
        "description": {
            "zh-CN": "允许 W&B SDK 采集 CPU、RAM 和 GPU 等运行时指标。",
            "en-US": "Allow the W&B SDK to collect runtime CPU, RAM, and GPU metrics.",
        },
    },
    "checkpoints": {
        "title": {"zh-CN": "Checkpoint", "en-US": "Checkpoints"},
        "description": {
            "zh-CN": "当前训练器尚未接入 W&B Artifact, 因此不能选择。",
            "en-US": "The current trainers do not publish W&B Artifacts, so this category is unavailable.",
        },
    },
    "videos": {
        "title": {"zh-CN": "评测视频", "en-US": "Evaluation videos"},
        "description": {
            "zh-CN": "仅 RL Token 的 TD3/PPO 训练器已经记录评测视频。",
            "en-US": "Only the RL Token TD3/PPO trainers currently emit evaluation videos.",
        },
    },
}


class WandbSecretStore:
    """Atomic, owner-only storage for the global W&B API key."""

    filename = "wandb_api_key"

    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.secrets_dir = self.state_dir / "secrets"
        self.path = self.secrets_dir / self.filename

    @staticmethod
    def _validate(value: str) -> str:
        value = value.strip()
        if not 8 <= len(value) <= 1024 or any(character.isspace() or ord(character) < 33 for character in value):
            raise ValueError("W&B API key must contain 8-1024 non-whitespace characters")
        return value

    def _ensure_secrets_dir(self) -> None:
        self.secrets_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = os.lstat(self.secrets_dir)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("W&B secrets path is not a regular directory")
        os.chmod(self.secrets_dir, 0o700)

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(self.secrets_dir, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def set_api_key(self, api_key: str) -> dict[str, bool]:
        value = self._validate(api_key)
        self._ensure_secrets_dir()
        descriptor, temporary_name = tempfile.mkstemp(prefix=".wandb-key-", dir=self.secrets_dir)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(value.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
            self._fsync_directory()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        return {"configured": True}

    def read_api_key(self) -> str | None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise RuntimeError("Unable to open the W&B credential file securely") from error
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError("W&B credential path is not a regular file")
            if stat.S_IMODE(info.st_mode) & 0o077:
                raise PermissionError("W&B credential file must not be accessible by group or other users")
            data = os.read(descriptor, 1025)
            if len(data) > 1024:
                raise RuntimeError("W&B credential file is invalid")
        finally:
            os.close(descriptor)
        try:
            return self._validate(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError("W&B credential file is invalid") from error

    def status(self) -> dict[str, bool]:
        return {"configured": self.read_api_key() is not None}

    def delete_api_key(self) -> dict[str, bool]:
        try:
            info = os.lstat(self.path)
        except FileNotFoundError:
            return {"configured": False}
        if stat.S_ISDIR(info.st_mode):
            raise RuntimeError("W&B credential path is not a regular file")
        self.path.unlink()
        if self.secrets_dir.is_dir():
            self._fsync_directory()
        return {"configured": False}


def normalize_wandb_config(value: Mapping[str, Any] | WandbRunConfig) -> dict[str, Any]:
    """Return a validated, credential-free structure suitable for snapshots."""

    if isinstance(value, WandbRunConfig):
        model = value
    else:
        model = WandbRunConfig.model_validate(dict(value))
    return model.model_dump(mode="json")


def wandb_category_catalog(combination: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Describe what the checked-in trainers can actually upload."""

    launcher = str((combination or {}).get("launcher", ""))
    algorithm = str((combination or {}).get("rl_algorithm") or (combination or {}).get("algorithm", ""))
    videos_supported = launcher == "rl_token" and algorithm in {"td3", "offpolicy", "ppo"}
    result: list[dict[str, Any]] = []
    for category in WANDB_CATEGORY_IDS:
        supported = category in {"metrics", "config", "system"} or (category == "videos" and videos_supported)
        item = {"id": category, "supported": supported, **_CATEGORY_TEXT[category]}
        if not supported:
            if category == "checkpoints":
                item["reason"] = {
                    "zh-CN": "尚未接入 W&B Artifact 上传。",
                    "en-US": "W&B Artifact publishing is not wired yet.",
                }
            elif category == "videos":
                item["reason"] = {
                    "zh-CN": "所选训练器不会生成 W&B 视频记录。",
                    "en-US": "The selected trainer does not emit W&B video records.",
                }
        result.append(item)
    return result


def unsupported_wandb_categories(config: Mapping[str, Any], combination: Mapping[str, Any]) -> list[str]:
    supported = {item["id"] for item in wandb_category_catalog(combination) if item["supported"]}
    if not config.get("enabled") or config.get("mode") == "disabled":
        return []
    return sorted(set(config.get("categories", [])) - supported)


def wandb_environment(config: Mapping[str, Any] | None, *, config_path: str | None = None) -> dict[str, str]:
    """Map a persistable run configuration to non-secret child variables."""

    if config is None:
        return {}
    value = normalize_wandb_config(config)
    enabled = bool(value["enabled"]) and value["mode"] != "disabled"
    categories = set(value["categories"]) if enabled else set()
    environment = {
        WANDB_CATEGORIES_ENV: ",".join(item for item in WANDB_CATEGORY_IDS if item in categories),
        "WANDB_MODE": value["mode"] if enabled else "disabled",
        # Content not represented by the five UI categories stays off.
        "WANDB_DISABLE_CODE": "true",
        "WANDB_DISABLE_GIT": "true",
        "WANDB_SAVE_CODE": "false",
        "WANDB_CONSOLE": "off",
    }
    if not enabled:
        return environment
    environment.update(
        {
            "WANDB_PROJECT": value["project"],
            "WANDB_ENTITY": value["entity"],
            "WANDB_RUN_GROUP": value["group"],
            "WANDB_JOB_TYPE": value["job_type"],
            "WANDB_TAGS": ",".join(value["tags"]),
            "WANDB_NOTES": value["notes"],
            # W&B maps a double underscore to its private x_disable_stats
            # setting. This is the SDK-supported switch used by 0.19.x.
            "WANDB__DISABLE_STATS": "false" if "system" in categories else "true",
        }
    )
    if value["run_name"]:
        environment["WANDB_NAME"] = value["run_name"]
    if "config" in categories and config_path:
        environment["WANDB_CONFIG_PATHS"] = config_path
    return environment


def wandb_requires_api_key(config_or_snapshot: Mapping[str, Any] | None) -> bool:
    if not config_or_snapshot:
        return False
    raw = config_or_snapshot.get("wandb")
    if raw is None and isinstance(config_or_snapshot.get("spec"), Mapping):
        raw = config_or_snapshot["spec"].get("wandb")
    if not isinstance(raw, Mapping):
        return False
    try:
        value = normalize_wandb_config(raw)
    except ValidationError:
        return False
    return bool(value["enabled"]) and value["mode"] == "online"
