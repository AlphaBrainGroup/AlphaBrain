from __future__ import annotations

import base64
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SSH_HOST = re.compile(r"^[A-Za-z0-9._:-]+$")
_SSH_USER = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class RemoteTrainingConfig:
    enabled: bool = False
    host: str = ""
    user: str = ""
    port: int = 22
    repo_root: str = ""
    identity_file: str = ""
    gpu_ids: tuple[int, ...] = ()
    setup_command: str = ""

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "RemoteTrainingConfig":
        return cls(
            enabled=bool(settings.get("remote_training_enabled", False)),
            host=str(settings.get("remote_training_host", "") or "").strip(),
            user=str(settings.get("remote_training_user", "") or "").strip(),
            port=int(settings.get("remote_training_port", 22) or 22),
            repo_root=str(settings.get("remote_training_repo_root", "") or "").strip().rstrip("/"),
            identity_file=str(settings.get("remote_training_identity_file", "") or "").strip(),
            gpu_ids=tuple(int(value) for value in settings.get("remote_training_gpu_ids", []) or []),
            setup_command=str(settings.get("remote_training_setup_command", "") or "").strip(),
        )

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def ssh_command(self, *, connect_timeout: int = 10) -> list[str]:
        command = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={connect_timeout}",
            "-o",
            "StrictHostKeyChecking=yes",
            "-p",
            str(self.port),
        ]
        if self.identity_file:
            command.extend(["-i", str(Path(self.identity_file).expanduser())])
        command.extend([self.target, "bash", "-s", "--"])
        return command


def validate_remote_training_config(config: RemoteTrainingConfig) -> None:
    if not 1 <= config.port <= 65535:
        raise ValueError("remote_training_port_invalid")
    if len(set(config.gpu_ids)) != len(config.gpu_ids) or any(value < 0 or value > 1024 for value in config.gpu_ids):
        raise ValueError("remote_training_gpu_ids_invalid")
    if config.host and not _SSH_HOST.fullmatch(config.host):
        raise ValueError("remote_training_host_invalid")
    if config.user and not _SSH_USER.fullmatch(config.user):
        raise ValueError("remote_training_user_invalid")
    if config.enabled:
        if not config.host:
            raise ValueError("remote_training_host_required")
        if not config.repo_root.startswith("/") or "\x00" in config.repo_root or "\n" in config.repo_root:
            raise ValueError("remote_training_repo_root_invalid")
        if not config.gpu_ids:
            raise ValueError("remote_training_gpu_ids_required")


def map_local_value(value: str, local_repo_root: Path, remote_repo_root: str) -> str:
    """Map repository-local paths embedded in arguments or config text."""
    resolved = str(local_repo_root.resolve())
    result = value
    for local_prefix in {resolved, resolved.replace("\\", "/")}:
        if local_prefix:
            result = result.replace(local_prefix, remote_repo_root)
    return result.replace("\\", "/") if result.startswith(remote_repo_root) else result


def build_remote_script(
    *,
    config: RemoteTrainingConfig,
    command: Sequence[str],
    cwd: str,
    environment: Mapping[str, str],
    local_repo_root: Path,
    local_state_dir: Path,
) -> tuple[list[str], bytes]:
    """Build an SSH argv and a stdin script without placing secrets in argv."""
    validate_remote_training_config(config)
    remote_command = [
        map_local_value(str(argument), local_repo_root, config.repo_root)
        for argument in command
    ]
    remote_cwd = map_local_value(str(cwd), local_repo_root, config.repo_root)
    if remote_cwd == str(cwd):
        remote_cwd = config.repo_root

    lines = ["set -e"]
    config_root = (local_state_dir / "configs").resolve()
    for argument in command:
        candidate = Path(str(argument)).expanduser()
        try:
            candidate = candidate.resolve(strict=True)
        except OSError:
            continue
        if not candidate.is_file() or not candidate.is_relative_to(config_root):
            continue
        remote_path = map_local_value(str(candidate), local_repo_root, config.repo_root)
        content = candidate.read_bytes()
        # Generated snapshots contain absolute local paths; translate textual
        # configs before sending while leaving arbitrary binary files alone.
        try:
            translated = map_local_value(content.decode("utf-8"), local_repo_root, config.repo_root).encode("utf-8")
        except UnicodeDecodeError:
            translated = content
        encoded = base64.b64encode(translated).decode("ascii")
        lines.append(f"mkdir -p {shlex.quote(str(Path(remote_path).parent).replace(chr(92), '/'))}")
        lines.append(f"printf %s {shlex.quote(encoded)} | base64 --decode > {shlex.quote(remote_path)}")

    lines.append(f"cd {shlex.quote(remote_cwd)}")
    lines.append("if [ -f .env ]; then set -a; . ./.env; set +a; fi")
    if config.setup_command:
        lines.append(config.setup_command)
    for key, value in environment.items():
        if _ENVIRONMENT_KEY.fullmatch(str(key)):
            mapped = map_local_value(str(value), local_repo_root, config.repo_root)
            lines.append(f"export {key}={shlex.quote(mapped)}")
    lines.append(f"exec {shlex.join(remote_command)}")
    return config.ssh_command(), ("\n".join(lines) + "\n").encode("utf-8")
