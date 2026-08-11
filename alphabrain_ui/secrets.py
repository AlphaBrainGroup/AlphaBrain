"""Small, filesystem-backed credential stores used by the UI.

Secret values are intentionally kept out of SQLite and API responses.  Files
are replaced atomically and are only readable by the account running the UI.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path


class SecureSecretStore:
    def __init__(self, state_dir: str | Path, namespace: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", namespace):
            raise ValueError("invalid secret namespace")
        self.root = Path(state_dir).expanduser().resolve() / "secrets" / namespace

    @staticmethod
    def _key_path(root: Path, key: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError("invalid secret key")
        return root / key

    def _ensure_root(self) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = os.lstat(self.root)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("secret store path is not a regular directory")
        os.chmod(self.root, 0o700)

    def _fsync_root(self) -> None:
        if os.name == "nt":
            # Windows has no portable directory fsync primitive. os.replace
            # remains atomic on the same volume.
            return
        descriptor = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def set(self, key: str, value: str, *, min_length: int = 8, max_length: int = 4096) -> None:
        value = value.strip()
        if not min_length <= len(value) <= max_length or any(c.isspace() or ord(c) < 33 for c in value):
            raise ValueError("invalid secret value")
        self._ensure_root()
        path = self._key_path(self.root, key)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".secret-", dir=self.root)
        temporary = Path(temporary_name)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            else:
                os.chmod(temporary, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(value.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            self._fsync_root()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def read(self, key: str) -> str | None:
        path = self._key_path(self.root, key)
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return None
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or (
                os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise PermissionError("secret file permissions are unsafe")
            data = os.read(descriptor, 4097)
            if len(data) > 4096:
                raise RuntimeError("secret value is too large")
        finally:
            os.close(descriptor)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("secret value is not valid UTF-8") from exc

    def delete(self, key: str) -> None:
        path = self._key_path(self.root, key)
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("secret path is not a regular file")
        path.unlink()
        self._fsync_root()

    def configured(self, key: str) -> bool:
        return self.read(key) is not None


class HuggingFaceSecretStore:
    """Global download credential plus one publishing credential per user."""

    def __init__(self, state_dir: str | Path):
        self.store = SecureSecretStore(state_dir, "huggingface")

    def set_global_download_token(self, token: str) -> dict[str, bool]:
        self.store.set("global-download", token)
        return {"configured": True}

    def read_global_download_token(self) -> str | None:
        return self.store.read("global-download")

    def delete_global_download_token(self) -> dict[str, bool]:
        self.store.delete("global-download")
        return {"configured": False}

    def global_status(self) -> dict[str, bool]:
        return {"configured": self.store.configured("global-download")}

    def set_user_publish_token(self, user_id: str, token: str) -> dict[str, bool]:
        self.store.set(f"user-{user_id}", token)
        return {"configured": True}

    def read_user_publish_token(self, user_id: str) -> str | None:
        return self.store.read(f"user-{user_id}")

    def delete_user_publish_token(self, user_id: str) -> dict[str, bool]:
        self.store.delete(f"user-{user_id}")
        return {"configured": False}

    def user_status(self, user_id: str) -> dict[str, bool]:
        return {"configured": self.store.configured(f"user-{user_id}")}


class HFGlobalDownloadSecretStore:
    """Compatibility-focused view of the administrator-managed HF credential."""

    def __init__(self, state_dir: str | Path):
        self._store = HuggingFaceSecretStore(state_dir)

    def set_token(self, token: str) -> dict[str, bool]:
        return self._store.set_global_download_token(token)

    def read_token(self) -> str | None:
        return self._store.read_global_download_token()

    def delete_token(self) -> dict[str, bool]:
        return self._store.delete_global_download_token()

    def status(self) -> dict[str, bool]:
        return self._store.global_status()


class HFUserPublishSecretStore:
    """Per-user Hugging Face publishing credential view."""

    def __init__(self, state_dir: str | Path):
        self._store = HuggingFaceSecretStore(state_dir)

    def set_token(self, user_id: str, token: str) -> dict[str, bool]:
        return self._store.set_user_publish_token(user_id, token)

    def read_token(self, user_id: str) -> str | None:
        return self._store.read_user_publish_token(user_id)

    def delete_token(self, user_id: str) -> dict[str, bool]:
        return self._store.delete_user_publish_token(user_id)

    def status(self, user_id: str) -> dict[str, bool]:
        return self._store.user_status(user_id)
