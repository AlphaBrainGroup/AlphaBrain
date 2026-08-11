from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta, timezone
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .database import AuditEvent, AuthSession, SystemSetting, User, utcnow

DEFAULT_SETTINGS: dict[str, Any] = {
    "initialized": False,
    "deployment_mode": "personal",
    "experimental_globally_enabled": False,
    "environment": {},
    "results_roots": ["results"],
    # Empty keeps the historical behavior: monitor the first results root.
    "storage_monitor_path": "",
    "dataset_roots": ["data"],
    "managed_dataset_root": ".alphabrain-ui/datasets",
    "pretrained_root": "",
    "resource_paths": {},
    "cpu_utility_concurrency": 2,
    "disk_min_free_gib": 100.0,
    "disk_min_free_percent": 10.0,
    "secure_cookies": False,
    "model_server_python": "",
    "remote_training_enabled": False,
    "remote_training_host": "",
    "remote_training_user": "",
    "remote_training_port": 22,
    "remote_training_repo_root": "",
    "remote_training_identity_file": "",
    "remote_training_gpu_ids": [],
    "remote_training_setup_command": "",
}


class SettingsService:
    def __init__(self, db: Session):
        self.db = db

    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.get(SystemSetting, key)
        if row is not None:
            return row.value
        return DEFAULT_SETTINGS.get(key, default)

    def all(self) -> dict[str, Any]:
        result = dict(DEFAULT_SETTINGS)
        rows = self.db.execute(select(SystemSetting)).scalars().all()
        result.update({row.key: row.value for row in rows})
        return result

    def set(self, key: str, value: Any, actor_id: str | None = None) -> None:
        row = self.db.get(SystemSetting, key)
        if row is None:
            self.db.add(SystemSetting(key=key, value=value, updated_by=actor_id))
        else:
            row.value = value
            row.updated_by = actor_id

    def update(self, values: dict[str, Any], actor_id: str | None = None) -> dict[str, Any]:
        for key, value in values.items():
            if value is not None and key in DEFAULT_SETTINGS:
                self.set(key, value, actor_id)
        return self.all()


def add_audit(
    db: Session,
    action: str,
    *,
    actor_id: str | None = None,
    target_type: str = "",
    target_id: str = "",
    detail: dict[str, Any] | None = None,
) -> None:
    db.add(
        AuditEvent(
            actor_id=actor_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            detail=detail or {},
        )
    )


class AuthService:
    cookie_name = "alphabrain_session"
    csrf_cookie_name = "alphabrain_csrf"

    def __init__(self, db: Session, session_days: int = 7):
        self.db = db
        self.session_days = session_days
        self.password_hasher = PasswordHasher()

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def hash_password(self, password: str) -> str:
        return self.password_hasher.hash(password)

    def verify_password(self, encoded: str, password: str) -> bool:
        if not encoded:
            return False
        try:
            return self.password_hasher.verify(encoded, password)
        except (VerifyMismatchError, InvalidHashError):
            return False

    def authenticate(self, username: str, password: str) -> User | None:
        user = self.db.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if user is None or not user.is_active or not self.verify_password(user.password_hash, password):
            return None
        if self.password_hasher.check_needs_rehash(user.password_hash):
            user.password_hash = self.hash_password(password)
        return user

    def create_session(self, user: User) -> tuple[str, AuthSession]:
        raw = secrets.token_urlsafe(40)
        row = AuthSession(
            token_hash=self.hash_token(raw),
            csrf_token=secrets.token_urlsafe(32),
            user_id=user.id,
            expires_at=utcnow() + timedelta(days=self.session_days),
        )
        self.db.add(row)
        self.db.flush()
        return raw, row

    def get_session(self, raw_token: str | None) -> AuthSession | None:
        if not raw_token:
            return None
        row = self.db.execute(
            select(AuthSession).where(AuthSession.token_hash == self.hash_token(raw_token))
        ).scalar_one_or_none()
        if row is None:
            return None
        expires = row.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= utcnow():
            self.db.delete(row)
            return None
        row.last_seen_at = utcnow()
        return row

    def revoke(self, raw_token: str | None) -> None:
        if raw_token:
            self.db.execute(delete(AuthSession).where(AuthSession.token_hash == self.hash_token(raw_token)))

    def purge_expired(self) -> None:
        self.db.execute(delete(AuthSession).where(AuthSession.expires_at <= utcnow()))


def redact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Redact values likely to be credentials before persistence or display."""
    result: dict[str, Any] = {}
    for key, value in values.items():
        upper = key.upper()
        if any(token in upper for token in ("TOKEN", "PASSWORD", "SECRET", "API_KEY", "CREDENTIAL")):
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_mapping(value)
        else:
            result[key] = value
    return result
