from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import User
from .services import AuthService, SettingsService


def get_db(request: Request) -> Generator[Session, None, None]:
    db = request.app.state.database.SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    settings = SettingsService(db)
    if not settings.get("initialized", False):
        raise HTTPException(status_code=status.HTTP_428_PRECONDITION_REQUIRED, detail="setup_required")

    if settings.get("deployment_mode") == "personal":
        user = db.execute(select(User).where(User.is_local.is_(True), User.is_active.is_(True))).scalar_one_or_none()
    else:
        raw = request.cookies.get(AuthService.cookie_name)
        session = AuthService(db, request.app.state.config.session_days).get_session(raw)
        user = session.user if session is not None else None
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication_required")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "administrator":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="administrator_required")
    return user


def assert_owner_or_admin(user: User, owner_id: str) -> None:
    if user.role != "administrator" and user.id != owner_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="owner_or_administrator_required")
