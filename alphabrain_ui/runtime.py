from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find an AlphaBrain checkout without depending on git being installed."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "AlphaBrain").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find the AlphaBrain repository root. Run the UI from the repository or set ALPHABRAIN_ROOT."
    )


@dataclass(frozen=True)
class RuntimeConfig:
    repo_root: Path
    state_dir: Path
    database_path: Path
    frontend_dist: Path
    host: str = "127.0.0.1"
    port: int = 8000
    initial_mode: str = "personal"
    secure_cookies: bool = False
    session_days: int = 7
    scheduler_interval: float = 2.0
    stop_grace_seconds: int = 30
    demo_mode: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        repo_root: str | Path | None = None,
        state_dir: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> "RuntimeConfig":
        root_value = repo_root or os.environ.get("ALPHABRAIN_ROOT")
        root = find_repo_root(Path(root_value)) if root_value else find_repo_root()
        demo_mode = os.environ.get("ALPHABRAIN_UI_DEMO", "0") == "1"
        state_value = state_dir or os.environ.get("ALPHABRAIN_UI_HOME")
        state = (
            Path(state_value).expanduser().resolve()
            if state_value
            else root / (".alphabrain-demo" if demo_mode else ".alphabrain-ui")
        )
        initial_mode = os.environ.get("ALPHABRAIN_UI_MODE", "personal").strip().lower()
        if initial_mode not in {"personal", "lab"}:
            raise RuntimeError("ALPHABRAIN_UI_MODE must be 'personal' or 'lab'.")
        resolved_host = host or os.environ.get("ALPHABRAIN_UI_HOST", "127.0.0.1")
        if demo_mode and resolved_host not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("Demo mode is local-only; use 127.0.0.1, localhost, or ::1.")
        return cls(
            repo_root=root,
            state_dir=state,
            database_path=state / "ui.sqlite3",
            frontend_dist=root / "ui" / "frontend" / "dist",
            host=resolved_host,
            port=int(port or os.environ.get("ALPHABRAIN_UI_PORT", "8100" if demo_mode else "8000")),
            initial_mode=initial_mode,
            secure_cookies=os.environ.get("ALPHABRAIN_UI_SECURE_COOKIES", "0") == "1",
            session_days=int(os.environ.get("ALPHABRAIN_UI_SESSION_DAYS", "7")),
            scheduler_interval=float(os.environ.get("ALPHABRAIN_UI_SCHEDULER_INTERVAL", "2")),
            stop_grace_seconds=int(os.environ.get("ALPHABRAIN_UI_STOP_GRACE_SECONDS", "30")),
            demo_mode=demo_mode,
        )

    def ensure_directories(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        for child in ("logs", "configs", "artifacts"):
            (self.state_dir / child).mkdir(parents=True, exist_ok=True)
