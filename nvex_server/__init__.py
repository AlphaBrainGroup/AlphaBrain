from typing import Any

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
	if name in {"app", "create_app"}:
		from .app import app, create_app

		return {"app": app, "create_app": create_app}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")