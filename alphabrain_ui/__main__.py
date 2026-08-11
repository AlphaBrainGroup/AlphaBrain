from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaBrain research console")
    parser.add_argument("--host", default=None, help="Bind host (use 0.0.0.0 for lab LAN access)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 8000)")
    parser.add_argument("--repo-root", default=None, help="AlphaBrain repository root")
    parser.add_argument("--state-dir", default=None, help="Persistent UI state directory")
    parser.add_argument(
        "--mode",
        choices=("personal", "lab"),
        default=None,
        help="Initial mode for an uninitialized state directory (default: personal; saved settings win later)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable development auto-reload")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run an isolated local demo with simulated GPUs, training, deployment, inference, and evaluation",
    )
    args = parser.parse_args()

    # Configuration is passed through environment because uvicorn imports the
    # application factory in a separate reload process when --reload is used.
    import os

    if args.repo_root:
        os.environ["ALPHABRAIN_ROOT"] = args.repo_root
    if args.state_dir:
        os.environ["ALPHABRAIN_UI_HOME"] = args.state_dir
    if args.host:
        os.environ["ALPHABRAIN_UI_HOST"] = args.host
    if args.port:
        os.environ["ALPHABRAIN_UI_PORT"] = str(args.port)
    if args.mode:
        os.environ["ALPHABRAIN_UI_MODE"] = args.mode
    if args.demo:
        os.environ["ALPHABRAIN_UI_DEMO"] = "1"
        os.environ.setdefault("ALPHABRAIN_UI_HOST", "127.0.0.1")
        os.environ.setdefault("ALPHABRAIN_UI_PORT", "8100")

    host = args.host or os.environ.get("ALPHABRAIN_UI_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("ALPHABRAIN_UI_PORT", "8100" if args.demo else "8000"))
    uvicorn.run(
        "alphabrain_ui.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=args.reload,
        workers=1,
    )


if __name__ == "__main__":
    main()
