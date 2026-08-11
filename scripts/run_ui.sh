#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"AlphaBrain UI requires Python >= 3.10; current interpreter is {sys.version.split()[0]}"
    )
missing = []
for module in (
    "fastapi", "uvicorn", "sqlalchemy", "alembic", "argon2", "psutil", "pynvml",
    "multipart", "PIL", "websockets", "huggingface_hub", "yaml", "numpy",
):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing AlphaBrain UI dependencies: " + ", ".join(missing)
        + "\nInstall them in the current AlphaBrain environment with: "
          "pip install -r requirements-ui.txt"
    )
PY

if [ ! -f "ui/frontend/dist/index.html" ]; then
    echo "[error] AlphaBrain UI frontend has not been built." >&2
    echo "Run: cd ui/frontend && npm install && npm run build" >&2
    exit 1
fi

exec python -m alphabrain_ui "$@"
