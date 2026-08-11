---
name: alphabrain-setup
description: >-
  Set up and verify AlphaBrain command-line, API-backend, training, model-server, and benchmark Python environments
  without browser automation. Use when Codex must create or complete an environment, install AlphaBrain or backend
  dependencies, configure local paths, start `python -m alphabrain_ui`, initialize personal or laboratory API mode,
  or verify runtime readiness; also trigger for Chinese requests such as “安装 AlphaBrain”, “配置运行环境”,
  “启动后端 API”, “设置 Python 路径”, or “这个环境还缺什么包”.
---

# Set up AlphaBrain CLI and API runtimes

Build only the runtime the user needs, preserve existing environments, and verify each interpreter independently.
Use the HTTP backend without requiring the browser bundle.

## Path convention

- Resolve `<repo-root>` as the directory containing `pyproject.toml`, `AlphaBrain/`, `configs/`, and
  `alphabrain_ui/`. Run repository commands from there.
- Use `<runtime-python>` for the interpreter that will train or serve models, `<backend-python>` for the API service,
  and `<benchmark-python>` for a simulator environment. Never assume they are the same executable.
- Use an explicit, user-approved Conda environment name and path. Do not modify a shared or existing environment
  until its package state has been inspected and the user has authorized the change.
- Keep state and generated files under user-approved writable paths. Default backend state is
  `<repo-root>/.alphabrain-ui/`; demo state is `<repo-root>/.alphabrain-demo/`.

## Where to find canonical answers

| Concern | Canonical source |
|---|---|
| Supported baseline and install sequence | `docs/quickstart/installation.md`, `requirements.txt`, `pyproject.toml` |
| Backend-only dependencies | `requirements-ui.txt`, `docs/quickstart/web_ui.md` |
| Backend flags and defaults | `alphabrain_ui/__main__.py`, `alphabrain_ui/runtime.py`, `scripts/run_ui.sh` |
| Local path variables | `.env.example`, `configs/finetune_config.yaml` |
| Settings and safe environment keys | `alphabrain_ui/schemas.py`, `SAFE_ENV_KEYS` in `alphabrain_ui/app.py` |
| Deployment interpreter imports | `alphabrain_ui/deployment_preflight.py` |
| Benchmark interpreter imports | `alphabrain_ui/evaluation_preflight.py` and the benchmark quickstart |
| Live HTTP contract | `GET /api/openapi.json` |

## Choose the runtime boundary

| Goal | Install into the selected interpreter |
|---|---|
| API backend only | `requirements-ui.txt` and editable AlphaBrain package |
| Training CLI plus API backend | `requirements.txt`, `requirements-ui.txt`, and editable AlphaBrain package |
| Managed model server | A runtime that imports `torch`, `transformers`, `websockets`, `msgpack`, `numpy`, and `AlphaBrain` |
| LIBERO/RoboCasa evaluation | A separate benchmark environment with the simulator-specific imports and the configured Python path |

Use Python 3.10 or newer because the package and preflight accept that range. If the user does not choose a version,
default a new baseline environment to Python 3.10 to match the installation guide. Honor a requested newer compatible
version; do not downgrade an existing interpreter merely to match the example.

## Workflow at a glance

1. Inspect before installing:

```bash
git status --short --branch
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
python -m pip check
nvidia-smi
```

Record the selected interpreter, Python version, CUDA driver visibility, installed Torch/Transformers/LeRobot
versions, and any existing dependency conflicts. Never resolve a conflict by silently downgrading unrelated packages.

2. Create a new environment only when requested or when isolation is necessary. Use the user's environment name; a
new baseline example is:

```bash
conda create -n <environment-name> python=3.10 -y
conda run -n <environment-name> python -m pip --version
```

3. Install the minimum selected stack from `<repo-root>` with that environment's Python:

```bash
<backend-python> -m pip install -r requirements-ui.txt
<backend-python> -m pip install -e .
```

For a combined training/backend runtime, install `requirements.txt` before `requirements-ui.txt`, then install the
repository editable. Install `flash-attn --no-build-isolation` only after confirming the selected recipe needs it and
the active Torch/CUDA ABI matches. Do not force-reinstall it as a routine setup step.

4. Configure paths without overwriting local state. Copy `.env.example` to `.env` only if `.env` does not already
exist, then fill absolute paths such as `PRETRAINED_MODELS_DIR`, `LEROBOT_LIBERO_DATA_DIR`, `LIBERO_HOME`, and the
benchmark Python paths required by the chosen workflow. Keep credentials out of generic environment settings and out
of committed files. `.env` and `.alphabrain-ui/` are ignored by this repository, but verify `git status` anyway.

5. Start the API backend with one worker:

```bash
<backend-python> -m alphabrain_ui --repo-root <repo-root> --state-dir <state-dir> --host 127.0.0.1 --port 8000
```

Use `bash scripts/run_ui.sh` as the repository wrapper when its defaults are suitable. Bind to `0.0.0.0` only when
the user explicitly requests remote access and the deployment has appropriate network controls. Do not run a second
Uvicorn worker: the SQLite-backed scheduler and GPU reservations assume one backend worker.

6. Probe the backend before initialization:

```bash
python scripts/agent/alphabrain_api.py --base-url http://127.0.0.1:8000 status
python scripts/agent/alphabrain_api.py --base-url http://127.0.0.1:8000 schema GET /api/v1/setup/status
```

If state is uninitialized, obtain an explicit choice between personal and laboratory mode before calling
`POST /api/v1/setup`. Personal mode can use an empty password. Laboratory mode requires an administrator password of
at least eight characters; collect it with a non-echoing prompt and never place it in argv, an environment variable,
logs, or a retained JSON file. For laboratory setup, generate the one-time request through an anonymous pipe so only
the JSON producer sees the typed password:

```bash
<backend-python> -c '
import getpass, json, sys
print("Administrator username [admin]: ", end="", file=sys.stderr, flush=True)
username = sys.stdin.readline().strip() or "admin"
password = getpass.getpass("Administrator password: ")
json.dump({"mode": "lab", "username": username, "display_name": "Administrator", "password": password}, sys.stdout)
' | <backend-python> scripts/agent/alphabrain_api.py request POST /api/v1/setup --body - --send
```

Run this only against loopback, HTTPS, or an SSH tunnel. By default, the client refuses every mutating request,
credential exchange, and `--secret-output` response over non-loopback plain HTTP in both Personal and Lab modes.
Treat `--allow-insecure-http` as an explicit security override, not a connectivity fix. For later protected calls,
provide `--username` and let the shared client collect the password interactively or via `--password-stdin`; it keeps
the session cookie and CSRF token only in memory for that invocation.

7. Configure a separate model-server Python or benchmark Python only after probing it. Use the settings API contract
from OpenAPI; do not guess request fields. Re-query settings and run the relevant deployment or evaluation preflight
after saving a path.

8. Verify the selected boundary:

```bash
<backend-python> -c "import AlphaBrain, alphabrain_ui; print('backend imports ok')"
<runtime-python> -c "import torch, transformers, websockets, msgpack, numpy, AlphaBrain; print('model runtime imports ok')"
python scripts/agent/alphabrain_api.py status
```

For a training runtime, also run `python -m pip check` and a configuration/preflight smoke check. Do not start real
training, download a large model, or allocate a GPU merely to verify installation.

## Things not obvious from the docs

- `pip install -e .` alone is not a complete install because `pyproject.toml` declares no core runtime dependencies.
- `requirements-ui.txt` is sufficient for the lightweight Python API backend, not for training or a managed model
  server. Browser assets additionally need Node/npm, but they are outside this API-only skill.
- LeRobot is not pinned in the core or UI requirements. Install it only for a workflow that imports it, after checking
  the requested AlphaBrain path, Python version, Transformers constraint, and checkpoint compatibility.
- The saved deployment mode wins after a state directory is initialized. A later `--mode` flag does not bypass an
  existing laboratory login configuration.
- Personal mode still requires one-time setup but then resolves a local user without login. Laboratory mode uses an
  HTTP-only session cookie and requires the matching CSRF header for unsafe API requests.
- Job environment precedence is repository `.env`, then the environment that started the backend, then non-secret
  values saved in settings.
- Demo mode is loopback-only, defaults to port 8100, uses simulated workloads, and must not be treated as proof that a
  real model or simulator environment is ready.
- The shared API client ignores inherited proxy and custom-CA environment variables by default so loopback requests
  cannot be diverted by workstation proxy settings. Add `--trust-env` only when the chosen remote backend genuinely
  requires those variables and the operator has reviewed them.
- Benchmark dependencies intentionally belong in separate interpreters. Do not repair a simulator import by
  destabilizing the training environment.

## Related skills

- Use `$alphabrain-codebase-nav` to trace an unknown installer, config field, API route, or runtime boundary.
- Use `$alphabrain-env-troubleshoot` when an existing setup fails imports, CUDA checks, authentication, or preflight.
- Use `$alphabrain-data-resources` before downloading or registering models and datasets.
- Use `$alphabrain-training`, `$alphabrain-deployment`, or `$alphabrain-evaluation` for the first real workload after
  setup.
