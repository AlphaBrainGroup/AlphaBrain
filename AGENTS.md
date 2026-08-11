# AlphaBrain agent guide

Use this file as the compact repository map and operating policy. Load the task-specific Skill under
`.agents/skills/` for procedural detail; do not expand this file into a second copy of those workflows.

## Repository map

- `AlphaBrain/`: model, dataloader, training, reinforcement-learning, and continual-learning implementation.
- `configs/`: unified finetune modes plus model, dataset, trainer, DeepSpeed, RL, and continual-learning defaults.
- `scripts/`: checked-in CLI launchers, config parsing, resource utilities, and `scripts/agent/alphabrain_api.py`.
- `benchmarks/`: LIBERO, LIBERO-plus, RoboCasa, and RoboCasa365 adapters and preparation scripts.
- `alphabrain_ui/`: FastAPI backend, schemas, registry, resolvers, preflight, scheduler, deployment, and evaluation.
- `ui/frontend/`: React/Vite browser client. Browser and desktop UI automation are outside this agent guide.
- `tests/`: backend, API-client, training, deployment, and evaluation contract tests.
- `docs/`: operator quickstarts and API/code documentation.

Resolve the repository root from `pyproject.toml`, `AlphaBrain/`, `configs/`, and `alphabrain_ui/`; do not hardcode a
checkout path. Inspect `git status --short --branch` before editing and preserve unrelated user changes.

## Prefer the managed HTTP API

Use the backend API when health is reachable and the registry exposes the requested workflow. It provides canonical
resolution, preflight, immutable snapshots, authorization, and the shared FIFO for training, deployment, evaluation,
and GPU utilities.

Run the checked-in client with a Python that has `requirements-ui.txt` installed:

```bash
python scripts/agent/alphabrain_api.py status
python scripts/agent/alphabrain_api.py schema
python scripts/agent/alphabrain_api.py schema POST /api/v1/experiments/preflight
python scripts/agent/alphabrain_api.py request GET /api/v1/gpus
python scripts/agent/alphabrain_api.py watch job JOB_ID --timeout 86400
```

Treat live `GET /api/openapi.json` and the Pydantic models in `alphabrain_ui/schemas.py` as the HTTP contract. Do not
guess payloads from UI labels or copy endpoint schemas into agent documentation. Run resolve/inspection and preflight
before submission; never submit when `ok` or `can_submit` is false.

Use a bounded watch timeout that fits the observation window. When it expires, re-query and resume watching the same
workload ID; a local observer timeout is not a workload failure and never authorizes duplicate submission.

Use direct CLI only when the API does not represent the task, source-level debugging requires it, or the user
explicitly requests CLI execution. Start from the checked-in wrapper for that capability, such as
`scripts/run_finetune.sh`, `scripts/run_eval.sh`, or the relevant `scripts/run_*` README. A direct CLI process bypasses
the backend's global GPU FIFO and may be invisible to its reservations: inspect both `/api/v1/gpus` and `nvidia-smi`,
then disclose this scheduling boundary before launch. Set `ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1` while resolving or
diagnosing a finetune command so resource acquisition remains a separate action.

Do not operate the browser UI or a Windows/macOS desktop app in this skill set. Achieve supported workflows through
the API or CLI; report that a separate computer-use/browser Skill and tool capability are required when visual UI
operation is explicitly requested. User administration, secret-setting management, Registry Overlay changes, model
publication, and audit-log workflows are also outside these seven core research Skills; do not infer authority for
them from a training, deployment, evaluation, or data request.

## Safety boundaries

- Start read-only. Re-query the exact workload, checkpoint, dataset, resource, and target path before any mutation.
- Download a model or dataset only when the user requests or approves that acquisition. Resolve the repository ID,
  destination, existing partial data, free space, credentials, and expected size first.
- Treat job submission and lifecycle controls as scoped mutations. The client requires `--send` for non-GET requests
  and `--dangerous` for DELETE, cancel, stop, terminate, force-kill, restart/retry, and key rotation; these flags do
  not replace user authorization. Use force-kill, deletion, restart/retry, or rotation only when explicitly requested
  for a freshly verified ID.
- Never overwrite existing datasets, run directories, snapshots, checkpoints, environments, or local configuration
  as an incidental repair. Prefer a new run/output or isolated environment.
- Keep passwords, cookies, Hugging Face/W&B tokens, SSH private keys, API keys, and other credentials out of argv,
  generic settings, JSON examples, logs, snapshots, commits, and final responses. Use interactive input or
  `--password-stdin` for laboratory login.
- Deployment plaintext API keys are returned only once. Save a requested key with `--secret-output <file>` to a new,
  user-approved untracked location; the client refuses an existing secret path and writes mode `0600`. Normal output
  remains redacted.
- Regular `--output` downloads also refuse an existing file by default. Use `--overwrite` only after verifying the
  exact destination and explicitly intending replacement; it never permits replacing a symlink or secret output.
- Identify checkpoint formats through filenames and YAML/JSON metadata. Do not deserialize weights merely to inspect
  compatibility.
- Verify completion from managed terminal state, logs, and expected final artifacts. A PID exit, queue record, output
  directory, or partial checkpoint alone is not success.

## Common validation

Install the relevant test/dev dependencies first, then run the smallest checks that cover the change:

```bash
python -m pytest tests/test_agent_api_client.py tests/test_agent_skills.py tests/ui
python -m pytest tests/deployment tests/test_evaluation_registry.py tests/test_evaluation_result.py
python scripts/agent/sync_skills.py --check
python -m ruff check alphabrain_ui scripts/agent tests/ui tests/test_agent_api_client.py tests/test_agent_skills.py
cd ui/frontend
npm test
npm run build
cd ../..
git diff --check
```

Do not start real training, download large resources, allocate GPUs, or launch a simulator as a generic smoke test.
Use demo or temporary state only when the test specifically needs backend lifecycle coverage.

## Task Skills

- [AlphaBrain codebase navigation](.agents/skills/alphabrain-codebase-nav/SKILL.md): trace configs, routes, resolvers,
  launchers, adapters, and tests.
- [AlphaBrain setup](.agents/skills/alphabrain-setup/SKILL.md): create and verify CLI/API runtime boundaries.
- [AlphaBrain environment troubleshooting](.agents/skills/alphabrain-env-troubleshoot/SKILL.md): diagnose Python,
  CUDA, dependency, path, auth, and preflight failures.
- [AlphaBrain data and resources](.agents/skills/alphabrain-data-resources/SKILL.md): manage models, datasets,
  mixtures, templates, and downloads.
- [AlphaBrain training](.agents/skills/alphabrain-training/SKILL.md): resolve, preflight, submit, monitor, and verify
  training.
- [AlphaBrain deployment](.agents/skills/alphabrain-deployment/SKILL.md): inspect checkpoints, serve models, run
  inference, and control deployment lifecycle.
- [AlphaBrain evaluation](.agents/skills/alphabrain-evaluation/SKILL.md): inspect, preflight, run, and verify benchmark
  evaluations and artifacts.
