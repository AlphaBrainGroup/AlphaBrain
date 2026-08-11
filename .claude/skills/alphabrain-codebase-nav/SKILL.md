---
name: alphabrain-codebase-nav
description: >-
  Navigate the AlphaBrain repository from a user-visible CLI or HTTP API behavior to its canonical configuration,
  schema, resolver, launcher, runtime, checkpoint adapter, and tests. Use when Codex must answer where AlphaBrain
  behavior is implemented, trace configuration precedence, find an API contract, assess whether a preset is really
  wired, or locate the correct extension point; also trigger for Chinese requests such as “哪个文件控制这个功能”,
  “配置如何合并”, “这个接口在哪里”, or “帮我看一下代码入口”.
---

# Navigate the AlphaBrain codebase

Trace behavior from the public entrypoint to the code that actually executes it. Prefer live source and generated API
schema over names, labels, screenshots, or stale documentation.

## Path convention

- Resolve `<repo-root>` before searching. Treat a directory containing `pyproject.toml`, `AlphaBrain/`, `configs/`,
  and `alphabrain_ui/` as an AlphaBrain checkout.
- Run commands from `<repo-root>` unless a document explicitly changes directories. Do not hardcode a checkout such
  as `/share/lrj/ab_ui` into an answer or patch.
- Record the inspected revision before drawing conclusions:

```bash
git status --short --branch
git rev-parse HEAD
```

- Use repository-relative paths in commands and clickable absolute paths when reporting results to the user.
- Represent the backend origin as `<base-url>`. The shared client at `scripts/agent/alphabrain_api.py` resolves its
  base URL and authentication; do not duplicate that logic in ad hoc curl commands.

## Where to find canonical answers

| Question | Start here | Follow through |
|---|---|---|
| Unified CLI mode and defaults | `configs/finetune_config.yaml` | `configs/models/`, `configs/datasets/`, `configs/trainer/`, `scripts/parse_config.py` |
| Training implementation | `scripts/run_finetune.sh` | `AlphaBrain/training/train_alphabrain.py`, capability-specific wrappers under `scripts/` |
| API experiment resolution | `alphabrain_ui/configuration.py` | `alphabrain_ui/capabilities.py`, `alphabrain_ui/workflows.py`, `alphabrain_ui/launchers.py` |
| Supported combinations | `alphabrain_ui/registry/catalog.yaml` | `alphabrain_ui/registry/`, `alphabrain_ui/builtin_presets.py`, resolver and launcher code |
| HTTP request/response contract | `GET /api/openapi.json` | `alphabrain_ui/app.py`, `alphabrain_ui/schemas.py` |
| Authentication and authorization | `alphabrain_ui/dependencies.py` | setup/login routes in `alphabrain_ui/app.py`, `alphabrain_ui/services.py` |
| Queue and process behavior | `alphabrain_ui/jobs.py` | deployment/evaluation/utility managers and `alphabrain_ui/process_control.py` |
| Checkpoint compatibility | deployment/evaluation inspection endpoints | `alphabrain_ui/deployment_registry.py`, `alphabrain_ui/evaluation_registry.py`, `alphabrain_ui/checkpoint_tools.py` |
| Data and model resources | resource/dataset endpoints | `alphabrain_ui/resource_catalog.py`, `alphabrain_ui/dataset_registry.py`, `alphabrain_ui/datasets.py` |
| Intended operator behavior | `docs/quickstart/` | the source paths above and focused tests under `tests/ui/` |

Inspect the live schema when the backend is available:

```bash
python scripts/agent/alphabrain_api.py status
python scripts/agent/alphabrain_api.py schema POST /api/v1/experiments/preflight
```

## Workflow at a glance

1. Restate the exact observable behavior: CLI invocation, API method/path, configuration field, artifact, or error.
2. Search narrowly with `rg`; inspect the public entrypoint and its tests before reading broad implementation files.
3. Trace one complete route:
   - For CLI, follow shell argument parsing to config parsing, Python entrypoint, runtime construction, and outputs.
   - For HTTP, follow route to Pydantic schema, capability/config resolver, preflight, launcher/manager, and persisted
     record or artifact.
   - For a preset, follow the registry entry to its resolver, launcher or adapter, source configs, and preflight.
4. Resolve configuration in actual precedence order. For finetune modes, read model, dataset, and trainer defaults;
   optional recipe config; global environment/seed; mapped mode fields; direct mode blocks; then `extra_args` dotlist.
   For API experiments, continue through parameter, workflow, expert, and resume overlays in
   `alphabrain_ui/configuration.py`.
5. Verify claims with the smallest relevant test, static resolver call, or read-only API request. Do not submit a job
   merely to learn how a route is wired.
6. Report the effective behavior, the controlling source, precedence or compatibility constraints, and any mismatch
   between docs, labels, and runtime.

## Things not obvious from the docs

- Do not equate a catalog entry, built-in template, or model directory with runnable support. Verify resource
  readiness, compatibility resolution, a concrete launcher or adapter, and preflight separately.
- Treat `configs/train_recipes_old_version/` as legacy input only when active code explicitly references it. The
  unified entrypoint is `configs/finetune_config.yaml` plus `scripts/run_finetune.sh`.
- Treat resolved experiment and evaluation snapshots as immutable execution records. Fix source configuration or
  create a new run; do not edit a submitted snapshot in place.
- Distinguish checkpoint completeness from deployability and evaluability. Each has a separate inspection contract.
- Keep checkpoint inspection metadata-only. Do not deserialize model weights just to identify a format.
- Distinguish the UI-backend interpreter, training interpreter, model-server interpreter, and benchmark interpreter;
  they may intentionally be different environments.
- Prefer OpenAPI and Pydantic models for HTTP shapes. Do not copy all endpoint schemas into a skill or client.

## Related skills

- Use `$alphabrain-setup` to install or start the CLI/API runtimes.
- Use `$alphabrain-env-troubleshoot` when the route is known but an environment, import, CUDA, or preflight check
  fails.
- Use `$alphabrain-data-resources` for datasets, resources, mixtures, templates, and downloads.
- Use `$alphabrain-training` for resolving, preflighting, submitting, and monitoring training.
- Use `$alphabrain-deployment` for checkpoint serving and inference.
- Use `$alphabrain-evaluation` for benchmark inspection, preflight, execution, and result verification.
