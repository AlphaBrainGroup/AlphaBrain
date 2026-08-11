---
name: alphabrain-data-resources
description: >
  Inspect and manage AlphaBrain model resources, downloads, server-local dataset
  registrations, validation, previews, statistics, weighted mixtures, and experiment
  templates through the backend HTTP API or checked-in CLI utilities. Use when Codex
  needs to download or register a model, inspect or register a dataset, create a data
  mixture, compute dataset statistics, manage templates, or diagnose duplicate-template
  and resource-readiness errors; also trigger for 模型资源, 下载模型, 登记/验证数据集,
  数据组合, 数据集统计, 模板中心, or 重复模板 requests.
---

# Manage AlphaBrain Data and Resources

## Path convention

- Start from the repository root containing `pyproject.toml`, `AlphaBrain/`, and `alphabrain_ui/`.
- Use `python scripts/agent/alphabrain_api.py` for backend operations.
- Read the live `/api/openapi.json` contract through the client's `schema` command before constructing a mutation body. Do not reuse a body from another checkout.
- Keep request files and downloaded outputs in a user-approved writable location. Never place credentials in request JSON, argv, logs, templates, or dataset metadata.

## Where to find answers

| Question | Authoritative source |
|---|---|
| Available resource IDs, install targets, and preprocessors | `alphabrain_ui/resource_catalog.py` and `GET /api/v1/resources` |
| Dataset format detection and builder support | `alphabrain_ui/dataset_registry.py` |
| Dataset, mixture, template, and utility request contracts | `/api/openapi.json`, `alphabrain_ui/schemas.py`, and `alphabrain_ui/app.py` |
| Template identity and duplicate rules | `alphabrain_ui/template_fingerprints.py` |
| Manual model prefetch behavior | `scripts/download_pretrained.py` and `scripts/run_base_vla/README.md` |

## Choose an execution path

1. Run `python scripts/agent/alphabrain_api.py status`.
2. Prefer the HTTP API when the backend is reachable; preserve its root checks, ownership rules, audit records, and utility queues.
3. Use the checked-in CLI only for explicit model prefetch or a workflow that the API does not expose. Do not call `alphabrain_ui.utility_worker` directly.
4. Use `$alphabrain-setup` when the backend, authentication, configured roots, or Python environment is not ready.
5. Obtain explicit user approval before starting a large download, managed dataset copy, GPU preprocessing run, cancellation, update, unregister, or delete operation.

For any route below, inspect its current request schema first, for example:

```bash
python scripts/agent/alphabrain_api.py schema POST /api/v1/datasets/inspect
```

Send mutations only from a reviewed JSON file or stdin and include `--send`:

```bash
python scripts/agent/alphabrain_api.py request POST /api/v1/datasets/inspect \
  --body /workspace/alphabrain-request.json --send
```

## Manage model and world-model resources

Follow these route orders exactly:

```text
Install:     GET /api/v1/resources
          -> GET /api/v1/storage
          -> POST /api/v1/resources/{resource_id}/install
          -> watch utility {run_id}
          -> GET /api/v1/resources

Register:    GET /api/v1/resources
          -> PUT /api/v1/resources/{resource_id}/path
          -> GET /api/v1/resources

Preprocess:  GET /api/v1/resources
          -> POST /api/v1/resources/{resource_id}/preprocess
          -> watch utility {run_id}
          -> GET /api/v1/resources
```

- Read the selected catalog record before acting. Verify its exact `id`, `status`, `installable` or `registerable` flag, `target_path`, token requirement, and `active_run`.
- Confirm the destination, available storage, source repository, and download authorization with the user. Require an administrator for install and path-registration routes.
- Treat `hf_download_token_required` as a configuration blocker. Stop and route the user to an administrator-approved, out-of-band credential setup; this Skill and the shared client intentionally do not set Hugging Face credentials. Do not ask for or save a token through generic settings or inline JSON.
- Register paths only for catalog entries that advertise world-model path registration. Require an existing, readable, non-empty directory.
- Require a new output path under a configured dataset root, pretrained root, or UI artifact root before preprocessing. Never overwrite an existing preprocessing output.
- Treat GPU preprocessing as part of the global GPU FIFO. Do not promise immediate execution.
- Run `python scripts/agent/alphabrain_api.py watch utility RUN_ID --timeout 86400` after a utility is created. Choose a bounded timeout appropriate to the observation window. If it expires, re-query and resume watching the same run; a watcher timeout is not a utility failure and never authorizes duplicate submission. Treat only `completed` as success; inspect `/api/v1/utilities/{run_id}/log` and report `failed`, `cancelled`, `stopped`, or `interrupted` as non-success.
- Observe utilities through `GET /api/v1/utilities` and the run log route. Request `POST /api/v1/utilities/{run_id}/cancel` with `--dangerous` only after the user identifies the exact run and authorizes cancellation.
- Do not request `/api/v1/utilities/{run_id}/output` for model downloads, dataset copies, statistics, or preprocessing; that download route accepts only completed checkpoint/training packages.
- Re-read the resource catalog after completion and require `status: installed`; do not infer completeness from a process exit or directory alone.

## Register and inspect datasets

Follow this route order for every new registration:

```text
POST /api/v1/datasets/inspect
 -> review valid, format_family, builder_support, issues, fingerprint, and size_bytes
 -> POST /api/v1/datasets
 -> [managed_copy only] watch utility {copy_run_id}
 -> GET /api/v1/datasets/{dataset_id}
 -> GET /api/v1/datasets/{dataset_id}/preview?limit=N
```

- Inspect first. Require the source to be inside an administrator-configured dataset root; do not bypass a `dataset_path_outside_configured_roots` response.
- Report the detected format and its actual builder support:
  - Treat `lerobot` with `builder_support: direct` as directly selectable by the experiment builder.
  - Treat `lerobot_collection` with `builder_support: mixture_only` as requiring mixture-aware use.
  - Treat `cosmos` and `vlm_json` with `builder_support: inventory_only` as registered inventory, not generic builder-ready training data.
  - Reject `unknown` or any report with `valid: false`.
- Ask the user to choose `reference` or `managed_copy` and `private` or `shared` when the intent does not determine them. Preserve the source; registration and unregistering do not authorize source-data deletion.
- Wait for a managed copy to finish, then require the registration status to become `ready`. Do not use a `copying` or `invalid` registration in a mixture or experiment.
- Preview only bounded metadata through the preview route; do not open or echo arbitrary dataset payloads unless the user requests source-level inspection.
- Revalidate changed data in this order:

```text
GET /api/v1/datasets/{dataset_id}
 -> POST /api/v1/datasets/{dataset_id}/validate
 -> GET /api/v1/datasets/{dataset_id}
```

- Compare the old and new fingerprint. Treat changed fingerprints and `stats_status: stale` as provenance changes that require review before reuse.
- Compute statistics only for a `ready` registration:

```text
POST /api/v1/datasets/{dataset_id}/stats
 -> watch utility {run_id} --timeout 86400
 -> GET /api/v1/datasets/{dataset_id}
```

- Require both utility `completed` and registration `stats_status: ready` before reporting statistics as available.

## Manage weighted mixtures

Follow this route order:

```text
GET /api/v1/datasets
 -> GET /api/v1/datasets/mixtures
 -> POST /api/v1/datasets/mixtures
 -> GET /api/v1/datasets/mixtures
```

- Select only visible registrations whose status is `ready`.
- Reference members by `registration_id`; never submit arbitrary server paths through an experiment `mixture_spec`.
- Preserve each requested `pattern`, positive `weight`, `robot_type`, and optional `trajectory_limit`. Do not invent robot types, glob semantics, weights, or sampling intent.
- Review `resolved_members`, `mixture_spec`, and the monotonic `version` after creation or patching. Experiment resolution freezes a single registration's fingerprint, but a mixture source freezes only its ID/version and expanded paths/sampling fields. Record each selected member's current fingerprint separately before training because it is not embedded in the mixture snapshot.
- Re-query the exact mixture immediately before a training preflight. Use `$alphabrain-training` for experiment resolution and submission.
- Patch or delete a mixture only after resolving its exact ID, owner, members, and current version and receiving explicit authorization.

## Manage experiment templates

Follow this route order:

```text
GET /api/v1/templates
 -> POST /api/v1/templates
 -> GET /api/v1/templates/{template_id}
```

- Treat built-in templates as read-only. Copy their `spec` into a new user-owned template rather than patching or deleting them.
- Submit `allow_duplicate: false` by default. Let the server compare normalized same-owner names and canonical scientific/operational specifications against visible and built-in templates.
- On HTTP 409 with `detail.code: duplicate_template`, show the returned matches and the `same_name`/`same_spec` reasons. Do not silently rename the template or retry with `allow_duplicate: true`.
- Set `allow_duplicate: true` only after the user explicitly chooses to keep the duplicate. Expect the override and matched IDs to be audited.
- Re-run duplicate checks on identity-changing patches. Do not treat description or visibility-only edits as a new template identity.
- Resolve the exact ID and confirm ownership before patching or deleting. Never attempt to mutate a built-in template.

## Use the CLI fallback safely

Discover required registered pretrained models without downloading them:

```bash
ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1 python scripts/download_pretrained.py \
  --config configs/finetune_config.yaml --mode MODE
```

After the user approves the exact models and destination, run the same command without `ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1`, with `PRETRAINED_MODELS_DIR` already set. Alternatively, prefetch explicit registered names with `--names NAME...`.

- Verify the target root, disk capacity, gated-repository access, and current partial files before download.
- Never print `HF_TOKEN` or place it on the command line.
- Verify model sentinels and re-run the disabled discovery command after completion.
- Prefer `POST /api/v1/resources/dataset.libero/install` for the four-suite LIBERO download; do not rely on a training wrapper's interactive download prompts.

## Things not obvious from the docs

- Treat dataset validity and builder readiness as different facts. A Cosmos or VLM JSON directory can be valid inventory while remaining unavailable to the generic experiment builder.
- Treat a single-registration fingerprint as content provenance. Treat a mixture version as composition provenance only; it does not replace member fingerprints or automatically change when registered content changes. Revalidation does not retroactively rewrite an existing experiment snapshot.
- Expect dataset-statistics work and downloads to use the utility subsystem. Expect world-model preprocessing to join the shared GPU FIFO instead of the CPU/network utility queue.
- Expect utility processes to become `interrupted` if the UI service restarts; verify artifacts before retrying.
- Expect template duplicate identity to ignore per-run presentation/allocation fields while retaining scientific and operational choices.

## Related skills

- Use `$alphabrain-training` after datasets, mixtures, templates, and resources are ready.
- Use `$alphabrain-env-troubleshoot` for Python, CUDA, Transformers, LeRobot, Hugging Face, or filesystem failures.
- Use `$alphabrain-codebase-nav` to trace a catalog entry, loader, schema, or configuration merge beyond these workflows.
