---
name: alphabrain-deployment
description: >-
  Inspect, preflight, launch, monitor, test, and stop AlphaBrain checkpoint deployments through the managed HTTP API, with a guarded command-line fallback for repository model servers. Use for checkpoint format or deployment compatibility questions, OpenPI/LeRobot/AlphaBrain Pi0.5 source identification, deployment capabilities, WebSocket serving, one-time deployment API keys, managed multipart inference, deployment logs, or deployment lifecycle failures. 适用于 checkpoint 格式识别、部署兼容性检查、模型服务启动、推理验证、日志监控和正常停止。
---

# Deploy AlphaBrain checkpoints

Use the managed API whenever the backend is available. It performs static checkpoint inspection, runtime preflight, one-time key handling, and scheduling in the shared GPU FIFO. Use direct server commands only when the API cannot represent the task or the user explicitly requests the CLI path.

## Path convention

Run repository commands from the checkout root containing `pyproject.toml`, `alphabrain_ui/`, `deployment/`, and `scripts/`. Resolve that root before interpreting a relative checkpoint path. Treat all checkpoint paths as server-local paths; never reinterpret a Windows or macOS client path as a Linux server path.

Use `scripts/agent/alphabrain_api.py` for HTTP operations. Start with:

```bash
python scripts/agent/alphabrain_api.py status
python scripts/agent/alphabrain_api.py schema GET /api/v1/deployment/capabilities
python scripts/agent/alphabrain_api.py schema POST /api/v1/deployments
```

Let that client handle Personal/Lab authentication, cookies, CSRF, HTTPS policy, redaction, and exit codes. Do not reproduce authentication logic with ad-hoc `curl` commands.

## Where to find answers

| Question | Source of truth |
|---|---|
| Wired adapters, combinations, checkpoint kinds, parameters | `alphabrain_ui/registry/catalog.yaml` |
| Safe metadata-only checkpoint detection | `alphabrain_ui/deployment_registry.py` |
| Runtime, dependency, GPU, disk, host, and port checks | `alphabrain_ui/deployment_preflight.py` |
| Current request and response contracts | Live `/api/openapi.json`, then `alphabrain_ui/schemas.py` and `alphabrain_ui/app.py` |
| Model-server flags and health behavior | `deployment/model_server/` and its `README.md` |
| Managed lifecycle and queue semantics | `alphabrain_ui/deployments.py` and `docs/quickstart/web_ui.md` |

Read the live OpenAPI schema before constructing a mutation. Do not copy an old request example when the running backend and checkout differ.

## Identify the checkpoint before launching

1. Distinguish an indexed source from a local source:
   - Use `{"kind":"indexed","checkpoint_id":"..."}` only for an ID returned by `GET /api/v1/checkpoints`.
   - Use `{"kind":"local","path":"/absolute/server/path"}` for a server-local file or directory.
2. For an indexed source, read `GET /api/v1/checkpoints/{checkpoint_id}` and inspect its `inspection` object. Require `complete=true` for a training checkpoint.
3. Interpret the inspection fields separately:
   - Treat `checkpoint.format` as the on-disk loading layout, such as `self_contained`, `legacy_file`, `lerobot_pi05`, `openpi_pi05`, or `cosmos_policy`.
   - Treat `checkpoint.checkpoint_format` or `checkpoint_origin` as producer provenance: `alphabrain`, `lerobot`, `openpi`, or `unknown`.
   - Treat `detected.combination_id`, `adapter_id`, and `candidate_combination_ids` as the actual wired runtime choices.
4. For an arbitrary local path, use the metadata-only `inspect_checkpoint()` helper in `alphabrain_ui/deployment_registry.py` when provenance must be reported. The public deployment API has no separate local-path inspection endpoint; use deployment preflight for the authoritative launch decision.
5. Never infer format from a directory name. Report Pi0.5 explicitly as AlphaBrain BaseFramework, native LeRobot, OpenPI, or unknown.

Apply the current adapter boundary:

| Detected source | Allowed route |
|---|---|
| AlphaBrain self-contained or legacy BaseFramework checkpoint, including AlphaBrain Pi0.5 | `base_framework_websocket` when inspection selects a wired combination |
| Native LeRobot 0.6.0 Pi0.5 checkpoint | `lerobot_pi05_websocket` |
| Cosmos Policy bundle | `cosmos_policy_websocket` |
| OpenPI Pi0.5 checkpoint | Report unsupported unless the live catalog exposes a matching adapter; never relabel it as LeRobot |
| RL actor/critic/encoder bundle | Report non-deployable unless a live compatible adapter exists |

Do not deserialize weights merely to identify a checkpoint. Inspect filenames and YAML/JSON metadata only.

## Follow the managed API workflow

1. Query `GET /api/v1/deployment/capabilities`. Select only a returned combination and adapter. Require explicit user intent plus `acknowledge_experimental=true` for an experimental choice.
2. Build a `DeploymentRequest` from the live schema. Keep `combination_id` aligned with inspection, select `resources.strategy` deliberately, and default `endpoint.scope` to `local`. Use LAN scope only when the user asks for remote access and supplies a reachable advertised host.
3. Submit the exact body to `POST /api/v1/deployments/preflight` first. Continue only when `ok` and `can_submit` are true and no issue has level `error`. Present unresolved warnings when they affect precision, reachability, dependencies, storage, or model behavior.
4. Reuse the identical body for `POST /api/v1/deployments`; do not edit it between preflight and create. Put `--secret-output` on this same create command because the plaintext key cannot be recovered afterward:

```bash
python scripts/agent/alphabrain_api.py request POST /api/v1/deployments \
  --body /workspace/alphabrain-deployment.json --send \
  --secret-output /absolute/untracked/deployment-create.json
```

5. Capture the one-time `api_key` only in that new, user-authorized file outside version control. Require mode `0600`; the shared client applies it atomically, refuses to overwrite an existing secret file, and redacts normal stdout. Never print, `tee`, paste, log, commit, or place the plaintext key in a JSON request, shell argument, run snapshot, or final response. If the create call omitted `--secret-output`, the plaintext key is irretrievably lost. Explain that rotation is a separate destructive credential operation; do not rotate it implicitly.
6. Read the returned deployment ID, then run:

```bash
python scripts/agent/alphabrain_api.py watch deployment DEPLOYMENT_ID \
  --until running --timeout 86400 --follow-logs
```

7. Re-query `GET /api/v1/deployments/{id}`. Require `status=running`, a valid port, assigned GPU IDs, no `error`, and `managed_inference_available=true` before claiming readiness. The manager's running state already includes its service readiness check; use the model port's unauthenticated `GET /healthz` only as an additional network check. If the bounded watcher expires, re-query and resume watching the same deployment ID; do not create a duplicate.

Training, deployment, evaluation, and GPU utilities share one FIFO. Treat `queued` and `starting` as progress, not success. Treat `failed`, `cancelled`, and `stopped` before readiness as failure outcomes.

## Test managed inference

Use the managed inference endpoint only after the deployment is running:

```bash
python scripts/agent/alphabrain_api.py infer DEPLOYMENT_ID \
  --instruction "pick up the red block" \
  --image /absolute/path/to/camera.png \
  --states /absolute/path/to/states.json
```

Treat that command as a schematic single-view example and the selected adapter's input contract as authoritative. Native LeRobot Pi0.5 requires exactly three ordered views (`front`, `left_wrist`, `right_wrist`) per instruction and state shape `[B, 7]` or `[B, T, 7]`; for one instruction, pass those three images in that order and a seven-value state vector. Do not infer an AlphaBrain BaseFramework checkpoint's input contract from the LeRobot adapter merely because both are labeled Pi0.5, and inspect every future adapter rather than reusing these assumptions.

Let the client construct the endpoint's multipart form (`instructions`, optional `states`, optional `image_counts`, and `images`). With multiple instructions, omit `--image-count` to broadcast every uploaded image to every instruction. To assign distinct groups, repeat `--image-count N` once per instruction and make the counts sum to the number of `--image` arguments. Keep input retention off by default; add `--save-inputs` only when the user explicitly requests persistent copies. Do not send the deployment's plaintext API key to this endpoint: the backend uses its private controller credential for managed inference.

Require an HTTP-success response whose inference run has `status=completed`, a nonempty model output, and no error. Use `GET /api/v1/inference-runs/{run_id}` for the durable result. Do not expose saved image paths or model outputs beyond the user's requested scope.

## Stop without escalating

Preserve a persistent deployment after validation unless the user requests shutdown. To stop an owned running deployment:

1. Re-query the ID and confirm its name, owner, and current status.
2. Send `POST /api/v1/deployments/{id}/stop` through the shared client with `--send --dangerous`.
3. Watch that deployment with `--until stopped`; verify that its GPU reservation is released.

Use `cancel` only for a specifically identified queued deployment. Do not delete, restart, rotate keys, or force-kill as part of a normal workflow. Route those operations to explicit user authorization; force-kill is administrator-only.

## Use the CLI fallback carefully

Use the direct model-server path only after checkpoint inspection and only from the repository root:

- Select `deployment/model_server/server_policy.py` for a wired BaseFramework checkpoint.
- Select `deployment/model_server/server_policy_lerobot_pi05.py` for native LeRobot Pi0.5.
- Select `deployment/model_server/server_policy_cosmos.py` for Cosmos Policy and supply both checkpoint and local base-model directories.

Run the selected entrypoint with `--help` and follow its current flags. Bind to `127.0.0.1` by default, choose a confirmed free port, inspect GPU ownership, and verify `GET /healthz` before inference. Direct launches do not join the UI's GPU FIFO and do not receive managed controller-key storage, audit records, idle lifecycle, or API cleanup. Prefer the managed API for LAN exposure or persistent services.

Do not force an incompatible adapter, patch a checkpoint, download a missing base model, or start an unauthenticated LAN service merely to make a smoke test pass.

## Things not obvious from the docs

- A checkpoint can be structurally valid yet not deployable because its model combination has no wired adapter.
- An ambiguous inspection requires an explicit `combination_id`; do not select the first candidate.
- Health probes do not reset the model server's idle timeout.
- The public deployment key is returned only on create or explicit rotation. SQLite retains only its digest and prefix.
- Managed inference uses a separate backend-only controller credential; it does not reveal or require the public deployment key.
- A running deployment holds its GPU reservation until stop, failure, or idle timeout.

## Related skills

- Use `$alphabrain-setup` when the backend, Python environment, or model-server interpreter is not ready.
- Use `$alphabrain-env-troubleshoot` when preflight or startup reports dependency, CUDA, import, or checkpoint-load failures.
- Use `$alphabrain-evaluation` to run LIBERO or RoboCasa against a checkpoint or this managed deployment.
- Use `$alphabrain-codebase-nav` to trace a new adapter or checkpoint detector without operating a deployment.
