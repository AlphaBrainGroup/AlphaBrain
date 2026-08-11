---
name: alphabrain-training
description: >
  Resolve, preflight, submit, monitor, stop, and verify AlphaBrain training through
  the backend HTTP API or the repository's checked-in CLI launchers, including base
  VLA finetuning, NeuroVLA pretraining and R-STDP, continual learning, RL-Token, and
  world-model training. Use when Codex needs to prepare or start a training experiment,
  select GPUs, inspect a command preview, follow logs and metrics, resume a run, verify
  checkpoints, or diagnose a failed submission; also trigger for 训练, 微调, 提交实验,
  训练队列, 训练日志, 断点续训, or checkpoint 完成验证 requests.
---

# Run AlphaBrain Training

## Path convention

- Start from the repository root containing `pyproject.toml`, `AlphaBrain/`, and `alphabrain_ui/`.
- Use `python scripts/agent/alphabrain_api.py` for backend operations.
- Read request schemas from the live `/api/openapi.json` through the client's `schema` command. Treat the current capability catalog and preflight response as authoritative over remembered presets.
- Keep credentials out of experiment specs, request files, argv, logs, command previews, and immutable run snapshots.

## Where to find answers

| Question | Authoritative source |
|---|---|
| Wired workflows, components, combinations, and compatibility | `alphabrain_ui/registry/catalog.yaml`, `alphabrain_ui/capabilities.py`, and `GET /api/v1/capabilities` |
| Experiment request and response contracts | `/api/openapi.json` and `alphabrain_ui/schemas.py` |
| Resolution, merge order, validation, and static preflight | `alphabrain_ui/configuration.py` and `alphabrain_ui/preflight.py` |
| API-to-CLI launch stages and immutable snapshots | `alphabrain_ui/launchers.py` |
| Job states, FIFO scheduling, events, and checkpoint indexing | `alphabrain_ui/jobs.py` and `alphabrain_ui/app.py` |
| Native launch arguments and family-specific constraints | `scripts/run_finetune.sh` and the README beside each specialized wrapper |

## Choose an execution path

1. Run `python scripts/agent/alphabrain_api.py status`.
2. Prefer the backend API when it is reachable. Preserve its capability resolution, immutable snapshots, disk/GPU preflight, ownership checks, audit trail, and global GPU FIFO shared with deployments, evaluations, and GPU utilities.
3. Use a native CLI launcher only when the API does not cover the workflow, source-level debugging requires it, or the user explicitly requests CLI execution.
4. Do not launch the same run through both paths. Do not expect a direct CLI process to appear in API jobs, hold an API GPU reservation, or follow API lifecycle controls.
5. Obtain explicit user authorization for the final resolved configuration before submitting a new run, resuming from a checkpoint, or stopping an existing job.

Use `$alphabrain-data-resources` first when a required dataset, mixture, pretrained model, or template is not ready. Use `$alphabrain-env-troubleshoot` when imports, CUDA, storage, remote execution, or dependency versions fail.

## Run the API-first workflow

### 1. Discover the live training surface

Follow this read-only order:

```text
GET /api/v1/capabilities
 -> GET /api/v1/training/target
 -> GET /api/v1/gpus                         [local target]
 -> GET /api/v1/remote-training/metrics      [remote target]
 -> GET /api/v1/storage
```

Use the repository client, for example:

```bash
python scripts/agent/alphabrain_api.py request GET /api/v1/capabilities
python scripts/agent/alphabrain_api.py request GET /api/v1/training/target
python scripts/agent/alphabrain_api.py request GET /api/v1/gpus
```

- Select only a catalog combination and workflow returned for the current checkout and user.
- Request experimental catalog entries only when the user explicitly asks for them and experimental access is enabled.
- Honor `training/target.mode`. Treat remote GPU IDs, paths, Python, and environment as remote-host values rather than local server values.
- Review existing reservations and `/api/v1/workloads`; do not equate low utilization with scheduler availability.
- Reuse a visible server template or construct a request from catalog choices. Reference registered data with `dataset.registration_id` or `dataset.mixture_id`; never inject an arbitrary `dataset.mixture_spec`.
- When using a weighted mixture, re-read `GET /api/v1/datasets/mixtures`, select the exact ID/version, and query
  `GET /api/v1/datasets/{registration_id}` for every member before resolve. Require each member to remain visible and
  `ready`; review its current fingerprint, format/builder support, path, pattern, weight, robot type, and trajectory
  limit. Retain those member fingerprints as separate evidence because the current mixture snapshot freezes its
  ID/version and expanded paths/options, not each member fingerprint.

### 2. Resolve the experiment without submitting it

Inspect the schema, then send a reviewed `ExperimentRequest` containing `name`, `spec`, and the default `acknowledge_experimental: false`:

```bash
python scripts/agent/alphabrain_api.py schema POST /api/v1/experiments/resolve
python scripts/agent/alphabrain_api.py request POST /api/v1/experiments/resolve \
  --body /workspace/alphabrain-experiment.json --send
```

- Review `resolved`, `compatibility`, every issue, `diff`, and every `command_preview` stage.
- Verify the model/action-head combination, dataset source and frozen provenance, training mode, resume mode/checkpoint, batch and step counts, precision, GPU request, output root, run ID, W&B mode, and specialized parameters.
- Treat `resolved` as a server-generated preview. Keep the original reviewed request as the input to preflight and submit; do not edit the resolved response and submit it as a new spec.
- Require an explicit risk acknowledgement before changing the request to `acknowledge_experimental: true` for an experimental combination.

### 3. Preflight the identical request

Follow resolution with:

```bash
python scripts/agent/alphabrain_api.py schema POST /api/v1/experiments/preflight
python scripts/agent/alphabrain_api.py request POST /api/v1/experiments/preflight \
  --body /workspace/alphabrain-experiment.json --send
```

- Require `can_submit: true`. Never submit around an error or reconstruct a command from `command_preview` to bypass the gate.
- Resolve every error at its authoritative source, then rerun both resolve and preflight.
- Report warnings and obtain the user's decision when they affect reproducibility, experimental status, resource use, resume semantics, or outputs.
- Reconfirm the final request if server resolution, catalog state, dataset fingerprint/mixture version, target, or GPU availability changed.

### 4. Submit once and capture all identifiers

Submit the same reviewed file only after preflight succeeds:

```bash
python scripts/agent/alphabrain_api.py schema POST /api/v1/experiments/submit
python scripts/agent/alphabrain_api.py request POST /api/v1/experiments/submit \
  --body /workspace/alphabrain-experiment.json --send
```

- Record `experiment_id`, every returned job ID, warning, and output directory. Immediately query `GET /api/v1/experiments/{experiment_id}` to record the resolved spec and snapshot path that are not returned directly by submit.
- Do not retry a timed-out submit blindly. Query `GET /api/v1/experiments` and match the requested name, owner, time, and resolved run ID before deciding whether a retry is safe.
- Expect dependency stages to start as `blocked` and advance only after their prerequisite job reaches `completed`.
- Expect the API launcher to set `ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1`; treat missing resources as preflight failures instead of allowing training to download weights implicitly.

### 5. Monitor jobs and preserve FIFO semantics

For each returned job, run:

```bash
python scripts/agent/alphabrain_api.py watch job JOB_ID \
  --timeout 86400 --follow-logs
```

Use this observation order when checking status manually:

```text
GET /api/v1/experiments/{experiment_id}
 -> GET /api/v1/jobs/{job_id}
 -> GET /api/v1/jobs/{job_id}/events
 -> repeat for every stage
```

- Let the client retain SSE `Last-Event-ID` across bounded reconnects. Do not start a duplicate because an event stream disconnects.
- Choose a bounded watch timeout appropriate to the observation window. If it expires, re-query the same job and experiment, then resume watching the same ID; timeout is not terminal state and never authorizes resubmission.
- Treat `queued`, `blocked`, `starting`, `running`, and `stopping` as nonterminal.
- Treat only `completed` as successful. Treat `failed`, `stopped`, `cancelled`, `dependency_failed`, and `interrupted` as non-success and report `error`, `exit_code`, the final log tail, and the failed stage.
- Require all jobs to be `completed` and the experiment status to be `completed`; a successful first stage does not complete a multi-stage experiment.
- Stop or cancel only after re-querying the exact job ID and owner. Use the client's `--dangerous` gate for `cancel`, `stop`, `terminate`, or `force-kill`; use force-kill only under explicit administrator instruction.

### 6. Verify checkpoint artifacts

After all jobs complete, follow this order:

```text
GET /api/v1/checkpoints?experiment_id={experiment_id}
 -> GET /api/v1/checkpoints/{checkpoint_id}
 -> GET /api/v1/experiments/{experiment_id}
```

- Require the expected final checkpoint to appear with `is_complete: true` (or its `complete` alias), a nonzero `size_bytes`, the expected name/final sentinel, and a path inside the recorded job output directory. Match a numeric step when the index provides one; do not invent a step for a format such as NeuroVLA `final_model` that identifies final output by name instead. For that format, prove the completed horizon from the resolved `max_train_steps`, terminal metrics/logs, and final-save marker.
- Verify the local path and framework-specific metadata/configuration files when the backend and training output share a filesystem. Do not deserialize weight tensors merely to prove completion.
- Distinguish `is_resumable` from `is_complete`. Do not claim a complete inference artifact is resumable unless training state or resume metadata is present.
- Report training as incomplete when the job completed but the expected checkpoint contract is absent. Do not use a W&B run, output directory, or checkpoint-like folder name as a completion marker by itself.
- Use `$alphabrain-deployment` or `$alphabrain-evaluation` only after checkpoint verification; do not infer deployability or benchmark compatibility from training success.

## Use native CLI launchers safely

Inspect the selected wrapper and its adjacent README before launching. Prefer these checked-in entrypoints:

| Workflow | Entrypoint |
|---|---|
| Standard/base VLA finetune | `scripts/run_finetune.sh` or `scripts/run_base_vla/train.sh` |
| NeuroVLA pretrain / R-STDP | `scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh` / `run_stdp_finetune.sh` |
| Continual learning | `scripts/run_continual_learning_scripts/run_cl_train.sh` |
| RL-Token RLT / RLT-a | `scripts/run_rl_scripts/run_rlt_pretrain.sh` then `run_rlt_rl.sh`; set `TRACK=rlt_a` only for the RLT-a flow documented in `AlphaBrain/training/reinforcement_learning/algos/RLT_a/README.md` |
| World Model / Cosmos Policy | `scripts/run_world_model/train/run_world_model.sh` / `run_cosmos_policy.sh` |

Run standard finetuning with reliable pipeline failure propagation:

```bash
ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1 bash -o pipefail scripts/run_finetune.sh \
  configs/finetune_config.yaml --mode MODE
```

- Verify the active Python/Conda environment, `AlphaBrain.__file__`, required imports, config mode, pretrained paths, dataset paths, checkpoint compatibility, output/run ID, free port, disk space, and actual GPU ownership before launch.
- Run the disabled discovery command from `$alphabrain-data-resources` before standard finetuning. Prepare every missing model and dataset as a separate user-approved operation.
- Do not launch an unattended standard wrapper while resources are missing. `ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1` disables `scripts/download_pretrained.py`, but the legacy shell wrapper can still offer interactive model or dataset downloads after parsing the config.
- Use `bash -o pipefail` for `scripts/run_finetune.sh`; its final training process is piped through `grep` and `tee`, so plain `bash` can otherwise mask the trainer's nonzero status.
- Treat `scripts/run_base_vla/train.sh MODE` as a convenience wrapper. Use the unified command above when passing a custom config or when exact failure propagation matters.
- Set family-specific arguments only as documented by the selected wrapper. Preserve NeuroVLA's recommended `sdpa` attention unless the user explicitly accepts the research change; require `--pretrained` for an intended R-STDP continuation.
- Inspect `/api/v1/workloads` when the backend is running and inspect system GPU processes directly. Never assume the API scheduler protects a GPU used by a direct CLI run.
- Capture the process, command, environment identity, output path, log, and exit status. Verify completion from the final checkpoint contract as well as logs and process state.
- Do not use API stop, events, or checkpoint indexing as lifecycle guarantees for a direct CLI run. Perform read-only artifact inspection directly and report that the run is unmanaged.

## Things not obvious from the docs

- Expect API resolution to apply model, dataset, workflow, expert, and runtime layers before freezing a server-side snapshot. Do not reproduce that merge manually from a template name.
- Expect `/experiments/resolve`, `/preflight`, and `/submit` to resolve against current server state each time. Reuse one reviewed request and re-check changed output rather than assuming the first preview is permanent.
- Expect a direct registration source to freeze its fingerprint. Expect a mixture source to freeze the mixture ID/version and expanded paths/options; the current snapshot does not embed every member fingerprint, so preserve the pre-resolve member-fingerprint check separately. Reject a client-supplied arbitrary `mixture_spec`.
- Expect all managed training, deployment, evaluation, and GPU preprocessing workloads to share one global FIFO. Do not bypass it with a previewed CLI command.
- Expect backend restarts to reconcile managed training by recorded process identity; treat `interrupted` as a failure requiring investigation, not an automatic resume signal.
- Expect checkpoint indexing to mark completeness from recognized weight sentinels and resumability from separate training-state metadata. Verify both according to the user's goal.

## Related skills

- Use `$alphabrain-data-resources` to prepare models, datasets, mixtures, statistics, and templates.
- Use `$alphabrain-deployment` to inspect and serve a completed checkpoint.
- Use `$alphabrain-evaluation` to benchmark a verified checkpoint.
- Use `$alphabrain-env-troubleshoot` for runtime failures and `$alphabrain-codebase-nav` for configuration or launcher tracing.
