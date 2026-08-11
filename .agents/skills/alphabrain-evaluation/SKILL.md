---
name: alphabrain-evaluation
description: >-
  Inspect compatibility, preflight, submit, monitor, and verify AlphaBrain benchmark evaluations through the managed HTTP API, with repository CLI wrappers as a fallback. Use for LIBERO, RoboCasa365, RoboCasa Tabletop, or LIBERO-plus evaluation; checkpoint or managed-deployment sources; quick, standard, full, custom, batch, continual-learning matrix, RL-iteration, Online-STDP, or world-model evaluation; evaluation logs, results, matrices, series, artifacts, videos, or completion diagnosis. 适用于评测 checkpoint 检查、benchmark 选择、预检、提交、监控、结果与视频完整性验证。
---

# Evaluate AlphaBrain policies

Use the managed API whenever the backend is available. It resolves checkpoint adapters and benchmark contracts, validates the simulator environment, schedules work in the shared GPU FIFO, and normalizes results and artifacts. Use `scripts/run_eval.sh` only when the API cannot express the requested run or the user explicitly requests CLI execution.

## Path convention

Run repository commands from the checkout root containing `pyproject.toml`, `alphabrain_ui/`, `benchmarks/`, and `scripts/run_eval.sh`. Resolve all checkpoint, config, result, simulator, and dataset paths on the server. Never substitute a browser-client path for a server-local path.

Use `scripts/agent/alphabrain_api.py` for HTTP operations. Start with:

```bash
python scripts/agent/alphabrain_api.py status
python scripts/agent/alphabrain_api.py schema GET /api/v1/evaluation-benchmarks
python scripts/agent/alphabrain_api.py schema POST /api/v1/evaluations/preflight
```

Let that client handle authentication, CSRF, HTTPS policy, redaction, SSE cursors, and exit codes. Do not recreate session handling with ad-hoc HTTP calls.

## Where to find answers

| Question | Source of truth |
|---|---|
| Benchmarks, suites/task sets, presets, parameter schemas, compatibility | `alphabrain_ui/evaluation_registry.py` |
| Checkpoint, simulator, Python, GPU, disk, W&B, and deployment-source preflight | `alphabrain_ui/evaluation_preflight.py` |
| Request and result schemas | Live `/api/openapi.json`, then `alphabrain_ui/schemas.py` |
| Group expansion and specialized workflows | `alphabrain_ui/evaluation_groups.py` and `alphabrain_ui/specialized_evaluation.py` |
| Managed lifecycle and completion rules | `alphabrain_ui/evaluations.py` and `alphabrain_ui/evaluation_result.py` |
| CLI orchestration and cleanup | `scripts/run_eval.sh`, `scripts/parse_config.py`, and `configs/finetune_config.yaml` |
| Simulator-specific behavior | The selected directory under `benchmarks/` |

Query the live catalog and OpenAPI schema before preparing a run. Do not assume that a benchmark, suite, parameter, or model combination remains available because it appears in an older README.

## Choose the source deliberately

Use `source_kind=temporary_checkpoint` to evaluate an indexed or absolute server-local checkpoint. The runner starts a loopback-only temporary policy server, waits for readiness, runs the simulator, and attempts cleanup on every exit.

Use `source_kind=managed_deployment` to reuse an existing UI-managed deployment. Require the deployment to be `running`, have valid assigned GPUs and a loopback service port, and expose a benchmark-compatible combination. This route reuses its GPU reservation and private controller credential; it must not request another model GPU or stop the deployment afterward.

Apply the request-model restrictions from the live schema. In particular, managed sources support only the kinds currently accepted by `EvaluationRequest`, and specialized multi-GPU requests are restricted to their declared kinds. Never work around validation by placing source fields inside `parameters`.

## Follow the managed API workflow

1. For a temporary checkpoint, call `POST /api/v1/evaluations/inspect-checkpoint` with an indexed ID or absolute local path. Require `ok=true`, inspect every issue, and select only a returned candidate combination. If multiple candidates exist, obtain or preserve an explicit `combination_id`; never choose the first candidate silently.
2. Read `compatible_benchmark_ids` from inspection. Do not use a model with a benchmark solely because their names look related.
3. Call `GET /api/v1/evaluation-benchmarks`. Select a returned benchmark, suite/task set, split, preset, and parameters. Check each benchmark's `status`, `readiness`, environment requirements, parameter schema, and compatible combination IDs.
4. Prefer verified integrations. Treat experimental benchmarks or combinations as opt-in: require the user's experimental setting, explicit intent, and `acknowledge_experimental=true`.
5. Build an `EvaluationRequest` from the live schema. Keep simulator controls in `parameters` and model-server controls in `model_parameters`. Never place passwords, API keys, W&B tokens, or deployment keys in either object.
6. Send the exact body to `POST /api/v1/evaluations/preflight`. Continue only when `ok` and, when present, `can_submit` are true, every child preflight is valid, and no issue has level `error`. Resolve warnings that affect scientific scope, task selection, checkpoint format, action chunk, environment readiness, GPU allocation, disk, or result completeness.
7. Reuse the identical body for `POST /api/v1/evaluations` with `--send`. Do not change the seed, suite, episode count, checkpoint, or action settings after preflight.
8. For a standard response, capture `evaluation.id` and run:

```bash
python scripts/agent/alphabrain_api.py watch evaluation EVALUATION_ID \
  --until completed --timeout 86400 --follow-logs
```

9. For a grouped response, capture `group.id` and every child evaluation ID. Watch every child to a terminal state, then read `GET /api/v1/evaluation-groups/{group_id}` and `/result`. Group `partial` is not success. If a bounded watcher expires, re-query and resume watching the same child ID; timeout is not terminal state and never authorizes duplicate submission.

Treat `queued`, `starting`, `running`, and `stopping` as nonterminal. Treat `failed`, `stopped`, `cancelled`, and `interrupted` as unsuccessful evaluation outcomes. The shared queue can legitimately leave a healthy request queued behind training, deployment, or GPU utility work.

## Select workload scope explicitly

Use catalog presets according to their returned definitions:

- Use `quick` only as a pipeline smoke test, not as a benchmark score.
- Use `standard` for the catalog's standard task coverage and episode count.
- Use `full` for the complete benchmark settings exposed by the catalog.
- Use `custom` only with an explicit task subset, episode/trial count, seed, views, action steps, parallelism, and maximum-step policy where the benchmark schema exposes them.

Preserve the requested evaluation unit. A single LIBERO suite is not equivalent to `libero_all`; a RoboCasa task set or split is not interchangeable with another. For comparisons, keep checkpoint, suite/task signature, task limits, episode count, seed policy, views, and execution settings aligned.

Use `kind=batch`, `cl_matrix`, `rl_iterations`, `online_stdp`, or `world_model_video` only when the live schema and preflight expand the request into valid children. Verify the child count and purpose before submitting potentially large runs.

## Verify completion strongly

Never declare success from a created output directory, an exited process, a final log line, or an HTTP 200 alone.

For a standard evaluation, require all of the following:

1. `GET /api/v1/evaluations/{id}` reports `status=completed`, no error, a finished timestamp, and an exit code consistent with success.
2. `GET /api/v1/evaluations/{id}/result` exists, has schema `evaluation-result-v1` or `evaluation-result-v2`, reports `status=completed`, contains no result error, and identifies the expected benchmark/checkpoint/suite.
3. The summary is internally consistent: successes do not exceed episodes, rates match counts within numeric tolerance, and task/episode coverage matches the resolved request. Do not invent expected counts for a specialized schema; validate its declared matrices, series, comparisons, children, or artifacts instead.
4. `GET /api/v1/evaluations/{id}/artifacts` includes the normalized result and configuration snapshot plus the logs/artifacts promised by the resolved workflow. Match declared video/artifact paths in the result to the authenticated artifact list; require the expected rollout-video coverage only when the selected client and configuration promise it.
5. Download artifacts through `/artifacts/{artifact_id}` with the shared client's explicit `--output` option. It
   refuses an existing file by default; use `--overwrite` only after re-verifying the exact destination and explicitly
   intending replacement. Never read an arbitrary server path supplied by an API response, and never treat a partial
   download as complete.

For a group, additionally require every child to complete, `group.status=completed`, a valid `evaluation-result-v2`, matching child cardinality, and the expected matrices/series/comparisons for the requested specialized kind. Report `partial` and mixed child states exactly rather than averaging them into success.

Treat W&B upload separately from simulation. A failed optional upload does not invalidate a completed evaluation; report `wandb_status` and `wandb_error` separately. Do not configure or accept W&B credentials in this skill.

## Cancel or stop only by explicit target

Use `POST /api/v1/evaluations/{id}/cancel` only for a re-queried queued run and `POST /api/v1/evaluations/{id}/stop` only for a re-queried active run. Require `--send --dangerous` with the shared client and verify the owner, name, and status immediately beforehand. Do not force-kill, manage credentials, retry W&B uploads, or cancel an entire group without explicit user authorization.

## Use the CLI fallback carefully

Use the repository wrapper rather than manually coordinating server and simulator processes:

```bash
bash scripts/run_eval.sh MODE CONFIG_FILE
```

1. Inspect `configs/finetune_config.yaml` for current evaluation modes and the selected benchmark's README for its environment contract. Common modes include LIBERO and RoboCasa wrappers, but accept only modes present in the current config.
2. Create a separate, reviewable config for changed checkpoint or task settings; do not overwrite a tracked shared config or a prior run snapshot.
3. Parse and inspect the mode before launch with `scripts/parse_config.py`. Confirm the checkpoint, server entrypoint, simulator Python, host/port, benchmark, suite/task set, episode count, seed, action steps, GPU, and output directory.
4. Confirm GPU ownership and port availability. Direct CLI runs do not join the UI FIFO.
5. Let `scripts/run_eval.sh` start the temporary server, wait for readiness, run the correct benchmark client, write the normalized result, and clean up the server. Do not replace it with old two-terminal scripts unless diagnosing the wrapper itself.
6. Verify `evaluation-result-v1.json`, server/evaluation logs, exit status, task coverage, and artifacts with the same completion rules used for API runs. Require the progress artifact only when the managed launcher or direct CLI configuration sets `EVAL_PROGRESS_PATH`; an unmanaged wrapper run does not create it unconditionally.

Keep the policy server and simulator in their required Python environments. Do not install benchmark dependencies into the AlphaBrain training environment merely to make one CLI run start.

## Things not obvious from the docs

- Verified integrations currently include LIBERO and RoboCasa365; RoboCasa Tabletop and LIBERO-plus remain experimental unless the live catalog says otherwise.
- Temporary checkpoint evaluation binds its policy server to loopback and cleans it up. Managed-deployment evaluation reuses a running service and must not stop it.
- LIBERO and LIBERO-plus cap trials per task because their clients index a finite initial-state set.
- Parallel RoboCasa recording may intentionally limit video writers to avoid corruption.
- A completed simulation and a successful optional W&B upload are separate outcomes.
- Grouped evaluations use result v2; standard runs normally use result v1. Validate the schema declared by the created run rather than assuming one file shape.

## Related skills

- Use `$alphabrain-deployment` to create or diagnose a managed policy source.
- Use `$alphabrain-setup` when simulator paths or Python interpreters are not configured.
- Use `$alphabrain-env-troubleshoot` for import, CUDA, MuJoCo, headless-rendering, or dependency failures.
- Use `$alphabrain-codebase-nav` to trace a benchmark adapter or specialized result generator without launching an evaluation.
