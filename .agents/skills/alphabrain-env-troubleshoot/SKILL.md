---
name: alphabrain-env-troubleshoot
description: >-
  Diagnose AlphaBrain CLI and HTTP-backend failures involving Python or Conda selection, missing packages, CUDA/NVML,
  Torch/FlashAttention/Transformers/LeRobot compatibility, configuration resolution, authentication, paths, GPU
  reservations, checkpoints, and benchmark runtimes. Use for tracebacks, failed imports, failed preflight, jobs that
  cannot start, or inconsistent runtime behavior; also trigger for Chinese requests such as “为什么启动失败”,
  “缺少什么 package”, “CUDA 看不到”, “preflight 报错”, or “这个环境还能不能用”.
---

# Troubleshoot AlphaBrain environments

Preserve evidence, identify the exact failing interpreter and contract, and apply the smallest verified repair. Treat
environment rebuilds, dependency downgrades, cache deletion, and checkpoint changes as high-impact actions that need
explicit user approval.

## Path convention

- Resolve `<repo-root>` from `pyproject.toml`, `AlphaBrain/`, `configs/`, and `alphabrain_ui/`; run repository probes
  there.
- Name every interpreter explicitly: `<backend-python>`, `<training-python>`, `<model-server-python>`, or
  `<benchmark-python>`. Do not use an unqualified `pip` when diagnosing multiple environments.
- Resolve backend state from `--state-dir`, `ALPHABRAIN_UI_HOME`, or the default `<repo-root>/.alphabrain-ui/` before
  reading logs. Do not assume a state directory or output directory proves completion.
- Use `scripts/agent/alphabrain_api.py` for authenticated API inspection. Keep secrets out of shell history, argv,
  diagnostic bundles, and final answers.

## Where to find canonical answers

| Failure area | Canonical source |
|---|---|
| Training preflight codes | `alphabrain_ui/preflight.py` |
| Deployment Python/checkpoint/GPU checks | `alphabrain_ui/deployment_preflight.py`, `alphabrain_ui/deployment_registry.py` |
| Evaluation interpreter and simulator checks | `alphabrain_ui/evaluation_preflight.py`, `alphabrain_ui/evaluation_registry.py` |
| Configuration resolution | `alphabrain_ui/configuration.py`, `configs/finetune_config.yaml` |
| Command construction | `alphabrain_ui/launchers.py`, `scripts/run_finetune.sh`, capability wrapper |
| Authentication/CSRF | `alphabrain_ui/dependencies.py`, setup/login middleware in `alphabrain_ui/app.py` |
| GPU reservations and process state | jobs/deployments/evaluations/utility managers plus their HTTP detail/events routes |
| Package expectations | `requirements.txt`, `requirements-ui.txt`, selected workflow documentation |

## Workflow at a glance

1. Capture the exact failing command or method/path, full redacted error, timestamp, workload ID, and last relevant
log lines. Record source and environment identity before changing anything:

```bash
git status --short --branch
git rev-parse HEAD
<failing-python> --version
<failing-python> -c "import os,sys; print(sys.executable); print(os.getcwd())"
<failing-python> -m pip check
```

2. Identify the runtime boundary. Query the backend rather than assuming its configured target:

```bash
python scripts/agent/alphabrain_api.py status
python scripts/agent/alphabrain_api.py request GET /api/v1/training/target
python scripts/agent/alphabrain_api.py request GET /api/v1/gpus
```

Use the job/deployment/evaluation detail and events endpoint for a known workload ID. Cross-check PID/process state,
GPU ownership, logs, terminal status, and expected artifacts before calling a workload failed or complete.

3. Re-run the relevant read-only resolve, inspection, or preflight request with the original payload. Read structured
`code`, `field`, `detail`, and `can_submit` values; do not infer a cause from the translated message alone. Never
submit a workload while diagnosing preflight.

4. Probe only the failing layer:

```bash
<failing-python> -c "from importlib.util import find_spec; import sys; print(sys.executable); print(find_spec('AlphaBrain'))"
<failing-python> -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
<failing-python> -c "import transformers; print(transformers.__version__)"
<failing-python> -m pip show torch transformers flash-attn lerobot
nvidia-smi
```

Run the model-server import contract in the configured model-server interpreter:

```bash
<model-server-python> -c "import torch, transformers, websockets, msgpack, numpy, AlphaBrain; print('imports ok')"
```

For evaluation, use the benchmark interpreter and the module list declared for that benchmark in
`alphabrain_ui/evaluation_preflight.py`; do not test simulator packages in the training interpreter.

5. Explain the minimal repair and its compatibility impact before changing state. Obtain explicit approval before
changing an existing environment's pinned package, rebuilding FlashAttention, deleting an environment or cache,
removing an output directory, replacing a checkpoint, or changing persistent backend settings.

6. Apply one repair at a time with the target interpreter's `python -m pip` or the approved Conda command. Preserve a
before/after package record. Do not combine a Transformers change, LeRobot installation, and Torch rebuild into one
unreviewable operation.

7. Re-run the exact import or preflight that failed, then a nearby contract test. Report what is verified and what
still requires a real GPU/model/simulator run.

## Error signatures and first checks

| Signature or issue code | First check | Safe direction |
|---|---|---|
| `ModuleNotFoundError` / wrong module version | `sys.executable`, `find_spec`, `python -m pip show` | Install only into the interpreter that executes the failing stage |
| `python_version` | Actual backend Python | Use Python 3.10+; create an isolated environment instead of in-place replacement |
| `model_server_python_not_found` | Saved executable path and execute bit | Point settings to an existing absolute interpreter |
| `model_server_dependencies_missing` | Six-module model-server import probe | Add only missing/incompatible runtime packages, then rerun deployment preflight |
| `config_resolution_failed` / `unknown_training_mode` | Mode name and all source YAML paths | Fix the source mode or request payload; do not edit immutable run snapshots |
| `launcher_missing` | Resolved command preview and repository revision | Restore/use the wired wrapper or choose a supported combination |
| `dataset_root_missing` / `dataset_not_found` | Effective data root and server-side registration | Inspect/register the correct supported dataset through the data-resource workflow |
| `pretrained_root_missing` / `checkpoint_not_found` | Effective path, format, and mount visibility | Correct the path or acquire the resource separately; do not fabricate metadata |
| `output_exists` | Resolved run ID and output root | Choose a new run ID/output; never delete or overwrite an existing run as a quick fix |
| `nvml_unavailable` / GPU visibility codes | `nvidia-smi`, `CUDA_VISIBLE_DEVICES`, backend NVML, API reservations | Separate driver/visibility failure from a scheduler reservation or expected queue |
| FlashAttention undefined symbol or ABI error | Torch, CUDA, FlashAttention build versions | Use a recipe-supported SDPA override or rebuild only after approval |
| Transformers missing/unexpected weight keys | Checkpoint producer/format and expected Transformers version | Test the required version in an isolated clone; never blindly downgrade a working environment |
| LeRobot import/API mismatch | Installed distribution source/version and dataloader import path | Match the workflow's tested contract; do not assume a package name implies checkpoint compatibility |
| HTTP `428` / `401` / `403` | setup status, deployment mode, login cookie, CSRF cookie/header | Initialize once, authenticate in lab mode, and let the shared client carry CSRF state |
| Proxy or `socksio` error before loopback connection | inherited `ALL_PROXY`/`HTTP_PROXY`/`HTTPS_PROXY` and client flags | Keep the client's default proxy isolation; use `--trust-env` only for a reviewed remote proxy/custom-CA requirement |
| Evaluation interpreter/import failure | Benchmark-specific Python and required modules | Repair the isolated simulator environment, not the training environment |
| Remote training connection failure | SSH target, host key, repo path, GPU IDs, noninteractive activation | Reproduce with a read-only SSH probe; never disable host-key verification |

## Things not obvious from the docs

- A successful backend import does not prove the training, model-server, or benchmark interpreter works. Probe each
  executable that actually owns a stage.
- A busy GPU may belong to a direct CLI process that the API FIFO cannot see. Compare API reservations with
  `nvidia-smi` and process ownership before changing queue state.
- The CLI finetune wrapper may download missing weights unless `ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1` is set. Disable
  implicit download during diagnosis and handle acquisition as a separate authorized step.
- `requirements-ui.txt` intentionally omits training packages. Missing Torch or LeRobot in a backend-only environment
  is not itself a backend defect.
- FlashAttention is ABI-sensitive to Torch/CUDA. A force reinstall is not a harmless generic fix.
- A checkpoint directory can be complete yet use an unsupported deployment/evaluation adapter. Use metadata-only
  inspection and preserve the checkpoint.
- Preflight `gpu_queue` or `deployment_will_queue_for_gpu` is informational, not a dependency failure.
- Never mark success from a created directory, a queued record, or a zero-exit wrapper alone; require the workload's
  success terminal state and expected final artifacts.

## Escalate with a useful report

If the minimal repair does not resolve the failure, provide a redacted report containing the Git revision, exact
entrypoint/API route, interpreter paths and versions, relevant package versions, structured preflight codes, workload
ID/status, GPU visibility/reservations, last meaningful log lines, and the checks already attempted. Exclude passwords,
tokens, cookies, SSH keys, complete environment dumps, and checkpoint contents.

## Related skills

- Use `$alphabrain-codebase-nav` when the controlling route, resolver, launcher, or adapter is unclear.
- Use `$alphabrain-setup` when the diagnosis calls for a new isolated runtime or first-time backend initialization.
- Use `$alphabrain-data-resources` for missing model/data acquisition and validation.
- Return to `$alphabrain-training`, `$alphabrain-deployment`, or `$alphabrain-evaluation` to rerun the exact preflight
  and verify the repaired workflow.
