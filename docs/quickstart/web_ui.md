# AlphaBrain Web UI

AlphaBrain includes a browser-based research console for composing and running
training jobs on one multi-GPU server. It wraps the existing launch scripts;
all command-line workflows remain available.

## Install

Use Python 3.10+. The UI backend can run in the training environment or in a
smaller UI-only environment. Building the browser bundle requires Node.js
20.19+ and npm 10+.

```bash
pip install -r requirements-ui.txt
cd ui/frontend
npm install
npm run build
cd ../..
```

The additional Python packages are intentionally kept in
`requirements-ui.txt` instead of being added to the core training dependency
set.

Managed model deployment requires a Python environment that can import
`torch`, `transformers`, `websockets`, `msgpack`, `numpy`, and `AlphaBrain`.
If the UI runs in a lightweight environment, set the training/runtime
interpreter under **Settings → Environment → Model-server Python**, for
example:

```text
/path/to/conda/envs/alphabrain-runtime/bin/python
```

Preflight checks this interpreter before accepting a deployment. It does not
install packages or download model files automatically.

## Start

Personal use on the local machine:

```bash
bash scripts/run_ui.sh
```

Laboratory use over the LAN:

```bash
bash scripts/run_ui.sh --mode lab --host 0.0.0.0 --port 8000
```

Open `http://<server-ip>:8000`. On first launch, choose **Personal** or
**Laboratory** mode. Laboratory mode asks for the first administrator account;
additional researcher accounts are created from Settings.

Personal mode is the first-launch default. `--mode lab` changes the suggested
mode only while the selected state directory is uninitialized. After setup,
the mode saved in Settings takes precedence on every later start; this avoids
silently bypassing an existing account/password configuration. Administrators
can switch modes from **Settings → General** when no jobs are active.

The server must run with one backend worker because its SQLite-backed FIFO
scheduler owns GPU reservations. Training processes run in independent process
groups and continue if the UI service is restarted; the UI reconciles them by
PID when it returns.

## Configure resources

Existing repository `.env` values are loaded when each UI job starts.
Administrators can additionally configure non-secret path variables from the
UI, including:

- `PRETRAINED_MODELS_DIR`
- `LIBERO_DATA_ROOT` and `LEROBOT_LIBERO_DATA_DIR`
- `LIBERO_HOME`
- `LIBERO_PYTHON`, `LIBERO_PLUS_HOME`, and `LIBERO_PLUS_PYTHON`
- `ROBOCASA365_PYTHON` and `ROBOCASA_TABLETOP_PYTHON`
- `ROBOCASA_TABLETOP_DATA_ROOT` and `ROBOCASA365_DATA_ROOT`

Generic environment settings deliberately reject credentials so they never
enter the UI database or snapshots. Provide download credentials such as
`HF_TOKEN` through the dedicated global Hugging Face download-token control in
**Resources**. Each researcher can separately save a personal Hugging Face
publishing token; neither token is returned by the API or written to SQLite,
argv, or logs. Save
`WANDB_API_KEY` only with the dedicated write-only control under
**Settings → Environment**; it is kept in a permission-restricted secret file
and injected only into the W&B process that needs it.

For jobs started by the Web UI, environment precedence is deterministic:
repository `.env`, then the environment that started the UI service, then
non-secret values saved in UI Settings. Existing command-line launches keep
their original `.env` behavior.

Missing datasets and pretrained weights block submission with a preflight
error. The first UI release does not automatically download large resources.
Multiple result roots can be configured under **Settings → Environment**. The
first is the default for new work, while checkpoint indexing and publishing
remain confined to the configured set. Configured roots are checked when saved;
unavailable mounts are shown
as structured dashboard/preflight errors instead of crashing the service.

## Dataset and resource centers

**Resources** inventories pretrained models, benchmark data, world-model
dependencies, and repository preprocessing utilities. CPU/network utilities
use a bounded independent queue; GPU preprocessing and LoRA merge tasks join
the same global GPU FIFO as training, deployment, and evaluation.

**Datasets** registers a validated server-local LeRobot dataset either by
referencing its original path or explicitly copying it into managed storage.
Administrators configure one or more allowed dataset roots. Registrations can
be private or laboratory-shared and retain a validation fingerprint. Weighted
dataset mixtures store member IDs, glob patterns, robot types, and a monotonic
version. The experiment builder resolves these IDs on the server and freezes
the registration fingerprint or mixture version into the immutable run
snapshot; clients cannot bypass root and visibility checks with arbitrary
`mixture_spec` paths.

## Built-in local presets

The Experiment and Template centers include read-only starter configurations
for model resources detected under `PRETRAINED_MODELS_DIR` (or the repository
default `data/pretrained_models/`). Resource badges distinguish a ready recipe
from one that still needs benchmark data or another dependency. A built-in
template can be used directly or copied into a personal/laboratory template,
but cannot be edited or deleted in place.

The Checkpoint center also discovers curated local model packages without
inserting fake training history into SQLite. Package completeness and current
deployment-adapter support are displayed separately. Only checkpoints that
pass the existing read-only deployment inspection expose Deploy/Evaluate
actions; deployable packages also appear as one-click presets on the Model
deployments page. Adding files to a known directory changes these views on the
next API refresh, while actual experiment, checkpoint, and deployment records
remain unchanged.

## Runtime state

UI state is stored under `.alphabrain-ui/` by default:

```text
.alphabrain-ui/
├── ui.sqlite3
├── configs/       # immutable experiment and evaluation snapshots
├── logs/          # scheduler-owned process logs
├── datasets/      # optional managed dataset copies
├── registry/      # optional constrained overlay.yaml (0600)
└── secrets/       # write-only W&B/HF/deployment-controller credentials
```

Override this location with `--state-dir` or `ALPHABRAIN_UI_HOME`. Native model
outputs remain under the existing `results/` and `results/Checkpoints/`
locations.

Checkpoint files are never deleted automatically. An owner or administrator
can delete an inactive experiment's checkpoint from the Checkpoint center only
after typing the experiment name. Deletion is restricted to that job's output
directory and the configured result roots.

## Managed model deployment

Open **Model deployments** to serve a completed checkpoint through the
AlphaBrain msgpack WebSocket policy protocol. A deployment can use a checkpoint
indexed by the UI or an absolute server-local path. Before submission, the UI:

- reads only checkpoint filenames plus YAML/JSON metadata and never
  deserializes model weights;
- verifies that the Backbone + Action Head combination has a registered,
  enabled deployment adapter;
- checks the checkpoint file contract, local base-model dependencies, model
  Python imports, GPU visibility, and endpoint conflicts;
- rejects RL actor/critic/encoder checkpoints until a compatible inference
  adapter is implemented.

Training jobs and deployments share one global FIFO GPU queue. A running model
service retains its GPU reservation until it stops, fails, or reaches its idle
timeout. The default idle timeout is 1800 seconds; health checks do not extend
it. No running workload is preempted.

The UI generates an API key for each deployment. Its plaintext is returned
only once, when the deployment is created or the key is rotated; SQLite stores
only its SHA-256 digest and a short display prefix. Clients authenticate the
WebSocket handshake with:

```text
Authorization: Api-Key <plaintext-key>
```

`GET /healthz` on the model-service port remains unauthenticated and exposes
only public health metadata. Use **Local only** for loopback access or **Local
network** with a hostname/IP reachable by laboratory clients. Stopping,
restarting, key rotation, live logs, and administrator force-kill are available
from the deployment detail page.

Deployment options are registry-driven. To add a future model without
rewriting the page, declare its adapter (entrypoint, checkpoint kinds, protocol,
and JSON parameter schema) and wired Backbone + Action Head combinations in
`alphabrain_ui/registry/catalog.yaml`, then implement the corresponding loader.
Unsupported combinations remain hidden; experimental combinations require both
the global and per-user experimental switches plus an explicit risk
acknowledgement.

The **Model registry** page exposes the effective component, training, and
deployment inventory. Administrators may install a data-only Registry Overlay
to derive new experimental entries from checked-in bases or disable existing
entries. Overlay entries cannot supply commands, Python entrypoints,
configuration paths, environment variables, or credentials; executable fields
are inherited from a trusted built-in base. Saving an overlay is atomic and
audited. It affects experiment resolution, deployment inspection/startup, and
evaluation compatibility for that UI state directory without monkeypatching a
process-global catalog.

## Managed benchmark evaluation

Open **Evaluations** to test an indexed checkpoint, a server-local checkpoint
path, or a running UI-managed deployment. LIBERO and RoboCasa365 are the
verified benchmark integrations.
RoboCasa Tabletop and LIBERO-plus are shown only when both experimental switches
are enabled, and every experimental submission requires a separate risk
acknowledgement.

As soon as a checkpoint is selected, the first wizard step statically inspects
its YAML/JSON metadata, filters the model-combination selector to wired
candidates, and automatically selects the candidate when it is unambiguous.
Missing runtime dependencies, including an unresolved
`PRETRAINED_MODELS_DIR`, are therefore shown before the Benchmark step rather
than only during final preflight. This inspection never deserializes weights.

For a checkpoint source, evaluation starts a temporary policy server, waits for
it to become ready, runs the simulator client, and always attempts to stop the
temporary service at the end. The temporary unauthenticated policy server is
forced to bind to `127.0.0.1` and is not exposed to the LAN. A managed source
instead reuses the running deployment's loopback port and existing GPU
reservation, without starting or stopping its model server or requesting a
second GPU. Its separate UI controller credential is injected only into the
benchmark child environment and is never stored in the evaluation config,
argv, SQLite, logs, or browser response.

Training, deployment, evaluation, and GPU utilities use the same FIFO queue.
Standard simulator runs use one GPU; on a one-GPU machine the selector is
hidden and the GPU is assigned automatically. Specialized CL-matrix and
RL-iteration evaluations may atomically request multiple GPUs.

Four workload profiles are available:

- **Quick check**: one episode on the first task in each selected suite/task set;
- **Standard**: all selected tasks with 10 episodes per task;
- **Full benchmark**: all selected tasks with 50 episodes per task;
- **Custom**: typed expert controls for task subsets, episode count, seed,
  parallel environments, action chunk, views, and maximum steps.

The creation dialog also exposes repository-backed specialized workflows:

- checkpoint × suite batches;
- continual-learning matrices with ASR, BWT, and forgetting summaries;
- RL iteration curves;
- Online-STDP baseline/adapted comparisons;
- world-model predicted/ground-truth video comparisons.

Multi-run requests are represented as evaluation groups. Child runs retain
their individual logs and results, while the group produces a normalized
`evaluation-result-v2` artifact with matrices, series, comparisons, videos,
and generic downloadable artifacts. LIBERO-plus category summaries are
rendered explicitly rather than flattened into one overall score.

LIBERO and LIBERO-plus are capped at 50 trials per task because their clients
index the benchmark's finite initial-state set directly. RoboCasa vector runs
seed Gymnasium resets deterministically; when multiple environments run in
parallel, replay recording is limited to the first worker to avoid concurrent
writers corrupting a shared video directory.

Evaluation environments are not installed automatically. Configure the
benchmark-specific Python executable and dataset/repository roots under
**Settings → Environment**. Each configured simulator interpreter must use
Python 3.10 or newer. Preflight validates the checkpoint adapter,
benchmark compatibility, Python imports, configured paths, GPU visibility,
disk space, and optional W&B credentials before a run can be submitted.

Each run writes to a unique directory under `results/evaluation/ui/` and keeps
`evaluation-result-v1.json`, a configuration snapshot, launcher/server/client
logs, progress events, and rollout videos together. The result file contains
overall and per-task success counts and rates plus episode/video manifests.
Artifacts are served through authenticated, directory-confined UI endpoints;
the browser never receives permission to read an arbitrary server path.

Evaluation results can optionally upload summary metrics, per-task metrics,
configuration, and videos to W&B using the write-only API key stored in
Settings. An upload failure does not change a successful simulation into a
failed evaluation, and the upload can be retried without rerunning simulation.

The **Website reference results** page reads a versioned local snapshot of
values displayed at `alphabrain-platform.com`. It never fetches the website at
runtime. Every snapshot includes its source URL, capture timestamp,
benchmark/signature constraints, and a disclaimer that the values were not
reproduced by the local UI. Exact-match APIs return references only when all
versioned signature fields agree; the UI never guesses a comparison.

## LAN security

Personal mode should remain bound to `127.0.0.1`. For laboratory deployment,
trusted-LAN HTTP is suitable only when the network itself is trusted. Use an
HTTPS reverse proxy when credentials cross an untrusted network, and set:

```bash
export ALPHABRAIN_UI_SECURE_COOKIES=1
bash scripts/run_ui.sh --host 127.0.0.1 --port 8000
```

Expose the reverse proxy, not the backend port, to other machines. The backend
uses one worker so the SQLite FIFO scheduler remains the sole GPU owner.
The same Secure-cookie policy is visible under **Settings → General**. The
environment flag is a security floor and cannot be disabled from the page.

## Current boundary

The Web UI covers schema-driven imitation learning, continual learning, STDP,
RL-token, world-model, co-training, and VLM-only workflows; dataset/resource
management; checkpoint tooling and publishing; managed WebSocket deployment
and Playground inference; and grouped simulator evaluation with structured
metrics, logs, artifacts, and rollout videos.

It deliberately does not install simulator environments, download large
benchmark datasets without an explicit Resource-center action, control a real
robot, or expose repository features that still lack a complete runnable chain.
CALVIN, RoboTwin, SimplerEnv, BEHAVIOR, Isaac Sim, StarVLA-FAST, ACT, and
simulator-replay conversion remain outside the UI until their repository paths
are complete and verifiable.
