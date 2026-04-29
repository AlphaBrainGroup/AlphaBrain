# Continual Learning

One-command wrappers for AlphaBrain's **Continual Learning (CL)** pipeline:
train a single Vision-Language-Action (VLA) backbone sequentially over a
stream of manipulation tasks, then evaluate the final checkpoint on the
full task matrix.

CL algorithms plug into a single `CLAlgorithm` hook interface and are
selected entirely via YAML — switching methods needs no code edits.
The four LIBERO suites (Goal / Spatial / Object / Long) ship in-tree as
first-class CL streams; additional LeRobot-format mixtures can be
plugged in via the [Custom task streams](#custom-task-streams) section.

## What's new

| Addition                                           | Where                                                                            |
|:---------------------------------------------------|:---------------------------------------------------------------------------------|
| **MIR** (Maximally Interfered Retrieval) — replay  | `algorithms/rehearsal_based/mir.py`, see [§MIR](#mir-maximally-interfered-retrieval--rehearsal_based) |
| **LIBERO-Long (libero_10)** — 10 long-horizon tasks | `qwengr00t_er_lora_libero_long.yaml`                                            |
| `compute_cl_matrix_metrics.py` — ASR / BWT / F     | computes Lopez-Paz BWT + Chaudhry forgetting from a 10×10 eval matrix            |
| `compute_dataset_statistics.py` — generic ds_stats | recomputes / repairs `dataset_statistics.json` for any LeRobot mixture           |

Four VLA architectures are supported, each with full-parameter and
**Low-Rank Adaptation (LoRA)** variants:

| Architecture  | Parameters | Backbone                               |
|:--------------|:-----------|:---------------------------------------|
| QwenGR00T     | ~3.8 B     | Qwen2.5-VL-3B + Flow-Matching DiT head |
| NeuroVLA      | ~3.0 B     | Qwen2.5-VL-3B + Q-Former + SNN head    |
| LlamaOFT      | ~11 B      | Llama-3.2-11B-Vision + MLP head        |
| PaliGemmaOFT  | ~3.0 B     | PaliGemma-3B + MLP head                |

---

## Supported CL algorithms

Selectable via `continual_learning.algorithm.name` in YAML.

| Algorithm | Category          | One-line idea                                                          |
|:----------|:------------------|:-----------------------------------------------------------------------|
| **ER**    | rehearsal_based   | Reservoir replay buffer; mixes past-task samples into each batch        |
| **MIR**   | rehearsal_based   | Replays the top-k samples that a virtual SGD step would hurt the most   |

All algorithms share the single :class:`CLAlgorithm` hook interface
(`observe`, `modify_batch`, `compute_penalty`, `after_backward`,
`on_task_start`, `on_task_end`) — a new method only needs to override the
hooks it cares about.  Code lives under:

```
AlphaBrain/training/continual_learning/algorithms/
├── base.py                      # CLAlgorithm + CLContext
├── rehearsal_based/             # replay methods
│   ├── er.py                    #   class ER
│   └── mir.py                   #   class MIR(ER)
├── regularization_based/        # loss-penalty methods
│   └── ewc.py                   #   class EWC
└── dynamic_based/               # per-task architecture changes (planned)
```

### ER (Experience Replay) — rehearsal_based

**Reference.** Chaudhry et al. 2019 (baseline rehearsal); Ratcliff 1990 (original).
**Mechanism.** A fixed-size per-task reservoir buffer.  At each training
step a configurable fraction of the current batch is replaced with samples
drawn uniformly (or balanced per task) from the buffer.  Buffer is
populated once at each task-end.

Core hyperparameters:

| Key                       | Default | Effect |
|:--------------------------|:--------|:-------|
| `buffer_size_per_task`    | 500     | Max samples per task; total buffer grows linearly with # tasks |
| `replay_batch_ratio`      | 0.3     | Fraction of each batch replaced by replay samples (≥ 2 enforced) |
| `balanced_sampling`       | false   | True → draw equally from every stored task; False → uniform over all samples |

YAML (generic-algorithm style):

```yaml
continual_learning:
  algorithm:
    name: er
    buffer_size_per_task: 500
    replay_batch_ratio: 0.3
    balanced_sampling: false
```

Or legacy replay-style (still supported):

```yaml
continual_learning:
  replay:
    enabled: true
    method: experience_replay
    buffer_size_per_task: 500
    replay_batch_ratio: 0.3
    balanced_sampling: false
```

### MIR (Maximally Interfered Retrieval) — rehearsal_based

**Reference.** Aljundi et al., NeurIPS 2019 (arXiv:1908.04742).
**Mechanism.** Same storage as ER, but the sampling policy is *interference-aware*.
Every `mir_refresh_interval` steps, MIR:

1. Draws `|C|` candidates uniformly from the buffer.
2. Forwards each candidate at the current parameters → `L(θ, c)`.
3. Takes a virtual SGD step θ_v ← θ − η · g_curr (g_curr is piggy-backed
   from the trainer's own backward via an `after_backward` hook — no extra
   reduction, DeepSpeed-safe).
4. Forwards each candidate at θ_v → `L(θ_v, c)`.
5. Restores θ, ranks candidates by Δ_c = L(θ_v, c) − L(θ, c), caches
   the top-k.  Subsequent steps replay from that cached top-k until the
   next refresh.

This amortized variant runs ≈ 10 % slower than plain ER at defaults — the
naive "every-step MIR" would be 50-100× slower on 3 B-parameter VLAs.

Gradient capture uses `torch.nn.Parameter.register_hook`, which fires
*during* autograd (before DeepSpeed ZeRO-2 moves gradients into its
contiguous buffer and clears `param.grad`).  This is why MIR works under
ZeRO-2 with `contiguous_gradients: true` in the DeepSpeed config.

Hyperparameters:

| Key                     | Default | Effect |
|:------------------------|:--------|:-------|
| `buffer_size_per_task`  | 500     | Same as ER |
| `replay_batch_ratio`    | 0.3     | Same as ER |
| `balanced_sampling`     | false   | Fallback uniform sampling while cache is empty |
| `mir_refresh_interval`  | 200     | Lower → fresher cache, higher overhead |
| `mir_candidate_size`    | 16      | How many candidates scored per refresh (|C|) |
| `mir_top_k`             | 8       | Cache size; must be ≤ `mir_candidate_size` |
| `mir_virtual_lr`        | null    | Virtual SGD lr; `null` → `MIR.DEFAULT_VIRTUAL_LR` (1e-4) |
| `mir_lora_only`         | true    | Virtual step only on LoRA params (essential on 3B+ models) |

YAML:

```yaml
continual_learning:
  algorithm:
    name: mir
    buffer_size_per_task: 500
    replay_batch_ratio: 0.3
    balanced_sampling: false
    mir_refresh_interval: 200
    mir_candidate_size: 16
    mir_top_k: 8
    mir_virtual_lr: null
    mir_lora_only: true
```

---

## Results

Backbone **QwenGR00T-3B + LoRA (r=32)**; **50 rollouts × 10 tasks**;
final-checkpoint matrix eval.

<div align="center">

| Method                     | LIBERO-Goal     | LIBERO-Long |
|:---------------------------|:---------------:|:-----------:|
|                            | ASR / BWT (pp)  | ASR         |
| Sequential FT              | 9.8 / —         | —           |
| ER (aligned, buf=1000)     | ~52 / —         | 16.0        |
| **MIR (refresh50 recipe)** | **77.0 / −7.8** | **30.0**    |

</div>

`—` rows are in-flight or out of current scope.  BWT requires a full
T×T matrix eval (drop `--last-only`); the LIBERO-Long row is a 50-trial
× 10-task final-checkpoint single-row eval.

**Cross-architecture reference** (LIBERO-Goal):

| Backbone          | Method        | ASR    |
|:------------------|:--------------|:------:|
| QwenGR00T (full)  | ER            | 51.6 % |

### Reproduce the 77 % MIR recipe (LIBERO-Goal, full 10×10 matrix)

```bash
# 1. Train (~17 h on 4× A800 80 GB)
bash scripts/run_continual_learning_scripts/run_cl_train.sh \
    --yaml configs/continual_learning/qwengr00t_mir_lora_libero_refresh50.yaml \
    --gpus 0,1,2,3

# 2. Eval (50 trials × 10 tasks; final ckpt only)
bash scripts/run_continual_learning_scripts/run_cl_eval.sh \
    --run-id qwengr00t_mir_lora_libero_goal_refresh50_v1 \
    --base-config configs/continual_learning/qwengr00t_mir_lora_libero_refresh50.yaml \
    --gpus 0,1 --trials 50 --last-only
```

For the BWT/F numbers a full 10×10 matrix eval is required — drop
`--last-only` to evaluate every per-task checkpoint, then run
`compute_cl_matrix_metrics.py results/eval_cl/<run_id>`.

### Metrics

**ASR** = mean success rate across the 10 tasks at the final checkpoint.
**BWT** = `1/(N−1) · Σ_{i<N} (a_{N,i} − a_{i,i})` in pp (Lopez-Paz &
Ranzato 2017); `0` = no forgetting, negative = forgetting.  Recompute
both for any run via
[`compute_cl_matrix_metrics.py`](compute_cl_matrix_metrics.py).

---

## Quick start

All examples assume a fresh clone and `conda activate alphabrain`. Each
block `cd`s into the repo root so commands can be pasted verbatim.

### Training

```bash
cd /path/to/AlphaBrain-CL

# Default — QwenGR00T LoRA + ER on LIBERO-Goal (~15 h on 2× A800)
bash scripts/run_continual_learning_scripts/run_cl_train.sh

# Smoke test — 5 steps × 10 tasks, ~3 min (pipeline check, not convergence)
bash scripts/run_continual_learning_scripts/run_cl_train.sh --smoke

# Switch CL method via yaml
bash scripts/run_continual_learning_scripts/run_cl_train.sh --smoke \
    --yaml configs/continual_learning/qwengr00t_mir_lora_test.yaml

# NeuroVLA full-parameter + ER
bash scripts/run_continual_learning_scripts/run_cl_train.sh \
    --yaml configs/continual_learning/neurovla_er_libero.yaml \
    --run-id neurovla_cl_run_v1

# Pin GPUs + custom step budget
bash scripts/run_continual_learning_scripts/run_cl_train.sh --gpus 1,2 -- \
    --continual_learning.steps_per_task=20000
```

Checkpoints are written to `results/Checkpoints/<run_id>/checkpoints/`:

| Variant           | Artifacts per task                                                              |
|:------------------|:--------------------------------------------------------------------------------|
| LoRA              | `task_<k>_id<k>_steps_<N>_{lora_adapter/, action_model.pt, _cl_state.json}`     |
| Full-parameter    | `task_<k>_id<k>_steps_<N>_pytorch_model.pt` (+ `_cl_state.json`)                |

The `_cl_state.json` sidecar stores per-algorithm metadata (e.g. ER
buffer sizes).  Tensor-heavy state (replay samples) is **not** serialized
— it is rebuilt on resume by replaying `on_task_end` for each completed
task.

### Evaluation

```bash
# LIBERO — final ckpt × 50 trials per task
bash scripts/run_continual_learning_scripts/run_cl_eval.sh \
    --run-id qwengr00t_mir_lora_libero_goal_v1 \
    --base-config configs/continual_learning/qwengr00t_mir_lora_libero.yaml \
    --gpus 0,1 --trials 50 --last-only
```

Per-task results land in `results/eval_cl/<run_id>/`.  Drop
`--last-only` to evaluate every per-task checkpoint and produce the
full T×T matrix needed for BWT/F.

> **`dataset_statistics.json` repair.**  If a run dir is missing
> `dataset_statistics.json` (legacy runs) or the file came from a
> different mixture, eval will mis-denormalize actions and SR collapses
> to ~0%.  Recompute on the fly:
> ```bash
> python scripts/run_continual_learning_scripts/compute_dataset_statistics.py \
>     --config configs/continual_learning/<your_config>.yaml \
>     --out results/Checkpoints/<run_id>/dataset_statistics.json
> ```
> The script auto-detects binary-gripper axes (`q01==0 ∧ q99==1`, the
> LIBERO convention) and forces `mask=False` on those, matching the
> eval-side `unnormalize_actions` binarize-then-passthrough branch.

---

## Custom task streams

The same `run_cl_train.sh` handles task streams beyond the LIBERO
suites.  Define a stream inline in a YAML config — no Python changes
required:

```yaml
continual_learning:
  task_sequence:
    base_data_mix: my_custom_mix        # must exist in DATASET_NAMED_MIXTURES
    num_tasks: 5
    task_order: [2, 0, 4, 1, 3]         # optional visit order
  task_stream_mode: by_dataset          # | by_task_index | auto
  steps_per_task: 5000
```

Partitioning modes:

| `task_stream_mode` | Semantics                                                                    |
|:-------------------|:-----------------------------------------------------------------------------|
| `by_task_index`    | LIBERO default: partition one multi-task parquet by its `task_index` column. |
| `by_dataset`       | Each sub-dataset in the mixture is one CL task.                              |
| `auto`             | Try `by_task_index`; fall back to `by_dataset` if it yields < 2 tasks.       |

Implementation notes and guidance for adding new benchmarks are
collected in [`README_custom_streams.md`](README_custom_streams.md).


---

## Prerequisites

```bash
conda activate alphabrain
cp .env.example .env
```

Edit `.env` with your local paths. Required:

| Variable                   | Purpose                                                                                    |
|:---------------------------|:-------------------------------------------------------------------------------------------|
| `PRETRAINED_MODELS_DIR`    | Parent directory holding `Qwen2.5-VL-3B-Instruct/`, `Llama-3.2-11B-Vision-Instruct/`, etc. |
| `LEROBOT_LIBERO_DATA_DIR`  | LeRobot-format LIBERO data root.                                                           |
| `LIBERO_PYTHON`            | Python from a separate conda env containing `robosuite` and `libero` (eval-only).          |
| `LIBERO_HOME`              | LIBERO project root (for simulator configuration paths).                                   |

Optional env vars are documented inline in `.env.example` for any
custom streams beyond LIBERO.

---

## CLI reference

### `run_cl_train.sh`

| Flag              | Description                                                                  | Default                                                    |
|:------------------|:-----------------------------------------------------------------------------|:-----------------------------------------------------------|
| `--yaml PATH`     | CL config yaml (relative or absolute).                                       | `configs/continual_learning/qwengr00t_er_lora_libero.yaml` |
| `--run-id ID`     | Override the yaml's `run_id` (controls the checkpoint directory name).       | from yaml                                                  |
| `--gpus SPEC`     | Either a count (`"2"`) or a comma-separated id list (`"1,2,3"`). A list pins `CUDA_VISIBLE_DEVICES`. | auto-detect                          |
| `--port N`        | `accelerate` main process port.                                              | auto-select a free port                                    |
| `--smoke`         | 5 steps × all tasks × batch 4 — verifies the pipeline end-to-end.            | off                                                        |
| `--`              | Pass-through OmegaConf overrides (e.g. `--lora.rank=16`).                    | —                                                          |
| `-h`, `--help`    | Full help text.                                                              | —                                                          |

**Available yaml presets** (under `configs/continual_learning/`):

| Yaml                                          | Backbone     | Variant         | Algo | Stream             |
|:----------------------------------------------|:-------------|:----------------|:-----|:-------------------|
| `qwengr00t_er_lora_libero.yaml` (default)     | QwenGR00T    | LoRA (r=32)     | ER   | LIBERO-Goal        |
| `qwengr00t_er_lora_libero_long.yaml`          | QwenGR00T    | LoRA            | ER   | LIBERO-Long (10)   |
| `qwengr00t_mir_lora_libero.yaml`              | QwenGR00T    | LoRA            | MIR  | LIBERO-Goal        |
| `qwengr00t_mir_lora_libero_refresh50.yaml`    | QwenGR00T    | LoRA            | MIR  | LIBERO-Goal (77%)  |
| `qwengr00t_er_libero.yaml`                    | QwenGR00T    | Full-parameter  | ER   | LIBERO-Goal        |
| `neurovla_er_lora_libero.yaml`                | NeuroVLA     | LoRA            | ER   | LIBERO-Goal        |
| `llamaoft_er_lora_libero.yaml`                | LlamaOFT     | LoRA (r=16)     | ER   | LIBERO-Goal        |
| `paligemma_oft_er_libero.yaml`                | PaliGemmaOFT | Full-parameter  | ER   | LIBERO-Goal        |

`qwengr00t_mir_lora_libero_refresh50.yaml` is the 77 % LIBERO-Goal recipe —
buffer 1000, replay ratio 0.5, balanced sampling, MIR refresh every 50
steps.  See [Reproduce the 77 % MIR recipe](#reproduce-the-77--mir-recipe-libero-goal-full-1010-matrix).

**Algorithm selection** (in yaml):

```yaml
continual_learning:
  algorithm:
    name: er          # er | mir | ewc
    # method-specific knobs — see per-algorithm sections above
```

### `run_cl_eval.sh`

| Flag                  | Description                                                                  | Default       |
|:----------------------|:-----------------------------------------------------------------------------|:--------------|
| `--run-id ID`         | **Required.** Run directory under `results/Checkpoints/`.                    | —             |
| `--base-config PATH`  | **Required for LoRA runs** — base yaml used to merge the adapter.            | —             |
| `--gpus LIST`         | Comma-separated GPU id list; determines parallelism.                         | `0`           |
| `--suite NAME`        | `libero_goal`, `libero_spatial`, `libero_object`, or `libero_10`.            | `libero_goal` |
| `--trials N`          | Rollouts per task (production = 50).                                         | `10`          |
| `--port-base N`       | Starting port (each parallel worker gets `+i`).                              | `5694`        |
| `--output-base PATH`  | Eval results root.                                                           | `results/eval_cl/<run_id>` |
| `--last-only`         | Evaluate only the final task checkpoint.                                     | off           |

The evaluator automatically:

1. Discovers every `task_*_lora_adapter/` or `task_*_pytorch_model.pt` under `<run_id>/checkpoints/`.
2. Detects LoRA runs and merges adapters into full checkpoints on demand (cached as `*_merged.pt`).
3. Parallelises across `--gpus` — each worker owns a dedicated policy server + port.
4. Emits per-checkpoint `eval.log` + `server.log` under the output base.

---

## Architecture

```
scripts/run_continual_learning_scripts/run_cl_train.sh     (self-contained wrapper)
                                     │
                                     │  resolves --yaml, loads .env,
                                     │  probes framework + base VLM + CL method,
                                     │  exec accelerate launch
                                     ▼
AlphaBrain/training/continual_learning/train.py            (trainer)
        │
        │  outer loop: tasks 0..N-1
        │  inner loop: standard VLA training on each task
        │  dispatches through CLAlgorithm hooks
        │
        ├── algorithms/
        │   ├── base.py                  CLAlgorithm protocol + CLContext
        │   ├── rehearsal_based/         ER, MIR
        │   ├── regularization_based/    EWC
        │   └── dynamic_based/           (planned: DWE / Weight Merge / PackNet)
        │
        ├── datasets/                    TaskFilteredDataset + task_sequences
        └── trainer_utils/peft/
            apply_lora() · save_lora_checkpoint() · load_and_merge()
            merge_lora_checkpoint  (CLI for post-hoc adapter merging)
```

### CLAlgorithm hook protocol

Every algorithm subclass overrides only the hooks it needs.  Per-step
hooks run in the inner loop; task-level hooks bracket each CL task:

| Hook                            | When                                          | Typical override                                    |
|:--------------------------------|:----------------------------------------------|:----------------------------------------------------|
| `observe(batch, task_id)`       | Per step, before forward                      | Online bookkeeping (SI, streaming reservoir)        |
| `modify_batch(batch, task_id)`  | Per step, before forward                      | ER / MIR inject replay samples                      |
| `compute_penalty(model)`        | Per step, inside autocast block               | EWC / SI return λ · regularizer tensor              |
| `after_backward(model)`         | Per step, after `accelerator.backward()`      | MIR snapshots gradients (DeepSpeed-safe)            |
| `on_task_start(ctx)`            | Before each task's inner loop begins          | DWE expands model; MIR installs grad hooks          |
| `on_task_end(ctx)`              | After each task's inner loop finishes         | ER populates buffer; EWC computes Fisher            |

`ctx` is a `CLContext` dataclass carrying `task_id`, `model`,
`task_dataset`, `task_dataloader`, and `accelerator` — the algorithm
picks the handles it needs.

---

## Related components

| Component                                    | Path                                                                    |
|:---------------------------------------------|:------------------------------------------------------------------------|
| CL trainer                                   | `AlphaBrain/training/continual_learning/train.py`                       |
| Custom-stream extensions (add-only)          | `AlphaBrain/training/continual_learning/{train_custom,datasets/custom_streams}.py` |
| CL algorithms — base / rehearsal / regularization / dynamic | `AlphaBrain/training/continual_learning/algorithms/`     |
| Task sequences + `TaskFilteredDataset`       | `AlphaBrain/training/continual_learning/datasets/task_sequences.py`     |
| LoRA helpers (inject / save / load & merge)  | `AlphaBrain/training/trainer_utils/peft/`                               |
| YAML configurations                          | `configs/continual_learning/`                                           |
| Documentation hub                            | `docs/continual_learning/`                                              |
| mkdocs quickstart                            | `docs/quickstart/continual_learning.md`                                 |

---

## Tips and caveats

- **flash-attn ABI mismatch.** If the active environment has `torch ≥ 2.6`
  but `flash-attn` was built against `torch 2.2`, the default
  `attn_implementation: flash_attention_2` crashes at model load.
  Workaround — override to SDPA:
  ```bash
  bash run_cl_train.sh -- --framework.qwenvl.attn_implementation=sdpa
  ```
  or reinstall: `pip install flash-attn --no-build-isolation --force-reinstall`.
- **Eval uses a separate conda env.** `LIBERO_PYTHON` in `.env` must
  point to an interpreter with `robosuite` installed (distinct from the
  training env). The wrapper auto-detects and falls back to a
  conventional `vlacl_engine_eval` install when the configured path
  lacks the dependency.
- **LoRA evaluation caches merged checkpoints.** The first evaluation
  call merges each LoRA adapter into a `*_merged.pt` file (~7 GB per
  task). Subsequent calls reuse the cache; remove the file to force a
  re-merge.
- **DeepSpeed ZeRO-2 + `contiguous_gradients: true`.** After
  `accelerator.backward()` the parameter `.grad` attribute is **None** —
  gradients live in DeepSpeed's internal buffer.  Algorithms that need
  per-step gradient access (currently: MIR) use
  `torch.nn.Parameter.register_hook` to capture gradients *during*
  autograd instead.  If you add a new algorithm that needs gradients,
  follow MIR's pattern.
- **Default run takes ~15 h on 2× A800 80 GB.** Use `--smoke` to verify
  the pipeline in three minutes before committing a full run.

---

## Further reading

- Full experiment record and 10×10 matrices: [`docs/continual_learning/EXPERIMENTS.md`](../../docs/continual_learning/EXPERIMENTS.md)
- Annotated experiment matrix: [`docs/continual_learning/EXPERIMENT_MATRIX.md`](../../docs/continual_learning/EXPERIMENT_MATRIX.md)
- Source-layout index: [`docs/continual_learning/CODE_LAYOUT.md`](../../docs/continual_learning/CODE_LAYOUT.md)
- Hosted quickstart (mkdocs): [`docs/quickstart/continual_learning.md`](../../docs/quickstart/continual_learning.md)
- Custom-stream implementation notes: [`README_custom_streams.md`](README_custom_streams.md)
