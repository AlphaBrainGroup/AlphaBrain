# CL Experiment Matrix

Source of truth for every continual-learning experiment we intend to run
(planned / running / done).  Update the `Status` column as runs start,
complete, and get evaluated.

---

## TL;DR status dashboard

| # | Group                           | Benchmark      | Algorithm | Configs | Status          | Wall-clock | Blocker |
|---|:--------------------------------|:--------------:|:---------:|:-------:|:----------------|:-----------|:--------|
| 1 | LIBERO-Goal · ER baseline       | LIBERO-Goal    | ER        | 1       | ⏳ pending       | ~3.5 h     | —       |
| 2 | LIBERO-Goal · EWC **coarse** λ   | LIBERO-Goal    | EWC       | 4       | 🏃 **running**   | ~7 h       | Round 2 queued |
| 3 | LIBERO-Goal · EWC **fine** λ     | LIBERO-Goal    | EWC       | 4       | ⏳ pending       | ~7 h       | Awaits #2 winner |
| 4 | LIBERO-Goal · MIR scout         | LIBERO-Goal    | MIR       | 1       | 🏃 **running**   | ~3.5 h     | —       |
| 5 | LIBERO-Goal · MIR extended      | LIBERO-Goal    | MIR       | 8       | ⏳ pending       | ~10.5 h    | Awaits #4 + free GPUs |
| 6 | Robocasa-atomic10 · ER baseline | Robocasa-A10   | ER        | 1       | ⏳ pending       | ~3 h       | Pipeline smoke-tested ✓ |
| 7 | Robocasa-atomic10 · MIR / EWC   | Robocasa-A10   | MIR + EWC | 2       | ⏳ pending       | ~6 h       | Awaits #6 ER |
| 8 | LIBERO-Goal · Eval matrix       | LIBERO-Goal    | all       | ~18     | ⏳ pending       | ~15 h      | Awaits training done |
| 9 | Robocasa eval wrapper           | Robocasa       | n/a       | infra   | ⏳ pending       | ~0.5 day dev | Not written yet |

Total training compute (groups 1-7): **~40 h** on 2× A800 serial, roughly
**~15 h wall-clock** with 3 pairs parallel once EWC coarse finishes.
Evaluation adds ~15 h on top.

Legend: 🏃 running · ⏳ pending · ✅ complete · 🔍 evaled · ❌ failed / dropped

---

## 1. LIBERO-Goal · ER baseline (`ER-baseline`)

**Purpose.** Sets the floor that EWC/MIR must beat.  Same step budget and
model as the hyperparameter sweeps so comparison is apples-to-apples.

| Run ID                              | YAML                                            | Overrides (beyond YAML)                  | GPUs  |
|:------------------------------------|:------------------------------------------------|:-----------------------------------------|:------|
| `er_baseline_stepsPerTask_2000`     | `qwengr00t_cl_lora_libero.yaml`                 | `--continual_learning.steps_per_task=2000` | 1 pair |

**Command:**
```bash
bash scripts/run_continual_learning_scripts/run_cl_train.sh \
    --yaml configs/continual_learning/qwengr00t_cl_lora_libero.yaml \
    --run-id er_baseline_stepsPerTask_2000 \
    --gpus <free_pair> -- \
    --continual_learning.steps_per_task=2000
```

---

## 2. LIBERO-Goal · EWC **coarse** λ sweep  🏃

**Purpose.** Identify which decade of λ is best on LIBERO-Goal before
fine sweeping.

| λ       | γ   | Run ID                                     | Status        | Notes                 |
|:--------|:----|:-------------------------------------------|:--------------|:----------------------|
| 1e3     | 1.0 | `ewc_lambda_1e3_stepsPerTask_2000`         | 🏃 ~80 %       | GPU 3,4               |
| 1e4     | 1.0 | `ewc_lambda_1e4_stepsPerTask_2000`         | 🏃 ~80 %       | GPU 5,6               |
| 1e5     | 1.0 | `ewc_lambda_1e5_stepsPerTask_2000`         | ⏳ queued      | Round 2 on 3,4        |
| 1e6     | 1.0 | `ewc_lambda_1e6_stepsPerTask_2000`         | ⏳ queued      | Round 2 on 5,6        |

**Command (already launched):**
```bash
PARALLEL=1 STEPS_PER_TASK=2000 GPUS_A=3,4 GPUS_B=5,6 \
    bash scripts/run_continual_learning_scripts/run_ewc_lambda_sweep.sh
```

**Shared hyperparameters** (from `qwengr00t_cl_lora_ewc_libero.yaml`):
`lora_only=true, fisher_num_batches=100, fisher_clip=1e4, grad_clip_per_sample=100, per_device_batch_size=4, LoRA r=32`.

---

## 3. LIBERO-Goal · EWC **fine** λ sweep  ⏳

**Purpose.** Once #2 identifies a winner λ★ (by Avg SR / NBT), test
`{λ★/3, λ★, λ★×3, λ★ with γ=0.9}`.  The γ=0.9 cell tests online-EWC
against the additive default.

| λ      | γ   | Notes                                |
|:-------|:----|:-------------------------------------|
| λ★/3   | 1.0 | halo below winner                    |
| λ★     | 1.0 | confirm winner at 2000 steps holds   |
| λ★×3   | 1.0 | halo above winner                    |
| λ★     | 0.9 | online-EWC variant at the best λ    |

**Command template** (substitute concrete λ★ after #2 is done):
```bash
GPU_PAIRS="1,2 3,4 5,6" \
LAMBDAS="<λ★/3> <λ★> <λ★×3> <λ★>" \
GAMMAS="1.0 1.0 1.0 0.9" \
STEPS_PER_TASK=2000 \
    bash scripts/run_continual_learning_scripts/run_ewc_lambda_sweep.sh
```

---

## 4. LIBERO-Goal · MIR scout  🏃

**Purpose.** Verify MIR production pipeline on a single default config
before committing to the wider sweep.

| Run ID                                         | refresh | cands | top_k | v_lr  | Status |
|:-----------------------------------------------|:--------|:------|:------|:------|:-------|
| `mir_ri200_vlrdefault_stepsPerTask_2000`       | 200     | 16    | 8     | 1e-4  | 🏃 task 3/10 |

**Command (already launched):**
```bash
MIR_CONFIGS="mir_ri200_vlrdefault" STEPS_PER_TASK=2000 GPUS_A=1,2 \
    bash scripts/run_continual_learning_scripts/run_mir_sweep.sh
```

---

## 5. LIBERO-Goal · MIR extended sweep  ⏳

**Purpose.** Map the three major MIR hyperparameter dimensions.  Runs
the scout's baseline + 8 variants (3 MIR-knob, 3 replay-ratio, 2 buffer-size).

| Config                  | refresh | cands | top_k | v_lr   | ratio | buffer | Dimension tested          |
|:------------------------|:--------|:------|:------|:-------|:------|:-------|:--------------------------|
| mir_ri200_vlrdefault ✓   | 200     | 16    | 8     | 1e-4   | 0.3   | 500    | — (scout baseline)        |
| mir_ri50_vlrmatched     | **50**  | 16    | 8     | **2.5e-5** | 0.3 | 500  | fresher cache + matched lr |
| mir_ri200_wide          | 200     | **32** | **16** | 2.5e-5 | 0.3 | 500  | larger candidate pool     |
| mir_ri500_slow          | **500** | 16    | 8     | 2.5e-5 | 0.3   | 500    | less overhead             |
| mir_ratio010            | 200     | 16    | 8     | 2.5e-5 | **0.1** | 500  | replay ratio low          |
| mir_ratio050            | 200     | 16    | 8     | 2.5e-5 | **0.5** | 500  | replay ratio high         |
| mir_ratio070            | 200     | 16    | 8     | 2.5e-5 | **0.7** | 500  | replay ratio extreme      |
| mir_buf200              | 200     | 16    | 8     | 2.5e-5 | 0.3   | **200** | smaller buffer           |
| mir_buf1000             | 200     | 16    | 8     | 2.5e-5 | 0.3   | **1000** | larger buffer           |

**Command template** (launch after #4 finishes + EWC Round 2 frees 3-6):
```bash
GPU_PAIRS="1,2 3,4 5,6" \
MIR_CONFIGS="mir_ri50_vlrmatched mir_ri200_wide mir_ri500_slow \
             mir_ratio010 mir_ratio050 mir_ratio070 \
             mir_buf200 mir_buf1000" \
STEPS_PER_TASK=2000 \
    bash scripts/run_continual_learning_scripts/run_mir_sweep.sh
```

---

## 6. Robocasa-atomic10 · ER baseline  ⏳

**Purpose.** Smoke-tested pipeline on a 10-task Robocasa subset.  This
is our first non-LIBERO CL experiment.  Task order (by_dataset mode):

| Task id | Task name                    | Category              |
|:--------|:-----------------------------|:----------------------|
| 0       | NavigateKitchen              | navigate              |
| 1       | OpenDrawer                   | open storage          |
| 2       | OpenCabinet                  | open storage          |
| 3       | CloseFridge                  | close storage         |
| 4       | CloseBlenderLid              | close manipulable     |
| 5       | CoffeeSetupMug               | tool manipulation     |
| 6       | PickPlaceCounterToCabinet    | pick-place (put in)   |
| 7       | PickPlaceSinkToCounter       | pick-place (take out) |
| 8       | TurnOnMicrowave              | appliance on          |
| 9       | TurnOffStove                 | appliance off         |

| Run ID                                        | YAML                                                       | Wall-clock |
|:----------------------------------------------|:-----------------------------------------------------------|:-----------|
| `robocasa_atomic10_er_stepsPerTask_2000`      | `qwengr00t_cl_lora_robocasa_atomic10.yaml`                 | ~3 h       |

**Command:**
```bash
bash scripts/run_continual_learning_scripts/run_cl_train.sh \
    --yaml configs/continual_learning/qwengr00t_cl_lora_robocasa_atomic10.yaml \
    --run-id robocasa_atomic10_er_stepsPerTask_2000 \
    --gpus <free_pair>
```

**Status.** Pipeline end-to-end smoke passed (3 task boundaries, ER
on_task_end firing, action_dit_loss≈1.14 at step 10).  Waiting for a
free GPU pair to start the proper 2000-step run.

---

## 7. Robocasa-atomic10 · MIR + EWC  ⏳

**Purpose.** Confirm MIR/EWC also work on Robocasa (not just LIBERO)
and gather first non-LIBERO CL numbers.

| Run ID                                        | Algorithm | YAML-delta vs #6                              |
|:----------------------------------------------|:---------:|:----------------------------------------------|
| `robocasa_atomic10_mir_stepsPerTask_2000`     | MIR       | swap `replay:` block for `algorithm: name=mir` |
| `robocasa_atomic10_ewc_stepsPerTask_2000`     | EWC       | swap `replay:` block for `algorithm: name=ewc` + Fisher defaults |

**YAML prep needed:** copy `qwengr00t_cl_lora_robocasa_atomic10.yaml` →
`…_mir.yaml` and `…_ewc.yaml`, change the `continual_learning` block to
match the LIBERO EWC/MIR YAML style.

---

## 8. Evaluation matrix (LIBERO-Goal)  ⏳

**Protocol.** For each trained LIBERO-Goal run, evaluate its final
checkpoint against all 10 LIBERO-Goal tasks × 10 rollouts/task → 10×10
success matrix → Avg SR + NBT (negative backward transfer).

| Eval target        | # runs                 | Estimated wall-clock (2 GPU parallel) |
|:-------------------|:----------------------:|:-------------------------------------:|
| ER baseline (#1)   | 1                      | ~50 min                                |
| EWC coarse (#2)    | 4                      | ~3.5 h                                 |
| EWC fine (#3)      | 4                      | ~3.5 h                                 |
| MIR scout (#4)     | 1                      | ~50 min                                |
| MIR extended (#5)  | 8                      | ~7 h                                   |
| **Total**          | **18**                 | **~15 h** on 2 GPUs                    |

**Per-run command:**
```bash
bash scripts/run_continual_learning_scripts/run_cl_eval.sh \
    --run-id <run_id> \
    --base-config configs/continual_learning/qwengr00t_cl_lora_libero.yaml \
    --gpus 0,1
```

**Result aggregation.** After eval completes, a comparison script
(TODO, see §9) will produce one table with `Run · Avg SR · NBT · λ/refresh/...`.

---

## 9. Robocasa evaluation wrapper  ⏳

**Purpose.** `run_cl_eval.sh` currently launches only the LIBERO sim.
Robocasa requires a separate wrapper that drives the Robocasa MuJoCo
sim + their `scripts/run_eval.py`.

**Status.** Not yet written.  Current Robocasa runs (#6, #7) will
produce checkpoints but we can't auto-evaluate them without this wrapper.
Estimated ~0.5 day of dev + testing against the upstream Robocasa repo
already cloned at `/share/chenziyang/continual-vla-rl/`.

---

## 10. Planned — future algorithms

Documented in each algorithm's docstring; tracked here for planning.

| Algorithm  | Category              | Est. effort     | Dependency / blocker           |
|:-----------|:----------------------|:----------------|:-------------------------------|
| DER / DER++ | rehearsal_based       | 1–2 days        | HDF5 logit cache infra         |
| Weight Merge | dynamic_based        | ~1 day          | load-time PEFT merge path      |
| DWE (per-task adapter) | dynamic_based | 3–5 days       | model/optimizer/eval refactor  |
| SLCA (layered LR) | regularization_based (realized via param-group LRs) | ~1 day | YAML-only once ER/MIR/EWC baselines exist |

---

## 11. Compute budget summary

Assumptions: QwenGR00T LoRA r=32, bf16, batch 4 per GPU, 2 GPUs per run,
0.5–0.65 s/step at 2000 steps/task × 10 tasks.

**Wall-clock, LIBERO-Goal only (#1–#5, #8):**

| Serial on 1 pair                  | With 3-pair parallel (6 GPUs)      |
|:----------------------------------|:-----------------------------------|
| Training: ~68 h                   | Training: ~24 h                    |
| Eval:     ~15 h                   | Eval:     ~8 h                     |
| **Total: ~83 h**                  | **Total: ~32 h**                   |

**Robocasa atomic10 (#6, #7):**

| Serial on 1 pair                  | With 2-pair parallel               |
|:----------------------------------|:-----------------------------------|
| Training: ~9 h                    | Training: ~4.5 h                   |
| Eval:     depends on §9           | —                                  |

---

## 12. Metrics recorded per run

| Metric                  | Source                                                   | Notes                              |
|:------------------------|:---------------------------------------------------------|:-----------------------------------|
| `action_dit_loss`       | training jsonl stream                                    | Per-step training loss             |
| `task_id`, `task_step`  | training jsonl stream                                    | Task boundary tagging              |
| `ewc_num_tasks_consolidated` | EWC metrics via step_metrics                        | Fisher snapshots so far            |
| `mir_refresh_count`     | MIR metrics via step_metrics                             | # virtual-step refreshes           |
| `er_buffer_size`        | ER metrics via step_metrics                              | Total samples in replay buffer     |
| **Avg SR**              | `run_cl_eval.sh` → `results/eval_cl/<run_id>/matrix.json` | Mean success across all 10×10      |
| **NBT**                 | same                                                     | ΔSR on prior tasks after new task |

---

## 13. Update log

| Date       | Change                                                                 |
|:-----------|:-----------------------------------------------------------------------|
| 2026-04-23 | Initial draft: EWC coarse + MIR scout running; Robocasa smoke passed.  |

---

## How to update this file

- When a run finishes: flip its row's Status emoji and fill the wall-clock.
- When you launch a new experiment: add a row (don't delete old ones —
  this is a log).
- Winning λ★ / other hyperparameter decisions should land in the
  relevant Run ID column and be referenced in the commit message.
- The 13. Update log section is append-only per date.
