# `run_rl_scripts/` — VLA Online RL launchers (LIBERO)

End-to-end pipeline:

```
[VLA finetune]  →  [Phase-1 encoder pretrain]  →  [Phase-2 TD3 RL]  →  [Eval]
   pi05 only       run_rlt_pretrain.sh           run_rlt_rl.sh        run_eval_*.sh
```

Two algorithm tracks live under `AlphaBrain/training/reinforcement_learning/algos/`:

| Track | Encoder input | Phase-2 launcher |
|:------|:--------------|:-----------------|
| **`RLT/`** | full VLM token sequence `(B, L, H)` | `run_rlt_rl.sh` |
| **`RLT_a/`** | action-query slice `(B, M, H)` | (training script not in this dir; eval via `run_eval_action_token.sh`) |

Two VLA backbones supported on the `RLT/` track:

| Backbone | Framework | Use |
|:---------|:----------|:----|
| **Qwen** | `Qwenvl_OFT` (Qwen2.5-VL-3B + MLP head) | `BACKBONE=qwen` |
| **Pi05** | `PaliGemmaPi05` (SigLIP + Gemma 2B + flow matching) | `BACKBONE=pi05 VARIANT={1traj,5traj}` |

---

## File inventory

```
.
├── README.md
├── run_pi05_finetune.sh       # VLA finetune (any variant, Pi05)
├── run_rlt_pretrain.sh        # Phase-1: RLT encoder pretrain
├── run_rlt_rl.sh              # Phase-2: TD3 RL (qwen or pi05)
│
├── pi05_eval.yaml             # configs for 3 Pi05 VLA-only eval modes
├── run_eval_pi05_latest_zhanghe.sh    # auto-pick latest Pi05 ckpt → eval
├── run_eval_action_token.sh   # offline eval, RLT_a (action_token) policy
├── run_eval_rlt.sh            # offline eval, RLT policy — single iter
├── run_eval_rlt_all_iters.sh  # offline eval, RLT policy — all ckpts of one run
│
├── example_results/           # reference plots / summaries
└── example_scripts/           # legacy / one-off launchers
```

---

## Quick start

### Prerequisites

- LIBERO installed in a separate conda env. Set `LIBERO_PYTHON` and `LIBERO_HOME` (or put them in `.env`) so launchers can spawn worker subprocesses.
- For Pi05: local PaliGemma tokenizer dir at `$PALIGEMMA_TOKENIZER_PATH` (default `/datasets/peligemma`). Otherwise tokenizer init falls back to HF hub fetch and then to `sentencepiece` (often absent).
- Disk: pretrain + RL output lands under `results/rlt_training/<run_name>_<timestamp>/`.

### 1 — Finetune the VLA (Pi05 only; Qwen ckpts assumed pre-existing)

```bash
VARIANT=1traj bash scripts/run_rl_scripts/run_pi05_finetune.sh   # 1 traj/task
VARIANT=5traj bash scripts/run_rl_scripts/run_pi05_finetune.sh   # 5 traj/task
VARIANT=task0 bash scripts/run_rl_scripts/run_pi05_finetune.sh   # task 0 only
```

Each maps to a mode block in `configs/finetune_config.yaml`. Output: `results/training/Pi05-goal-<VARIANT>-openpi/checkpoints/steps_<N>/`.

### 2 — Phase-1: pretrain the RLT encoder

```bash
bash scripts/run_rl_scripts/run_rlt_pretrain.sh [GPU_ID]
```

Override `CKPT_PATH` / `OUTPUT_DIR` via env if you want a non-default VLA. Output: `results/rlt_training/<tag>/pretrain/checkpoints/pretrain_best/encoder.pt`.

### 3 — Phase-2: RL fine-tune

```bash
# Qwen track (default)
bash scripts/run_rl_scripts/run_rlt_rl.sh [GPU_ID]

# Pi05 track
BACKBONE=pi05 VARIANT=1traj TASK_ID=0 bash scripts/run_rl_scripts/run_rlt_rl.sh 0
BACKBONE=pi05 VARIANT=5traj TASK_ID=3 bash scripts/run_rl_scripts/run_rlt_rl.sh 0
```

`ENCODER_PATH` is auto-discovered (latest matching pretrain dir). Override `CKPT_PATH` / `ENCODER_PATH` via env if needed.

### 4 — Eval

```bash
# VLA-only (no RL): server + LIBERO client
bash scripts/run_base_vla/eval.sh pi05_goal_5traj_eval scripts/run_rl_scripts/pi05_eval.yaml

# VLA-only, latest-ckpt wrapper (rewrites the yaml `checkpoint:` line in place)
RUN_DIR=results/training/Pi05-goal-5traj-openpi \
MODE=pi05_goal_5traj_eval \
    bash scripts/run_rl_scripts/run_eval_pi05_latest_zhanghe.sh

# Offline eval of one RLT RL ckpt (50 ep / task, sharded if multi-task)
bash scripts/run_rl_scripts/run_eval_rlt.sh

# Offline eval of EVERY iter ckpt in one RL run (parallel across GPUs)
RUN_DIR=results/rlt_training/<run>/rl_offpolicy \
VLA_CKPT=results/training/Pi05-goal-5traj-openpi/checkpoints/steps_30000 \
GPUS="0 1 2" TASK_IDS=0 N_EPS=50 \
    bash scripts/run_rl_scripts/run_eval_rlt_all_iters.sh

# Offline eval, RLT_a (action_token) policy — 10 tasks split across 3 GPUs
bash scripts/run_rl_scripts/run_eval_action_token.sh <RUN_DIR>
```

---

## Output layout

```
results/
├── training/
│   ├── Pi05-goal-{1traj,5traj,task0}-openpi/checkpoints/steps_<N>/   # VLA finetune ckpts
│   └── QwenOFT-5traj-libero_goal/final_model/                        # Qwen VLA
└── rlt_training/
    ├── <pretrain_tag>/pretrain/checkpoints/pretrain_best/encoder.pt   # Phase-1
    └── <rl_tag>_<timestamp>/rl_offpolicy/
        ├── checkpoints/rl_offpolicy_iter_<N>/                         # Phase-2 ckpts
        ├── train.log
        └── eval_iter_<N>_rlt/summary.json                             # offline eval
```

---

## Tips & gotchas

- **`num_envs` is GPU-memory-bounded** (~0.5 GB activation/env). H100 80 GB → ≈48–64; A100 40 GB → ≈24–32.
- **Host RAM also matters**: each LIBERO env subprocess is ~600 MB MuJoCo + a few GB of Python overhead. Running multiple RL trainings simultaneously in the same cgroup can OOM-kill workers (check `cat /sys/fs/cgroup/memory.max`).
- **Pi05 tokenizer**: always export `PALIGEMMA_TOKENIZER_PATH` in the launching shell (or rely on the launcher's default `/datasets/peligemma`). Without it, rollout's first VLA call dies with `ModuleNotFoundError: sentencepiece`.
- **Step-lock (`--use_steplock`) is faster** than free-run rollout because it batches all envs through one VLA forward.
- **Async eval during training** uses socket-IPC env workers (matches rollout); the older pipe-IPC `LiberoEnv` deadlocks on some container setups.
- **Encoder must be pretrained on the same VLA you fine-tune on**. Mixing a 1-traj-VLA encoder with 5-traj-VLA RL silently degrades.
- **`[Warning]: datasets path ... does not exist!`** flooding the log: each env reset checks LIBERO's `datasets/` dir, which RL doesn't need. Silence by creating an empty dir:
  ```bash
  DATASETS_DIR=$("$LIBERO_PYTHON" -c "import libero.libero, os; print(os.path.realpath(os.path.join(os.path.dirname(libero.libero.__file__), '..', 'datasets')))")
  mkdir -p "$DATASETS_DIR"
  ```

---

## ⚠️ Disclaimer

A few honest notes on what this release is — and what it isn't:

- **Why we open-source RLT first.** One of the reasons we open-source RLT first is that we believe the idea itself — compressing VLA hidden states through an information bottleneck and then editing a reference policy with residual actions — is novel and genuinely promising, and worth sharing with the community at an early stage. This does **not** mean we consider GRPO / PPO any less important; on the contrary, we view them as core algorithms for VLA online RL, and will progressively update and release our implementations of them.
- **On reproducing the RL Token paper in simulation.** Faithfully reproducing every detail of the original RL Token paper inside a simulator is hard — in particular the carefully curated tuning datasets and the timely human-in-the-loop interventions described in the paper are difficult to replicate one-for-one in a purely automated sim setup. The recipe we ship here is therefore a best-effort simulation adaptation, not a line-by-line reproduction. See `AlphaBrain/training/reinforcement_learning/algos/RLT/README.md` for the concrete deltas.
- **What we believe matters most.** Collecting high-quality positive trajectories is one of the most critical open problems in this area — good positives do far more than clever loss tricks. Going forward we plan to:
  1. broaden the online-RL algorithm coverage (GRPO, PPO, and others);
  2. improve tooling for positive-sample collection, filtering, and curation on sim and real-world data;
  3. release stronger, better-documented baselines and more reproducible recipes.

This sub-module is a living research snapshot; APIs, configs, and numbers may change between releases. Issues and PRs are welcome.
