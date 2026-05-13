# RLT_a — Online RL Fine-tuning for VLA

Minimal pipeline for RLT_a on LIBERO:

1. **`run_action_token_5traj_alltasks.sh`** — End-to-end training (Phase 1 encoder pretrain + Phase 2 off-policy TD3 RL)
2. **`run_eval_action_token.sh`** — Parallel per-task eval over a training run

> **Naming note.** The module directory was renamed from `RLT` to
> `RLT_a` to avoid implying a line-by-line reproduction of the
> RL Token paper (Physical Intelligence, 2026). This release is inspired by
> the paper but deviates in several concrete places (input, bottleneck,
> decoder, pretrain data). See
> [`AlphaBrain/training/reinforcement_learning/algos/RLT_a/README.md`](../../AlphaBrain/training/reinforcement_learning/algos/RLT_a/README.md)
> for the full design notes and the delta against the paper.

---

## Quick Start

### Prerequisites

- A pretrained QwenOFT VLA checkpoint, e.g. `results/training/QwenOFT-5traj-libero_goal/final_model`
- LIBERO installed in a separate conda env. Set `LIBERO_PYTHON` and `LIBERO_HOME` to point to it.
- 6 GPUs (5 rollout + 1 train) for the default run; smaller setups work by reducing `num_envs_per_task`.

### Step 1 — Train (Phase 1 + Phase 2)

```bash
# default: GPUs 0,2,3,4,5 for rollout, 1 for train
bash scripts/run_rl_scripts/run_action_token_5traj_alltasks.sh

# override GPU layout
bash scripts/run_rl_scripts/run_action_token_5traj_alltasks.sh "0,1,2,3,4,5"
```


**Checkpoints** land in `results/action_token_training_TD3/<run_name>_<timestamp>/rl_offpolicy/checkpoints/rl_offpolicy_iter_NNNNN/`.

### Step 2 — Eval

```bash
# default eval GPUs: 0,1,2
bash scripts/run_rl_scripts/run_eval_action_token.sh \
    results/action_token_training_TD3/action_token_5traj_alltasks_release_0414_1727/rl_offpolicy

# override eval GPUs
bash scripts/run_rl_scripts/run_eval_action_token.sh \
    results/action_token_training_TD3/action_token_5traj_alltasks_release_0414_1727/rl_offpolicy \
    "3,4,5"
```

The 10 tasks are split across the given GPUs and aggregated inline. Output lands at
`<RUN_DIR>/eval_rl_offpolicy_iter_<NNNNN>/summary.json` (per-task + overall SR).

---

## CLI Reference

Key knobs inside `run_action_token_5traj_alltasks.sh` (edit if your hardware differs):

| Flag | Default | Meaning |
|:-----|:--------|:--------|
| `--rollout_gpus` | `0,1,2,3,4` | GPUs used for env rollout |
| `--train_gpu` | `5` | GPU used for TD updates (split from rollout) |
| `--num_envs_per_task` | `10` | Parallel envs per task per rollout GPU (memory-bounded) |
| `--G_per_task` | `30` | Episodes per task per `main_iter` (auto-chunked when > num_envs) |
| `--use_steplock` | on | Step-lock rollout: all envs advance in lockstep, batched VLA forward |
| `--utd_ratio` | `10.0` | Update-to-data ratio (TD updates per collected step) |
| `--td_updates_per_iter` | `10000` | Hard cap on TD updates per `main_iter` |
| `--max_iter` | `400` | Total training iterations |
| `--eval_interval` | `20` | Async-eval cadence (every N `main_iter`) |
| `--reward_coef` | `5.0` | Sparse reward scaling |
| `--beta` | `1.0` | BC regularization weight |
| `--ref_dropout` | `0.5` | Reference-action dropout probability |
| `--fixed_std` | `0.1` | Gaussian exploration std |

All other hyper-parameters are set inside the script or via `configs/rl_recipes/QwenOFT_LIBERO_ActionToken.yaml`.

---

## How the knobs translate (rollout + TD math)

The CLI flags above are per-task / ratios. The physical counts — total envs, trajectories per iter, TD updates per iter — are derived at runtime.

### Env pool sizing

```
max_tasks_per_gpu = ceil(n_tasks / n_rollout_gpus)      # round-robin task→GPU
envs_per_gpu      = num_envs_per_task × max_tasks_per_gpu
total_envs        = envs_per_gpu × n_rollout_gpus
```

**Default config** (10 `libero_goal` tasks, 5 rollout GPUs, `num_envs_per_task=10`):

| Quantity | Value |
|:---------|:------|
| tasks per rollout GPU | `ceil(10 / 5) = 2` |
| envs per rollout GPU | `10 × 2 = 20` |
| **total parallel envs** | **`20 × 5 = 100`** |

### Trajectories per iter

```
trajs_per_iter   = G_per_task × n_tasks
rollout_passes   = ceil(G_per_task / num_envs_per_task)    # sequential chunks if G > num_envs
```

**Default**: `G_per_task=30`, 10 tasks → `300 trajectories/main_iter`, run as `ceil(30/10) = 3` sequential rollout passes per iter.

---

## Tips & gotchas

- **`num_envs_per_task` is GPU-memory-bounded**: ~0.5 GB activation per env. H100 80 GB → ≈48–64; A100 40 GB → ≈24–32; RTX 4090 24 GB → ≈16–24.
- **Host RAM also matters**: each env is a separate MuJoCo subprocess (~3–4 GB each); 100 envs ≈ 300–400 GB. Check your cgroup limit before scaling up.
- **`G_per_task` should be large (~60–128)** for stable TD3 convergence. Auto-chunking handles `G > num_envs` transparently.
- **Step-lock (`--use_steplock`) is faster** than free-run rollout because it batches all envs through one VLA forward.
- **Async eval doesn't block training** — eval results are drained on the next `main_iter`. Keep `--eval_n_episodes` small (≤20) during training and run a 50-eps final eval separately.
- **Encoder must be pretrained on the same VLA you fine-tune on**. Mixing 1-traj-VLA encoder with 5-traj-VLA RL training will silently degrade.
- **LIBERO `[Warning]: datasets path ... does not exist!` flooding the log**: each env reset calls `get_libero_path()` which prints a warning for every non-existent path in LIBERO's config. RL rollout doesn't need the `datasets` dir — just create an empty one to silence it. Use `LIBERO_PYTHON` from `.env` to resolve the right location (varies per install):
  ```bash
  DATASETS_DIR=$("$LIBERO_PYTHON" -c "import libero.libero, os; print(os.path.realpath(os.path.join(os.path.dirname(libero.libero.__file__), '..', 'datasets')))")
  mkdir -p "$DATASETS_DIR"
  ```

---

## ⚠️ Disclaimer

A few honest notes on what this release is — and what it isn't:

- **Why we open-source RLT_a first.** One of the reasons we open-source RLT_a is that we believe the idea itself — compressing VLA hidden states through an information bottleneck and then editing a reference policy with residual actions — is novel and genuinely promising, and worth sharing with the community at an early stage. This does **not** mean we consider GRPO / PPO any less important; on the contrary, we view them as core algorithms for VLA online RL, and we will progressively update and release our implementations of them in subsequent versions.
- **On reproducing the RL Token paper in simulation.** Faithfully reproducing every detail from the original RL Token paper inside a simulator is genuinely hard — in particular the carefully curated tuning datasets and the timely human-in-the-loop interventions described in the paper are difficult to replicate one-for-one in a purely automated sim setup. The recipe we ship here (`RLT_a`) is therefore a best-effort simulation adaptation, not a line-by-line reproduction of the paper. See the algo README under `AlphaBrain/training/reinforcement_learning/algos/RLT_a/` for the concrete deltas.
- **What we believe matters most.** Even so, we believe that collecting high-quality positive trajectories is one of the most critical open problems in this area — good positives do far more than clever loss tricks. Going forward we plan to:
  1. broaden the online-RL algorithm coverage (GRPO, PPO, and RLTginal);
  2. improve tooling for positive-sample collection, filtering, and curation on both sim and real-world data;
  3. release stronger, better-documented baselines and more reproducible recipes as the work matures.

This sub-module is a living research snapshot: APIs, configs, and numbers may change between releases. Issues, pull requests, and discussion are very welcome.
