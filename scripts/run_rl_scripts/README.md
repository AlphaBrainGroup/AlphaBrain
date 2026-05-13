# `run_rl_scripts/` — VLA Online RL launchers (LIBERO)

End-to-end RL pipeline (assumes a pretrained VLA checkpoint is already on disk):

```
[Phase-1 encoder pretrain]  →  [Phase-2 TD3 RL]  →  [Eval]
   run_rlt_pretrain.sh         run_rlt_rl.sh         run_eval_rlt{,_a}.sh
```

VLA finetune itself isn't part of this directory's flow — it's a one-time
upstream step. See `example_scripts/run_pi05_finetune.sh` for the Pi05
recipe, or `configs/finetune_config.yaml` for Qwen.

---

## Two algorithm tracks: `RLT_a` (first release) and `RLT` (later, paper-faithful)

The repo ships two encoder/actor recipes side-by-side under
`AlphaBrain/training/reinforcement_learning/algos/`:

| Track | Encoder input | When/why we use it |
|:------|:--------------|:-------------------|
| **`RLT_a/`** *(first released)* | action-query slice `(B, M=chunk_len, H)` projected to a `D=256` bottleneck | the practical default — deliberately deviates from the RL Token paper so the recipe scales to multi-task and stays portable across VLM backbones |
| **`RLT/`** *(added later)* | full VLM token sequence `(B, L, H)`, `z_rl` kept at VLA `H` | a paper-faithful reference track, useful for research-side comparisons; closer to the original Eq. 1/2 construction |

### Why `RLT_a` deviates from the paper

Two concrete reasons:

1. **Language is in the loop — multi-task potential.**
   `RLT` (per the paper's footnote 1) drops language tokens from the
   encoder input on the assumption that each task has a fixed
   instruction. That's fine for a single task but loses the natural
   conditioning signal for multi-task training. `RLT_a` feeds the
   encoder the action-query slice, which is already downstream of the
   VLA's image-language attention — so language is implicitly baked
   into every hidden state and the same actor can cover many tasks.

2. **Encoder training is easier across VLMs.**
   Action queries are exposed in roughly the same way by every VLA
   (one `get_vla_action`-style call), so `RLT_a`'s encoder ports across
   backbones with only a small adapter. `RLT`'s encoder consumes the
   full VLM token sequence, which means each new VLA needs its own
   feature-extraction code that understands that VLA's token layout
   (cf. `algos/RLT/vla_features.py` for Qwen vs
   `vla_features_pi05.py` for Pi05 — two largely independent
   implementations).

`RLT` is the more recent addition: it's closer to the paper's exact
construction (full VLM tokens in, no extra projection, encoder-decoder
cross-attention) and exists for fair side-by-side comparisons. The two
tracks share the trainer, rollout, and eval infrastructure; only the
encoder/decoder class and the `--encoder_mode {action_token, rlt}` flag
differ. Full design notes for each are in their `algos/<track>/README.md`.

### Backbone × track support matrix

|                 | Qwen (`Qwenvl_OFT`) | Pi05 (`PaliGemmaPi05`) |
|:----------------|:--------------------|:-----------------------|
| **`RLT`**       | ✓                   | ✓                      |
| **`RLT_a`**     | ✓                   | roadmapped — not wired yet |

`RLT_a` × Pi05 isn't currently supported: `RLT_a`'s encoder consumes the
VLA's action-query hidden states (which `Qwenvl_OFT` exposes via
`get_vla_action`), but `PaliGemmaPi05`'s flow-matching head outputs
actions directly without an "action-query slice" equivalent — that
adapter is on the roadmap.

---

## File inventory

```
.
├── README.md
├── run_rlt_pretrain.sh        # Phase-1: RLT encoder pretrain
├── run_rlt_rl.sh              # Phase-2: TD3 RL on the RLT track (Qwen or Pi05)
│
├── run_eval_rlt.sh            # offline eval, RLT policy (single iter or all iters)
├── run_eval_rlt_a.sh          # offline eval, RLT_a policy (Qwen only)
│
├── example_results/           # reference plots / summaries
└── example_scripts/
    ├── run_pi05_finetune.sh   # upstream VLA finetune (not in the RL flow)
    ├── pi05_eval.yaml         # VLA-only (no RL) eval configs
    └── ...                    # legacy / one-off launchers
```

(End-to-end `RLT_a` training launcher isn't in this dir; see
`AlphaBrain/training/reinforcement_learning/algos/RLT_a/README.md`.)

---

## Prerequisites

- A pretrained VLA checkpoint on disk. The releases assume either
  `QwenOFT-5traj-libero_goal/final_model` (Qwen) or
  `Pi05-goal-{1traj,5traj}-openpi/checkpoints/steps_30000` (Pi05). If you
  don't have one yet, see `example_scripts/run_pi05_finetune.sh` (Pi05)
  or `configs/finetune_config.yaml` (Qwen).
- LIBERO installed in a separate conda env. Set `LIBERO_PYTHON` and
  `LIBERO_HOME` (in `.env` is fine) so launchers can spawn env workers.
- **Pi05 only**: local PaliGemma tokenizer dir at
  `$PALIGEMMA_TOKENIZER_PATH` (default `/datasets/peligemma`). Without
  this the first rollout dies with `ModuleNotFoundError: sentencepiece`.

---

## Walkthrough: train + eval on `libero_goal` task 0 with Pi05-5traj

Concrete commands for one full run, end-to-end. Assumes the VLA is
already at `results/training/Pi05-goal-5traj-openpi/checkpoints/steps_30000`.

### Step 1 — Phase-1: pretrain the RLT encoder

Compress the VLA's hidden states into a 1-token information bottleneck.
Frozen VLA, trainable encoder/decoder, MSE reconstruction loss.

```bash
CKPT_PATH=results/training/Pi05-goal-5traj-openpi/checkpoints/steps_30000 \
    bash scripts/run_rl_scripts/run_rlt_pretrain.sh 0
```

Output: `results/rlt_training/pi05_5traj_openpi_strict_<MMDD_HHMM>/pretrain/checkpoints/pretrain_best/encoder.pt`

### Step 2 — Phase-2: TD3 RL with the frozen encoder

Frozen VLA + frozen encoder (from Step 1) + trainable actor & critic.
TD updates fed by transitions from parallel LIBERO env rollouts. The
launcher auto-discovers the latest `encoder.pt` produced by Step 1.

```bash
BACKBONE=pi05 VARIANT=5traj TASK_ID=0 \
    bash scripts/run_rl_scripts/run_rlt_rl.sh 0
```

Output: `results/rlt_training/rlt_rl_t0_release_pi05_5traj_<MMDD_HHMM>/rl_offpolicy/`
containing `train.log`, `checkpoints/rl_offpolicy_iter_<N>/`, and online
`eval_iter_<N>_rlt/summary.json`.

The default training schedule runs `--max_iter 300` with a ckpt every
25 iters (12 ckpts total) and an async eval every 10 iters.

### Step 3 — Offline eval across all saved ckpts

The in-training eval is fine for monitoring but uses only 20 episodes.
For paper-grade numbers, re-eval each ckpt with 50 episodes:

```bash
RUN_DIR=results/rlt_training/rlt_rl_t0_release_pi05_5traj_<MMDD_HHMM>/rl_offpolicy \
VLA_CKPT=results/training/Pi05-goal-5traj-openpi/checkpoints/steps_30000 \
GPUS="0 1 2" TASK_IDS=0 N_EPS=50 \
    bash scripts/run_rl_scripts/run_eval_rlt.sh
```

This shards the 12 ckpts across the 3 GPUs (~50 min wall time). Output:
one `eval_iter_<N>_rlt/summary.json` per ckpt + an aggregate table
printed at the end + saved at `<RUN_DIR>/eval_all_iters_rlt/all_iters_summary.json`.

To eval a single ckpt only:
```bash
ITER=00300 RUN_DIR=... VLA_CKPT=... GPUS=0 \
    bash scripts/run_rl_scripts/run_eval_rlt.sh
```

---

## Switching backbones / tracks

```bash
# Same flow, but Qwen backbone (default for run_rlt_rl.sh)
bash scripts/run_rl_scripts/run_rlt_pretrain.sh 0
bash scripts/run_rl_scripts/run_rlt_rl.sh 0
# RLT eval works the same

# Same flow, but Pi05 1traj instead of 5traj
BACKBONE=pi05 VARIANT=1traj TASK_ID=0 bash scripts/run_rl_scripts/run_rlt_rl.sh 0

# RLT_a track (action-token encoder) — Qwen only currently
# Training entry point isn't in this dir; see algos/RLT_a/README.md.
bash scripts/run_rl_scripts/run_eval_rlt_a.sh <RUN_DIR>
```

For VLA-only eval (baseline SR before any RL), point the standard
LIBERO server at the finetune ckpt:

```bash
bash scripts/run_base_vla/eval.sh pi05_goal_5traj_eval \
    scripts/run_rl_scripts/example_scripts/pi05_eval.yaml
```

---

## Output layout

```
results/
├── training/
│   ├── Pi05-goal-{1traj,5traj,task0}-openpi/checkpoints/steps_<N>/   # Pi05 VLA finetune
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
- **Host RAM also matters**: each LIBERO env subprocess is ~600 MB MuJoCo + a few GB of Python overhead. Multiple RL trainings in the same cgroup can OOM-kill workers — check `cat /sys/fs/cgroup/memory.max` before scaling out.
- **Pi05 tokenizer**: always export `PALIGEMMA_TOKENIZER_PATH` (or rely on the launcher default `/datasets/peligemma`). Without it the first rollout VLA call dies with `ModuleNotFoundError: sentencepiece`.
- **Step-lock (`--use_steplock`) is faster** than free-run rollout because it batches all envs through one VLA forward.
- **Async eval during training** uses socket-IPC env workers (matches rollout); the older pipe-IPC `LiberoEnv` deadlocks on some container setups.
- **Encoder must be pretrained on the same VLA you fine-tune on.** Mixing a 1-traj-VLA encoder with 5-traj-VLA RL silently degrades.
- **`[Warning]: datasets path ... does not exist!`** flooding the log: each env reset checks LIBERO's `datasets/` dir, which RL doesn't need. Silence by creating an empty dir:
  ```bash
  DATASETS_DIR=$("$LIBERO_PYTHON" -c "import libero.libero, os; print(os.path.realpath(os.path.join(os.path.dirname(libero.libero.__file__), '..', 'datasets')))")
  mkdir -p "$DATASETS_DIR"
  ```

---

## ⚠️ Disclaimer

A few honest notes on what this release is — and what it isn't:

- **Why we open-source `RLT_a` first.** One of the reasons we open-source `RLT_a` first is that we believe the core idea — compressing VLA hidden states through an information bottleneck and then editing a reference policy with residual actions — is novel and worth sharing with the community at an early stage, *especially in the multi-task, multi-VLM-backbone setting* that the two deviations above are designed for. This does **not** mean we consider GRPO / PPO any less important; on the contrary, we view them as core algorithms for VLA online RL and will progressively release our implementations of them.
- **On reproducing the RL Token paper in simulation.** Faithfully reproducing every detail of the original RL Token paper inside a simulator is hard — in particular the carefully curated tuning datasets and the timely human-in-the-loop interventions described in the paper are difficult to replicate one-for-one in a purely automated sim setup. The `RLT_a` recipe is therefore a best-effort simulation adaptation, not a line-by-line reproduction. The newer `RLT` track is closer to the paper, but still differs in pretrain data, base VLA, and the absence of human-in-the-loop. See each algo's `README.md` for the concrete deltas.
- **What we believe matters most.** Collecting high-quality positive trajectories is one of the most critical open problems in this area — good positives do far more than clever loss tricks. Going forward we plan to:
  1. broaden the online-RL algorithm coverage (GRPO, PPO, and others);
  2. improve tooling for positive-sample collection, filtering, and curation on sim and real-world data;
  3. release stronger, better-documented baselines and more reproducible recipes.

This sub-module is a living research snapshot; APIs, configs, and numbers may change between releases. Issues and PRs are welcome.
