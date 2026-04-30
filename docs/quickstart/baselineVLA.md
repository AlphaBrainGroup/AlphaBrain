# Baseline VLA

Train and evaluate AlphaBrain's three base VLA frameworks — **PaliGemmaOFT**, **PaliGemmaPi05**, **LlamaOFT** — on LIBERO.

---

## Prerequisites

```bash
conda activate alphabrain
cp .env.example .env                      # fill in the paths below
```

```bash
PRETRAINED_MODELS_DIR=data/pretrained_models
LIBERO_DATA_ROOT=/path/to/lerobot/libero
LIBERO_HOME=/path/to/LIBERO
LIBERO_PYTHON=/path/to/miniconda3/envs/libero/bin/python
```

See [Installation](installation.md) for the full environment setup.

---

## Train

```bash
# PaliGemmaPi05 multi-task (4 GPU, BS=256, 60k steps)
bash scripts/run_base_vla/train.sh paligemma_pi05_openpi_aligned_v3

# PaliGemmaOFT multi-task (4 GPU, BS=128, 150k steps)
bash scripts/run_base_vla/train.sh paligemma_oft_all_150k

# LlamaOFT multi-task (4 GPU, BS=128, 1.2M steps; LM frozen)
bash scripts/run_base_vla/train.sh llama_oft_all_150k

# Single-task examples
bash scripts/run_base_vla/train.sh paligemma_oft_goal
bash scripts/run_base_vla/train.sh llama_oft_goal
```

Checkpoints: `results/training/<run_id>/checkpoints/steps_*`.

## Evaluate

```bash
bash scripts/run_base_vla/eval.sh paligemma_pi05_eval
bash scripts/run_base_vla/eval.sh paligemma_oft_bs128_goal_eval
bash scripts/run_base_vla/eval.sh llama_oft_eval
```

Results: per-task + overall success rate.

---

## Next Steps

- [NeuroVLA](neurovla.md) — brain-inspired SNN action head
- [RL-Token](rl_token.md) — online RL fine-tuning
- [World Model](world_model.md) — V-JEPA / Cosmos / Wan backbones
- [Continual Learning](continual_learning.md) — sequential finetuning across 10 tasks

Full mode list, perf table, and custom-config overrides: [`scripts/run_base_vla/README.md`](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/scripts/run_base_vla/README.md).
