# Quickstart

From a blank machine to a trained-and-evaluated VLA checkpoint — then into any of the five capability-specific paths.

!!! tip "Read top to bottom"
    Work through **Installation → Baseline VLA** first. Everything below that is optional and can be tackled in any order.

---

## The path

| Step | Page | What it does | Default hardware |
|:---:|:-----|:-------------|:-----------------|
| 0 | [**Installation**](installation.md) | Conda envs, Flash Attention, pretrained weights, LIBERO data, `.env` config. | — |
| 1 | [**Baseline VLA**](baselineVLA.md) | PaliGemmaOFT / Pi05 / LlamaOFT finetune + LIBERO eval — the first trained checkpoint from the default recipe. | 4× A800 |
| 2 | [**NeuroVLA**](neurovla.md) | Pretrain a brain-inspired VLA, then R-STDP fine-tune the SNN action head. | 4× A800 |
| 3 | [**ActionToken**](action_token.md) | Off-policy TD3 RL on a pretrained QwenOFT (encoder pretrain + rollout + TD updates). | 6× A800 |
| 4 | [**World Model**](world_model.md) | V-JEPA / Cosmos / Wan backbones × GR00T / OFT / PI decoders (12 combinations) + CosmosPolicy. | 4× A800 |
| 5 | [**Continual Learning**](continual_learning.md) | Sequential finetuning across 10 LIBERO tasks with Experience Replay (LoRA + full-param). | 2× A800 |

Every capability page follows the same layout — Overview → Prerequisites → Quick Start → Tips → Pointers — so you can skim one and recognize the shape of the next.

---

## Minimum requirements

- Python 3.10+
- CUDA 11.8 / 12.x with matching PyTorch 2.1+
- At least one A100/H100-class GPU (4× A800 recommended for the default Baseline VLA recipe)
- ~100 GB of local disk for LIBERO datasets + pretrained VLMs

Detailed environment setup lives in [Installation](installation.md).

---

## Shared conventions

Every quickstart assumes the same project layout:

```
AlphaBrain/
├── .env                               # local paths (see Installation)
├── configs/
│   ├── finetune_config.yaml          # single-entry training/eval modes
│   ├── continual_learning/*.yaml     # CL-specific configs
│   └── rl_recipes/*.yaml             # RL-specific configs
├── scripts/
│   ├── run_finetune.sh               # training launcher
│   ├── run_eval.sh                   # eval launcher
│   ├── run_base_vla/                 # Baseline VLA wrappers
│   ├── run_brain_inspired_scripts/   # NeuroVLA wrappers
│   ├── run_rl_scripts/               # ActionToken wrappers
│   ├── run_continual_learning_scripts/
│   └── run_world_model/              # World Model wrappers
└── results/
    ├── training/<run_id>/            # finetune outputs
    └── evaluation/<run_id>/          # eval outputs
```

All launchers:

1. Source `.env` at the project root
2. Read a mode block from YAML (mode name → resolved config)
3. Create the output dir and snapshot the config for reproducibility
4. Call `accelerate launch` (training) or start a server + client (eval)

---

## Shared prerequisites

Run these once before any capability page:

1. Complete [Installation](installation.md) — main conda env + flash-attn + eval env + LIBERO data.
2. Fill out `.env` with `PRETRAINED_MODELS_DIR`, `LEROBOT_LIBERO_DATA_DIR`, `LIBERO_DATA_ROOT`, `LIBERO_HOME`, `LIBERO_PYTHON`.
3. Download the backbone(s) you need (see each page's Prerequisites section).

---

## Which page should I read?

- **"I just installed — does training work?"** → [Baseline VLA](baselineVLA.md)
- **"How do I finetune a different backbone?"** → [Baseline VLA](baselineVLA.md) § Switch Backbone
- **"Is there a bio-plausible fine-tuning path?"** → [NeuroVLA](neurovla.md)
- **"Can I improve a pretrained VLA with RL?"** → [ActionToken](action_token.md)
- **"I want the SOTA visual backbone."** → [World Model](world_model.md) (Cosmos 2.0 + GR00T currently leads)
- **"How do I handle task drift across 10 tasks?"** → [Continual Learning](continual_learning.md)

---

## Getting help

- Full installation reference → [Installation](installation.md)
- First trained-and-evaluated checkpoint → [Baseline VLA](baselineVLA.md)
- Source-level reference → [API Reference](../api/index.md)
- File an issue → [github.com/AlphaBrainGroup/AlphaBrain/issues](https://github.com/AlphaBrainGroup/AlphaBrain/issues)
