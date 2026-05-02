# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaBrain is a modular PyTorch framework for embodied intelligence research — specifically Vision-Language-Action (VLA) models for robot manipulation. It unifies multiple VLA architectures (GR00T, OFT, PI, NeuroVLA, CosmosPolicy, etc.), world model backbones, continual learning, and RL fine-tuning under one stack.

## Environment Setup

```bash
conda create -n alphabrain python=3.10 -y && conda activate alphabrain
pip install -r requirements.txt && pip install -e .
pip install flash-attn --no-build-isolation
cp .env.example .env   # fill in paths below
```

Key `.env` variables:
```bash
PRETRAINED_MODELS_DIR=/path/to/pretrained_models   # Qwen2.5-VL, Qwen3-VL weights
LEROBOT_LIBERO_DATA_DIR=/path/to/lerobot/libero    # LeRobot-format training data
LIBERO_DATA_ROOT=/path/to/IPEC-COMMUNITY            # RLDS-format eval data
LIBERO_HOME=/path/to/LIBERO                         # LIBERO simulation env
LIBERO_PYTHON=/path/to/envs/libero/bin/python       # Separate eval conda env
```

Evaluation (LIBERO) requires a **separate conda environment** — see `docs/quickstart/installation.md`.

## Commands

**Lint / format:**
```bash
ruff check AlphaBrain/          # lint
black AlphaBrain/               # format (line-length 121)
```

**Verify install:**
```bash
python -c "import AlphaBrain; print('ok')"
```

**Training (unified entry point):**
```bash
bash scripts/run_finetune.sh <mode>
# e.g.:
bash scripts/run_finetune.sh qwen_oft_goal
bash scripts/run_finetune.sh paligemma_oft_all_150k
```

**Benchmark-specific training/eval:**
```bash
bash scripts/run_base_vla/train.sh <mode>
bash scripts/run_base_vla/eval.sh <mode>

bash scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh
bash scripts/run_brain_inspired_scripts/run_stdp_finetune.sh --pretrained <ckpt>

bash scripts/run_continual_learning_scripts/run_cl_train.sh
bash scripts/run_continual_learning_scripts/run_cl_eval.sh --run-id <id>

MODEL=cos2 bash scripts/run_world_model/train/run_world_model.sh
bash scripts/run_rl_scripts/run_action_token_5traj_alltasks.sh
```

## Architecture

### Package Layout

```
AlphaBrain/
├── model/
│   ├── framework/      # VLA model implementations (one file per architecture)
│   └── modules/        # Shared sub-modules: action_model, vlm, projector, dino, world_model
├── training/
│   ├── train_alphabrain.py          # Main trainer (Accelerate + DeepSpeed)
│   ├── train_alphabrain_vlm.py      # VLM co-training variant
│   ├── train_alphabrain_cotrain.py  # Co-train variant
│   ├── train_stdp.py                # NeuroVLA STDP training
│   ├── continual_learning/          # Experience-replay CL trainer
│   ├── reinforcement_learning/      # RLActionToken TD3 trainer
│   └── trainer_utils/               # Config loading, LR groups, PEFT helpers
└── dataloader/
    ├── lerobot_datasets.py          # LeRobot-format datasets (main)
    ├── gr00t_lerobot/               # GR00T-specific data pipeline
    ├── vlm_datasets.py              # VLM co-training data
    └── cosmos_datasets.py           # World model datasets
```

### Config System (priority low → high)

```
configs/models/<model>.yaml       # architecture defaults (action_dim, hidden dims, etc.)
configs/datasets/<dataset>.yaml   # dataset paths & task lists
configs/trainer/default.yaml      # optimizer, LR schedule, save intervals
configs/finetune_config.yaml      # modes section: per-run overrides (highest priority)
CLI args                          # dot-list overrides, e.g. trainer.learning_rate.base=1e-5
```

`configs/finetune_config.yaml` is the main entry point: it defines named **modes**, each pointing at a `model:`, `dataset:`, and optional overrides. `scripts/parse_config.py` resolves this into shell env-vars that `run_finetune.sh` reads.

### Model Framework

All VLA architectures extend from `AlphaBrain/model/framework/base_framework.py` (`BaseFramework`), which handles:
- Loading YAML config + normalization stats from checkpoints
- Action normalization / un-normalization
- Discovering trainable sub-modules

New frameworks register via `FRAMEWORK_REGISTRY` (see `AlphaBrain/model/tools.py`) and `build_framework()` dispatches by `cfg.framework.name`.

**VLM backends** are detected via `_VLM_REGISTRY` in `base_framework.py`: `paligemma` → `vlm_interface`, `llamavl` → `llama_vl_interface`, `qwenvl` → `qwen_vl_interface`.

### Training Stack

The trainer (`train_alphabrain.py`) uses **Accelerate + DeepSpeed** (ZeRO-2 by default). Key design choices:
- Multiple `DataLoader`s for heterogeneous task mixtures
- Per-module learning rates configured in `configs/trainer/default.yaml` (`learning_rate.base`, `learning_rate.action_model`, `learning_rate.qwen_vl_interface`)
- Checkpoints saved to `results/training/<run_id>/checkpoints/steps_*/`
- W&B logging enabled by default (`wandb_mode: online`)

### Capability Modules

| Module | Entry Point | Notes |
|---|---|---|
| NeuroVLA (SNN + STDP) | `train_stdp.py` | QFormer → LIF neurons; R-STDP and online STDP modes |
| RLActionToken (TD3) | `reinforcement_learning/` | Needs 6 GPUs (5 rollout + 1 train) |
| Continual Learning | `continual_learning/` | Experience replay; LoRA (~6% trainable params) |
| World Model | `WorldModelVLA.py` framework | Cosmos 2/2.5, Wan 2.2, V-JEPA 2.1 backbones; requires text embedding precomputation |

### Adding a New Framework

1. Create `AlphaBrain/model/framework/MyFramework.py` implementing `BaseFramework`
2. Add a `configs/models/my_framework.yaml` with `framework.name: MyFramework`
3. Either register via `@FRAMEWORK_REGISTRY.register("MyFramework")` or add an `elif` branch in `build_framework()`
4. Add a mode entry in `configs/finetune_config.yaml`
