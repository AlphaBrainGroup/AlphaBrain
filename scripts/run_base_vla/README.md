# Base VLA Scripts

Unified training & evaluation pipeline for the three base VLA frameworks in AlphaBrain.

1. **`train.sh`** — Train any base VLA mode (multi-task or single-task)
2. **`eval.sh`** — Evaluate a trained checkpoint on LIBERO benchmarks

---

## Architecture Overview

| Framework | VLM Backbone | Action Head |
|:----------|:-------------|:------------|
| **PaliGemmaOFT** | PaliGemma 3B | DiT regression |
| **PaliGemmaPi05** | PaliGemma 3B | Pi0.5 flow matching expert (300M) |
| **LlamaOFT** | Llama 3.2 Vision 11B | DiT regression |

### Switching VLM Backbone

- **PaliGemma**: Use modes prefixed with `paligemma_*`
- **Llama 3.2 Vision**: Use modes prefixed with `llama_*`

### Switching Framework (Action Head)

- **OFT (DiT regression)**: Use modes with `_oft_` in the name
- **Pi0.5 (flow matching)**: Use modes with `_pi0_` in the name

---

## Performance (LIBERO Benchmark)

| Model | VLM | Action Head | Steps | Finetune Format | EMA | BS | Acc | LIBERO-Goal | LIBERO-Spatial | LIBERO-Object | LIBERO-10 (Long) |
|:------|:----|:------------|:------|:----------------|:----|:---|:----|:------------|:---------------|:--------------|:-----------------|
| OpenPi Pi05 (official) | PaliGemma 3B | Flow Matching 300M | 30k | multi-task | full ema | 256 | - | **98.0** | **98.8** | **98.2** | **92.4** |
| AlphaBrain+PaliGemmaPi05 (pretrained) | PaliGemma 3B | Flow Matching 300M | 22k | multi-task | first 7k w/ ema | 32 | 2 | 94.7 | 93.3 | 98.0 | 92.0 |
| AlphaBrain+PaliGemmaPi05 (no finetune) | PaliGemma 3B | Flow Matching 300M | - | - | - | - | - | 0 | 0 | 0 | 0 |
| AlphaBrain+LlamaPi0 | Llama 11B | MLP | 30k | multi-task | - | 256 | 4 | TBD | TBD | TBD | TBD |
| LlamaOFT | Llama 11B | MLP | 30k | single-task | - | 128 | 1 | 92.0 | 79.0 | 94.0 | 44.0 |
| PaliGemmaOFT | PaliGemma 3B | MLP | 30k | single-task | - | 128 | 1 | 88.0 | 77.0 | 96.0 | 43.0 |

---

## Environment Setup

Before running any script, create a `.env` file in the project root:

```bash
# .env (project root)

# Required: LIBERO dataset root directory
LIBERO_DATA_ROOT=/path/to/IPEC-COMMUNITY

# Required: LIBERO source code directory (for eval client)
LIBERO_HOME=/path/to/LIBERO

# Required for PaliGemmaPi05 models: PaliGemma tokenizer path
# Must use the original Gemma tokenizer (vocab_size=256000)
# Do NOT use the PaliGemma VLM checkpoint tokenizer (vocab_size=257152)
PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma_tokenizer
```

> **⚠️ Flash Attention**: If `flash_attn` is not installed (e.g., on NVIDIA B200/Blackwell GPUs), the code automatically falls back to SDPA. No manual config changes needed.

---

## Gripper Handling

Different model types use different gripper conventions. The eval client (`model2libero_interface.py`) auto-detects the correct handling based on the checkpoint's `framework_config.yaml`:

| Model Type | Detection | Gripper Processing |
|:-----------|:----------|:-------------------|
| **PaliGemmaOFT / LlamaOFT** | `framework.name` not Pi0-family | Standard q99 unnorm (no extra processing) |
| **Pi0/Pi0.5 with MEAN_STD norm** (our pipeline finetuned) | `framework.name` ∈ Pi0-family, `normalization.enabled=true` | `skip_client_unnorm=True`, gripper inverted by client (`-gripper`). Framework's `predict_action` maps gripper via `1 - 2g` before returning. |
| **Pi0/Pi0.5 without norm** (OpenPI official checkpoints) | `framework.name` ∈ Pi0-family, no normalization | `skip_client_unnorm=False`, q99 unnorm then `1.0 - gripper` (`invert_gripper_after_unnorm=True`) |

This is fully automatic — no manual flags needed. Just point to the checkpoint and run eval.

---

## Quick Start

> **Environment**: Activate your project virtual environment before running any script.

### Training

```bash
# ── Multi-task (libero_all) ──────────────────────────────────

# PaliGemmaPi05: 4 GPU, BS=256, 60k steps (aligned with OpenPi official)
bash scripts/run_base_vla/train.sh paligemma_pi0_openpi_aligned_v3

# PaliGemmaOFT: 4 GPU, BS=128, 150k steps
bash scripts/run_base_vla/train.sh paligemma_oft_all_150k

# LlamaOFT: 4 GPU, BS=128, 1.2M steps (LM frozen)
bash scripts/run_base_vla/train.sh llama_oft_all_150k

# ── Single-task ──────────────────────────────────────────────

# PaliGemmaOFT single-task
bash scripts/run_base_vla/train.sh paligemma_oft_goal
bash scripts/run_base_vla/train.sh paligemma_oft_spatial
bash scripts/run_base_vla/train.sh paligemma_oft_object
bash scripts/run_base_vla/train.sh paligemma_oft_long

# LlamaOFT single-task
bash scripts/run_base_vla/train.sh llama_oft_goal
bash scripts/run_base_vla/train.sh llama_oft_spatial
bash scripts/run_base_vla/train.sh llama_oft_object
bash scripts/run_base_vla/train.sh llama_oft_long
```

### Evaluation

```bash
# PaliGemmaPi05 eval (aligned_v2 checkpoint)
bash scripts/run_base_vla/eval.sh paligemma_pi0_v2_goal_eval

# PaliGemmaOFT eval
bash scripts/run_base_vla/eval.sh paligemma_oft_eval
bash scripts/run_base_vla/eval.sh paligemma_oft_mlp_goal30k_eval
bash scripts/run_base_vla/eval.sh paligemma_oft_mlp_long50k_eval
bash scripts/run_base_vla/eval.sh paligemma_oft_bs128_goal_eval
bash scripts/run_base_vla/eval.sh paligemma_oft_bs128_spatial_eval
bash scripts/run_base_vla/eval.sh paligemma_oft_bs128_object_eval
bash scripts/run_base_vla/eval.sh paligemma_oft_bs128_long_eval

# LlamaOFT eval
bash scripts/run_base_vla/eval.sh llama_oft_eval
```

---

## Mode Reference

### Multi-task Training Modes (libero_all)

| Mode | Framework | GPUs | Effective BS | Steps | Notes |
|:-----|:----------|:-----|:-------------|:------|:------|
| `paligemma_pi0_openpi_aligned_v3` | PaliGemmaPi05 | 4 | 256 (32×4, acc=2) | 60k | Aligned with OpenPi official; no EMA |
| `paligemma_oft_all_150k` | PaliGemmaOFT | 4 | 128 (32×4, acc=1) | 150k | LR 2.4e-4 base, 8e-4 action |
| `llama_oft_all_150k` | LlamaOFT | 4 | 128 (4×4, acc=8) | 1.2M | LM frozen; LR 2.4e-4 base |

### Single-task Training Modes

| Suite | PaliGemmaOFT | LlamaOFT |
|:------|:-------------|:---------|
| libero_goal | `paligemma_oft_goal` | `llama_oft_goal` |
| libero_spatial | `paligemma_oft_spatial` | `llama_oft_spatial` |
| libero_object | `paligemma_oft_object` | `llama_oft_object` |
| libero_long (10) | `paligemma_oft_long` | `llama_oft_long` |

> **Note**: PaliGemmaPi05 currently only has a multi-task mode (`paligemma_pi0_openpi_aligned_v3`). To add single-task Pi05 modes, define new entries in `configs/finetune_config.yaml` with the desired `dataset_mix`.

### How to Switch Between libero_all and Single-task

Simply choose the corresponding mode name. Multi-task modes use `dataset_mix: "libero_all"`, single-task modes use `dataset_mix: "libero_goal"` / `"libero_spatial"` / etc. All configurations are defined in `configs/finetune_config.yaml`.

### Eval Modes

| Mode | Framework | Checkpoint | Benchmark |
|:-----|:----------|:-----------|:----------|
| `paligemma_pi0_v2_goal_eval` | PaliGemmaPi05 | aligned_v2 final_model | libero_goal |
| `paligemma_oft_eval` | PaliGemmaOFT | custom | libero_goal |
| `paligemma_oft_mlp_goal30k_eval` | PaliGemmaOFT | mlp_goal_30k | libero_goal |
| `paligemma_oft_mlp_long50k_eval` | PaliGemmaOFT | mlp_long_50k | libero_10 |
| `paligemma_oft_bs128_goal_eval` | PaliGemmaOFT | bs128 | libero_goal |
| `paligemma_oft_bs128_spatial_eval` | PaliGemmaOFT | bs128 | libero_spatial |
| `paligemma_oft_bs128_object_eval` | PaliGemmaOFT | bs128 | libero_object |
| `paligemma_oft_bs128_long_eval` | PaliGemmaOFT | bs128 | libero_10 |
| `llama_oft_eval` | LlamaOFT | custom | libero_goal |

> To add more eval modes, define new entries in `configs/finetune_config.yaml` with `type: "eval"`, specifying `checkpoint`, `task_suite`, `num_trials`, `host`, `port`, and `gpu_id`.

---

## Custom Config Override

Both scripts accept an optional second argument to use a custom config file:

```bash
bash scripts/run_base_vla/train.sh my_custom_mode configs/my_custom_config.yaml
```

All other hyper-parameters (LR, batch size, DeepSpeed config, dataset paths…) are set in the YAML mode definition.

---

## Related Components

| Component | Path |
|:----------|:-----|
| PaliGemmaOFT framework | `AlphaBrain/model/framework/PaliGemmaOFT.py` |
| PaliGemmaPi05 framework | `AlphaBrain/model/framework/PaliGemmaPi05.py` |
| LlamaOFT framework | `AlphaBrain/model/framework/LlamaOFT.py` |
| Config (mode router) | `configs/finetune_config.yaml` |
| Model architecture defaults | `configs/models/paligemma_oft.yaml`, `configs/models/paligemma_pi0.yaml`, `configs/models/llama_oft.yaml` |
| Training entrypoint | `AlphaBrain/training/train_alphabrain.py` |
| Eval entrypoint | `AlphaBrain/evaluation/` |
