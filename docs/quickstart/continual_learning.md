# Continual Learning

Train a single VLA backbone sequentially over the four LIBERO task
suites with a pluggable family of CL algorithms (ER / MIR / EWC)
selectable from YAML.  Supports 4 architectures × LoRA / full-parameter.

---

## Prerequisites

```bash
conda activate alphabrain
cp .env.example .env
vim .env           # fill in paths below
```

Required env vars: `PRETRAINED_MODELS_DIR`, `LEROBOT_LIBERO_DATA_DIR`,
`LIBERO_PYTHON`, `LIBERO_HOME`.

---

## Train

```bash
# Default: QwenGR00T LoRA + ER on LIBERO-Goal (~15 h on 2× A800)
bash scripts/run_continual_learning_scripts/run_cl_train.sh

# Smoke test (~3 min, pipeline verification)
bash scripts/run_continual_learning_scripts/run_cl_train.sh --smoke

# Switch CL algorithm (MIR 77% LIBERO-Goal recipe)
bash scripts/run_continual_learning_scripts/run_cl_train.sh \
    --yaml configs/continual_learning/qwengr00t_mir_lora_libero_refresh50.yaml \
    --gpus 0,1,2,3

# Switch suite (LIBERO-Long with ER)
bash scripts/run_continual_learning_scripts/run_cl_train.sh \
    --yaml configs/continual_learning/qwengr00t_er_lora_libero_long.yaml

# Switch backbone (NeuroVLA + LoRA + ER on LIBERO-Goal)
bash scripts/run_continual_learning_scripts/run_cl_train.sh \
    --yaml configs/continual_learning/neurovla_er_lora_libero.yaml \
    --run-id my_neurovla_run
```

Checkpoints: `results/Checkpoints/<run_id>/`.

## Evaluate

```bash
# LIBERO LoRA run — base-config required for adapter merge
bash scripts/run_continual_learning_scripts/run_cl_eval.sh \
    --run-id qwengr00t_er_lora_libero_goal_v1 \
    --base-config configs/continual_learning/qwengr00t_er_lora_libero.yaml \
    --gpus 0,1 --trials 50

# Full-param run (no LoRA merge)
bash scripts/run_continual_learning_scripts/run_cl_eval.sh \
    --run-id neurovla_er_libero_goal_v1 --gpus 1
```

Per-task SR + 10×10 matrix: `results/eval_cl/<run_id>/`.  Aggregate
ASR / BWT / forgetting: `python scripts/run_continual_learning_scripts/compute_cl_matrix_metrics.py results/eval_cl/<run_id>`.

---

Full yaml preset list, CLI flags, algorithm descriptions, and headline
results: [`scripts/run_continual_learning_scripts/README.md`](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/scripts/run_continual_learning_scripts/README.md).
