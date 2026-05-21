# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo has two distinct layers:

1. **AlphaBrain** — a modular PyTorch framework for VLA (Vision-Language-Action) robot models (training, continual learning, RL, world models).
2. **Nvex** — a product layer built on top: a FastAPI backend (`nvex_server/`) and a React investor demo (`demo/`) that showcases the autonomous self-improvement loop.

Active development is focused on Nvex. AlphaBrain is the execution engine Nvex orchestrates.

## Nvex Demo Stack

### Running locally

Start the backend and frontend in two terminals:

```bash
# Terminal 1 — FastAPI backend (auto-reloads, seeds demo state on startup)
.venv/bin/python -m uvicorn nvex_server.app:app --reload --port 8000

# Terminal 2 — React dev server (proxies /api and /health to :8000)
cd demo && npm install && npm run dev   # → http://127.0.0.1:5173
```

Standalone demo (no backend needed):
```bash
open demo/nvex-demo.html
```

### Frontend commands

```bash
cd demo
npm run dev      # dev server with hot reload
npm run build    # production build → dist/
npm run lint     # ESLint
npm run preview  # serve the built dist/
```

### Backend lint

```bash
ruff check nvex_server/
black nvex_server/   # line-length 121
```

## Nvex Architecture

### `nvex_server/` — FastAPI backend

All state is held in `InMemoryStore` (defined in `app.py`) — no database. `create_app()` seeds demo state on startup using fixture files in `nvex_server/examples/`.

| File | Role |
|---|---|
| `app.py` | FastAPI app factory, `InMemoryStore`, all route handlers, `seed_demo_state()` |
| `schemas.py` | All Pydantic models — `EvalRun`, `PatchPlan`, `IterationJob`, `AgentRunState`, `AgentEvent`, etc. |
| `agent.py` | `SelfImprovementAgent` — autonomous eval→diagnose→plan→dispatch→verify→memory loop |
| `dispatcher.py` | `JobDispatcher` — builds and launches AlphaBrain shell commands; writes per-job metadata to `results/nvex_jobs/` |
| `exporters.py` | `EvalArtifactExporter` — parses LIBERO, RoboCasa, and generic JSON eval artifacts into `EvalRun` |
| `patch_plan_generator.py` | Rule-based `PatchPlanGenerator` — maps failure clusters to training strategies |
| `llm_narrator.py` | `LLMNarrator` — optional LLM-generated narration for diagnosis steps |

**Key API routes:**

| Route | Purpose |
|---|---|
| `GET /api/demo/state` | Returns full seeded dashboard state for the React app |
| `GET /api/demo/agent` | Returns the pre-seeded `AgentRunState` |
| `POST /api/agent/run` | Start a new autonomous improvement run |
| `POST /api/agent/{id}/advance` | Advance demo agent by one step (called by the UI stream loop) |
| `POST /api/eval/import` | Import benchmark artifacts into `EvalRun` |
| `POST /api/plan/generate` | Generate a `PatchPlan` from an `EvalRun` |
| `POST /api/iteration/start` | Create and optionally run an `IterationJob` |

### Agent demo flow

`SelfImprovementAgent` runs in two modes:

- **simulate=True** (default for demos): A precomputed 4-loop sequence is stored in `_DEMO_LOOPS` in `agent.py`. Each call to `advance_step` marks the current running step complete and activates the next. Loop 3 demonstrates a regression + rollback (81% → 79%) before recovering to 85% in loop 4.
- **simulate=False** (real mode): Drives `JobDispatcher` → AlphaBrain shell commands in a background thread. Real-mode loop body is a M4 extension point.

The React frontend calls `POST /api/agent/{id}/advance` on a timer driven by `step.expected_duration_ms`. The stream loop lives in `NvexRuntimeContext.jsx`.

### `demo/src/` — React frontend

`NvexRuntimeContext.jsx` is the central data layer: it fetches `/api/demo/state`, normalizes the snake_case API response into camelCase dashboard shape (`normalizeDemoState`), and exposes stream controls (`streamAgentRun`, `advanceAgentStep`, `stopAgentStream`).

Pages map to the 7-page investor narrative: Home → Overview → Failure Map → Patch Plan → Iteration Runner → Improvement Report → Platform Memory.

No UI framework — pure CSS with dark `#07090f` theme and indigo-violet gradients.

---

## AlphaBrain (ML Framework)

### Environment Setup

```bash
conda create -n alphabrain python=3.10 -y && conda activate alphabrain
pip install -r requirements.txt && pip install -e .
pip install flash-attn --no-build-isolation
cp .env.example .env
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

### Commands

```bash
ruff check AlphaBrain/          # lint
black AlphaBrain/               # format (line-length 121)
python -c "import AlphaBrain; print('ok')"   # verify install

bash scripts/run_finetune.sh <mode>                          # main training entry point
bash scripts/run_continual_learning_scripts/run_cl_train.sh  # CL training
bash scripts/run_continual_learning_scripts/run_cl_eval.sh --run-id <id>
bash scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh
MODEL=cos2 bash scripts/run_world_model/train/run_world_model.sh
bash scripts/run_rl_scripts/run_action_token_5traj_alltasks.sh
```

### Package Layout

```
AlphaBrain/
├── model/
│   ├── framework/      # VLA model implementations (one file per architecture)
│   └── modules/        # Shared sub-modules: action_model, vlm, projector, dino, world_model
├── training/
│   ├── train_alphabrain.py          # Main trainer (Accelerate + DeepSpeed ZeRO-2)
│   ├── train_alphabrain_vlm.py      # VLM co-training variant
│   ├── continual_learning/          # Experience-replay CL trainer
│   └── reinforcement_learning/      # RLActionToken TD3 (needs 6 GPUs)
└── dataloader/
    ├── lerobot_datasets.py          # LeRobot-format datasets (main)
    └── cosmos_datasets.py           # World model datasets
```

### Config System (priority low → high)

```
configs/models/<model>.yaml       # architecture defaults
configs/datasets/<dataset>.yaml   # dataset paths & task lists
configs/trainer/default.yaml      # optimizer, LR schedule, save intervals
configs/finetune_config.yaml      # named modes with per-run overrides (highest priority)
CLI args                          # dot-list overrides, e.g. trainer.learning_rate.base=1e-5
```

`configs/finetune_config.yaml` defines named **modes**. `scripts/parse_config.py` resolves them into shell env-vars consumed by `run_finetune.sh`.

### Model Framework

All VLA architectures extend `BaseFramework` (`AlphaBrain/model/framework/base_framework.py`), which handles config loading, action normalization, and trainable-module discovery. New frameworks register via `@FRAMEWORK_REGISTRY.register("MyFramework")` in `AlphaBrain/model/tools.py`.

VLM backends are detected via `_VLM_REGISTRY` in `base_framework.py`: `paligemma` → `vlm_interface`, `llamavl` → `llama_vl_interface`, `qwenvl` → `qwen_vl_interface`.

### Capability Modules

| Module | Entry Point | Notes |
|---|---|---|
| NeuroVLA (SNN + STDP) | `train_stdp.py` | QFormer → LIF neurons; R-STDP and online STDP modes |
| RLActionToken (TD3) | `reinforcement_learning/` | Needs 6 GPUs (5 rollout + 1 train) |
| Continual Learning | `continual_learning/` | Experience replay; LoRA (~6% trainable params) |
| World Model | `WorldModelVLA.py` | Cosmos 2/2.5, Wan 2.2, V-JEPA 2.1 backbones |

### Dispatcher → AlphaBrain bridge

`JobDispatcher._build_command()` maps `ExecutionBackend` values to AlphaBrain shell commands:
- `alphabrain_cl` → `bash scripts/run_continual_learning_scripts/run_cl_train.sh`
- `alphabrain_finetune` / `alphabrain_vlm_cotrain` → `bash scripts/run_finetune.sh <mode>`
- `alphabrain_eval` → `bash scripts/run_eval.sh <mode>`

Job logs and metadata land in `results/nvex_jobs/<iteration_id>/`.
