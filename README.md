<div align="center">

# Nvex × AlphaBrain

### The Self-Improving Physical AI Stack — for Teams That Ship Robots

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/Docs-Online-green.svg)](https://alphabraingroup.github.io/AlphaBrain/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-orange.svg)](https://huggingface.co/AlphaBrainGroup)
[![WeChat](https://img.shields.io/badge/WeChat-Group-07C160.svg)](assets/wechat.jpg)

<p align="center">
  <img src="assets/main_fig.png" width="100%" alt="Nvex × AlphaBrain Architecture Overview"/>
</p>

**When a Physical AI policy fails, Nvex identifies the failure pattern, diagnoses the capability gap, generates a targeted patch plan, and orchestrates AlphaBrain to deliver a verifiable checkpoint improvement — closing the loop from failure to fix, autonomously.**

[What is Nvex](#-what-is-nvex) · [Who It's For](#-who-its-for) · [Self-Improving Agent](#-self-improving-agent) · [AlphaBrain Framework](#-alphabrain-execution-layer) · [Demo](#-demo) · [Quick Start](#-quick-start) · [Community](#-community)

</div>

---

## 🧩 What is Nvex

Most Physical AI teams hit the same wall after initial training: the policy fails in deployment, and the path to improvement is murky. Teams add more data blindly, rely on intuition to diagnose root causes, and run disconnected cycles of annotation, training, and evaluation with no unified loop.

**Nvex is the orchestration layer that closes this loop.**

It sits above the execution runtime and drives the full intelligence cycle:

```
eval → failure diagnosis → gap analysis → data targeting → post-training → re-eval
```

At each step, Nvex produces structured, actionable outputs — not dashboards full of charts, but decisions: what failed, why it failed, what data to target, which training strategy to apply, and how to verify the fix. Every iteration compounds into reusable platform assets: recipes, templates, failure ontologies, and verification setups.

**Nvex is not a training framework. It is not an annotation tool. It is the intelligence layer that decides what to do next — and executes it.**

---

## 👥 Who It's For

**Robotics & Physical AI teams** who have a trained policy and need to improve it faster:
- You're running evals and getting 60–70% success — and can't tell exactly why it's failing
- You're spending weeks on manual root-cause analysis between training runs
- You want to close the gap between "something broke in deployment" and "here's the targeted fix"
- You need every iteration to build institutional knowledge, not just a new checkpoint

**Investors & decision-makers** evaluating the Physical AI infrastructure landscape:
- You want to understand what a compound, platform-grade post-training system looks like
- You're asking why Nvex is not just a wrapper around AlphaBrain or an MLOps dashboard
- You want to see a measurable, repeatable failure-to-fix loop

---

## 🏗 System Architecture

The stack has two distinct layers:

```
┌─────────────────────────────────────────────────────┐
│                    Nvex                              │
│         Orchestration / Intelligence Layer           │
│                                                      │
│  Failure Map → Patch Plan → Iteration Runner         │
│  Improvement Report → Platform Memory               │
└─────────────────────┬───────────────────────────────┘
                      │  job dispatch / artifact consumption
┌─────────────────────▼───────────────────────────────┐
│                  AlphaBrain                          │
│           Execution / Runtime Layer                  │
│                                                      │
│  VLA train/eval · Continual Learning · World Model   │
│  RL fine-tuning · Benchmark suites                   │
└─────────────────────────────────────────────────────┘
```

**Nvex** owns the intelligence: failure analysis, patch planning, experiment orchestration, result presentation, and platform memory accumulation.

**AlphaBrain** owns the execution: VLA training, evaluation, continual learning, world model runs, and benchmark artifact generation.

Nvex does not reimplement training runtimes. It consumes AlphaBrain's capabilities through a standardized job interface and transforms the outputs into structured intelligence.

---

## 🔁 The Failure-to-Fix Loop

The core demo narrative follows a LIBERO Kitchen Pick-and-Place scenario:

> `NeuroVLA-LIBERO-ckpt_v0.7` is running at 62% success. Nvex diagnoses that failures cluster around occlusion-heavy scenes and missing recovery trajectories. It generates a structured patch plan — 120 targeted episodes, teleop corrections, a continual learning training pass — and dispatches AlphaBrain to execute. `ckpt_v0.8` comes back at 74% (+12%). The patch recipe is saved to Platform Memory for reuse on future projects.

### The Five Flows

| Flow | What happens |
|:-----|:-------------|
| **Project Intake** | Load a checkpoint and eval result; get a project summary, risk flags, and recommended next action |
| **Failure Map** | Compress raw benchmark outputs into structured failure clusters, root-cause hypotheses, and prioritized gaps |
| **Patch Plan** | Generate a structured fix: target data spec, training strategy, verification setup, expected uplift, confidence |
| **Iteration Runner** | Dispatch a continual learning or fine-tune job to AlphaBrain; track stages in real time |
| **Improvement Report + Platform Memory** | Show before/after KPI uplift; save recipes, templates, and failure patterns as reusable platform assets |

---

## 🤖 Self-Improving Agent

The most powerful demo of Nvex is watching it close the loop **without human intervention**. The self-improvement agent runs the full cycle autonomously:

```
1. Load a checkpoint + eval results
2. Nvex diagnoses failure clusters and root causes
3. Nvex generates a structured patch plan (data spec, training strategy, verification)
4. AlphaBrain executes the patch (continual learning or fine-tune run)
5. Nvex re-evaluates and confirms improvement
6. Assets are saved to Platform Memory for future iterations
```

In the LIBERO Kitchen scenario this takes the policy from **62% → 74% success in a single autonomous loop** — with no human deciding what data to collect or which training strategy to apply.

See [`SELF_IMPROVEMENT_AGENT.md`](SELF_IMPROVEMENT_AGENT.md) for the full design, demo modes, and roadmap for the autonomous agent.

---

## ⚡ AlphaBrain — Execution Layer

AlphaBrain is a modular PyTorch framework for embodied intelligence research. It unifies multiple VLA architectures, world model backbones, biologically-inspired learning, and RL fine-tuning under one extensible stack.

### VLA Frameworks

| Framework | Action Decoding | Typical Use |
|:----------|:----------------|:------------|
| **OFT** | MLP action head, parallel continuous decoding | Fast prototyping, baseline alignment |
| **GR00T** | System1 + Flow-Matching DiT System2 | High-precision manipulation, long-horizon planning |
| **PI** | Flow-Matching action prediction | Diffusion-style policies |
| **Adapter** | Lightweight Adapter decoding | Parameter-efficient fine-tuning |
| **NeuroVLA** | Bio-inspired spiking + STDP | Brain-inspired control |
| **CosmosPolicy** | Latent-space video diffusion | World-model-native policy |

### Capability Modules

**Brain-Inspired VLA (NeuroVLA + STDP)** — The first open-source biologically-inspired VLA. QFormer extracts layer-wise features from VLM hidden states; an SNN action head with LIF neurons produces spike-based actions; R-STDP supports both hybrid (backprop + STDP) and pure online STDP modes for test-time adaptation with zero backpropagation.

**RLActionToken** — A novel architecture that compresses VLA hidden states through an information bottleneck and applies off-policy Actor-Critic RL. The RL gradient update phase operates on a highly lightweight parameter set, making online fine-tuning practical.

**Continual Learning** — Experience-replay CL for sequential task acquisition. LoRA integration keeps ~6% trainable params (~3× memory savings). Cross-architecture: the same CL algorithm drops directly onto different VLA frameworks.

**World Model Integration** — Native support for 4 backbones:

| Backbone | Params | Mode |
|:---------|:-------|:-----|
| V-JEPA 2.1 | ~1.8B | `world_model_vjepa` |
| Cosmos Predict 2.5 | ~2.1B | `world_model_cosmos` |
| Cosmos Predict 2 | ~2.1B | `world_model_cosmos2` |
| Wan 2.2 | ~5B | `world_model_wan` |

### Benchmarks

| Benchmark | Focus |
|:----------|:------|
| **LIBERO** | Spatial / Object / Goal / Long-horizon (4 task suites) |
| **LIBERO-plus** | Zero-shot robustness: camera shift, robot swap, lighting, language variation |
| **RoboCasa** | Tabletop and kitchen manipulation, real-world scene diversity |
| **RoboCasa365** | 365-day large-scale kitchen task collection |

---

## 🖥 Demo

The Nvex demo is now a 7-page interactive experience with a working Milestone 2 backend path behind it.

The UI covers: Project Hub → Project Overview → Failure Map → Patch Plan → Iteration Runner → Improvement Report → Platform Memory.

### Run The Full Demo Locally

Start the backend first:

```bash
./.venv/bin/python -m uvicorn nvex_server.app:app --reload --port 8000
```

Then start the React app:

```bash
cd demo
npm install
npm run dev   # http://127.0.0.1:5173
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`, so the React pages consume live demo-state endpoints rather than reading only static mock data.

### Backend Endpoints

The local backend currently exposes:

- `GET /health`
- `GET /api/demo/state`
- `POST /api/eval/import`
- `POST /api/plan/generate`
- `POST /api/iteration/start`
- `GET /api/iteration/{id}/status`
- `GET /api/report/{iteration_id}`

### Standalone Demo

If you only want the static investor narrative, you can still open the standalone HTML directly:

```bash
open demo/nvex-demo.html
```

That standalone file is still useful for quick walkthroughs, but the React app is now the primary M2 demo surface.

---

## 🚀 Quick Start

### AlphaBrain Setup

```bash
conda create -n alphabrain python=3.10 -y && conda activate alphabrain
pip install -r requirements.txt && pip install -e .
pip install flash-attn --no-build-isolation
cp .env.example .env   # fill in model and data paths
```

Key `.env` variables:

```bash
PRETRAINED_MODELS_DIR=/path/to/pretrained_models   # Qwen2.5-VL / Qwen3-VL weights
LEROBOT_LIBERO_DATA_DIR=/path/to/lerobot/libero    # LeRobot-format training data
LIBERO_DATA_ROOT=/path/to/IPEC-COMMUNITY            # RLDS-format eval data
LIBERO_HOME=/path/to/LIBERO                         # LIBERO simulation env
LIBERO_PYTHON=/path/to/envs/libero/bin/python       # Separate eval conda env
```

Evaluation requires a **separate conda env** — see [Installation Guide](https://alphabraingroup.github.io/AlphaBrain/).

### Training

```bash
# Unified entry point
bash scripts/run_finetune.sh <mode>
# e.g.:
bash scripts/run_finetune.sh qwen_oft_goal
bash scripts/run_finetune.sh paligemma_oft_all_150k

# NeuroVLA / STDP
bash scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh
bash scripts/run_brain_inspired_scripts/run_stdp_finetune.sh --pretrained <ckpt>

# Continual Learning
bash scripts/run_continual_learning_scripts/run_cl_train.sh

# World Model
MODEL=cos2 bash scripts/run_world_model/train/run_world_model.sh

# RL
bash scripts/run_rl_scripts/run_action_token_5traj_alltasks.sh
```

### Verify Install

```bash
python -c "import AlphaBrain; print('ok')"
```

### Milestone 2 Backend Notes

Milestone 2 is now implemented in the repo:

- `nvex_server/` provides schema contracts, artifact import, patch-plan generation, iteration dispatch, status polling, improvement reports, and a seeded demo state.
- `benchmarks/LIBERO/eval/eval_libero.py` now writes a structured `eval_results.json` artifact that can be imported as an `EvalRun`.
- The current demo uses seeded before/after LIBERO Kitchen artifacts (`62% -> 74%`) to drive the failure map, patch plan, runner, report, and platform memory pages through the backend.

Full documentation: **[alphabraingroup.github.io/AlphaBrain](https://alphabraingroup.github.io/AlphaBrain/)**

---

## 🤝 Community

We welcome contributions — new VLA frameworks, benchmark adapters, bug fixes, and improvements that push benchmark performance further. Outstanding contributors may be invited to join as core members.

| Channel | Link |
|:--------|:-----|
| GitHub Issues | [Report bugs & request features](https://github.com/AlphaBrainGroup/AlphaBrain/issues) |
| HuggingFace | [Models](https://huggingface.co/AlphaBrainGroup) |
| WeChat Group | [Scan to join](assets/wechat.jpg) |

### Acknowledgments

AlphaBrain is forked from [starVLA](https://github.com/starVLA/starVLA) and builds on a rich open-source ecosystem. We are grateful to the authors of:

[starVLA](https://github.com/starVLA/starVLA) · [OpenVLA](https://github.com/openvla/openvla) · [openvla-oft](https://github.com/moojink/openvla-oft) · [openpi](https://github.com/Physical-Intelligence/openpi) · [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) · [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) · [Cosmos 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) · [Wan 2.2](https://github.com/Wan-Video/Wan2.2) · [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [RoboCasa](https://github.com/robocasa/robocasa) · [NeuroVLA](https://github.com/guoweiyu/NeuroVLA)

---

## 📝 Citation

```bibtex
@article{AgenticAIEval2026,
  title   = {Evaluating the Autonomous Mind, A Multi-Dimensional Framework for Agentic AI Readiness},
  author  = {Fei Wang, Salon Ren and Eric Wang},
  year    = {2026},
  url     = {https://github.com/Alchedata/agentic-ai-evaluation},
  license = {MIT}
}
```

```bibtex
@article{RLEnvEval2026,
  title   = {The Environment Layer: Building Infrastructure for Agentic AI Training},
  author  = {Fei Wang, Salon Ren, Eric Wang and Michael Zhang},
  year    = {2026},
  url     = {https://github.com/Alchedata/rl-env-white-papern},
  license = {MIT}
}
```

```bibtex
@software{AlphaBrain2026,
  title   = {AlphaBrain: A Modular Open-Source Framework for Embodied Intelligence Research},
  author  = {AlphaBrain Community},
  year    = {2026},
  url     = {https://github.com/AlphaBrainGroup/AlphaBrain},
  license = {MIT}
}
```




---

## 📄 License

[MIT License](LICENSE)

<div align="center">
<sub>Nvex orchestrates. AlphaBrain executes. Together they close the loop.</sub>
</div>
