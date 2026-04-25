# Nvex Self-Improving Agent — Design & Demo Brainstorm

**Last Updated:** April 25, 2026

---

## Core Idea

The self-improvement agent is the clearest proof that Nvex is not a dashboard. It takes a failing policy and — without human intervention at each step — runs the full diagnosis → plan → training → verification loop until the policy meets a target KPI or hits a stopping condition.

This is the "aha moment" for both investors and customers:

> You upload a checkpoint. You set a target (e.g., 75% success on LIBERO Kitchen). The agent runs. You come back and the policy is better — and it can tell you exactly why, what it did, and what it learned.

---

## What "Self-Improving" Actually Means

The agent doesn't retrain from scratch. It does **targeted, incremental improvement**:

1. **Identify** — Run eval, cluster failures, rank root causes by impact
2. **Decide** — Choose the highest-leverage intervention (CL patch, SFT, data augmentation, env verification)
3. **Execute** — Dispatch the job to AlphaBrain
4. **Verify** — Re-run eval on the new checkpoint
5. **Store** — Save the recipe, the failure pattern, and the improvement delta to Platform Memory
6. **Loop** — If target not met and improvement is positive, go again

The agent terminates when:
- Target KPI is reached
- Max iterations exceeded
- Improvement delta falls below a threshold (diminishing returns)
- A blocking failure is detected (e.g., data source unavailable)

---

## Demo Modes

### Mode A — Precomputed Replay (Milestone 1 / Investor Demo)
The fastest, most reliable demo. All results are pre-generated from a real AlphaBrain run.

**Flow:**
1. User clicks "Run Self-Improvement" on the Iteration Runner page
2. The UI animates through each agent step with realistic timing (2–5 seconds per step)
3. Failure Map updates: cluster sizes shrink as the agent identifies and patches them
4. Improvement Report appears: 62% → 74%, with failure cluster diffs
5. Platform Memory gains a new recipe

**Why it works:** The underlying AlphaBrain results are real. The "live" execution is a replay. This is fully stable for investor demos and customer presentations.

**Implementation:** `demo/nvex-demo.html` already supports this mode — just add the animation trigger and step-by-step state updates.

---

### Mode B — Live Agent with Real AlphaBrain (Milestone 3)
For customer POCs and hands-on demos where the policy is the customer's own checkpoint.

**Flow:**
1. Customer provides checkpoint + eval results (or runs eval inside Nvex)
2. `SelfImprovementAgent` starts: calls `diagnose_failures()`, shows intermediate reasoning
3. Agent proposes patch plan — customer can approve or override before execution
4. AlphaBrain CL job runs (typically 30–60 min for a small LIBERO patch)
5. Re-eval runs automatically
6. Results appear in Improvement Report

**Why it works for customers:** They see their own data moving. The improvement is real. The patch recipe is saved for their next project.

**Caution:** Don't run this live at an investor meeting — too much depends on training stability. Use Mode A for investor demos, Mode B for customer POCs.

---

### Mode C — Multi-Iteration Compound View (Milestone 3+)
Shows the compounding effect across multiple loops — the "platform moat" visual.

**Flow:**
- Loop 1: 62% → 74% (occlusion patch)
- Loop 2: 74% → 81% (recovery trajectory patch)
- Loop 3: 81% → 85% (lighting variation patch, smaller gain — agent detects diminishing returns and stops)

**What it proves:**
- Each loop uses a recipe from Platform Memory — getting faster over time
- The agent knows when to stop (stopping criteria)
- The platform gets smarter across projects, not just within one

---

## Agent Architecture

### Core Components

```
SelfImprovementAgent
├── EvalRunner          — triggers AlphaBrain eval, returns EvalRun artifact
├── FailureDiagnoser    — clusters failures, ranks root causes, returns FailureDiagnosis
├── PatchPlanner        — maps diagnosis to PatchPlan (rule-based v1, LLM-enhanced v2)
├── JobDispatcher       — submits IterationJob to AlphaBrain, polls status
├── Comparator          — diffs before/after EvalRun, produces ImprovementReport
├── MemoryWriter        — saves reusable assets to Platform Memory
└── LoopController      — manages iteration state, stopping criteria, convergence check
```

### Agent Tools (function-calling interface)

| Tool | Input | Output |
|------|-------|--------|
| `run_eval` | checkpoint_path, benchmark_suite | EvalRun |
| `diagnose_failures` | EvalRun | FailureDiagnosis |
| `generate_patch_plan` | FailureDiagnosis, memory_context | PatchPlan |
| `dispatch_training` | PatchPlan, execution_backend | IterationJob |
| `poll_job_status` | job_id | JobStatus |
| `compare_checkpoints` | EvalRun (before), EvalRun (after) | ImprovementReport |
| `save_to_memory` | ImprovementReport, PatchPlan | ReusableAsset |
| `check_stopping` | ImprovementReport, loop_state | ShouldStop (bool + reason) |

### PatchPlanner — Strategy Selection Rules (v1)

| Failure Pattern | Recommended Strategy | AlphaBrain Backend |
|----------------|---------------------|-------------------|
| Occlusion / object visibility | CL patch with occluded scene episodes | `alphabrain_cl` |
| Recovery / error correction | Fine-tune on teleop correction trajectories | `alphabrain_finetune` |
| Language variation | VLM co-training with augmented instructions | `alphabrain_vlm_cotrain` |
| Lighting / appearance shift | CL patch with lighting-augmented data | `alphabrain_cl` |
| Long-horizon failure (step N) | World model rollout verification + targeted re-train | `alphabrain_world_model` |
| Generalization across robots | Cross-architecture CL | `alphabrain_cl` (cross-arch) |

### Memory Context in Planning

Platform Memory makes each loop smarter:
- If a similar failure pattern was patched before, the agent reuses that recipe
- Recipe confidence score: how many times it was applied successfully
- The agent favors high-confidence recipes and experiments on low-confidence ones

---

## What to Show in the Demo UI

### Iteration Runner Page Additions
- **"Auto-Improve" toggle** — switches from manual to autonomous mode
- **Agent Reasoning Panel** — shows the agent's step-by-step decisions in plain language:
  - *"Failure clusters detected: occlusion (38%), recovery (24%). Targeting occlusion first — highest impact."*
  - *"Found matching recipe in Platform Memory: occlusion_recovery_v1 (used 3 times, avg +9% uplift). Applying."*
  - *"Training job dispatched to AlphaBrain CL. Estimated time: 45 min."*
- **Loop Progress Bar** — shows current iteration (1/3), current success rate, target
- **Stop / Override button** — lets the user intervene, inspect the plan, and resume

### Improvement Report Page Additions
- **Multi-loop comparison chart** — success rate over iterations
- **"Why did it stop?"** — agent explains the stopping reason
- **Asset trail** — which Platform Memory assets were used and which new ones were created

---

## Technical Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Training run takes too long for live demo | Use Mode A (precomputed replay) for demos; Mode B only for async customer POCs |
| Agent makes a bad plan (wrong strategy) | v1 is rule-based, not fully autonomous — customer can review and approve before execution |
| Improvement is negative on first loop | Build in retry logic; if delta < 0, agent tries a different strategy from the strategy library |
| AlphaBrain job fails mid-run | Job dispatcher catches failures, saves partial state, allows resume |
| Customers don't trust "autonomous" decisions | Make agent reasoning fully transparent — every decision is logged and explainable |

---

## Roadmap

| Phase | What | When |
|-------|------|------|
| **M1** | Precomputed replay animation in demo HTML | Now |
| **M2** | Real eval artifact ingestion + rule-based PatchPlanner + JobDispatcher | Next sprint |
| **M3A** | `SelfImprovementAgent` skeleton with tool registry | ~4 weeks |
| **M3B** | Agent reasoning panel in demo UI | ~5 weeks |
| **M3C** | Multi-iteration compound view | ~6 weeks |
| **M3D** | LLM-enhanced PatchPlanner (natural language reasoning) | ~8 weeks |
| **M4** | Customer-uploadable checkpoints, async POC mode | ~12 weeks |

---

## Key Message for Demos

> "Most tools tell you what happened. Nvex tells you what to do — and then does it."

The self-improvement agent is proof that Nvex is an intelligence layer, not a dashboard. Every loop makes the platform smarter. Every project compounds the knowledge. That's the moat.
