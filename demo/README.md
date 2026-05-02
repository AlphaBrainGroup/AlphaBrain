# Nvex Physical AI Demo

A 7-page interactive web demo showcasing Nvex as the **self-improving Physical AI orchestration layer** — for investor presentations, customer discovery calls, and technical evaluations.

The React demo is now backed by the Milestone 2 Nvex server path. Failure Map, Patch Plan, Iteration Runner, Improvement Report, and Platform Memory all consume live API responses from `nvex_server.app`, seeded with a LIBERO Kitchen before/after improvement case.

## Audience Modes

| Audience | Focus | Key Pages |
|----------|-------|-----------|
| **Investors** | Platform moat, compound value, orchestration layer vs. training framework | Home → Platform Memory |
| **Potential customers** | "What happens to my failing policy?" | Overview → Failure Map → Patch Plan → Improvement Report |
| **Technical evaluators** | AlphaBrain execution depth, benchmark results, agent reasoning | Failure Map → Iteration Runner |

## Pages

| Route | Page | What it proves |
|-------|------|----------------|
| Home | Project Hub — intelligence loop diagram, platform metrics, project list | Nvex is a platform, not a one-off tool |
| Overview | Project Overview — KPI cards, task breakdown, loop position, next action | Eval is the starting point, not the end |
| Failure Map | Interactive failure clusters, radar chart, root-cause diagnosis | Nvex knows *why* the policy failed |
| Patch Plan | Data targeting, training strategy, verification, expected uplift | Nvex decides *what to do next* |
| Iteration Runner | Animated timeline, live console, artifact tracker | AlphaBrain executes; Nvex orchestrates |
| Improvement Report | Before/after metrics, assets created, next iteration suggestion | Measurable, verifiable improvement |
| Platform Memory | Recipes, pipeline templates, failure ontology, compounding chart | Every loop makes the platform smarter |

## Quick Start

**Standalone (no build needed):**
```bash
open nvex-demo.html   # or just double-click it
```

**React dev server with local backend:**
```bash
cd ..
./.venv/bin/python -m uvicorn nvex_server.app:app --reload --port 8000

cd demo
npm install
npm run dev       # http://127.0.0.1:5173
npm run build     # production build → dist/
```

Vite proxies `/api` and `/health` to `http://127.0.0.1:8000`, so no extra frontend env var is required for local development.

## API-Backed Demo Flow

The current local demo uses these backend endpoints:

- `GET /api/demo/state` — returns the seeded full dashboard state used by the React app
- `POST /api/eval/import` — imports benchmark artifacts into `EvalRun`
- `POST /api/plan/generate` — produces a rule-based `PatchPlan`
- `POST /api/iteration/start` — creates an `IterationJob`
- `GET /api/iteration/{id}/status` — polls file-backed job state
- `GET /api/report/{iteration_id}` — returns the resulting `ImprovementReport`

The seeded scenario uses `nvex_server/examples/libero_kitchen_before_eval.json` and `nvex_server/examples/libero_kitchen_after_eval.json` to demonstrate a real structured loop from `62%` to `74%` success.

## Stack

- `nvex-demo.html` — self-contained single-file demo, no dependencies
- React 19 + Vite 8 app in `src/`
- Pure CSS (no UI framework) — dark `#07090f` theme, indigo-violet gradients
- SVG for Intelligence Loop diagram and radar chart
- `src/data/NvexRuntimeContext.jsx` adapts backend demo state into the dashboard data shape
- `nvex_server/` provides the M2 backend orchestration path
- Real artifact import is supported for LIBERO `eval_results.json`, RoboCasa365 `aggregate_stats.json`, RoboCasa tabletop stats JSON, generic JSON, and LIBERO logs

## Story

Demo follows the LIBERO Kitchen Pick-and-Place scenario:

> `NeuroVLA-LIBERO-ckpt_v0.7` at **62% success** → Nvex diagnoses failure clusters (occlusion 38%, recovery 24%) → generates targeted patch plan → AlphaBrain CL update → `ckpt_v0.8` at **74% (+12%)** → recipe saved to Platform Memory.

For local M2 development, this story is served through the backend, not hardcoded page-by-page in the React app.

For the self-improvement agent story (autonomous multi-loop), see [`../SELF_IMPROVEMENT_AGENT.md`](../SELF_IMPROVEMENT_AGENT.md).
