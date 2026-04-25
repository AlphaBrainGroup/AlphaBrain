# Nvex Physical AI Demo

A 7-page interactive web demo showcasing Nvex as the **self-improving Physical AI orchestration layer** — for investor presentations, customer discovery calls, and technical evaluations.

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

**React dev server:**
```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # production build → dist/
```

## Stack

- `nvex-demo.html` — self-contained single-file demo, no dependencies
- React 19 + Vite 8 app (in progress — see `src/`)
- Pure CSS (no UI framework) — dark `#07090f` theme, indigo-violet gradients
- SVG for Intelligence Loop diagram and radar chart
- All data mocked; real AlphaBrain artifacts wired in at Milestone 2

## Story

Demo follows the LIBERO Kitchen Pick-and-Place scenario:

> `NeuroVLA-LIBERO-ckpt_v0.7` at **62% success** → Nvex diagnoses failure clusters (occlusion 38%, recovery 24%) → generates targeted patch plan → AlphaBrain CL update → `ckpt_v0.8` at **74% (+12%)** → recipe saved to Platform Memory.

For the self-improvement agent story (autonomous multi-loop), see [`../SELF_IMPROVEMENT_AGENT.md`](../SELF_IMPROVEMENT_AGENT.md).
