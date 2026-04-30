"""Dynamic-architecture continual learning algorithms.

These methods adapt the model itself across tasks — either by adding new
parameters (per-task adapters, expert heads) or by reallocating capacity
within a fixed parameter budget (magnitude-based masks).

No implementations yet — this package is a placeholder for upcoming work:

- **DWE** — Dynamic Weight Expansion: spin up a fresh LoRA adapter per
  task and route forwards accordingly.
- **Weight Merge** — load-time linear merging of previous-task adapters
  into the base model (per continual-vla-rl / Wortsman et al.).
- **PackNet** — magnitude-based pruning + per-task weight ownership masks.
"""

__all__: list = []
