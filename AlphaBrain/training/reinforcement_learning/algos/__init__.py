"""RL algorithms.

Each subpackage implements one method on top of the shared `envs/` + `common/`
infrastructure. Currently:

- `RLT_a` — our off-policy TD3 variant with a frozen VLA and a bottleneck
  encoder-decoder. Inspired by the RL Token paper (Physical Intelligence) but
  diverges in several design choices (see `RLT_a/__init__.py`). The
  faithful paper implementation is still under test.

Future siblings (GRPO, PPO, …) drop in here without touching any other dir.
"""
from AlphaBrain.training.reinforcement_learning.algos import RLT_a

__all__ = ["RLT_a"]
