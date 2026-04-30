# API Reference

This section is auto-generated from the source tree under `AlphaBrain/` — class names, function signatures, docstrings, and the source itself are kept in sync with the codebase.

Pages are rendered by the [mkdocstrings](https://mkdocstrings.github.io/) Python handler, which uses [griffe](https://mkdocstrings.github.io/griffe/) for static analysis — no source code is executed and runtime import errors are avoided.

## Section layout

The reference is organised to mirror the top-level modules of the `AlphaBrain/` source tree:

| Section | Source path | Contents |
| --- | --- | --- |
| [Dataloader](./dataloader.md) | `AlphaBrain/dataloader/` | LeRobot / PaliGemma / VLM / Cosmos / GR00T / Qwen-VL LLaVA-JSON dataloaders and statistics utilities |
| [Model › Framework](./model_framework.md) | `AlphaBrain/model/framework/` | VLA frameworks (PaliGemma Pi0/Pi0.5, QwenOFT, NeuroVLA, ACT, CosmosPolicy, …) and the framework registry |
| [Model › Modules](./model_modules.md) | `AlphaBrain/model/modules/` | Building-block components: action heads, VLM backbones, world-model encoders, projectors, DINO |
| [Training › General](./training.md) | `AlphaBrain/training/` | `train_alphabrain*.py` entrypoints, `trainer_utils/`, and shared training scripts |
| [Training › Continual Learning](./training_continual_learning.md) | `AlphaBrain/training/continual_learning/` | Continual learning algorithms, replay buffer, and task sequences |
| [Training › Reinforcement Learning](./training_reinforcement_learning.md) | `AlphaBrain/training/reinforcement_learning/` | RLActionToken online RL training, environments, rollout, and algorithm implementations |

## How to read each page

Every page uses the `:::` directive so that mkdocstrings expands submodules automatically:

- Click the **Source** button to view the code inline.
- Signatures, inheritance, type annotations, and docstrings are rendered in Google style.
- Members whose names begin with `_` are filtered out by default.

To adjust how a specific symbol is displayed on a page (for example, hide the source, restrict members, or change the heading level), override the global defaults with an `options:` block, e.g.:

```markdown
::: AlphaBrain.model.framework.PaliGemmaPi
    options:
      show_source: false
      members:
        - PaliGemmaPi
```
