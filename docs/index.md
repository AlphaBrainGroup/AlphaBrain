# AlphaBrain Documentation

**AlphaBrain** is the world’s first all-in-one, open-source community for embodied intelligence, built to be ready out of the box. We unifies multiple VLA architectures, world model backbones, biologically-inspired learning algorithms, and reinforcement learning paradigms under a single, extensible framework. AlphaBrain brings embodied AI within everyone’s reach.

<p align="center">
  <img src="https://raw.githubusercontent.com/AlphaBrainGroup/AlphaBrain/main/assets/main_fig.png" width="95%" alt="AlphaBrain Architecture Overview"/>
</p>

---

## Where to go next

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Installation](quickstart/installation.md)**

    Set up conda envs, Flash Attention, pretrained weights, dataset, and `.env` — everything you need before the first run.

-   :material-flash: **[Quickstart](quickstart/index.md)**

    Start with Baseline VLA for the default finetune + eval, then pick a capability: NeuroVLA, ActionToken, World Model, or Continual Learning.

-   :material-api: **[API Reference](api/index.md)**

    API reference for every public class and function in `AlphaBrain/`.

</div>

---

## Key capabilities

| Capability | Summary | Quickstart |
|:-----------|:--------|:-----------|
| **Baseline VLA** | PaliGemmaOFT / PaliGemmaPi05 / LlamaOFT finetune + LIBERO eval. | [Baseline VLA](quickstart/baselineVLA.md) |
| **NeuroVLA** | Brain-inspired VLA with Spiking Neural Network action head and R-STDP learning. | [NeuroVLA](quickstart/neurovla.md) |
| **ActionToken** | Off-policy TD3 online RL fine-tuning with an information-bottleneck encoder. | [ActionToken](quickstart/action_token.md) |
| **World Model** | V-JEPA / Cosmos / Wan backbones with pluggable GR00T / OFT / PI decoders. | [World Model](quickstart/world_model.md) |
| **Continual Learning** | Experience-replay fine-tuning across 10 LIBERO tasks (LoRA or full-param). | [Continual Learning](quickstart/continual_learning.md) |

---

## Project links

- Source: [github.com/AlphaBrainGroup/AlphaBrain](https://github.com/AlphaBrainGroup/AlphaBrain)
- Issues: [Report bugs or request features](https://github.com/AlphaBrainGroup/AlphaBrain/issues)
- License: [MIT](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/LICENSE)
