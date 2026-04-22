<div align="center">

# AlphaBrain

### A Modular Open-Source Framework for Embodied Intelligence Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/Docs-Online-green.svg)](https://alphabraingroup.github.io/AlphaBrain/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-orange.svg)](https://huggingface.co/AlphaBrainGroup)
[![WeChat](https://img.shields.io/badge/WeChat-Group-07C160.svg)](assets/wechat.jpg)

<p align="center">
  <img src="assets/main_fig.png" width="100%" alt="AlphaBrain Architecture Overview"/>
</p>

**AlphaBrain** is an all-in-one, open-source community for embodied intelligence, built to be ready out of the box. We unifies multiple VLA architectures, world model backbones, biologically-inspired learning algorithms, and reinforcement learning paradigms under a single, extensible framework. AlphaBrain brings embodied AI within everyone’s reach.

[Quick Start & Documentation](#-quick-start--documentation) · [Key Features](#-key-features) · [Community](#-community) · [Citation](#-citation)

</div>

---

## Highlights

<table>
<tr>
<td width="50" align="center">🧠</td>
<td><b>Brain-Inspired VLA (NeuroVLA)</b> — The first open-source biologically-inspired VLA model, achieving <b>SOTA on brain-inspired control tasks</b>. Integrates spiking neural networks (SNN) with STDP learning rules, advancing embodied intelligence toward biological brain learning mechanisms.</td>
</tr>
<tr>
<td width="50" align="center">🔄</td>
<td><b>Cross-Architecture Continual Learning</b> — The first open-source continual learning algorithm designed for cross-architecture VLA, breaking architecture compatibility bottlenecks and supporting universal adaptation and knowledge accumulation across different VLA models.</td>
</tr>
<tr>
<td width="50" align="center">🎯</td>
<td><b>RLActionToken Training Paradigm</b> — The first open-source VLA training architecture based on <b>RL Token</b>, a novel architecture that compresses VLA hidden states through an information bottleneck, followed by off-policy Actor-Critic reinforcement learning.</td>

</tr>
<tr>
<td width="50" align="center">🌍</td>
<td><b>Native World Model Integration</b> — The first open-source VLA to natively integrate <b>Cosmos Policy</b> original weights, supporting flexible world model switching across Cosmos 2 / 2.5, Wan 2.2, and V-JEPA 2.1.</td>
</tr>
<tr>
<td width="50" align="center">📊</td>
<td><b>Comprehensive Benchmark Suite</b> — Full adaptation to the latest embodied benchmarks with open-source support for <b>long-horizon task execution and memory</b>: LIBERO, LIBERO-plus, RoboCasa, RoboCasa365 and more to come.</td>
</tr>
</table>

---

## 🚀 Quick Start & Documentation

Full setup, training, evaluation, and deployment instructions live in our documentation site. Step-by-step guides, configuration references, and troubleshooting notes are all maintained there.

👉 **[AlphaBrain Documentation →](https://alphabraingroup.github.io/AlphaBrain/)**

---

## 🔬 Key Features

AlphaBrain delivers five core capabilities on a single stack: the **VLA framework family** as the base, with **NeuroVLA / RLActionToken / Continual Learning / World Model** as composable capability modules. All capabilities share the same trainer, config system, and inference interface.

### VLA Frameworks

| Framework | Action Decoding | Typical Use |
|:----------|:----------------|:------------|
| **OFT** | MLP action head, parallel continuous decoding | Fast prototyping, baseline alignment |
| **GR00T** | System1 + Flow-Matching DiT System2 | High-precision manipulation, long-horizon planning |
| **PI** | Flow-Matching action prediction | Diffusion-style policies |
| **Adapter** | Lightweight Adapter decoding | Parameter-efficient fine-tuning |
| **NeuroVLA** | Bio-inspired spiking + STDP | Brain-inspired control |
| **CosmosPolicy** | Latent-space video diffusion | World-model-native policy |

### Brain-Inspired VLA (NeuroVLA + STDP)

NeuroVLA integrates spiking neural networks with biological learning rules into the VLA pipeline:

- **QFormer** extracts layer-wise action-relevant features from VLM hidden states;
- **SNN Action Head** with Leaky Integrate-and-Fire (LIF) neurons for spike-based action prediction;
- **R-STDP Training** — Reward-Modulated Spike-Timing-Dependent Plasticity, supporting both hybrid (backprop + STDP) and pure STDP modes;
- **Online STDP** — Test-time adaptation with zero backpropagation, using self-supervised reward signals from environment interaction.

### RLActionToken Online RL Fine-tuning

A novel architecture that compresses VLA hidden states through an information bottleneck, followed by off-policy Actor-Critic reinforcement learning:
- **Encoder-Decoder**: Extracts a compact action token from the VLA's internal features to serve as the state representation for RL.
- **Two-Phase Training**: An initial adaptation stage to expose the action token → RL fine-tuning with a frozen VLA.
- **Low Resource Requirements**: The actual reinforcement learning gradient update phase involves a highly lightweight parameters.

### Continual Learning

Experience-replay-based continual learning for sequential task acquisition:

- **Incremental design** — all changes are additive, no modification to base training code;
- **LoRA integration** — parameter-efficient fine-tuning (~6% trainable params, ~3× memory savings);
- **Replay buffer** with configurable per-task capacity;
- **Cross-architecture adaptation** — the same CL algorithm drops directly onto different VLA frameworks.

### World Model Integration

Native support for 4 world model backbones plus full CosmosPolicy finetuning:

| Backbone | Params | Mode Name | Text Encoder |
|:---------|:-------|:----------|:-------------|
| V-JEPA 2.1 | ~1.8B | `world_model_vjepa` | T5-small |
| Cosmos Predict 2.5 | ~2.1B | `world_model_cosmos` | Reason1-7B |
| Cosmos Predict 2 | ~2.1B | `world_model_cosmos2` | T5-XXL |
| Wan 2.2 | ~5B | `world_model_wan` | UMT5-XXL |

---

### Benchmarks

| Benchmark | Tasks | Highlights | Path |
|:----------|:------|:-----------|:-----|
| **LIBERO** | Spatial / Object / Goal / Long-horizon | Core evaluation suite, 4 task suites | `benchmarks/LIBERO/` |
| **LIBERO-plus** | Robustness (Camera, Robot, Language, Light, etc.) | Zero-shot generalization testing | `benchmarks/LIBERO-plus/` |
| **RoboCasa** | Tabletop & kitchen manipulation | Real-world scene diversity | `benchmarks/Robocasa_tabletop/` |
| **RoboCasa365** | 365-day kitchen task collection | Large-scale daily tasks | `benchmarks/Robocasa365/` |
| ... | | |

---

## 🤝 Community

We welcome contributions from the community — including new frameworks, benchmark adapters, bug fixes, and improvements that achieve stronger benchmark performance. Outstanding contributors may be invited to join the community as core members. Every contribution matters.

| Channel | Link |
|:--------|:-----|
| GitHub Issues | [Report bugs & request features](https://github.com/AlphaBrainGroup/AlphaBrain/issues) |
| HuggingFace | [Models](https://huggingface.co/AlphaBrainGroup) |
| WeChat Group | [Scan the QR code to join](assets/wechat.jpg) |

### Acknowledgments

AlphaBrain is mainly forked from [starVLA](https://github.com/starVLA/starVLA) and stands on the shoulders of an incredible open-source ecosystem. We are deeply grateful to the authors and maintainers of the following projects, whose code, models, datasets, and ideas directly enabled this work:

- [starVLA/starVLA](https://github.com/starVLA/starVLA)
- [openvla/openvla](https://github.com/openvla/openvla)
- [moojink/openvla-oft](https://github.com/moojink/openvla-oft)
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [robocasa/robocasa](https://github.com/robocasa/robocasa)
- [guoweiyu/NeuroVLA](https://github.com/guoweiyu/NeuroVLA)


---

## 📝 Citation

```bibtex
@software{AlphaBrain2026,
  title     = {AlphaBrain: a Modular Open-Source Framework for Embodied Intelligence Research},
  author    = {AlphaBrain Community},
  year      = {2026},
  url       = {https://github.com/AlphaBrainGroup/AlphaBrain},
  license   = {MIT},
  doi       = {}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

<div align="center">
<sub>Built with passion by the AlphaBrain Community upon starVLA</sub>
</div>
