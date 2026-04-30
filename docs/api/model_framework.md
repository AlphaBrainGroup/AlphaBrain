# Model › Framework

Source path: `AlphaBrain/model/framework/`

Each framework is an independent VLA model implementation. Frameworks are registered via `FRAMEWORK_REGISTRY` in `AlphaBrain.model.tools` and constructed by the `build_framework(cfg)` factory based on `cfg.framework.name`.

---

## Factory and registry

::: AlphaBrain.model.framework
    options:
      heading_level: 3
      members:
        - build_framework
      show_submodules: false

::: AlphaBrain.model.tools
    options:
      heading_level: 3
      members:
        - FRAMEWORK_REGISTRY

---

## Base class and config utilities

::: AlphaBrain.model.framework.base_framework
    options:
      heading_level: 3

::: AlphaBrain.model.framework.config_utils
    options:
      heading_level: 3

---

## ToyVLA

::: AlphaBrain.model.framework.ToyModel
    options:
      heading_level: 3

---

## ACT

::: AlphaBrain.model.framework.ACT
    options:
      heading_level: 3

---

## CosmosPolicy

::: AlphaBrain.model.framework.CosmosPolicy
    options:
      heading_level: 3

---

## NeuroVLA

::: AlphaBrain.model.framework.NeuroVLA
    options:
      heading_level: 3

---

## PaliGemma family

::: AlphaBrain.model.framework.PaliGemmaOFT
    options:
      heading_level: 3

::: AlphaBrain.model.framework.PaliGemmaPi
    options:
      heading_level: 3

---

## Llama OFT

::: AlphaBrain.model.framework.LlamaOFT
    options:
      heading_level: 3

---

## Qwen family

::: AlphaBrain.model.framework.QwenOFT
    options:
      heading_level: 3

::: AlphaBrain.model.framework.QwenPI
    options:
      heading_level: 3

::: AlphaBrain.model.framework.QwenGR00T
    options:
      heading_level: 3

---

## World Model VLA

::: AlphaBrain.model.framework.WorldModelVLA
    options:
      heading_level: 3
