"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., M1.py, QwenFast.py) should register itself:
    from AlphaBrain.model.framework.framework_registry import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("InternVLA-M1")
    def build_model_framework(config):
        return InternVLA_M1(config=config)
"""

import pkgutil
import importlib
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY

from AlphaBrain.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration
if pkg_path is not None:
    try:
        for _, module_name, _ in pkgutil.iter_modules(pkg_path):
            importlib.import_module(f"{__name__}.{module_name}")
    except Exception as e:
        logger.info(f"Warning: Failed to auto-import framework submodules: {e}")
        
def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "InternVLA-M1")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"): 
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml
        
    if cfg.framework.name == "ToyVLA":
        from AlphaBrain.model.framework.ToyModel import ToyVLA
        return ToyVLA(config=cfg)
    elif cfg.framework.name == "QwenOFT":
        from AlphaBrain.model.framework.QwenOFT import Qwenvl_OFT
        return Qwenvl_OFT(cfg)
    elif cfg.framework.name == "QwenFast":
        from AlphaBrain.model.framework.QwenFast import Qwenvl_Fast
        return Qwenvl_Fast(cfg)
    elif cfg.framework.name == "NeuroVLA":
        from AlphaBrain.model.framework.NeuroVLA import NeuroVLA
        return NeuroVLA(cfg)
    elif cfg.framework.name == "QwenGR00T":
        from AlphaBrain.model.framework.QwenGR00T import Qwen_GR00T
        return Qwen_GR00T(cfg)
    elif cfg.framework.name == "ACT":
        from AlphaBrain.model.framework.ACT import ACTModel
        return ACTModel(config=cfg)
    elif cfg.framework.name == "CosmosPolicy":
        from AlphaBrain.model.framework.CosmosPolicy import CosmosPolicy
        return CosmosPolicy(config=cfg)
    elif cfg.framework.name == "PaliGemmaOFT":
        from AlphaBrain.model.framework.PaliGemmaOFT import PaliGemma_OFT
        return PaliGemma_OFT(cfg)
    elif cfg.framework.name in ("PaliGemmaPi05", "LlamaPi05"):
        # Pi0.5 framework. The registry name only picks defaults (e.g.
        # gripper_remap defaults to true under "PaliGemmaPi05").
        from AlphaBrain.model.framework.PaliGemmaPi import PaliGemmaPi
        return PaliGemmaPi(cfg)
    elif cfg.framework.name == "LlamaOFT":
        from AlphaBrain.model.framework.LlamaOFT import Llama_OFT
        return Llama_OFT(cfg)

    # auto detect from registry
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.name} is not implemented. Plz, python yourframework_py to specify framework module.")
    
    MODEL_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODEL_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
