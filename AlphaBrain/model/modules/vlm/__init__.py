def get_vlm_model(config):

    # Detect VLM name from config (supports qwenvl, llamavl, paligemma backends)
    # Use framework.name to determine the primary VLM backend, avoiding false
    # matches when multiple VLM config blocks exist (e.g. default qwenvl + llamavl).
    vlm_name = None
    fw_name = getattr(config.framework, 'name', '')
    
    if fw_name in ('LlamaOFT', 'LlamaPi05') and hasattr(config.framework, 'llamavl'):
        vlm_name = config.framework.llamavl.base_vlm
    elif fw_name in ('PaliGemmaPi05', 'PaliGemmaOFT') and hasattr(config.framework, 'paligemma'):
        vlm_name = config.framework.paligemma.base_vlm
    elif hasattr(config.framework, 'qwenvl'):
        vlm_type = config.framework.qwenvl.get("vlm_type", None) if hasattr(config.framework.qwenvl, 'get') else None
        vlm_name = vlm_type if vlm_type else config.framework.qwenvl.base_vlm
    elif hasattr(config.framework, 'llamavl'):
        vlm_name = config.framework.llamavl.base_vlm
    elif hasattr(config.framework, 'paligemma'):
        vlm_name = config.framework.paligemma.base_vlm
    else:
        raise ValueError("No VLM config found in framework (expected qwenvl, llamavl, or paligemma)")

    if "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower(): # temp for some ckpt
        from .qwen2_5 import _QWen_VL_Interface
        return _QWen_VL_Interface(config)
    elif "Qwen3-VL" in vlm_name:
        from .qwen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)
    elif "Qwen3.5" in vlm_name:
        from .qwen3_5 import _QWen3_5_VL_Interface
        return _QWen3_5_VL_Interface(config)
    elif "florence" in vlm_name.lower(): # temp for some ckpt
        from .Florence2 import _Florence_Interface
        return _Florence_Interface(config)
    elif "world_model" in vlm_name.lower() or "vjepa" in vlm_name.lower() or "wan2" in vlm_name.lower() or "cosmos-predict" in vlm_name.lower():
        from AlphaBrain.model.modules.world_model import WorldModelVLMInterface
        return WorldModelVLMInterface(config)
    elif "cosmos-reason2" in vlm_name.lower():
        from .CosmosReason2 import _CosmosReason2_Interface
        return _CosmosReason2_Interface(config)
    elif 'Llama' in vlm_name or 'llama' in vlm_name:
        from .llama3_2 import _Llama_VL_Interface
        return _Llama_VL_Interface(config)
    elif 'paligemma' in vlm_name.lower() or 'PaliGemma' in vlm_name:
        from .paligemma import _PaliGemma_VL_Interface
        return _PaliGemma_VL_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")



