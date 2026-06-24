# from .auto import AutoDistributedTargetModel, AutoDraftModelConfig, AutoEagle3DraftModel
__all__ = []

try:
    from .auto import AutoDraftModelConfig, AutoEagle3DraftModel

    __all__.extend(["AutoDraftModelConfig", "AutoEagle3DraftModel"])
except ModuleNotFoundError:
    pass

try:
    from .draft.llama3_eagle import LlamaForCausalLMEagle3

    __all__.append("LlamaForCausalLMEagle3")
except ModuleNotFoundError:
    pass

try:
    from .target.eagle3_target_model import (
        CustomEagle3TargetModel,
        HFEagle3TargetModel,
        SGLangEagle3TargetModel,
        get_eagle3_target_model,
    )

    __all__.extend(
        [
            "SGLangEagle3TargetModel",
            "HFEagle3TargetModel",
            "CustomEagle3TargetModel",
            "get_eagle3_target_model",
        ]
    )
except ModuleNotFoundError:
    pass
