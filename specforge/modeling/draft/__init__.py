from .base import Eagle3DraftModel
from .dflash import (
    DFlashDraftModel,
    build_target_layer_ids,
    extract_context_feature,
    sample,
)

__all__ = [
    "Eagle3DraftModel",
    "DFlashDraftModel",
    "build_target_layer_ids",
    "extract_context_feature",
    "sample",
]

try:
    from .llama3_eagle import LlamaForCausalLMEagle3

    __all__.append("LlamaForCausalLMEagle3")
except ModuleNotFoundError:
    pass
