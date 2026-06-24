from .target_head import TargetHead

__all__ = ["TargetHead"]

try:
    from .eagle3_target_model import (
        CustomEagle3TargetModel,
        Eagle3TargetModel,
        HFEagle3TargetModel,
        SGLangEagle3TargetModel,
        get_eagle3_target_model,
    )

    __all__.extend(
        [
            "Eagle3TargetModel",
            "SGLangEagle3TargetModel",
            "HFEagle3TargetModel",
            "CustomEagle3TargetModel",
            "get_eagle3_target_model",
        ]
    )
except ModuleNotFoundError:
    pass
