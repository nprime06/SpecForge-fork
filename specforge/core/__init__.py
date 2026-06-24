from .dflash import OnlineDFlashModel

__all__ = ["OnlineDFlashModel"]

try:
    from .domino import OnlineDominoModel

    __all__.append("OnlineDominoModel")
except ModuleNotFoundError:
    pass

try:
    from .eagle3 import OnlineEagle3Model, QwenVLOnlineEagle3Model

    __all__.extend(["OnlineEagle3Model", "QwenVLOnlineEagle3Model"])
except ModuleNotFoundError:
    pass
