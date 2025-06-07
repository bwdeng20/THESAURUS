from .base import GnnGLoss
from .ow_losses import (
    GCDLoss,
    SelfDistillLoss,
    SelfLabelLoss,
    SupConLoss,
    InfoNCELoss,
    EntropyRegLoss,
    UNOLoss,
    SemiSupConLoss,
    G2MSELoss,
    G2CrossEntropyLoss,
    G2MAELoss
)

__all__ = [
    "GnnGLoss",
    "GCDLoss",
    "SelfDistillLoss",
    "SelfLabelLoss",
    "SupConLoss",
    "SemiSupConLoss",
    "InfoNCELoss",
    "EntropyRegLoss",
    "UNOLoss",
    "G2MSELoss",
    "G2CrossEntropyLoss",
    "G2MAELoss"
]
