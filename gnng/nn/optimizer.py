"""From lightning.pytorch.core.lr_scheduler."""

from typing import Dict, Any, Optional, Callable
import torch
from torch.optim import Optimizer


class MockOptimizer(Optimizer):
    """The `MockOptimizer` will be used inplace of an optimizer in the event that `None` is returned from
    `configure_optimizers`."""

    def __init__(self) -> None:
        super().__init__([torch.zeros(1)], {})

    def add_param_group(self, param_group: Dict[Any, Any]) -> None:
        pass  # Do Nothing

    def load_state_dict(self, state_dict: Dict[Any, Any]) -> None:
        pass  # Do Nothing

    def state_dict(self) -> Dict[str, Any]:
        return {}  # Return Empty

    def step(self, closure: Optional[Callable] = None) -> None:
        if closure is not None:
            closure()

    def zero_grad(self, set_to_none: Optional[bool] = False) -> None:
        pass  # Do Nothing

    def __repr__(self) -> str:
        return "No Optimizer"
