from typing import Optional
from torch import Tensor
from torch.nn.modules.loss import _Loss


class GnnGLoss(_Loss):
    def reduce_output(self, output: Tensor) -> Tensor:
        if self.reduction == "mean":
            res = output.mean()
        elif self.reduction == "sum":
            res = output.sum()
        elif self.reduction == "none":
            res = output
        else:
            raise ValueError
        return res


class GnnGWeightedLoss(GnnGLoss):
    """
    from torch/nn/modules/loss.py/_WeightedLoss
    """

    def __init__(
        self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = "mean"
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.weight: Optional[Tensor]
