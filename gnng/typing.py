from typing import Any, Union, List, Dict, Tuple, Sequence, Type, Optional
from functools import partial
from pathlib import Path
from numpy import ndarray
from torch import Tensor, BoolTensor, LongTensor
from torch.nn import Module, ModuleList
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # PyTorch < 2.0
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from torchmetrics import Metric, MetricCollection
from torch_geometric.typing import OptTensor
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.transforms import BaseTransform
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from gnng.nn.losses.base import GnnGLoss

ParsableLoss = Union[str, partial, _Loss, Type[_Loss], GnnGLoss, Type[GnnGLoss]]
ParsableOptimizer = Union[str, partial, Optimizer, Type[Optimizer], OptimizerCallable]
ParsableScheduler = Union[str, partial, LRScheduler, Type[LRScheduler], LRSchedulerCallable]
ParsableMetric = Union[str, partial, Metric, Type[Metric], MetricCollection, Type[MetricCollection]]
SplitRatio = Union[Sequence[int], Sequence[float], ndarray, Tensor]
PyGData = Union[Data, HeteroData, Batch]

ParsableModel = Union[str, partial, Module, Type[Module]]
ParsableTransform = Union[str, partial, BaseTransform, Type[BaseTransform]]

ParsableMultiModel = Union[  # homogeneous Sequence of parsable type
    Sequence[str], Sequence[partial], Sequence[Module], Sequence[Type[Module]], Type[ModuleList]
]

ParsableCkpt = Union[Path, str, Dict[str, Any]]

PyReal = Union[int, float]
PathStr = Union[Path, str]

# ChoirOutputElm is the return tuple of each model participating in distillation. This return tuple contains 6 parts:
# loit4aim, y4aim, aim_mask, logit4distill, xs4distill, distill_mask, see `graph_brewer/nn/steps_also4distill.py`
FullChoirOutputElm = Tuple[
    OptTensor, OptTensor, Optional[BoolTensor], OptTensor, Optional[List[Tensor]], Optional[BoolTensor]
]

# loit4aim, y4aim, _       , logit4distill, xs4distill,       _     , see `graph_brewer/distill/gmd_module.py`
ChoirOutputElm = Tuple[OptTensor, OptTensor, OptTensor, Optional[List[Tensor]]]

# ChoirOutput is the returned tuples of multiple models involved in distillation
ChoirOutput = Sequence[ChoirOutputElm]
ZippedChoirOutput = Tuple[Tuple[OptTensor], Tuple[OptTensor], Tuple[Tensor], Tuple[List[OptTensor]]]

DirectedReturn = Tuple[Tensor, None]
UndirectedReturn = Tuple[Tensor, Tensor]
