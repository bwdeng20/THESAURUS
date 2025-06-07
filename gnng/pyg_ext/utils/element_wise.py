from torch_sparse import SparseTensor
import torch
from torch import Tensor
from typing import Union


def power_ew(input: SparseTensor, exponent: Union[float, Tensor]):
    value = input.storage.value()
    if value is None:  # None edge_value means all values are ones
        out = input
    else:
        out = input.set_value(torch.pow(value, exponent), layout='coo')
    return out
