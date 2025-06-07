from typing import Optional, Union, Dict, Any
from torch_geometric.loader import NodeLoader
import logging
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch import BoolTensor, LongTensor
from lightning.fabric.wrappers import _FabricDataLoader
from lightning_utilities.core.apply_func import apply_to_collection
from gnng.utils import are_masks_disjoint, index_to_mask, to_mask, mask_to_index
import warnings
from torch_geometric.data import Data, HeteroData, Batch

PyGData = Union[Data, HeteroData, Batch]

py_logger = logging.getLogger(__name__)


def possible_split_names(split_name):
    if split_name in ("train", "test"):
        psn = [split_name]
    elif "val" in split_name:
        psn = ["val", "valid", "validation"]
    else:
        raise ValueError(f"{split_name} seems not a valid split markers")
    return psn


def infer_node_mask(
    data: PyGData, node_mask: Optional[Union[str, BoolTensor, LongTensor]] = None
) -> Optional[BoolTensor]:
    mask = None
    if node_mask is None:
        pass
        # mask = torch.ones(datasets.num_nodes, dtype=torch.bool, device=datasets.x.device)
    elif isinstance(node_mask, str):
        psn = possible_split_names(node_mask)
        for nm in psn:
            mask = data.get(nm + "_mask", data.get(nm + "_indices", None))
            if mask is not None:
                break
        if mask is None:
            warnings.warn(f"`{node_mask}_mask` or `{node_mask}_indices` not found, return None.")
    elif isinstance(node_mask, LongTensor):
        mask = index_to_mask(node_mask, num_nodes=data.get("num_nodes", data.get("x").shape[0]))
    elif isinstance(node_mask, BoolTensor):
        mask = node_mask
    else:
        raise ValueError
    return mask


def infer_all_splits(data) -> Dict[str, Any]:
    split_dict = {split_name: infer_node_mask(data, split_name) for split_name in ("train", "val", "test")}
    return split_dict


def infer_dataset_size_from_dataloader(dataloader, stage=None):
    if isinstance(dataloader, _FabricDataLoader):
        raw_dataloader = dataloader._dataloader  # noqa
    else:
        raw_dataloader = dataloader

    if isinstance(raw_dataloader, NodeLoader):
        dt_size = len(raw_dataloader.dataset)
    elif isinstance(raw_dataloader, DataLoader):
        if isinstance(raw_dataloader.dataset[0], PyGData) and len(raw_dataloader.dataset) == 1:
            dt_size = infer_node_mask(raw_dataloader.dataset[0], stage).sum()
        else:  # normal (x,y) samples or graph-level GNN dataset consiting of [PyGData0, PyGData1, ...]
            dt_size = len(raw_dataloader.dataset)
    else:
        raise NotImplementedError
    return dt_size


def merge_two_splits(split1, split2, num_nodes=None, return_long=True):
    is_bool1 = isinstance(split1, torch.BoolTensor)
    is_bool2 = isinstance(split2, torch.BoolTensor)

    if is_bool1 and is_bool2:
        assert split1.size(0) == split2.size(0)

    else:
        assert num_nodes is not None
        split1 = to_mask(split1, num_nodes)
        split2 = to_mask(split2, num_nodes)

    assert are_masks_disjoint(split1, split2), "Two masks should have been disjoint."
    merge_split = split1 + split2
    return merge_split if not return_long else mask_to_index(merge_split)


def convert_tensors_to_scalars(data: Any, strict: bool = False) -> Any:
    """
    Recursively walk through a collection and convert single-item tensors to scalar values.

    Args:
        data: the collection to apply the function to
        strict: If True, will raise error if any element of :obj:`data` collection is not a scalar; skip the
                non-scalar element otherwise.

    Returns: Any
        The data collection of tensors
    Raises:
        ValueError:
            If tensors inside ``metrics`` contains multiple elements, hence preventing conversion to a scalar.
    """

    def to_item(value: Tensor) -> Union[int, float, bool]:
        if value.numel() != 1:
            if strict:
                raise ValueError(
                    f"The metric `{value}` does not contain a single element, thus it cannot be converted to a scalar."
                )
            else:
                py_logger.warning(f"A {value.shape} non-scalar Tensor caught, skipped...")

        return value.item()

    return apply_to_collection(data, Tensor, to_item)
