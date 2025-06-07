from typing import Sequence, Union
from gnng.typing import PyGData
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset as PyGDataset
from gnng.utils import index2mask
from .basics import DataSplitterBase


def split_stratify(num_total, val_size=0.1, test_size=0.8, stratify=None, return_mask=False, seed=None):
    """This setting follows nettack/mettack, where we split the nodes into 10% training, 10% validation and 80% testing
    data_tools according to the labels of all nodes.

    Parameters
    ----------
    num_total : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data_tools is expected to split in a stratified fashion. So stratify should be labels.
    return_mask:
        If True, return BoolTensor mask; otherwise LongTensor indices
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    return_mask: bool
        If True, return mask

    Args:
        return_mask:
        return_mask:
        return_mask:
        return_mask:

    """
    idx = np.arange(num_total)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(
        idx,
        random_state=seed,
        train_size=1 - test_size,
        test_size=test_size,
        stratify=stratify,
    )

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    n_train = int(num_total*train_size)
    n_val = idx_train_and_val.size - n_train
    idx_train, idx_val = train_test_split(
        idx_train_and_val,
        random_state=seed,
        train_size=n_train,
        test_size=n_val,
        stratify=stratify,
    )

    if return_mask:
        return (
            index2mask(idx_train, num_total),
            index2mask(idx_val, num_total),
            index2mask(idx_test, num_total),
        )
    return idx_train, idx_val, idx_test


class StratifyDataSplitter(DataSplitterBase):
    def __init__(self, train_val_test: Sequence = (10, 10, 80), seed: int = None):
        super().__init__(train_val_test, seed)

    def __call__(self, pyg_data: Union[PyGData, PyGDataset]):
        if isinstance(pyg_data, PyGData):
            num_nodes = pyg_data.num_nodes
            self._check_ratio(num_nodes)
            train_marker, val_marker, test_marker = self.get_stratify_split(
                num_nodes, pyg_data.y, return_mask=True
            )
            pyg_data.train_mask = torch.as_tensor(train_marker)
            pyg_data.val_mask = torch.as_tensor(val_marker)
            pyg_data.test_mask = torch.as_tensor(test_marker)
        elif isinstance(pyg_data, PyGDataset):
            num_graphs = len(pyg_data)
            self._check_ratio(num_graphs)
            train_indices, val_indices, test_indices = self.get_stratify_split(
                num_graphs, pyg_data.y, return_mask=False
            )
            pyg_data.train_indices = torch.as_tensor(train_indices)
            pyg_data.val_indices = torch.as_tensor(val_indices)
            pyg_data.test_indices = torch.as_tensor(test_indices)
            num_all = len(train_indices) + len(val_indices) + len(test_indices)
            assert  num_all == num_graphs, f"{num_all} vs actual {num_graphs}."
        return pyg_data

    def get_stratify_split(self, num_total: int, stratify, return_mask=True):
        train_mask, val_mask, test_mask = split_stratify(
            num_total,
            self.val_ratio,
            self.test_ratio,
            stratify.cpu().numpy(),
            seed=self.seed,
            return_mask=return_mask,
        )

        return train_mask, val_mask, test_mask


class NettackDataSplitter(StratifyDataSplitter):
    pass
