from typing import Optional
from gnng.typing import SplitRatio, PyGData

import torch

from gnng.utils import are3masks_disjoint, mask2index
from .basics import DataSplitterBase


class GCNPerClassDataSplitter(DataSplitterBase):
    def __init__(self, train_val_test: SplitRatio = (20, 500, 1000), seed: Optional[int] = None):
        super().__init__(train_val_test, seed)

    @staticmethod
    def pre_process_ratio(train_val_test):  # no ratio-like assert
        assert (
            train_val_test[0] > 1 and train_val_test[1] > 1 and train_val_test[2] > 1
        ), f"The split ratio sum({train_val_test})>1 is invalid"
        return train_val_test

    def get_gcn_split(
        self,
        label,
        num_classes,
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        return_mask=True,
    ):
        train_mask = torch.zeros(label.shape, dtype=torch.bool)
        val_mask = torch.zeros_like(train_mask)
        test_mask = torch.zeros_like(train_mask)
        for c in range(num_classes):
            idx = (label == c).nonzero(as_tuple=False).view(-1)
            randperm = self.np_rng.permutation(idx.size(0))
            randperm = torch.as_tensor(randperm, dtype=torch.long)
            idx = idx[randperm[:num_train_per_class]]
            train_mask[idx] = True

        remaining = (~train_mask).nonzero(as_tuple=False).view(-1)

        randperm = self.np_rng.permutation(remaining.size(0))
        randperm = torch.as_tensor(randperm, dtype=torch.long)
        remaining = remaining[randperm]

        val_mask.fill_(False)
        val_mask[remaining[:num_val]] = True

        test_mask.fill_(False)
        test_mask[remaining[num_val : num_val + num_test]] = True

        are3masks_disjoint(train_mask, val_mask, test_mask)
        if return_mask:
            return train_mask, val_mask, test_mask
        else:
            return mask2index(train_mask), mask2index(val_mask), mask2index(test_mask)

    def __call__(self, pyg_data: PyGData):
        self._check_ratio(pyg_data.num_nodes)
        train_marker, val_marker, test_marker = self.get_gcn_split(
            pyg_data.y,
            pyg_data.num_classes,
            num_train_per_class=self.train_ratio,
            num_val=self.val_ratio,
            num_test=self.test_ratio,
            return_mask=True,
        )
        pyg_data.train_mask = torch.as_tensor(train_marker)
        pyg_data.val_mask = torch.as_tensor(val_marker)
        pyg_data.test_mask = torch.as_tensor(test_marker)
        return pyg_data
