from typing import Optional, List, Union
import torch
import numpy as np
from torch_geometric.data import Dataset as PyGDataset
from gnng.typing import SplitRatio, PyGData
from gnng.utils import are3masks_disjoint


class DataSplitterBase:
    def __init__(self, train_val_test: SplitRatio = (10, 10, 80), seed: Optional[int] = None):
        train_val_test = np.array(train_val_test)
        self._train_val_test = self.pre_process_ratio(train_val_test)
        self._min_ratio = min(self._train_val_test)
        self.np_rng = np.random.default_rng(seed)
        self.seed = seed

    @staticmethod
    def pre_process_ratio(train_val_test):
        if train_val_test[0] > 1:  # e.g., 60, 10, 30
            train_val_test = train_val_test / 100
        # else e.g., 0.6, 0.1, 0.3
        assert train_val_test.sum() <= 1, f"The split ratio sum({train_val_test})>1 is invalid"
        return train_val_test

    @staticmethod
    def _check_split(train_val_test):
        if sum(train_val_test) not in (100, 1):
            raise ValueError(f"the split rate {train_val_test} is wrong")

    def _check_ratio(self, num_total):
        if int(self._min_ratio * num_total) < 1:
            raise ValueError(
                f"The split ratio {self._min_ratio} is too small to get" f"one sample out of {num_total} samples"
            )

    @property
    def train_val_test(self):
        return self._train_val_test

    @train_val_test.setter
    def train_val_test(self, new_split):
        self._check_split(new_split)
        self._train_val_test = new_split

    @property
    def train_ratio(self):
        return self._train_val_test[0]

    @property
    def val_ratio(self):
        return self._train_val_test[1]

    @property
    def test_ratio(self):
        return self._train_val_test[2]

    def __call__(self, *args, **kwargs) -> Union[PyGData, PyGDataset]:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(train/val/test={self._train_val_test})"


class NoopDataSplitter(DataSplitterBase):
    def __call__(self, pyg_data: PyGData):  # the public split is fixed and loaded from pyg datasets
        return pyg_data

    def __repr__(self):
        return f"{self.__class__.__name__}(train/val/test=default_in)"


class RandomDataSplitter(DataSplitterBase):
    def __init__(self, train_val_test: SplitRatio = (10, 10, 80), seed: Optional[int] = None):
        super().__init__(train_val_test, seed)

    def __call__(self, pyg_data: PyGData):
        self._check_ratio(pyg_data.num_nodes)
        train_marker, val_marker, test_marker = self.get_random_split(pyg_data.num_nodes, return_mask=True)
        pyg_data.train_mask = torch.as_tensor(train_marker)
        pyg_data.val_mask = torch.as_tensor(val_marker)
        pyg_data.test_mask = torch.as_tensor(test_marker)
        return pyg_data

    def get_random_split(self, num_total: int, return_mask=True):
        shuffled_idx = torch.as_tensor(self.np_rng.permutation(num_total), dtype=torch.long)
        n1 = int(num_total * self.train_ratio)
        n2 = int(num_total * self.val_ratio)
        train_idx = shuffled_idx[:n1]
        val_idx = shuffled_idx[-n2:]

        if return_mask:
            train_marker = torch.zeros(num_total, dtype=torch.bool)
            val_marker = torch.zeros_like(train_marker)
            test_marker = torch.ones_like(train_marker)
            train_marker[train_idx] = True
            val_marker[val_idx] = True
            test_marker[train_marker] = False
            test_marker[val_marker] = False
            assert are3masks_disjoint(train_marker, val_marker, test_marker)
        else:
            train_marker = train_idx
            val_marker = val_idx
            test_marker = shuffled_idx[n1:n2]
        return train_marker, val_marker, test_marker
