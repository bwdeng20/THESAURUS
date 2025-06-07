from gnng.typing import SplitRatio, PyGData
import torch
import numpy as np
from torch_geometric.utils import degree
from gnng.utils import index2mask
from .basics import DataSplitterBase


def is_dict_value_overlapped(dictionary):
    all_entries = np.concatenate(list(dictionary.values()))
    unique_entries = np.unique(all_entries)
    return len(unique_entries) < len(all_entries)


def group_node_with_degree(degrees, fr=0.0, return_dict=False):
    assert 0 <= fr < 1
    nnode = len(degrees)
    sort_idx = np.argsort(degrees)

    # filter out fr% nodes with the lowest degrees and fr% nodes with the highest degrees
    num2filter = int(nnode * fr)
    if fr > 0:
        assert num2filter != 0, "Please choose smaller ratio of nodes to filter out"
    filtered_sort_idx = sort_idx[num2filter:-num2filter] if num2filter > 0 else sort_idx
    chunked_sort_idx = np.array_split(filtered_sort_idx, 3)  # three groups
    # -1, 0, 1, 2, 3 means:
    # -1: filtered-out and low-degree,
    #  0: low-degree, 1: medium degree, 2: high degree,
    #  3: filtered-out and high-degree
    if return_dict:
        degree_group = {-1: sort_idx[:num2filter], 3: sort_idx[-num2filter:]}
        for i, chunk in enumerate(chunked_sort_idx):
            degree_group[i] = chunk
    else:
        degree_group = np.ones(nnode) * -2
        for i, chunk in enumerate(chunked_sort_idx):
            degree_group[chunk] = i
        degree_group[sort_idx[:num2filter]] = -1
        degree_group[sort_idx[-num2filter:]] = 3
    return degree_group


def split_grb(
    degrees,
    train=0.6,
    val=0.1,
    test_per_partition=0.1,
    fr=0.05,
    mode="F",
    seed=None,
    return_mask=False,
):
    """`"Graph robustness benchmark: Benchmarking the adversarial robustness of graph machine learning".

    <https://arxiv.org/abs/1902.08412>`_ paper (NeurIPS2021)

    Step1: Rank all nodes in terms of the degrees and filter out  fr% nodes with the lowest degrees
            and fr% nodes with the highest degree (totally 2*fr% of ORIGINAL nodes).
    Step2: The rest nodes are divided into three equal partitions without overlapping, and randomly sample
            test_per_partition% nodes (without repetition) from each partition
            (totally 3*test_per_partition% of the REST nodes).
    Step3:  The test subsets with different levels of degree are marked as Small/Medium/Large/Full (‘S/M/L/F’)
            with ‘F’ containing all test nodes.
    Args:
        degrees: np.ndarray
        train:  float
        val:    float
        test_per_partition: float
        fr: float
        mode: str
            ("F", "S", "M", "L")
        seed: optional[int]
        return_mask: bool
            If True, return mask instead of indices

    Returns: List[np.ndarray]
        The train/val/test indices

    """
    assert mode in ("F", "S", "M", "L")
    np_rng = np.random.default_rng(seed)
    nnodes = len(degrees)

    degree_group_dict = group_node_with_degree(degrees, fr=fr, return_dict=True)
    assert len(degree_group_dict) == 5, "The nodes are not grouped into five partitions as GRB"

    assert not is_dict_value_overlapped(degree_group_dict)

    num_filter_node = nnodes - len(degree_group_dict[-1]) - len(degree_group_dict[3])
    num_test_per_part = int(num_filter_node * test_per_partition)

    test_node_group = []
    for i in range(3):
        nodes = degree_group_dict[i]
        test4part = np_rng.choice(nodes, num_test_per_part, replace=False)
        test_node_group.append(test4part)

    all_test_nodes = np.concatenate(test_node_group)
    all_non_test_nodes = np.setdiff1d(np.arange(nnodes), all_test_nodes)
    # remove nodes from filter-out group as GRB
    all_train_val_nodes = np.setdiff1d(all_non_test_nodes, degree_group_dict[-1])
    all_train_val_nodes = np.setdiff1d(all_train_val_nodes, degree_group_dict[3])

    idx_val = np_rng.choice(all_train_val_nodes, int(num_filter_node * val), replace=False)
    idx_train = np.setdiff1d(all_train_val_nodes, idx_val)
    if train is not None and 0 < train < 0.6:  # select from the rest nodes
        # train, val, testS, testM, testL = ?, 0.1, 0.1, 0.1, 0.1
        idx_train = np_rng.choice(idx_train, int(num_filter_node * train), replace=False)

    if mode == "F":
        test_nodes = all_test_nodes
    elif mode == "S":
        test_nodes = test_node_group[0]
    elif mode == "M":
        test_nodes = test_node_group[1]
    elif mode == "L":
        test_nodes = test_node_group[2]
    else:
        raise ValueError(f"{mode} is not valid")
    if return_mask:
        return (
            index2mask(idx_train, nnodes),
            index2mask(idx_val, nnodes),
            index2mask(test_nodes, nnodes),
        )
    return idx_train, idx_val, test_nodes


class GrbDataSplitter(DataSplitterBase):
    """`"Graph robustness benchmark: Benchmarking the adversarial robustness of graph machine learning".

    <https://arxiv.org/abs/1902.08412>`_ paper (NeurIPS2021)

    """

    def __init__(
        self,
        train_val_test: SplitRatio = (0.6, 0.1, 0.1),
        fr: float = 0.05,
        mode="F",
        seed: int = None,
    ):
        super().__init__(train_val_test, seed)
        self.fr = fr
        self.mode = mode

    def __call__(self, pyg_data: PyGData):
        self._check_ratio(pyg_data.num_nodes * (1 - self.fr * 2))
        degrees = degree(pyg_data.edge_index).cpu().numpy()
        train_marker, val_marker, test_marker = self.get_grb_split(degrees, return_mask=True)
        pyg_data.train_mask = torch.as_tensor(train_marker)
        pyg_data.val_mask = torch.as_tensor(val_marker)
        pyg_data.test_mask = torch.as_tensor(test_marker)
        return pyg_data

    def get_grb_split(self, degrees, return_mask=True):
        return split_grb(
            degrees,
            train=self.train_ratio,
            val=self.val_ratio,
            test_per_partition=self.test_ratio,
            fr=self.fr,
            mode=self.mode,
            seed=self.seed,
            return_mask=return_mask,
        )

    @staticmethod
    def parse_grb_str(setting):
        require_llc = "llc" in setting
        if "grbs" in setting:
            tmode = "S"
        elif "grbm" in setting:
            tmode = "M"
        elif "grbl" in setting:
            tmode = "L"
        else:
            tmode = "F"
        return tmode, require_llc
