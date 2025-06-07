import re
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.utils import subgraph, coalesce
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
from gnng.training.utils import infer_node_mask, merge_two_splits
from gnng.typing import PyGData


def edges_between_two_sets(
    subset: Union[Tensor, List[int]],
    subset2: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    two_direction: bool = False,
    num_nodes: Optional[int] = None,
    *,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, Tensor]]:
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)
    if isinstance(subset2, (list, tuple)):
        subset2 = torch.tensor(subset2, dtype=torch.long, device=device)

    if subset.dtype != torch.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)
    else:
        num_nodes = subset.size(0)
        node_mask = subset
        subset = node_mask.nonzero().view(-1)

    if subset2.dtype != torch.bool:
        num_nodes2 = maybe_num_nodes(edge_index, num_nodes)
        node_mask2 = index_to_mask(subset2, size=num_nodes2)
    else:
        num_nodes2 = subset2.size(0)
        node_mask2 = subset2
        subset2 = node_mask2.nonzero().view(-1)
    assert num_nodes2 == num_nodes

    num_unique_nodes = torch.cat([subset, subset2]).unique().shape[0]
    _disjoint_set_flag = num_unique_nodes == subset.shape[0] + subset2.shape[0]
    if not _disjoint_set_flag:  # between the same sets
        assert num_unique_nodes == subset.shape[0], "Overlapping sets are not considered now."

    new_edge_mask = node_mask[edge_index[0]] & node_mask2[edge_index[1]]

    new_edge_index = edge_index[:, new_edge_mask]
    new_edge_attr = edge_attr[new_edge_mask] if edge_attr is not None else None

    if two_direction:
        edge_mask21 = node_mask2[edge_index[0]] & node_mask[edge_index[1]]
        edge_index21 = edge_index[:, edge_mask21]
        edge_attr21 = edge_attr[edge_mask21] if edge_attr is not None else None

        new_edge_mask = torch.cat([new_edge_mask, edge_mask21])
        new_edge_index = torch.cat([new_edge_index, edge_index21], dim=-1)
        if edge_attr21 is not None:
            new_edge_attr = torch.cat([new_edge_attr, edge_attr21])

    if return_edge_mask:
        return new_edge_index, new_edge_attr, new_edge_mask
    else:
        return new_edge_index, new_edge_attr


def find_sort_pairs(string, pattern=r"(tr|te|va)2(tr|te|va)"):
    matched_pair_list = re.findall(pattern, string)
    return sorted(matched_pair_list)  # like [('te', 'va'), ('tr', 'te'), ('tr', 'va'), ('va', 'tr'), ('va', 'va')]


def data_from_edge_sets(data: PyGData, command, train_split=None, val_split=None, test_split=None):
    train_split = infer_node_mask(data, "train") if train_split is None else train_split
    val_split = infer_node_mask(data, "val") if val_split is None else val_split
    test_split = infer_node_mask(data, "test") if test_split is None else test_split

    split_idx_dict = {"train": train_split, "val": val_split, "test": test_split}
    split_idx_dict["tr"] = split_idx_dict["train"]
    split_idx_dict["va"] = split_idx_dict["val"]
    split_idx_dict["te"] = split_idx_dict["test"]

    # # for debug
    # print("VersionData DDDDDD:")
    # for s in ['tr', 'va', 'te']:
    #     print(f"{s:6s}: ", mask2index(split_idx_dict[s]))

    edge_index = data.edge_index
    pair_list = find_sort_pairs(command)
    edge_mask_list = []
    for src_set, dst_set in pair_list:
        if src_set == dst_set:
            _, _, e_mask = subgraph(
                split_idx_dict[src_set], edge_index, return_edge_mask=True, num_nodes=data.num_nodes
            )
        else:
            _, _, e_mask = edges_between_two_sets(
                split_idx_dict[src_set],
                split_idx_dict[dst_set],
                num_nodes=data.num_nodes,
                edge_index=edge_index,
                two_direction=False,
                return_edge_mask=True,
            )
        edge_mask_list.append(e_mask)
    e_mask_all = torch.stack(edge_mask_list)
    e_mask_merged = e_mask_all.any(dim=0)  # if one edge is selected by one edge set

    # new_data = data.edge_subgraph(mask2index(e_mask_merged))
    new_data = data.edge_subgraph(e_mask_merged)
    new_data = new_data.coalesce()
    sorted_command = "_".join(compose_pair_list(pair_list))
    new_data.command = sorted_command
    # return new_data, edge_mask_list  # for debug
    return new_data


def compose_edge_index(edge_index, split_idx_dict, num_nodes, command):
    pair_list = find_sort_pairs(command)
    split_idx_dict["tr"] = split_idx_dict["train"]
    split_idx_dict["va"] = split_idx_dict.get("val", split_idx_dict.get("valid"))
    split_idx_dict["te"] = split_idx_dict["test"]
    edge_list = []
    edge_mask_list = []
    # # for debug
    # print("VersionRaw DDDDDD:")
    # for s in ['tr', 'va', 'te']:
    #     print(f"{s:6s}: ", mask2index(split_idx_dict[s]))

    for src_set, dst_set in pair_list:
        if src_set == dst_set:
            edge, _, e_mask = subgraph(split_idx_dict[src_set], edge_index, num_nodes=num_nodes, return_edge_mask=True)
        else:
            edge, _, e_mask = edges_between_two_sets(
                split_idx_dict[src_set],
                split_idx_dict[dst_set],
                num_nodes=num_nodes,
                edge_index=edge_index,
                return_edge_mask=True,
                two_direction=False,
            )
        edge_list.append(edge)
        edge_mask_list.append(e_mask)

    composed_edge_index = torch.cat(edge_list, dim=-1)
    new_edge_index = coalesce(composed_edge_index.clone().detach(), num_nodes=num_nodes)
    assert new_edge_index.shape[-1] == composed_edge_index.shape[-1]
    involved_nodes = set(new_edge_index.unique().cpu().tolist())
    should_involved_node_sets = set([i for row in pair_list for i in row])
    max_involved_node = set(torch.cat([split_idx_dict[s] for s in should_involved_node_sets]).unique().cpu().tolist())
    assert len(max_involved_node.difference(involved_nodes)) >= 0
    sorted_command = "_".join(compose_pair_list(pair_list))
    # return new_edge_index, sorted_command, edge_mask_list  # for debug
    return new_edge_index, sorted_command


def decompose_pairname_list(pair_list):
    return [tuple(pair_name.split("2")) for pair_name in pair_list]


def compose_pair_list(pair_list):
    return [f"{src_set}2{dst_set}" for src_set, dst_set in pair_list]


class TVTGraph:  # generate train/val/test graphs
    def __call__(self, data, *args, **kwargs):
        raise NotImplementedError


class PieceGraph(TVTGraph):
    def __init__(self, piece_rule: str = None):
        super().__init__()
        self.piece_rule = piece_rule

    def __call__(self, full_data: PyGData, piece_rule: str = None):  # noqa
        piece_rule = piece_rule or self.piece_rule
        if piece_rule is None:
            raise ValueError(f"No piece rule is given.")
        split_dict = {
            split_marker: infer_node_mask(full_data, split_marker) for split_marker in ["train", "val", "test"]
        }
        train_idx = split_dict["train"]
        val_idx = split_dict["val"]
        test_idx = split_dict["test"]

        data_dict = {"train": None, "val": None, "test": None, "full": full_data}

        if piece_rule == "induc":  # classic validation way: the links between train and val nodes are used
            train_valid_idx = merge_two_splits(train_idx, val_idx)
            data_dict["train"] = full_data.subgraph(train_idx)
            data_dict["val"] = full_data.subgraph(train_valid_idx)
            data_dict["test"] = full_data

        elif piece_rule == "trans":
            data_dict["train"] = full_data
            data_dict["val"] = full_data
            data_dict["test"] = full_data

        elif piece_rule in ("true_induc", "true_induc_te2te"):
            data_dict["train"] = full_data.subgraph(train_idx)
            data_dict["val"] = full_data.subgraph(val_idx)
            data_dict["test"] = full_data.subgraph(test_idx)

        elif "true_induc" in piece_rule and "2" in piece_rule:
            data_dict["train"] = full_data.subgraph(train_idx)
            data_dict["val"] = full_data.subgraph(val_idx)
            data_dict["test"] = data_from_edge_sets(
                full_data, piece_rule, train_split=train_idx, val_split=val_idx, test_split=test_idx
            )
        else:
            raise ValueError

        return data_dict

    def __repr__(self):
        return f"{self.__class__.__name__}({self.piece_rule})"


class GCDGraph(PieceGraph):
    def __init__(self, piece_rule: str, known_classes: Union[List[int], Tensor, float, int]):
        super().__init__(piece_rule)
        self.known_classes = known_classes

    def __call__(self, full_data: PyGData, piece_rule: str = None):
        if isinstance(self.known_classes, float):  # the first portion of labels are taken as known classes
            all_classes = full_data.y.unique()
            known_classes = all_classes[: int(all_classes.size(0) * self.known_classes)]
        else:
            known_classes = self.known_classes

        old_mask = torch.zeros(full_data.y.shape, dtype=torch.bool)
        for c in known_classes:
            old_mask = old_mask.logical_or(full_data.y == c)
        num_old_nodes = old_mask.sum()
        full_data.old_mask = old_mask
        full_data.num_old_nodes = num_old_nodes
        return super().__call__(full_data, piece_rule)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.piece_rule}, known_classes={self.known_classes})"
