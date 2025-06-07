import torch
import warnings
from torch_sparse import SparseTensor
from torch_sparse.matmul import matmul as ts_matmul
import random
from itertools import combinations
from typing import List, Tuple, Union
from numba import njit
import numpy as np

def stable_cumsum(arr, dim=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.
    """
    if dim is None:
        arr = arr.flatten()
        dim = 0
    out = torch.cumsum(arr, dim=dim, dtype=torch.float64)
    expected = torch.sum(arr, dim=dim, dtype=torch.float64)
    if not torch.all(torch.isclose(out.take(torch.Tensor([-1]).long().to(arr.device)),
                                   expected, rtol=rtol,
                                   atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out


def index_add_deterministic(labels, X, shape=None, ):
    spm = SparseTensor(row=labels, sparse_sizes=shape,
                       col=torch.arange(labels.size(0), device=labels.device))

    return ts_matmul(spm, X, reduce='sum')


def index_op_deterministic(labels, X, shape=None, reduce="sum"):
    spm = SparseTensor(row=labels,
                       col=torch.arange(labels.size(0), device=X.device),
                       sparse_sizes=shape)

    return ts_matmul(spm, X, reduce=reduce)


# 初始化父节点数组和大小数组
@njit(nogil=True)
def initialize(num_nodes):
    parent = np.arange(num_nodes)
    rank = np.ones(num_nodes)
    return parent, rank


# 查找操作（带路径压缩）
@njit(nogil=True)
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]


# 合并操作（按大小合并）
@njit(nogil=True)
def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
            rank[root_y] += rank[root_x]
        else:
            parent[root_y] = root_x
            rank[root_x] += rank[root_y]


# 统计群组详细信息
@njit(nogil=True)
def get_group_detailed(parent):
    num_nodes = parent.size
    group_indicator = np.full(num_nodes, -1, dtype=np.int64)
    group_sizes = []

    for i in range(num_nodes):
        root_i = find(parent, i)
        if group_indicator[root_i] == -1:
            group_indicator[root_i] = len(group_sizes)
            group_sizes.append(0)
        group_indicator[i] = group_indicator[root_i]
        group_sizes[group_indicator[root_i]] += 1
    group_sizes = np.asarray(group_sizes)
    return group_indicator, group_sizes


# 统计群组

# 运行并查集操作
@njit(nogil=True)
def get_group_union_find(num_nodes, friendships=None):
    parent, rank = initialize(num_nodes)
    if friendships is not None:
        for i in range(friendships.shape[0]):
            x, y = friendships[i]
            union(parent, rank, x, y)
    return get_group_detailed(parent)


def generate_simulated_clustering_data(
        num_clusters: int = 3,
        points_per_cluster: int = 50,
        dimensions: int = 2,
        cluster_std: float = 1.0,
        must_link_ratio: float = 0.1,
        cannot_link_ratio: float = 0.1,
        random_seed: int = 42
) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    生成用于聚类任务的随机模拟数据集，并根据数据点之间的关系生成朋友（must_link）和敌人（cannot_link）。

    参数：
    - num_clusters: 簇的数量。
    - points_per_cluster: 每个簇中的数据点数量。
    - dimensions: 数据的维度。
    - cluster_std: 簇的标准差（决定簇的紧密程度）。
    - must_link_ratio: 生成must_link关系的比例（相对于所有可能的must_link对）。
    - cannot_link_ratio: 生成cannot_link关系的比例（相对于所有可能的cannot_link对）。
    - random_seed: 随机种子，确保结果可复现。

    返回：
    - data: 生成的数据点，形状为 (num_points, dimensions)。
    - must_link: must_link关系的列表，每个元素是一个数据点对的索引元组。
    - cannot_link: cannot_link关系的列表，每个元素是一个数据点对的索引元组。
    """

    torch.manual_seed(random_seed)
    random.seed(random_seed)

    data = []
    labels = []
    cluster_centers = torch.randn(num_clusters, dimensions) * 5  # 簇中心分布更广泛

    for cluster_idx in range(num_clusters):
        # 为每个簇生成数据点
        points = cluster_centers[cluster_idx] + torch.randn(points_per_cluster, dimensions) * cluster_std
        data.append(points)
        labels += [cluster_idx] * points_per_cluster

    data = torch.vstack(data)
    labels = torch.tensor(labels)

    num_points = data.shape[0]

    # 生成must_link关系：同一簇内的随机点对
    must_link_candidates = []
    for cluster_idx in range(num_clusters):
        cluster_points = (labels == cluster_idx).nonzero(as_tuple=True)[0].tolist()
        if len(cluster_points) < 2:
            continue
        # 生成所有可能的组合
        cluster_combinations = list(combinations(cluster_points, 2))
        must_link_candidates.extend(cluster_combinations)

    num_must_link = int(len(must_link_candidates) * must_link_ratio)
    must_link = random.sample(must_link_candidates, min(num_must_link, len(must_link_candidates)))

    # 生成cannot_link关系：不同簇的随机点对
    cannot_link_candidates = []
    for cluster_a in range(num_clusters):
        for cluster_b in range(cluster_a + 1, num_clusters):
            points_a = (labels == cluster_a).nonzero(as_tuple=True)[0].tolist()
            points_b = (labels == cluster_b).nonzero(as_tuple=True)[0].tolist()
            cross_combinations = list((a, b) for a in points_a for b in points_b)
            cannot_link_candidates.extend(cross_combinations)

    num_cannot_link = int(len(cannot_link_candidates) * cannot_link_ratio)
    cannot_link = random.sample(cannot_link_candidates, min(num_cannot_link, len(cannot_link_candidates)))

    return data, must_link, cannot_link


def parse_th_generator(generator: Union[int, torch.Generator, None] = None,
                       device=None):
    if isinstance(generator, torch.Generator):
        rgen = generator
    else:
        rgen = torch.Generator(device=device)
        generator = generator if isinstance(generator, int) else 0  # default seed 0
        rgen.manual_seed(generator)  # For reproducibility in tests
    return rgen

if __name__ == '__main__':
    num_nodes = 7
    friendships = np.array([(0, 1), (1, 2), (3, 4), (5, 6), (4, 5)], dtype=np.int32)
    gp, gp_sizes = get_group_union_find(num_nodes, torch.tensor(friendships).cpu().numpy())
    assert isinstance(gp, np.ndarray)
    print(gp.dtype, gp)  # expected [0 0 0 3 3 3 3] or [0 0 0 1 1 1 1]
    print(gp_sizes.dtype, gp_sizes)
