import torch
import numpy as np
import scipy as sp
from torch_geometric.utils import to_scipy_sparse_matrix, degree, scatter, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes


def eigenvector_centrality_numpy(edge_index, edge_weight=None, edge_attr=None, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    M = to_scipy_sparse_matrix(edge_index, edge_weight if edge_weight is not None else edge_attr, num_nodes)
    _, eigenvector = sp.sparse.linalg.eigs(M.T, k=1, which="LR", maxiter=50, tol=0.0)
    largest = eigenvector.flatten().real
    norm = np.sign(largest.sum()) * sp.linalg.norm(largest)
    return torch.tensor(largest / norm, dtype=torch.float32).to(edge_index.device)


def compute_pr(edge_index, damp: float = 0.85, k: int = 10, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes,), device=edge_index.device, dtype=torch.float32)
    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce="sum")
        x = (1 - damp) * x + damp * agg_msg
    return x


def drop_feature_weighted(x, w, drop_rate: float, threshold: float = 0.7):
    w = w / w.mean() * drop_rate
    w = w.where(w < threshold, threshold)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    x = x.clone()
    x[..., drop_mask] = 0.0
    return x, drop_mask


def feature_drop_uniform_mask(x: torch.Tensor, drop_rate: float):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_rate
    return drop_mask


def drop_weighted_mask(w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, threshold)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    return drop_mask


def feature_drop_weights_binary_sparse(x, node_c):
    x = x.to(torch.bool).to(torch.float32)  # 0,1 feature value
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())
    return s


def feature_drop_weights(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())
    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.0):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, threshold)
    sel_mask = torch.bernoulli(1.0 - edge_weights).bool()
    return edge_index[:, sel_mask], sel_mask


def degree_drop_weights(edge_index, num_nodes=None, x: torch.Tensor = None, no_edge: bool = False):
    edge_index_ = to_undirected(edge_index, num_nodes=num_nodes)
    deg = degree(edge_index_[1], num_nodes)
    drop_edge_weights = None
    drop_fea_weights = None
    if not no_edge:
        deg_col = deg[edge_index[1]].float()
        s_col = torch.log(deg_col)
        drop_edge_weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
    if x is not None:
        drop_fea_weights = feature_drop_weights(x, deg)
    return drop_edge_weights, drop_fea_weights


def pr_drop_weights(
    edge_index, aggr: str = "sink", k: int = 10, num_nodes=None, x: torch.Tensor = None, no_edge: bool = False
):
    pv = compute_pr(edge_index, k=k, num_nodes=num_nodes)
    drop_edge_weights = None
    drop_fea_weights = None
    if not no_edge:
        pv_row = pv[edge_index[0]].to(torch.float32)
        pv_col = pv[edge_index[1]].to(torch.float32)
        s_row = torch.log(pv_row)
        s_col = torch.log(pv_col)
        if aggr == "sink":
            s = s_col
        elif aggr == "source":
            s = s_row
        elif aggr == "mean":
            s = (s_col + s_row) * 0.5
        else:
            s = s_col
        drop_edge_weights = (s.max() - s) / (s.max() - s.mean())
    if x is not None:
        drop_fea_weights = feature_drop_weights(x, pv)
    return drop_edge_weights, drop_fea_weights


def evc_drop_weights(
    edge_index, edge_weight=None, edge_attr=None, num_nodes=None, x: torch.Tensor = None, no_edge: bool = False
):
    evc = eigenvector_centrality_numpy(edge_index, edge_weight, edge_attr, num_nodes)
    drop_edge_weights = None
    drop_fea_weights = None
    if not no_edge:
        evc_e = evc.where(evc > 0, 0.0)
        s = (evc_e + 1e-8).log()
        s_row, s_col = s[edge_index[0]], s[edge_index[1]]
        s = s_col
        drop_edge_weights = (s.max() - s) / (s.max() - s.mean())
    if x is not None:
        drop_fea_weights = feature_drop_weights(x, evc)
    return drop_edge_weights, drop_fea_weights


def add_gaussian_noise(x, noise_std=0.1):
    noise = torch.randn_like(x) * noise_std
    return x + noise


def feature_perturbation(x, perturbation_factor=0.05):
    perturbation = (torch.rand_like(x) - 0.5) * 2 * perturbation_factor
    return x * (1 + perturbation)


def random_masking(x, mask_ratio=0.1):
    mask = (torch.rand_like(x) > mask_ratio).float()
    return x * mask


def dimension_shuffling(x):
    perm = torch.randperm(x.size(-1))
    return x[:, perm]


def contrastive_stretching(x, stretch_factor=0.2):
    factors = 1 + (torch.rand_like(x) - 0.5) * 2 * stretch_factor
    return x * factors
