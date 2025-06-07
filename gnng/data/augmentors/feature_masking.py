from typing import Tuple
import torch
from torch import Tensor
from gnng.typing import PyGData
from torch_geometric.utils import to_undirected, degree
from gnng.data.augmentors.augmentor import Augmentor
from gnng.data.augmentors.functional import (
    eigenvector_centrality_numpy,
    drop_weighted_mask,
    compute_pr,
    feature_drop_weights,
)


class FeatureMasker(Augmentor):
    def __init__(self, p: float):
        super(FeatureMasker, self).__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Dropout probability has to be between 0 and 1 " f"(got {p}")
        self.p = p

    def get_drop_mask(self, g: PyGData) -> Tuple[Tensor, Tensor]:
        raise NotImplemented

    def augment(self, g: PyGData) -> PyGData:
        new_g = g.detach().clone()
        if self.p == 0.0:
            return new_g
        drop_mask = self.get_drop_mask(g)
        new_g.x[..., drop_mask] = 0.0
        return new_g


class UniformFeatureMasker(FeatureMasker):
    def get_drop_mask(self, g: PyGData) -> Tuple[Tensor]:
        mask: Tensor = torch.rand(g.x.shape[-1], device=g.x.device) >= self.p
        return mask


class WeightedFeatureMasker(FeatureMasker):
    def __init__(self, p: float, scheme: str = "evc", threshold: float = 1.0, cache: bool = False):
        assert scheme in ("degree", "evc", "pr")
        super(WeightedFeatureMasker, self).__init__(p)
        self.scheme = scheme
        self.threshold = threshold
        self.cache_weights = cache
        self.drop_weights = None

    def compute_drop_fea_weights(self, g: PyGData):
        if self.scheme == "degree":
            edge_index_ = to_undirected(g.edge_index)
            node_c = degree(edge_index_[1], g.num_nodes)
        elif self.scheme == "pr":
            node_c = compute_pr(g.edge_index)
        elif self.scheme == "evc":
            node_c = eigenvector_centrality_numpy(g.edge_index, g.edge_weight, g.edge_attr, g.num_nodes)
        else:
            raise TypeError(f"Unrecognized feature masking scheme {self.scheme}")

        drop_weights = feature_drop_weights(g.x, node_c)
        return drop_weights

    def get_drop_mask(self, g: PyGData) -> Tuple[Tensor, Tensor]:
        if self.drop_weights is None:
            drop_weights = self.compute_drop_fea_weights(g)
            if self.cache_weights:
                self.drop_weights = drop_weights
        else:
            drop_weights = self.drop_weights
        return drop_weighted_mask(drop_weights, p=self.p, threshold=self.threshold)
