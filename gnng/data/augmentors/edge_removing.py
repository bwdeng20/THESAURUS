from typing import Tuple, Optional
import torch
from torch import Tensor
from gnng.typing import PyGData

from gnng.data.augmentors.augmentor import Augmentor
from gnng.data.augmentors.functional import pr_drop_weights, degree_drop_weights, evc_drop_weights, drop_weighted_mask


class EdgeRemover(Augmentor):
    def __init__(self, pe: float, px: float = 0.0):
        super(EdgeRemover, self).__init__()
        if pe < 0.0 or pe > 1.0:
            raise ValueError(f"Dropout probability has to be between 0 and 1 " f"(got {pe}")
        if px < 0.0 or px > 1.0:
            raise ValueError(f"Dropout probability has to be between 0 and 1 " f"(got {px}")
        self.pe = pe
        self.px = px
        self.drop_edge_flag = pe > 0.0
        self.mask_fea_flag = px > 0.0

    def get_drop_masks(self, g: PyGData) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        raise NotImplemented

    def augment(self, g: PyGData) -> PyGData:
        new_g = g.detach().clone()
        if self.pe == 0.0 and self.px == 0.0:
            return new_g
        e_mask, x_mask = self.get_drop_masks(g)
        if e_mask is not None:
            new_g.edge_index = new_g.edge_index[:, ~e_mask]
            new_g.edge_attr = new_g.edge_attr[~e_mask] if new_g.edge_attr is not None else None
        if x_mask is not None:
            new_g.x[..., x_mask] = 0.0
        return new_g


class UniformEdgeRemover(EdgeRemover):
    def get_drop_masks(self, g: PyGData) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        e_mask = None
        x_mask = None
        if self.drop_edge_flag:
            row = g.edge_index[0]
            e_mask = torch.rand(row.size(0), device=row.device) >= self.pe
        if self.mask_fea_flag:
            x_mask = torch.rand(g.x.shape[-1], device=g.x.device) >= self.px
        return e_mask, x_mask


class WeightedEdgeRemover(EdgeRemover):
    def __init__(
        self,
        pe: float,
        px: float = 0.0,
        scheme: str = "evc",
        e_threshold: float = 1.0,
        x_threshold: float = 0.7,
        cache: bool = False,
    ):
        assert scheme in ("degree", "evc", "pr")
        super(WeightedEdgeRemover, self).__init__(pe=pe, px=px)
        self.scheme = scheme
        self.e_threshold = e_threshold
        self.x_threshold = x_threshold
        self.cache_weights = cache
        self.drop_edge_weights = None
        self.mask_fea_weights = None

    def compute_drop_edge_weights(self, g: PyGData):
        x = g.x if self.mask_fea_flag else None
        if self.scheme == "degree":
            drop_edge_weights, mask_fea_weights = degree_drop_weights(
                g.edge_index, g.num_nodes, x, no_edge=not self.drop_edge_flag
            )
        elif self.scheme == "pr":
            drop_edge_weights, mask_fea_weights = pr_drop_weights(
                g.edge_index, aggr="sink", k=200, num_nodes=g.num_nodes, x=x, no_edge=not self.drop_edge_flag
            )
        elif self.scheme == "evc":
            drop_edge_weights, mask_fea_weights = evc_drop_weights(
                g.edge_index,
                g.get("edge_weight"),
                g.get("edge_attr"),
                g.num_nodes,
                x=x,
                no_edge=not self.drop_edge_flag,
            )
        else:
            raise TypeError(f"Unrecognized scheme {self.scheme}")
        return drop_edge_weights, mask_fea_weights

    def get_drop_masks(self, g: PyGData) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if (self.drop_edge_weights is None and self.drop_edge_flag) or (
            self.mask_fea_weights is None and self.mask_fea_flag
        ):
            drop_edge_weights, mask_fea_weights = self.compute_drop_edge_weights(g)
            if self.cache_weights:
                self.drop_edge_weights = drop_edge_weights
                self.mask_fea_weights = mask_fea_weights
        else:
            drop_edge_weights = self.drop_edge_weights
            mask_fea_weights = self.mask_fea_weights
        e_mask = None
        x_mask = None
        if drop_edge_weights is not None:
            e_mask = drop_weighted_mask(drop_edge_weights, p=self.pe, threshold=self.e_threshold)
        if mask_fea_weights is not None:
            x_mask = drop_weighted_mask(mask_fea_weights, p=self.px, threshold=self.x_threshold)
        return e_mask, x_mask
