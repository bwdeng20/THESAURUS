from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import spmm


class GPRConv(MessagePassing):
    r"""The graph convolutional operator from the `"Adaptive Universal Generalized PageRank Graph Neural Network"
    <https://openreview.net/forum?id=n6jl7fLxrP>`_ paper

    Args:
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on-the-fly.
            (default: :obj:`True`)

        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        K: int,
        alpha: float,
        init: str = "ppr",
        cached: bool = False,
        normalize: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.K = K
        self.alpha = alpha
        self.init = init.lower()

        self.cached = cached
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self.coeff = Parameter(self.init_coefficients(K, alpha, init))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.coeff.data = self.init_coefficients(self.K, self.alpha, self.init)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def init_coefficients(self, K, alpha, init):
        if init == "sgc":  # alpha has to be an integer indicating the peak
            temp = np.zeros(K + 1)
            temp[int(alpha)] = 1.0
        elif init == "ppr":  # PPR-like
            temp = alpha * (1 - alpha) ** np.arange(K + 1)
        elif init == "nppr":  # NPPR-like
            temp = (alpha) ** np.arange(K + 1)
            temp = temp / np.sum(np.abs(temp))
        elif init == "random":
            bound = np.sqrt(3 / (K + 1))
            temp = np.random.uniform(-bound, bound, K + 1)
            temp = temp / np.sum(np.abs(temp))
        else:
            raise TypeError(f"{init} is not an valid coefficient initialization strategy.")
        coeff = torch.from_numpy(temp).float()
        return coeff

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(
                f"'{self.__class__.__name__}' received a tuple "
                f"of node features as input while this layer "
                f"does not support bipartite message passing. "
                f"Please try other layers such as 'SAGEConv' or "
                f"'GraphConv' instead"
            )

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        add_self_loops=True,
                        flow=self.flow,
                        dtype=x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        add_self_loops=True,
                        flow=self.flow,
                        dtype=x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = x * (self.coeff[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            out = out + self.coeff[k + 1] * x
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return "{}(K={}, temp={})".format(self.__class__.__name__, self.K, self.coeff)
