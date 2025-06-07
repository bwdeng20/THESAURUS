"""Adopted from torch_geometric.nn.models.basic_gnn.py.

What we add:
1)  customized spectral normalization for all GNN and Linear layers
2)  linear layers before and/or after gnn convs
3)  GCN2 model support
4)  hidden feature tape for feature distillation

"""

import copy
import inspect
import warnings
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.nn.conv import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
)
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils._trim_to_layer import TrimToLayer
from torch.nn.utils.parametrize import remove_parametrizations
from gnng.nn.utils.parametrizations import spectral_norm, parse_spec_norm_arg
from gnng.nn.models.mlp import MLP
from gnng.nn.resolver_initializer import (
    activation_resolver,
    normalization_resolver,
)
from gnng.nn.models.base import GnnGGNNBase


class BasicGNN(GnnGGNNBase):
    r"""An abstract class for implementing basic GNN distill.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        tape (bool, optional): If True, the output features of all layers
            are recorded in xs and returned. If None, use the tape argument
            passed at init(). (default: :obj:`None`)
        num_pre_lin (int): If >0, a num_pre_lin-layer MLP is used to process
            the node features before GNN layers. (default: :obj:`0`)
        num_post_lin (int): If >0, a num_post_lin-layer MLP is used to process
            the node features after GNN layers. (default: :obj:`0`)
        spec_norm (str, float, int, optional): If None, no spectral normalization
            is applied. If a digit==1, native pytorch spectral normalization towards
            a spectral norm of 1. If a digit==0, actually no spectral normalization.
            If `auto` or a digit from (0,1), an intermediate spectral normalization.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.

    """

    supports_edge_weight: Final[bool]
    supports_edge_attr: Final[bool]
    supports_norm_batch: Final[bool]

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int,
            out_channels: Optional[int] = None,
            dropout: float = 0.0,
            act: Union[str, Callable, None] = "relu",
            act_first: bool = False,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Union[str, Callable, None] = None,
            norm_kwargs: Optional[Dict[str, Any]] = None,
            jk: Optional[str] = None,
            tape: bool = False,
            num_pre_lin: int = 0,
            num_post_lin: int = 0,
            spec_norm: Optional[Union[str, float, int]] = None,
            **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_pre_lin = num_pre_lin
        self.num_post_lin = num_post_lin

        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        self.tape = tape
        self.spec_norm_arg = parse_spec_norm_arg(spec_norm)

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.pre_lins = ModuleList()
        if num_pre_lin > 0:
            self.pre_lins.append(self.init_lin(in_channels, hidden_channels, spec_norm=self.spec_norm_arg))
            for i in range(1, num_pre_lin):
                self.pre_lins.append(self.init_lin(hidden_channels, hidden_channels, spec_norm=self.spec_norm_arg))
            in_channels = hidden_channels

        self.convs = ModuleList()
        if num_layers > 1:
            self.convs.append(self.init_conv(in_channels, hidden_channels, spec_norm=self.spec_norm_arg, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(in_channels, hidden_channels, spec_norm=self.spec_norm_arg, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels

        if out_channels is not None and jk is None and num_post_lin == 0:
            self._is_conv_to_out = True
            self.convs.append(self.init_conv(in_channels, out_channels, spec_norm=self.spec_norm_arg, **kwargs))
        else:
            self._is_conv_to_out = False
            self.convs.append(self.init_conv(in_channels, hidden_channels, spec_norm=self.spec_norm_arg, **kwargs))

        self.post_lins = ModuleList()
        self._is_post_lin_to_out = False
        if num_post_lin > 0:
            for i in range(1, num_post_lin):
                self.post_lins.append(self.init_lin(hidden_channels, hidden_channels, spec_norm=self.spec_norm_arg))

            if out_channels is not None and jk is None:
                self._is_post_lin_to_out = True
                self.post_lins.append(self.init_lin(hidden_channels, out_channels, spec_norm=self.spec_norm_arg))
            else:
                self.post_lins.append(self.init_lin(hidden_channels, hidden_channels, spec_norm=self.spec_norm_arg))

        self.norms = ModuleList()
        norm_layer = normalization_resolver(
            norm,
            False,
            hidden_channels,
            **(norm_kwargs or {}),
        )
        if norm_layer is None:
            norm_layer = torch.nn.Identity()

        self.supports_norm_batch = False
        if hasattr(norm_layer, "forward"):
            norm_params = inspect.signature(norm_layer.forward).parameters
            self.supports_norm_batch = "batch" in norm_params

        for _ in range(num_layers - 1):
            self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None:
            self.norms.append(copy.deepcopy(norm_layer))
        else:
            self.norms.append(torch.nn.Identity())

        if jk is not None and jk != "last":
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == "cat":
                in_channels = (num_layers + num_pre_lin + num_post_lin) * hidden_channels
            else:
                in_channels = hidden_channels
            self.jk_lin = Linear(in_channels, self.out_channels)  # never spectral normalization

            # We define `trim_to_layer` functionality as a module such that we can
        # still use `to_hetero` on-top.
        self._trim = TrimToLayer()
        # after initialize all layers, we have
        self.is_spectral_normalized = True if self.spec_norm_arg is not None else False

    def init_conv(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            spec_norm: Optional[Union[str, float, int]] = None,
            **kwargs,
    ) -> MessagePassing:
        raise NotImplementedError

    def init_lin(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            weight_initializer: Optional[str] = None,
            bias_initializer: Optional[str] = None,
            spec_norm: Optional[Union[str, float, int]] = None,
    ):
        lin = Linear(in_channels, out_channels, bias, weight_initializer, bias_initializer)
        if spec_norm is not None:
            lin = spectral_norm(lin)
        return lin

    def add_sn(self, scale):
        if self.is_spectral_normalized:
            warnings.warn(f"Op Ignored! Already spectral normalized with scale={self.spec_norm_arg}.")
            return
        scale = parse_spec_norm_arg(scale)
        self._add_sn4conv(scale)
        self._add_sn4lin(scale)
        self.is_spectral_normalized = True
        self.spec_norm_arg = scale
        return self

    def rm_sn(self):
        if not self.is_spectral_normalized:
            return
        self._rm_sn4conv()
        self._rm_sn4lin()
        self.is_spectral_normalized = False
        return self

    def _add_sn4conv(self, scale):
        raise NotImplementedError

    def _rm_sn4conv(self):
        raise NotImplementedError

    def _add_sn4lin(self, scale):
        scale = parse_spec_norm_arg(scale)
        if scale is not None:
            for i, lin in enumerate(self.pre_lins):
                self.pre_lins[i] = spectral_norm(lin, "weight", scale=scale)
            for i, lin in enumerate(self.post_lins):
                self.post_lins[i] = spectral_norm(lin, "weight", scale=scale)

    def _rm_sn4lin(self):
        for i, lin in enumerate(self.pre_lins):
            remove_parametrizations(lin, "weight")
        for i, lin in enumerate(self.post_lins):
            remove_parametrizations(lin, "weight")

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        if self.pre_lins is not None:
            for lin in self.pre_lins:
                lin.reset_parameters()
        if self.post_lins is not None:
            for lin in self.post_lins:
                lin.reset_parameters()
        if hasattr(self, "jk"):
            self.jk.reset_parameters()
        if hasattr(self, "jk_lin"):
            self.jk_lin.reset_parameters()

    def forward(  # noqa
            self,
            x: Tensor,
            edge_index: Tensor,  # TODO Support `SparseTensor` in type hint.
            edge_weight: OptTensor = None,
            edge_attr: OptTensor = None,
            batch: OptTensor = None,
            batch_size: Optional[int] = None,
            num_sampled_nodes_per_hop: Optional[List[int]] = None,
            num_sampled_edges_per_hop: Optional[List[int]] = None,
            tape: Optional[bool] = None,
            *args,
            **kwargs,
    ) -> Union[Tensor, List[Tensor]]:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:`~torch_geometric.loaders.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:`~torch_geometric.loaders.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            tape (bool, optional): If True, the output features of all layers
                are recorded in xs and returned. If None, use the tape argument
                passed at init(). (default: :obj:`None`)
        """
        if num_sampled_nodes_per_hop is not None and isinstance(edge_weight, Tensor) and isinstance(edge_attr, Tensor):
            raise NotImplementedError(
                "'trim_to_layer' functionality does not "
                "yet support trimming of both "
                "'edge_weight' and 'edge_attr'"
            )
        tape = self.tape if tape is None else tape
        xs: List[Tensor] = []

        for i, lin in enumerate(self.pre_lins):
            x = lin(x)
            x = self.act(x)
            x = self.dropout(x)
            if hasattr(self, "jk") or tape:
                xs.append(x)

        assert len(self.convs) == len(self.norms)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if num_sampled_nodes_per_hop is not None and not torch.jit.is_scripting():
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)

            if hasattr(self, "jk") or tape:
                xs.append(x)

        if len(self.post_lins) > 0:
            for i, lin in enumerate(self.post_lins[:-1]):
                x = lin(x)
                x = self.act(x)
                x = self.dropout(x)
                if hasattr(self, "jk") or tape:
                    xs.append(x)

            x = self.post_lins[-1](x)

        x = self.jk(xs) if hasattr(self, "jk") else x
        x = self.jk_lin(x) if hasattr(self, "jk_lin") else x

        if tape:  # the final predication is also taped
            xs.append(x)
        return x if not tape else xs

    @torch.no_grad()
    def inference_per_layer(
            self,
            layer: int,
            x: Tensor,
            edge_index: Adj,
            batch_size: int,
    ) -> Tensor:
        x = self.convs[layer](x, edge_index)[:batch_size]

        if layer == self.num_layers - 1 and self.jk_mode is None:
            return x

        if self.act is not None and self.act_first:
            x = self.act(x)
        if self.norms is not None:
            x = self.norms[layer](x)
        if self.act is not None and not self.act_first:
            x = self.act(x)
        if layer == self.num_layers - 1 and hasattr(self, "jk_lin"):
            x = self.jk_lin(x)

        return x

    @torch.no_grad()
    def inference(
            self,
            loader: NeighborLoader,
            device: Optional[Union[str, torch.device]] = None,
            embedding_device: Union[str, torch.device] = "cpu",
            progress_bar: bool = False,
            cache: bool = False,
    ) -> Tensor:
        r"""Performs layer-wise inference on large-graphs using a :class:`~torch_geometric.loaders.NeighborLoader`, where
        :class:`~torch_geometric.loaders.NeighborLoader` should sample the full neighborhood for only one layer. This is
        an efficient way to compute the output embeddings for all nodes in the graph. Only applicable in case
        :obj:`jk=None` or `jk='last'`.

        Args:
            loader (torch_geometric.loader.NeighborLoader): A neighbor loaders
                object that generates full 1-hop subgraphs, *i.e.*,
                :obj:`loaders.num_neighbors = [-1]`.
            device (torch.device, optional): The device to run the GNN on.
                (default: :obj:`None`)
            embedding_device (torch.device, optional): The device to store
                intermediate embeddings on. If intermediate embeddings fit on
                GPU, this option helps to avoid unnecessary device transfers.
                (default: :obj:`"cpu"`)
            progress_bar (bool, optional): If set to :obj:`True`, will print a
                progress bar during computation. (default: :obj:`False`)
            cache (bool, optional): If set to :obj:`True`, caches intermediate
                sampler outputs for usage in later epochs.
                This will avoid repeated sampling to accelerate inference.
                (default: :obj:`False`)

        """
        assert self.jk_mode is None or self.jk_mode == "last"
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.node_sampler.num_neighbors) == 1
        assert not self.training
        # assert not loaders.shuffle  # TODO (matthias) does not work :(

        x_all = loader.data.x.to(embedding_device)

        if cache:
            # Only cache necessary attributes:
            def transform(data: Data) -> Data:
                kwargs = dict(n_id=data.n_id, batch_size=data.batch_size)
                if hasattr(data, "adj_t"):
                    kwargs["adj_t"] = data.adj_t
                else:
                    kwargs["edge_index"] = data.edge_index

                return Data.from_dict(kwargs)

            loader = CachedLoader(loader, device=device, transform=transform)

        for i in range(self.num_pre_lin):  # TODO: replace print with logger
            print(f"preprocess node features with one {self.num_pre_lin}-layer MLP...")
            x_all = self.pre_lins[i](x_all)

        if progress_bar:
            pbar = tqdm(total=len(self.convs) * len(loader))
            pbar.set_description("Graph Inference")

        for i in range(self.num_layers):
            xs: List[Tensor] = []
            for batch in loader:
                x = x_all[batch.n_id].to(device)
                batch_size = batch.batch_size
                if hasattr(batch, "adj_t"):
                    edge_index = batch.adj_t.to(device)
                else:
                    edge_index = batch.edge_index.to(device)

                x = self.inference_per_layer(i, x, edge_index, batch_size)
                xs.append(x.to(embedding_device))

                if progress_bar:
                    pbar.update(1)

            x_all = torch.cat(xs, dim=0)

        if progress_bar:
            pbar.close()

        for i in range(self.num_post_lin):  # TODO: replace print with logger
            print(f"postprocess node features with one {self.num_post_lin}-layer MLP...")
            x_all = self.post_lins[i](x_all)

        print("Inference done")
        return x_all

    def jittable(self, use_sparse_tensor: bool = False) -> "BasicGNN":
        r"""Produces a new jittable instance module that can be used in combination with :meth:`torch.jit.script`."""

        class EdgeIndexJittable(torch.nn.Module):
            def __init__(self, child: BasicGNN):
                super().__init__()
                self.child = child

            def reset_parameters(self):
                self.child.reset_parameters()

            def forward(
                    self,
                    x: Tensor,
                    edge_index: Tensor,
                    edge_weight: OptTensor = None,
                    edge_attr: OptTensor = None,
                    batch: OptTensor = None,
                    batch_size: Optional[int] = None,
                    num_sampled_nodes_per_hop: Optional[List[int]] = None,
                    num_sampled_edges_per_hop: Optional[List[int]] = None,
            ) -> Tensor:
                return self.child(
                    x,
                    edge_index,
                    edge_weight,
                    edge_attr,
                    batch,
                    batch_size,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                )

            def __repr__(self) -> str:
                return str(self.child)

        class SparseTensorJittable(torch.nn.Module):
            def __init__(self, child: BasicGNN):
                super().__init__()
                self.child = child

            def reset_parameters(self):
                self.child.reset_parameters()

            def forward(
                    self,
                    x: Tensor,
                    edge_index: SparseTensor,
                    edge_weight: OptTensor = None,
                    edge_attr: OptTensor = None,
                    batch: OptTensor = None,
                    batch_size: Optional[int] = None,
                    num_sampled_nodes_per_hop: Optional[List[int]] = None,
                    num_sampled_edges_per_hop: Optional[List[int]] = None,
            ) -> Tensor:
                return self.child(
                    x,
                    edge_index,
                    edge_weight,
                    edge_attr,
                    batch,
                    batch_size,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                )

            def __repr__(self) -> str:
                return str(self.child)

        out = copy.deepcopy(self)
        convs = [conv.jittable() for conv in out.convs]
        out.convs = torch.nn.ModuleList(convs)
        out._trim = None  # TODO Trimming is currently not support in JIT mode.

        if use_sparse_tensor:
            return SparseTensorJittable(out)
        return EdgeIndexJittable(out)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.hidden_channels}, {self.out_channels},"
            f" layers={self.num_pre_lin}lins+{self.num_layers}gconvs+{self.num_post_lin}lins"
            f"+{1 if hasattr(self, 'jk_lin') else 0}jk_lin, jk={self.jk_mode}, tape={self.tape}, "
            f"spectral_normalized={self.is_spectral_normalized})"
        )


class GCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """

    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(
            self, in_channels: int, out_channels: int, spec_norm: Optional[Union[str, float, int]] = None, **kwargs
    ) -> MessagePassing:
        conv = GCNConv(in_channels, out_channels, **kwargs)
        if spec_norm is not None:
            conv.lin = spectral_norm(conv.lin, scale=spec_norm)
        return conv

    def _add_sn(self, scale):
        for conv in self.convs:
            conv.lin = spectral_norm(conv.lin, "weight", scale=parse_spec_norm_arg(scale))

    def _rm_sn(self):
        for conv in self.convs:
            remove_parametrizations(conv.lin, "weight")


class GraphSAGE(BasicGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            spec_norm: Optional[Union[str, float, int]] = None,
            **kwargs,
    ) -> MessagePassing:
        conv = SAGEConv(in_channels, out_channels, **kwargs)
        if spec_norm is not None:
            conv.lin_l = spectral_norm(conv.lin_l)
            if conv.root_weight:
                conv.lin_r = spectral_norm(conv.lin_r, scale=spec_norm)
        return conv

    def _add_sn(self, scale):
        scale = parse_spec_norm_arg(scale)
        for conv in self.convs:
            conv.lin_l = spectral_norm(conv.lin_l, scale=scale)
            if conv.root_weight:
                conv.lin_r = spectral_norm(conv.lin_r, scale=scale)

    def _rm_sn(self):
        for conv in self.convs:
            remove_parametrizations(conv.lin_l, "weight")
            conv.lin_l = spectral_norm(conv.lin_l)
            if conv.root_weight:
                remove_parametrizations(conv.lin_r, "weight")


class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, **kwargs)


class GAT(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(
            self, in_channels: Union[int, Tuple[int, int]], out_channels: int, spec_norm=None, **kwargs
    ) -> MessagePassing:
        v2 = kwargs.pop("v2", False)
        heads = kwargs.pop("heads", 1)
        concat = kwargs.pop("concat", True)
        bias = kwargs.pop("bias", True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, "_is_conv_to_out", False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(
                f"Ensure that the number of output channels of "
                f"'GATConv' (got '{out_channels}') is divisible "
                f"by the number of heads (got '{heads}')"
            )

        if concat:
            out_channels = out_channels // heads

        if v2:  # TODO: reverse this when all GAT baselines are finished
            self.version = "v2"
        else:
            self.version = "v1"
        Conv = GATConv if not v2 else GATv2Conv
        conv = Conv(in_channels, out_channels, heads=heads, concat=concat, dropout=self.dropout.p, bias=bias, **kwargs)
        if spec_norm is not None:
            conv.lin = spectral_norm(conv.lin, scale=spec_norm)
            if conv.lin_edge is not None:
                conv.lin_edge = spectral_norm(conv.lin_edge, scale=spec_norm)
        return conv

    def _add_sn(self, scale):
        for conv in self.convs:
            conv.lin = spectral_norm(conv.lin, "weight", scale=parse_spec_norm_arg(scale))
            if conv.lin_edge is not None:
                conv.lin_edge = spectral_norm(conv.lin_edge, scale=parse_spec_norm_arg(scale))

    def _rm_sn(self):
        for conv in self.convs:
            remove_parametrizations(conv.lin, "weight")
            if conv.lin_edge is not None:
                remove_parametrizations(conv.lin_edge, "weight")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}{self.version}({self.in_channels}, {self.hidden_channels}, {self.out_channels},"
            f" layers={self.num_pre_lin}lins+{self.num_layers}gconvs+{self.num_post_lin}lins"
            f"+{1 if hasattr(self, 'jk_lin') else 0}jk_lin, jk={self.jk_mode}, tape={self.tape}, "
            f"spectral_normalized={self.is_spectral_normalized})"
        )


class PNA(BasicGNN):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.PNAConv`.
    """

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int, spec_norm=None, **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels, **kwargs)


class EdgeCNN(BasicGNN):
    r"""The Graph Neural Network from the `"Dynamic Graph CNN for Learning on
    Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper, using the
    :class:`~torch_geometric.nn.conv.EdgeConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.EdgeConv`.
    """

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int, spec_norm=None, **kwargs) -> MessagePassing:
        mlp = MLP(
            [2 * in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return EdgeConv(mlp, **kwargs)


class SAGE(GraphSAGE):
    __doc__ += GraphSAGE.__doc__
    pass


__all__ = [
    "GCN",
    "GraphSAGE",
    "SAGE",
    "GIN",
    "GAT",
    "PNA",
    "EdgeCNN",
]
