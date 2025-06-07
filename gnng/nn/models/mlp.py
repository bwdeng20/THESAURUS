"""Adopted from torch_geometric.nn.models.mlp.py.

What we add:
1)  customized spectral normalization for layers
2)  hidden feature tape for feature distillation
3)  jump knowledge

"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch.nn.utils.parametrize import remove_parametrizations
from gnng.nn.utils.parametrizations import spectral_norm, parse_spec_norm_arg
from gnng.nn.resolver_initializer import (
    activation_resolver,
    normalization_resolver,
)
from .base import GnnGBase


class MLP(GnnGBase):
    r"""A Multi-Layer Perception (MLP) model. There exists two ways to instantiate an :class:`MLP`:

    1. By specifying explicit channel sizes, *e.g.*,

       .. code-block:: python

          mlp = MLP([16, 32, 64, 128])

       creates a three-layer MLP with **differently** sized hidden layers.

    1. By specifying fixed hidden channel sizes over a number of layers,
       *e.g.*,

       .. code-block:: python

          mlp = MLP(in_channels=16, hidden_channels=32, out_channels=128, num_layers=3)

       creates a three-layer MLP with **equally** sized hidden layers.

    Args:
        channel_list (List[int] or int, optional): List of input, intermediate
            and output channels such that :obj:`len(channel_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        in_channels (int, optional): Size of each input sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        hidden_channels (int, optional): Size of each hidden sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        dropout (float or List[float], optional): Dropout probability of each
            hidden embedding. If a list is provided, sets the dropout value per
            layer. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        plain_last (bool, optional): If set to :obj:`False`, will apply
            non-linearity, batch normalization and dropout to the last layer as
            well. (default: :obj:`True`)
        bias (bool or List[bool], optional): If set to :obj:`False`, the module
            will not learn additive biases. If a list is provided, sets the
            bias per layer. (default: :obj:`True`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        spec_norm (str, float, int, optional): If None, no spectral normalization
            is applied. If a digit==1, native pytorch spectral normalization towards
            a spectral norm of 1. If a digit==0, actually no spectral normalization.
            If `auto` or a digit from (0,1), an intermediate spectral normalization.
            (default: :obj:`None`)
        **kwargs (optional): Additional deprecated arguments of the MLP layer.

    """

    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Union[float, List[float]] = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        plain_last: bool = True,
        bias: Union[bool, List[bool]] = True,
        jk: Optional[str] = None,
        tape: bool = False,
        spec_norm: Optional[Union[str, float, int]] = None,
        **kwargs,
    ):
        super().__init__()

        # Backward compatibility:
        act_first = act_first or kwargs.get("relu_first", False)
        batch_norm = kwargs.get("batch_norm", None)
        if batch_norm is not None and isinstance(batch_norm, bool):
            warnings.warn("Argument `batch_norm` is deprecated, " "please use `norm` to specify normalization layer.")
            norm = "batch_norm" if batch_norm else None
            batch_norm_kwargs = kwargs.get("batch_norm_kwargs", None)
            norm_kwargs = batch_norm_kwargs or {}

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_channels is None:
                raise ValueError(f"Argument `hidden_channels` must be given " f"for `num_layers={num_layers}`")
            if out_channels is None:
                raise ValueError("Argument `out_channels` must be given")

            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = activation_resolver(act, return_cls=False, **(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
            if plain_last:
                dropout[-1] = 0.0
        if len(dropout) != len(channel_list) - 1:
            raise ValueError(
                f"Number of dropout values provided ({len(dropout)} does not "
                f"match the number of layers specified "
                f"({len(channel_list) - 1})"
            )
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list) - 1})"
            )

        self.jk_mode = jk
        self.tape = tape
        self.spec_norm_arg = parse_spec_norm_arg(spec_norm)

        self.lins = torch.nn.ModuleList()
        if jk is not None:
            channel_list[-1] = hidden_channels
        iterator = zip(channel_list[:-1], channel_list[1:], bias)
        for in_channels, out_channels, _bias in iterator:
            self.lins.append(self.init_lin(in_channels, out_channels, bias=_bias, spec_norm=self.spec_norm_arg))

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if norm is not None:
                norm_layer = normalization_resolver(
                    norm,
                    False,
                    hidden_channels,
                    **(norm_kwargs or {}),
                )
            else:
                norm_layer = Identity()
            self.norms.append(norm_layer)

        if jk is not None and jk != "last":
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == "cat":
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.jk_lin = Linear(in_channels, self.out_channels)  # never spectral normalization here

        self.reset_parameters()
        self.is_spectral_normalized = True if self.spec_norm_arg is not None else False

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
        if scale is not None:
            for i, lin in enumerate(self.lins):
                self.lins[i] = spectral_norm(lin, "weight", scale=scale)
        self.is_spectral_normalized = True
        self.spec_norm_arg = scale
        return self

    def rm_sn(self):
        if not self.is_spectral_normalized:
            return
        for i, lin in enumerate(self.pre_lins):
            remove_parametrizations(lin, "weight")
        self.is_spectral_normalized = False
        return self

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        if hasattr(self, "jk"):
            self.jk.reset_parameters()
        if hasattr(self, "jk_lin"):
            self.jk_lin.reset_parameters()

    def forward(self, x: Tensor, tape: Optional[bool] = None, *args, **kwargs) -> Tensor:
        tape = self.tape if tape is None else tape
        xs: List[Tensor] = []

        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            if hasattr(self, "jk") or tape:
                xs.append(x)

        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)
            if hasattr(self, "jk") or tape:
                xs.append(x)

        x = self.jk(xs) if hasattr(self, "jk") else x
        x = self.jk_lin(x) if hasattr(self, "jk_lin") else x
        if tape:  # the final predication of MLP or JK Linear is also taped
            xs.append(x)
        return xs if tape else x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({str(self.channel_list)[1:-1]}, "
            f"+{1 if hasattr(self, 'jk_lin') else 0}jk_lin, "
            f"jk={self.jk_mode}, tape={self.tape}, "
            f"spectral_normalized={self.is_spectral_normalized})"
        )
