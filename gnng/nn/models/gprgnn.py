from typing import Optional, Union, Callable, Dict, Any
import torch.nn.functional as F
from gnng.nn.convs.gpr_conv import GPRConv
from gnng.nn.models.mlp import MLP
from gnng.nn.models.base import GnnGGNNBase


class GPRGNN(GnnGGNNBase):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        out_channels: Optional[int] = None,
        K: int = 10,
        alpha: float = 0.1,
        init: str = "ppr",
        cached: bool = False,
        normalize: bool = True,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        spec_norm: Optional[Union[str, float, int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.init = init.lower()
        self.dropout = dropout
        self.fea_extractor = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            tape=False,
            spec_norm=spec_norm,
        )
        self.gpr_layer = GPRConv(K, alpha, init, cached, normalize, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.fea_extractor.reset_parameters()
        self.gpr_layer.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, *args, **kwargs):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fea_extractor(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gpr_layer(x, edge_index, edge_weight)
        return x

    def __repr__(self):
        return f"GPRGNN(K={self.K}, alpha={self.alpha}, init={self.init}, MLP={repr(self.fea_extractor)})"
