from typing import Any, Optional, Dict, Union

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn.dense.linear import Linear
from gnng.nn.models.clustering import KMeans
from gnng.clustering import TorchSemiKMeans
from gnng.nn.models.mlp import MLP
from gnng.nn import ContrastiveModel


class GCDBase(ContrastiveModel):
    pass


class NonParametricGCD(GCDBase):
    def forward(self, x, *args, **kwargs):
        """embed(). NonParametricGCD utilizes the backbone embeddings `z` and non-parametric methods such as
        K-means post it. So the model only outputs the backbone embeddings
        """
        z = self.embed(x, *args, **kwargs)
        return None, z


class VanillaGCD(NonParametricGCD):
    def predict(self, z, *args, **kwargs):
        km_predictor: Union[TorchSemiKMeans, KMeans] = self.predictor
        # you can only use train labels to make predictions
        labelled_mask: Optional["torch.BoolTensor"] = kwargs.get("y_access_mask", None)
        target = kwargs.get("target", None)
        if (
                labelled_mask is not None and target is not None and torch.any(labelled_mask)
        ):  # supervised k-means, the node ids are permuted
            ori_nid = torch.arange(z.shape[0], device=z.device)
            unlabelled_mask = torch.logical_not(labelled_mask)
            l_targets = target[labelled_mask]
            l_feats = z[labelled_mask]
            u_feats = z[unlabelled_mask]
            pred_labels, cluster_centers = km_predictor.fit_predict(u_feats, l_feats, l_targets, return_centroids=True)
            nid = torch.cat([ori_nid[labelled_mask], ori_nid[unlabelled_mask]], dim=0)
        else:  # unsupervised k-means
            pred_labels, cluster_centers = km_predictor.fit_predict_unsup(z)
            nid = None  # No supervised KMeans, no nid permutation

        if nid is not None:
            pred_labels[nid] = pred_labels.clone()  # inverse permutate to original nid
        return {"preds": pred_labels, "centers": cluster_centers, "labelled_mask": labelled_mask}


class ParametricGCD(GCDBase):
    """This must has a parametric predictor (e.g., MLP classifier) with trainable weights."""


class SimGCD(ParametricGCD):
    """This must be used with a SimGCD loss and
    has a parametric predictor (e.g., MLP classifier) with trainable weights."""


class PlainGNNGCD(ParametricGCD):
    """This only has a parametric GNN predictor with trainable weights. Like directly using GNN for GCD"""


class PlainGCD(ParametricGCD):
    """This only has a parametric MLP predictor with trainable weights. Like directly using MLP for GCD"""


class Prototypes(Module):
    def __init__(self, out_channels: int, num_prototypes: int):
        super().__init__()
        self.lin = Linear(out_channels, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.lin.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.lin.weight.copy_(w)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        return self.lin(x)


class UNOHead(Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            bb_out_channels: int,
            num_heads: int = 1,
            num_layers: int = 1,
            mlp_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        # MLP backbone of projectors
        mlp_kwargs = mlp_kwargs or {}
        self.projectors = torch.nn.ModuleList(
            [
                MLP(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=bb_out_channels,
                    num_layers=num_layers,
                    **mlp_kwargs,
                )
                for _ in range(num_heads)
            ]
        )
        self.out_channels = out_channels

        # Linear prototype classifier
        self.prototypes = torch.nn.ModuleList([Prototypes(bb_out_channels, out_channels) for _ in range(num_heads)])
        self.normalize_prototypes()

    def reset_parameters(self):
        for proj in self.projectors:
            proj.reset_parameters()
        for proto in self.prototypes:
            proto.reset_parameters()
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=1)
        # if not self.training:  # only for hack in GW clustering exp
        #     self.z11 = z.detach().clone()
        return self.prototypes[head_idx](z)

    def forward(self, z, head_train_loss_record=None, *args, **kwargs):
        # (batch_size, n_hid) --> n_head x (batch_size, n_hid)
        logit = [self.forward_head(h, z) for h in range(self.num_heads)]
        if head_train_loss_record is not None:  # in evaluation mode, pick the best head to predicate
            best_head_idx = torch.argmin(head_train_loss_record)
            return logit[best_head_idx]
        return torch.stack(logit)


class UNO(ParametricGCD):
    def __init__(self, backbone: Module, predictor: UNOHead, joint_training: bool = True):
        super().__init__(backbone, None, predictor, joint_training)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.predictor.normalize_prototypes()


class WNP(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.last_layer = torch.nn.utils.weight_norm(torch.nn.Linear(in_channels, out_channels, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def reset_parameters(self):
        torch.nn.utils.remove_weight_norm(self.last_layer)
        self.last_layer.reset_parameters()
        torch.nn.utils.weight_norm(self.last_layer)
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x, *args, **kwargs):
        x = torch.nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return logits


class WNP2(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.last_layer = torch.nn.utils.weight_norm(torch.nn.Linear(in_channels, out_channels, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)

    def reset_parameters(self):
        torch.nn.utils.remove_weight_norm(self.last_layer)
        self.last_layer.reset_parameters()
        torch.nn.utils.weight_norm(self.last_layer)
        self.last_layer.weight_g.data.fill_(1.0)

    def forward(self, x, *args, **kwargs):
        x = torch.nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return logits


class CASMLP(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 1024, norm_layer=True):
        super().__init__()
        self.f1 = torch.nn.utils.weight_norm(torch.nn.Linear(in_channels, hidden_channels, bias=False))
        self.f2 = torch.nn.Linear(hidden_channels, out_channels, bias=False)
        self.f1.weight_g.data.fill_(1)
        if norm_layer:
            self.f1.weight_g.requires_grad = False

    def forward(self, x, *args, **kwargs):
        x = torch.nn.functional.normalize(x, dim=-1, p=2)
        x_semantic = self.f1(x)
        logits = self.f2(x_semantic)
        return logits

    def reset_parameters(self):
        # 移除 weight_norm
        torch.nn.utils.remove_weight_norm(self.f1)
        # 重置参数
        self.f1.reset_parameters()
        # 重新应用 weight_norm
        torch.nn.utils.weight_norm(self.f1)
        self.f1.weight_g.data.fill_(1)
        self.f1.weight_g.requires_grad = False

        self.f2.reset_parameters()
