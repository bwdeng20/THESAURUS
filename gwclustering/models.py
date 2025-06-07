from typing import Any, Optional, Dict, Union

import torch
from torch.nn import Module
from torch import Tensor
from gnng.nn.models.clustering import KMeans
from gnng.ow.models import UNOHead, GCDBase
class GWGCDHead(UNOHead):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            bb_out_channels: int,
            num_heads: int = 1,
            num_layers: int = 1,
            mlp_kwargs: Optional[Dict[str, Any]] = None,
            init_proto_topo: Tensor = None,
            init_proto_weight: Tensor = None,
            proto_topo_identity_weight: float = 0.0,
            proto_topo_momentum: float = 0.99,
            proto_weight_momentum: float = 0.999,
            proto_topo_with_struc: bool = False,
    ):
        super().__init__(in_channels, hidden_channels, out_channels, bb_out_channels, num_heads, num_layers, mlp_kwargs)
        if init_proto_topo is None:
            self.register_buffer("proto_topo", torch.eye(out_channels)[None, :, :].repeat_interleave(num_heads, 0))
        else:
            self.register_buffer("proto_topo", init_proto_topo)

        self.proto_topo_with_struc = proto_topo_with_struc

        self.proto_topo_momentum = proto_topo_momentum
        self.proto_topo_identity_weight = proto_topo_identity_weight

        if init_proto_weight is None:
            self.register_buffer("proto_weight", torch.ones(num_heads, out_channels) / out_channels)
        else:
            self.register_buffer("proto_weight", init_proto_weight)

        self.proto_weight_momentum = proto_weight_momentum

    def forward(self, z, *args, **kwargs):
        # (batch_size, n_hid) --> n_head x (batch_size, n_hid)

        logit = [self.forward_head(h, z) for h in range(self.num_heads)]
        logit = torch.stack(logit)
        data_topo = kwargs.get("data_topo")
        if self.training:
            proto_assign = torch.softmax(logit.detach(), dim=-1) / logit.size(-2)
            new_proto_weight = proto_assign.sum(dim=-2)

            if self.proto_topo_with_struc:
                if isinstance(data_topo, Tensor):
                    new_proto_topo = (proto_assign.transpose(-1, -2).contiguous() @ data_topo) @ proto_assign
                else:
                    new_proto_topo = proto_assign.transpose(-1, -2).contiguous() @ (data_topo @ proto_assign)
            else:
                new_proto_topo = proto_assign.transpose(-1, -2).contiguous() @ proto_assign

            self.update_proto_topo(new_proto_topo / new_proto_topo.max())  # v03
            # self.update_proto_topo(new_proto_topo)  # v02
            self.update_proto_weight(new_proto_weight)

        return logit, self.proto_topo, self.proto_weight

    def update_proto_topo(self, proto_topo: Tensor):
        self.proto_topo = self.proto_topo_momentum * self.proto_topo + (1 - self.proto_topo_momentum) * proto_topo
        self.proto_topo = (
                                  1 - self.proto_topo_identity_weight
                          ) * self.proto_topo + self.proto_topo_identity_weight * torch.eye(
            self.proto_topo.shape[-1], device=self.proto_topo.device
        )[
                                                                                  None, :, :
                                                                                  ]

    def update_proto_weight(self, proto_weight: Tensor):
        self.proto_weight = (
                self.proto_weight_momentum * self.proto_weight + (1 - self.proto_weight_momentum) * proto_weight
        )


class GWClustering(GCDBase):  # v2: SupKmeans/ParametricClassifier after GW gwclustering-guided SSL
    def __init__(
            self, backbone: Module, projector: GWGCDHead, predictor: Union[KMeans, Module], joint_training: bool = True
    ):
        super().__init__(backbone, projector, predictor, joint_training=joint_training)
        self.is_parametric_classifier = isinstance(self.predictor, Module)

    @torch.no_grad()
    def normalize_prototypes(self):
        if hasattr(self.projector, "normalize_prototypes"):
            self.projector.normalize_prototypes()

        if hasattr(self.predictor, "normalize_prototypes"):
            self.predictor.normalize_prototypes()

    def _predict_p(self, z, *args, **kwargs):
        """
        Predicting according to the backbone outputs `z`
        Args:
            z: Tensor of shape (B, H_bb)
                The hidden fetures from backbone models
            *args: args for predictor.forward
            **kwargs: args for predictor.forward
        Returns:
            h: Tensor of shape (B, H_out) or (B,)
                The predicted logits or labels of the interested samples
        """
        if not self.joint_training:
            z = z.detach()
        logit_or_label = self.predictor(z, *args, **kwargs)
        return logit_or_label

    def _predict_nonp(self, z, *args, **kwargs):
        """
        Predicting according to the backbone outputs `z` of entire samples. NO batch style supported
        Args:
            z: Tensor of shape (B, H_bb)
                The hidden features from backbone models
            *args: args for predictor.forward
            **kwargs: args for predictor.forward
        Returns:
            h: Tensor of shape (B, H_out) or (B,)
                The predicted logits or labels of the interested samples
        """
        km_predictor: KMeans = self.predictor
        # you can only use train labels to make predictions
        labelled_mask: Optional["torch.BoolTensor"] = kwargs.get("labelled_mask", None)
        target = kwargs.get("target", None)
        if (
                labelled_mask is not None and target is not None and torch.any(labelled_mask)
        ):  # supervised k-means, the node ids are permuted
            ori_nid = torch.arange(z.shape[0], device=z.device)
            unlabelled_mask = torch.logical_not(labelled_mask)
            l_targets = target[labelled_mask]
            l_feats = z[labelled_mask]
            u_feats = z[unlabelled_mask]
            pred_labels, cluster_centers, inertia, n_iter = km_predictor.fit_mix(u_feats, l_feats, l_targets)
            nid = torch.cat([ori_nid[labelled_mask], ori_nid[unlabelled_mask]], dim=0)
        else:  # unsupervised k-means
            pred_labels, cluster_centers, inertia, n_iter = km_predictor.fit(z)
            nid = None  # No supervised KMeans, no nid permutation

        if nid is not None:
            pred_labels[nid] = pred_labels.clone()  # inverse permutate to original nid
        return {"preds": pred_labels, "centers": cluster_centers, "labelled_mask": labelled_mask}

    def _forward_nonp(self, x, *args, **kwargs):
        """embed(). NonParametricGCD utilizes the backbone embeddings `z` and non-parametric methods such as
        K-means post it. So the model only outputs the backbone embeddings
        """
        z = self.embed(x, *args, **kwargs)
        return None, z

    def _forward_p(self, x, *args, **kwargs):
        """embed + predict"""
        z = self.embed(x, *args, **kwargs)
        input2pred = z if z is not None else x
        pred = self.predict(input2pred, *args, **kwargs)
        return pred, z

    def predict(self, z, *args, **kwargs):
        if self.is_parametric_classifier:
            return self._predict_p(z, *args, **kwargs)
        else:
            return self._predict_nonp(z, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        if self.is_parametric_classifier:
            return self._forward_p(x, *args, **kwargs)
        else:
            return self._forward_nonp(x, *args, **kwargs)


class GWClusteringv2(GWClustering):
    def __init__(self, backbone: Module, projector: GWGCDHead, predictor: KMeans):
        super().__init__(backbone, projector, predictor, joint_training=False)
