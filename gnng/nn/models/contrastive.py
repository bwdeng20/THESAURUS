import torch.nn as nn
from gnng.nn.models.base import GnnGBase
from gnng.nn.models.clustering import KMeans
from gnng.clustering.base import KMeansBase
from typing import List, Optional, Tuple, Union

from torch import Tensor
from torch_geometric.typing import OptTensor, SparseTensor


class ContrastiveModel(GnnGBase):
    """
    Args:
        backbone: Module
        projector: Module
        predictor: Any
            The non-parametric (e.g., Supvervised K-means) or (e.g., MLP classifier) parametric predictor.
           joint_training: bool
        joint_training: bool
            If True, the backbone also receives the gradients from the predictor (for supervised learning).
            Otherwise, the backbone is only tuned with the gradients from the projector (for contrastive learning).
    """

    def __init__(
            self,
            backbone: Optional[nn.Module] = None,
            projector: Optional[nn.Module] = None,
            predictor: Optional[Union[nn.Module, KMeans, KMeansBase]] = None,
            joint_training: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.predictor = predictor
        self.joint_training = joint_training

    def reset_parameters(self):
        if self.backbone is not None:
            self.backbone.reset_parameters()
        if self.projector is not None:
            self.projector.reset_parameters()
        if self.predictor is not None and isinstance(self.predictor, nn.Module):
            if isinstance(self.predictor, (nn.ModuleList, List)):
                for sub_pred in self.predictor:
                    sub_pred.reset_parameters()
            else:
                self.predictor.reset_parameters()

    def embed(self, x, *args, **kwargs):
        """
        Get the backbone outputs of input features `x`
        Args:
            x: Tensor of shape (B, H_in)
                The input features of samples
            *args:
                Some args required by backbone forward defintion
            **kwargs:
                Some args required by backbone forward defintion
        Returns:
            z: Tensor of shape (B, H_bb)
                The sample embeddings from the backbone model
        """
        z = None
        if self.backbone is not None:
            z = self.backbone(x, *args, **kwargs)
        return z

    def project(self, z, *args, **kwargs):
        """
        Projecting the backbone outputs `z` for contrastive learning
        Args:
            z: Tensor of shape (B, H_bb)
                The hidden fetures from backbone models
            *args: args for projector.forward
            **kwargs: args for projector.forward
        Returns:
            h: Tensor of shape (B, H_con)
                The features for contrastive learning
        """
        h = None
        if self.projector is not None:
            h = self.projector(z, *args, **kwargs)
        return h

    def forward(self, x, *args, **kwargs):
        """embed + predict"""
        z = self.embed(x, *args, **kwargs)
        pred = None
        if self.predictor is not None:
            input2pred = z if z is not None else x
            pred = self.predict(input2pred, *args, **kwargs)
        return pred, z

    def predict(self, z, *args, **kwargs):
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

    def extra_repr(self) -> str:
        return f"joint={self.joint_training}"

    def brief_repr(self) -> str:
        desc = (
            f"{self.__class__.__name__}(joint={self.joint_training},backbone={self.backbone.__class__.__name__},"
            f"predictor={self.predictor.__class__.__name__},"
            f"projector={self.projector.__class__.__name__})"
        )
        return desc


class ContrastivePyGModel(GnnGBase):
    def __init__(self, backbone: nn.Module, projector: nn.Module, predictor: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.predictor = predictor

    def embed(
            self,
            x: Tensor,
            edge_index: Union[SparseTensor, Tensor],
            edge_weight: OptTensor = None,
            edge_attr: OptTensor = None,
            batch: OptTensor = None,
            batch_size: Optional[int] = None,
            num_sampled_nodes_per_hop: Optional[List[int]] = None,
            num_sampled_edges_per_hop: Optional[List[int]] = None,
    ):
        z = self.backbone(
            x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        )
        return z

    def project(
            self,
            z: Tensor,
            edge_index: Union[SparseTensor, Tensor],
            edge_weight: OptTensor = None,
            edge_attr: OptTensor = None,
            batch: OptTensor = None,
            batch_size: Optional[int] = None,
            num_sampled_nodes_per_hop: Optional[List[int]] = None,
            num_sampled_edges_per_hop: Optional[List[int]] = None,
    ):
        h = self.projector(
            z,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        )
        return h

    def forward(
            self,
            x: Tensor,
            edge_index: Union[SparseTensor, Tensor],
            edge_weight: OptTensor = None,
            edge_attr: OptTensor = None,
            batch: OptTensor = None,
            batch_size: Optional[int] = None,
            num_sampled_nodes_per_hop: Optional[List[int]] = None,
            num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tuple[OptTensor, Tensor]:
        z = self.embed(
            x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        )
        pred = None
        if self.predictor is not None:
            pred = self.predictor(
                z,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_attr=edge_attr,
                batch=batch,
                batch_size=batch_size,
                num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
                num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            )
        return pred, z
