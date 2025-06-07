from .base import GnnGBase, GnnGGNNBase
from .mlp import MLP
from .basic_gnn import GCN, GraphSAGE, GIN, GAT, PNA, EdgeCNN, BasicGNN, SAGE
from .gprgnn import GPRGNN
from .contrastive import ContrastiveModel, ContrastivePyGModel
from .clustering import KMeans, SupKMeans

NonGNNs = ["MLP", "KMeans", "SupKMeans"]
GNNs = ["BasicGNN", "GCN", "SAGE", "GraphSAGE", "GIN", "GAT", "PNA", "EdgeCNN", "GPRGNN"]
__all__ = classes = ["GnnGGNNBase", "GnnGBase"] + NonGNNs + GNNs + ["ContrastiveModel", "ContrastivePyGModel"]
