from .identity import Identity

from .gatedgcn_layer import GatedGCN_Player
from .gcn_conv_layer_e_p import GCNConvWithEdges, GCNWithEdges_Player
from .gcn_conv_layer_p import GCN_Player
from .gine_conv_layer_p import GINEConvESLapPE, GINE_Player

# see "Unlocking the Potential of Classic GNNs for Graph-level Tasks: Simple Architectures Meet Excellence"
__enhanced_layers__ = ["GatedGCN_Player", "GCNWithEdges_Player",
                       "GCN_Player", "GINE_Player"]

__all__ = ["Identity"] + __enhanced_layers__
