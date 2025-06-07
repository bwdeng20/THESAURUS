from torch_sparse import eye
from torch_geometric.utils import get_self_loop_attr


def diagonal(data):
    new_data = data.clone()
    num_nodes = new_data.num_nodes
    identity_edge_index = eye(num_nodes)[0]
    raw_edge_index = new_data.get("edge_index", new_data.get("adj_t"))
    new_data["edge_index"] = identity_edge_index
    if new_data.edge_weight is not None:
        new_data.edge_weight = get_self_loop_attr(raw_edge_index, new_data.edge_weight)
    if new_data.edge_attr is not None:
        new_data.edge_attr = get_self_loop_attr(raw_edge_index, new_data.edge_attr)
    return new_data
