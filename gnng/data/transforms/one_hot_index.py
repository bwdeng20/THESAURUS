import torch
import warnings
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.data.datapipes import functional_transform
from gnng.typing import PyGData


@functional_transform("one_hot_index")
class OneHotIndex(BaseTransform):
    def __call__(self, data: PyGData) -> PyGData:
        x = data.get("x", data.get("node_features", data.get("node_fea")))

        if x is not None:
            warnings.warn(f"The input {type(data)} has attribute `x`, concat it with one-hot index node features.")
            one_hot_emb = torch.eye(data.num_nodes, dtype=x.dtype, device=data.y.device)
            new_x = torch.concat([x, one_hot_emb])
        else:
            new_x = torch.eye(data.num_nodes, dtype=torch.float32, device=data.y.device)
        data.x = new_x
        return data
