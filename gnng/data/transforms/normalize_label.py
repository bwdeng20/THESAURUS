import torch
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.data.datapipes import functional_transform
from gnng.typing import PyGData
from sklearn.preprocessing import LabelEncoder


@functional_transform("normalize_label")
class NormalizeLabel(BaseTransform):
    def __call__(self, data: PyGData) -> PyGData:
        ori_label = data.get("y", data.get("label"))
        if ori_label is not None:
            return data

        old_y = data.y.cpu().numpy()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(old_y)
        new_y = self.label_encoder.transform(old_y)
        data.y = torch.as_tensor(new_y, device=data.y.device)
        data.num_classes = data.y.max() + 1
        return data
