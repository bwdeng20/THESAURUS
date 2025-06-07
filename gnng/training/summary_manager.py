import torch
from torch import Tensor
from numpy import ndarray
from collections.abc import MutableMapping


class SummaryManager(MutableMapping):
    def __init__(self, init_dict=None):
        self.data = {"scalars": {}, "tensors": {}, "objects": {}}
        init_dict = init_dict or {}
        self.add_elems(init_dict)

    def __getitem__(self, key):
        for category in self.data.values():
            if key in category:
                return category[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(value, (int, float)):
            self.data["scalars"][key] = torch.tensor(value)
        elif isinstance(value, Tensor):
            if value.numel() == 1:
                self.data["scalars"][key] = value
            else:
                self.data["tensors"][key] = value
        elif isinstance(value, ndarray):
            if value.size == 1:
                self.data["scalars"][key] = value
            else:
                self.data["tensors"][key] = torch.from_numpy(value)
        elif isinstance(value, (list, tuple)):
            if len(value) == 1:
                self.data["scalars"][key] = value[0]
            else:
                self.data["tensors"][key] = torch.tensor(value)
        else:
            self.data["objects"][key] = value

    def __delitem__(self, key):
        for category in self.data.values():
            if key in category:
                del category[key]
                return
        raise KeyError(key)

    def __iter__(self):
        for category in self.data.values():
            for key in category:
                yield key

    def __len__(self):
        return sum(len(category) for category in self.data.values())

    def add_elem(self, key, value):
        self[key] = value  # Use __setitem__ for consistency and reusability

    def add_elems(self, kv_dict, v_source: str = None):
        for key, value in kv_dict.items():
            self[key] = value

    def get_summary(self, *data_types):
        combined_summary = {}
        for data_type in data_types:
            if data_type in self.data:
                combined_summary.update(self.data[data_type])
            else:
                raise ValueError("Invalid data type specified. Use 'scalars', 'tensors', or 'objects'.")
        return combined_summary

    def __repr__(self):
        summary_repr = f"{self.__class__.__name__}("
        category_details = []
        for category_name, category in self.data.items():
            category_repr = "\n\t"
            items_details = [f"{key}: {self.describe_value(value)}" for key, value in category.items()]
            category_repr += f"{category_name}={ {', '.join(items_details)} }"
            category_details.append(category_repr)
        summary_repr += ", ".join(category_details) + "\n)"
        return summary_repr

    @staticmethod
    def describe_value(value):
        if isinstance(value, Tensor):
            return f"{value.dtype}Tensor{tuple(value.shape)}"
        elif isinstance(value, float):
            return f"{value: .4f}"
        elif isinstance(value, int):
            return f"{value : 4d}"
        else:
            return f"{type(value).__name__}"
