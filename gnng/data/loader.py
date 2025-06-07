import copy
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from gnng.data.dataset import SingleElementDataset

def get_loader_from_cfg(data, loader_name: str = "full", input_nodes=None, transform=None, loader_cfg=None):
    native_torch_loader_cfg = loader_cfg.get("torch")
    native_torch_loader_cfg = native_torch_loader_cfg or {}
    data = copy.deepcopy(data)
    if loader_name == "full":
        data = transform(data) if transform is not None else data
        native_torch_loader_cfg["batch_size"] = 1
        native_torch_loader_cfg["num_workers"] = 0
        native_torch_loader_cfg["pin_memory"] = 0
        data_loader = torch.utils.data.DataLoader(
            dataset=SingleElementDataset(data),
            shuffle=False,  # only one PyGData, Don't shuffle
            collate_fn=lambda xs: xs[0],
            **native_torch_loader_cfg,
        )

    elif loader_name == "sage_sampler":
        # sage_sampler requires on-the-fly args `data` and `input_nodes`bsides other in predefined `sage_sampler_cfg`
        sage_sampler_cfg = loader_cfg.get("sage_sampler", {})
        if transform is not None:
            sage_sampler_cfg["transform"] = transform
        data_loader = NeighborLoader(
            data=data,
            input_nodes=input_nodes,
            **sage_sampler_cfg,
            **native_torch_loader_cfg,
        )

    elif loader_name == "gcd_train_sage_sampler":
        # balanced old and new class samples in a training batch with weighted sampler arg of torch.loader
        old_mask = data.old_mask
        old_new_label = old_mask[input_nodes] if input_nodes is not None else old_mask
        old_new_label = old_new_label.int()
        old_new_class_weight = 1.0 / old_new_label.bincount()
        sampling_weight = old_new_class_weight[old_new_label]
        sampler = torch.utils.data.WeightedRandomSampler(
            sampling_weight, num_samples=old_new_label.numel(), replacement=True
        )
        native_torch_loader_cfg["sampler"] = sampler

        sage_sampler_cfg = loader_cfg["sage_sampler"]

        data_loader = NeighborLoader(data=data, input_nodes=input_nodes, **sage_sampler_cfg, **native_torch_loader_cfg)

    else:
        raise NotImplementedError

    return data_loader



def get_loader_from_cfg_new(data, loader_name: str = "full", input_nodes=None, transform=None, loader_cfg=None):
    native_torch_loader_cfg = loader_cfg.get("torch")
    native_torch_loader_cfg = native_torch_loader_cfg or {}
    data = copy.deepcopy(data)
    if loader_name == "full":
        data = transform(data) if transform is not None else data
        native_torch_loader_cfg["batch_size"] = 1
        native_torch_loader_cfg["num_workers"] = 0
        native_torch_loader_cfg["pin_memory"] = 0
        data_loader = torch.utils.data.DataLoader(
            [data],
            shuffle=False,  # only one PyGData, Don't shuffle
            collate_fn=lambda xs: xs[0],
            **native_torch_loader_cfg,
        )

    elif loader_name == "sage_sampler":
        # sage_sampler requires on-the-fly args `data` and `input_nodes`bsides other in predefined `sage_sampler_cfg`
        sage_sampler_cfg = loader_cfg["sage_sampler"]
        if transform is not None:
            sage_sampler_cfg["transform"] = transform
        data_loader = NeighborLoader(
            data=data,
            input_nodes=input_nodes,
            **sage_sampler_cfg,
            **native_torch_loader_cfg,
        )

    elif loader_name == "gcd_train_sage_sampler":
        # balanced old and new class samples in a training batch with weighted sampler arg of torch.loader
        old_mask = data.old_mask
        old_new_label = old_mask[input_nodes] if input_nodes is not None else old_mask
        old_new_label = old_new_label.int()
        old_new_class_weight = 1.0 / old_new_label.bincount()
        sampling_weight = old_new_class_weight[old_new_label]
        sampler = torch.utils.data.WeightedRandomSampler(
            sampling_weight, num_samples=old_new_label.numel(), replacement=True
        )
        native_torch_loader_cfg["sampler"] = sampler

        sage_sampler_cfg = loader_cfg["sage_sampler"]

        data_loader = NeighborLoader(data=data, input_nodes=input_nodes, **sage_sampler_cfg, **native_torch_loader_cfg)

    else:
        raise NotImplementedError

    return data_loader
