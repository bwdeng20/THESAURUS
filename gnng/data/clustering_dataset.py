from typing import Union, List, Tuple, Optional, Callable
import re
import logging
import os.path as osp
from pathlib import Path
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_undirected as to_undirected_fn

logger = logging.getLogger(__file__)


class ClusteringDataset(InMemoryDataset):
    url = "https://drive.google.com/drive/folders/1thSxtAexbvOyjx-bJre8D4OyFKsBe1bK"

    def __init__(
        self,
        root: Union["Path", str],
        name: str,
        transform: Optional[Union[str, Callable]] = None,
        pre_transform: Optional[Union[str, Callable]] = None,
        pre_filter: Optional[Callable] = None,
        to_undirected: bool = True,
        force_reload: bool = False,
    ):
        self.short_name = name.lower()
        self.name = f"{self.short_name}_clustering"
        self.to_undirected = to_undirected
        root = Path(root) / "ClusteringDatasets"
        super().__init__(str(root), transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])
        data = self.get(0)
        self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.short_name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.short_name, "processed")

    @property
    def processed_file_names(self):
        return [f"{self.name}.pt"]

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"{self.short_name}_feat.npy", f"{self.short_name}_label.npy", f"{self.short_name}_adj.npy"]

    def process(self):
        raw_dir = Path(self.raw_dir)
        loaded_dict = dict()
        for raw_file in self.raw_file_names:
            match = re.search(r"_(.*?)\.", raw_file)
            ft = match.group(1)
            loaded = np.load(raw_dir / raw_file, allow_pickle=True)
            if ft == "adj":
                np.fill_diagonal(loaded, 0)
                loaded = coo_matrix(loaded)
                row = torch.from_numpy(loaded.row).to(torch.long)
                col = torch.from_numpy(loaded.col).to(torch.long)
                edge_index = torch.stack([row, col], dim=0)
                if self.to_undirected:
                    edge_index = to_undirected_fn(edge_index, num_nodes=loaded.shape[0])
                loaded_dict[ft] = edge_index
            elif ft == "feat":
                loaded_dict[ft] = torch.from_numpy(loaded.astype(np.float32))
            elif ft == "label":
                loaded_dict[ft] = torch.from_numpy(loaded.astype(np.int64))
            else:
                raise TypeError
        data = Data(x=loaded_dict["feat"], edge_index=loaded_dict["adj"], y=loaded_dict["label"])
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.short_name})"

    # def download(self):
    #     raise NotImplementedError
