import logging
import os.path as osp
from pathlib import Path
from typing import Callable, Optional, Union, Dict, Any
from gnng.typing import PathStr
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import *
from gnng.utils import index2mask

logger = logging.getLogger(__file__)


class UnifiedNodeEdgeLevelDataset:
    PlanetoidDataNames = ["cora", "citeseer", "pubmed"]
    OgbnDataNames = ["ogbn-arxiv", "ogbn-products", "ogbn-proteins", "arxiv", "products", "proteins"]
    OtherDataNames = ["reddit", "reddit2", "polblogs"]
    AmazonDataNames = ["a-computers", "a-photo", "amazon-computers", "amazon-photo"]
    WikiDataNames = ["chameleon", "squirrel"]
    WebKBDataNames = ["cornell", "texas", "wisconsin"]
    CoauthorDataNames = ["coauthor-cs", "coauthor-physics", "c-cs", "c-phy"]
    CitationFUllDataNames = ["full-cora", "full-cora_ml", "full-citeseer", "full-dblp", "full-pubmed"]
    HeteroDataNames = ["roman-empire", "amazon-ratings", "minesweeper", "tolokers"]
    SupportedNodeDatasetNames = (
        PlanetoidDataNames
        + OtherDataNames
        + OgbnDataNames
        + AmazonDataNames
        + WikiDataNames
        + WebKBDataNames
        + CoauthorDataNames
        + CitationFUllDataNames
        +HeteroDataNames
    )

    WoLabelDataNames = AmazonDataNames

    def __init__(
        self,
        name: str,
        root: PathStr,
        transform: Optional[Union[str, Callable]] = None,
        pre_transform: Optional[Union[str, Callable]] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.name = name.lower()
        self.root = str(root)
        self.transform = transform
        self.pre_transform = pre_transform
        self.read_dataset = None
        self.dataset_kwargs = dataset_kwargs or {}

    def __repr__(self):
        desc = (
            f"{self.__class__.__name__}(name={self.name}, transform={self.transform}, "
            f"pre_transform={self.pre_transform})"
        )
        return desc

    @property
    def num_classes(self):
        return self.read_dataset.num_classes

    @property
    def num_features(self):
        return self.read_dataset.num_features

    def get_pyg_data(self):  # for node-level task
        # The "Data" of PyG built-in datasets has "train_mask", "val_mask",
        # "test_mask" that can be converted to indices by "LightningDataModule".
        if self.name in self.PlanetoidDataNames:
            self.read_dataset = (
                Planetoid(
                    self.root,
                    self.name,
                    "public",
                    transform=self.transform,
                    pre_transform=self.pre_transform,
                    **self.dataset_kwargs,
                )
                if self.read_dataset is None
                else self.read_dataset
            )
            pyg_data = self.read_dataset[0]

        elif self.name in self.OgbnDataNames:
            name = self.name
            if "ogbn-" not in self.name:
                name = f"ogbn-{name}"
            self.read_dataset = (
                PygNodePropPredDataset(name, self.root, self.transform, self.pre_transform, **self.dataset_kwargs)
                if self.read_dataset is None
                else self.read_dataset
            )
            pyg_data = self.read_dataset[0]
            pyg_data = self.ogb_idx_split2mask(self.read_dataset.get_idx_split(), pyg_data)

        elif self.name in self.OtherDataNames:
            if self.name == "reddit":
                self.read_dataset = (
                    Reddit(osp.join(self.root, "reddit"), self.transform, self.pre_transform, **self.dataset_kwargs)
                    if self.read_dataset is None
                    else self.read_dataset
                )
            elif self.name == "reddit2":  # a sparser version of reddit
                self.read_dataset = (
                    Reddit2(osp.join(self.root, "reddit2"), self.transform, self.pre_transform, **self.dataset_kwargs)
                    if self.read_dataset is None
                    else self.read_dataset
                )
            elif self.name == "polblogs":
                self.read_dataset = (
                    PolBlogs(osp.join(self.root, "polblogs"), self.transform, self.pre_transform, **self.dataset_kwargs)
                    if self.read_dataset is None
                    else self.read_dataset
                )
            else:
                raise ValueError
            pyg_data = self.read_dataset[0]
        elif self.name in self.AmazonDataNames:
            self.read_dataset = Amazon(
                self.root,
                self.name.split("-")[-1],  # "photo" pr "computers"
                self.transform,
                self.pre_transform,
                **self.dataset_kwargs,
            )
            pyg_data = self.read_dataset[0]
        elif self.name in self.CoauthorDataNames:
            self.read_dataset = Coauthor(
                self.root,
                self.name.split("-")[-1],  # "photo" pr "computers"
                self.transform,
                self.pre_transform,
                **self.dataset_kwargs,
            )
            pyg_data = self.read_dataset[0]

        elif self.name in self.WikiDataNames:
            self.read_dataset = (
                WikipediaNetwork(self.root, self.name, True, self.transform, self.pre_transform, **self.dataset_kwargs)
                if self.read_dataset is None
                else self.read_dataset
            )
            pyg_data = self.read_dataset[0]

        elif self.name in self.WebKBDataNames:
            self.read_dataset = (
                WebKB(self.root, self.name, self.transform, self.pre_transform, **self.dataset_kwargs)
                if self.read_dataset is None
                else self.read_dataset
            )
            pyg_data = self.read_dataset[0]

        elif self.name in self.CitationFUllDataNames:
            name = self.name.split("-")[-1]  # "full_cora" --> "cora"
            root = osp.join(self.root, "FullCitation")
            self.read_dataset = (
                CitationFull(str(root), name, self.transform, self.pre_transform, **self.dataset_kwargs)
                if self.read_dataset is None
                else self.read_dataset
            )
            pyg_data = self.read_dataset[0]

        else:
            raise ValueError(f"{self.name} is not a supported dataset from {self.SupportedNodeDatasetNames}")
        pyg_data["name"] = self.name
        pyg_data["num_classes"] = self.read_dataset.num_classes
        # some datasets, e.g., ogbn-arxiv has labels of shape (N,1)
        pyg_data.y.squeeze_(-1)
        return pyg_data

    def get_pyg_dataset(self):  # TODO cope with graph-level
        raise NotImplementedError

    @staticmethod
    def ogb_idx_split2mask(ogb_idx_split, pyg_data):
        """
        TODO: Support Heterogenerous
        """

        def normalize_valid_name(k):
            return k if k != "valid" else "val"

        num_nodes = pyg_data.num_nodes
        for k, idx in ogb_idx_split.items():
            nk = normalize_valid_name(k) + "_mask"
            pyg_data[nk] = index2mask(idx, num_nodes)
        return pyg_data

    @staticmethod
    def dump_pyg_data2npz(data, dump_root, dump_name):
        dump_root = Path(dump_root)
        dump_root.mkdir(exist_ok=True, parents=True)
        file_path = dump_root / f"{dump_name}.npz"

        edge_index = data.edge_index.cpu().numpy()
        x = data.x.cpu().numpy()
        y = data.y.cpu().numpy()

        train_mask = data.train_mask.cpu().numpy() if hasattr(data, "train_mask") else None
        val_mask = data.val_mask.cpu().numpy() if hasattr(data, "val_mask") else None
        test_mask = data.test_mask.cpu().numpy() if hasattr(data, "test_mask") else None
        edge_attr = data.edge_attr.cpu().numpy() if hasattr(data, "edge_attr") else None

        np.savez(
            file_path,
            edge_index=edge_index,
            node_attr=x,
            node_label=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            edge_attr=edge_attr,
        )

    def __getitem__(self, idx):
        assert idx == 0, "Only node level is supported now. So every dataset contains only one graph."
        return self.get_pyg_data()
