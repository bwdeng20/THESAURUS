from typing import List, Union, Dict, Optional, Callable, Any
from torch_geometric.typing import EdgeType
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.sampler import NeighborSampler
from dataclasses import dataclass
from lightning.pytorch.cli import LightningCLI

ShaDowKHopSamplerConfigDict = {
    "depth": None,  # 填写实际的值
    "num_neighbors": None,  # 填写实际的值
    "replace": False
}

SAGESamplerConfigDict = {
    "num_neighbors": [],  # 可以是 List[int] 或 Dict[EdgeType, List[int]]
    "replace": False,
    "subgraph_type": "directional",
    "disjoint": False,
    "temporal_strategy": "uniform",
    "time_attr": None,  # 可选字段
    "weight_attr": None,  # 可选字段
    "transform": None,  # 可选字段
    "transform_sampler_output": None,  # 可选字段
    "is_sorted": False,
    "filter_per_worker": None,  # 可选字段
    "neighbor_sampler": None  # 可选字段
}

NativeTorchLoaderConfig_dict = {
    "batch_size": 1,
    "num_workers": 0,
    "pin_memory": True,
    "drop_last": False,
    "shuffle": False,
    "persistent_workers": None  # 可选字段
}


@dataclass
class SAGESamplerConfig:
    num_neighbors: Union[List[int], Dict[EdgeType, List[int]]]
    # input_nodes: InputNodes = None
    # input_time: OptTensor = None
    replace: bool = False
    subgraph_type: str = "directional"
    disjoint: bool = False
    temporal_strategy: str = "uniform"
    time_attr: Optional[str] = None
    weight_attr: Optional[str] = None
    transform: Optional[Callable] = None
    transform_sampler_output: Optional[Callable] = None
    is_sorted: bool = False
    filter_per_worker: Optional[bool] = None
    neighbor_sampler: Optional[NeighborSampler] = None


@dataclass
class NativeTorchLoaderConfig:
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = False
    persistent_workers: Optional[bool] = None


@dataclass
class LoaderConfig:
    name: str = "full"
    torch: Optional[Dict[str, Any]] = None
    shadowhop: Optional[Dict[str, Any]] = None
    sage_sampler: Optional[Dict[str, Any]] = None
