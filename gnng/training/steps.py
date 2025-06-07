from typing import Union, Optional, List, Callable, Dict, Sequence

import torchmetrics
from functools import partial
from torch import BoolTensor, LongTensor
from torch.nn import Module, ModuleList
from lightning.fabric import Fabric
from torch._dynamo import OptimizedModule
from gnng.typing import PyGData
from gnng.utils import normalize_md_class_name
from gnng.nn.models import GnnGGNNBase, MLP
from gnng.training.loss_recorder import LossRecorder
from gnng.training.utils import infer_node_mask


def gnn_step_for_pyg_node(
    model: Module,
    data: PyGData,
    loss_fn: Optional[Callable] = None,
    aim_node_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    metric_manager: "torchmetrics.Metric" = None,
    loss_manager: "LossRecorder" = None,
    stage: str = None,  # train/val/test
    *args,
    **kwargs,
):
    if aim_node_mask is not None:
        aim_mask = infer_node_mask(data, aim_node_mask)
    else:
        aim_mask = infer_node_mask(data, stage)

    batch_size = data.get("batch_size", None)
    structure = data.get("adj_t", data.get("edge_index", None))
    edge_weight = data.get("edge_weight", None)
    edge_attr = data.get("edge_attr", None)
    label = data.get("y", None)
    tape_hid = getattr(model, "tape", False)
    if batch_size is None:  # Node-level GNN with Full Graph
        xs = model.forward(data.x, structure, edge_weight, edge_attr)
        logit = xs[-1] if tape_hid else xs
        logit4aim = logit[aim_mask] if aim_mask is not None else logit
        y4aim = label[aim_mask] if label is not None else None

    else:  # Node-level GNN with Neighbor Sampler
        if not isinstance(model, OptimizedModule):  # TODO Trimming is currently not support in JIT mode.
            num_sampled_nodes_per_hop = data.get("num_sampled_nodes_per_hop")
            num_sampled_edges_per_hop = data.get("num_sampled_edges_per_hop")
        else:
            num_sampled_nodes_per_hop = None
            num_sampled_edges_per_hop = None
        xs = model.forward(
            data.x,
            structure,
            edge_weight,
            edge_attr,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        )
        aim_mask = aim_mask[:batch_size] if aim_mask is not None else aim_mask
        logit = xs[-1][:batch_size] if tape_hid else xs[:batch_size]
        logit4aim = logit[aim_mask] if aim_mask is not None else logit
        if label is not None:
            y4aim = label[:batch_size][aim_mask] if aim_mask is not None else label[:batch_size]
        else:
            y4aim = None
    if y4aim is not None and loss_fn is not None:
        loss = loss_fn(logit4aim, y4aim)
    else:
        loss = None

    batch_return = {
        # "logit": logit4aim,
        # "y": y4aim,
        # "aim_mask": aim_mask,
        "loss": loss,
        "batch_size": logit4aim.shape[0],
    }
    loss_manager.update(loss, logit4aim.shape[0])
    batch_metric = metric_manager(logit4aim.detach(), y4aim)
    batch_return.update(batch_metric)
    return batch_return


def gnn_step_for_pyg_graph(
    model: Module,
    data: PyGData,
    loss_fn: Optional[Callable] = None,
    aim_graph_mask: Optional[Union[BoolTensor, LongTensor]] = None,
    batch: Optional[LongTensor] = None,
    metric_manager: "torchmetrics.Metric" = None,
    loss_manager: "LossRecorder" = None,
    stage: str = None,  # train/val/test
    *args,
    **kwargs,
):
    batch = batch if batch is not None else data.get("batch")
    edge_weight = data.get("edge_weight", None)
    edge_attr = data.get("edge_attr", None)
    structure = data.get("adj_t", data.get("edge_index", None))
    label = data.get("y", None)
    tape_hid = getattr(model, "tape", False)
    xs = model(data.x, structure, edge_weight, edge_attr, batch)
    logit = xs[-1] if tape_hid else xs
    logit4aim = logit[aim_graph_mask] if aim_graph_mask is not None else logit
    y4aim = label if label is not None else None

    if y4aim is not None and loss_fn is not None:
        loss = loss_fn(logit4aim, y4aim)
    else:
        loss = None
    loss_manager.update(loss, logit4aim.shape[0])
    batch_return = {
        # "logit": logit4aim,
        # "y": y4aim,
        # "aim_mask": aim_graph_mask,
        "loss": loss,
        "batch_size": logit4aim.shape[0],
    }
    batch_metric = metric_manager(logit4aim.detach(), y4aim)

    batch_return.update(batch_metric)
    return batch_return


def mlp_step_for_pyg_node(
    model: Module,
    data: PyGData,
    loss_fn: Optional[Callable] = None,
    aim_node_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    metric_manager: "torchmetrics.Metric" = None,
    loss_manager: "LossRecorder" = None,
    stage: str = None,  # train/val/test
    *args,
    **kwargs,
):
    if aim_node_mask is not None:
        aim_mask = infer_node_mask(data, aim_node_mask)
    else:
        aim_mask = infer_node_mask(data, stage)
    batch_size = data.get("batch_size", None)
    label = data.get("y", None)
    tape_hid = getattr(model, "tape", False)
    if batch_size is None:  # full
        x = data.get("x", data.get("node_fea", data.get("node_features")))
    else:
        aim_mask = aim_mask[:batch_size] if aim_mask is not None else None
        x = data.x[:batch_size]
        label = label[:batch_size] if label is not None else None

    xs = model.forward(x)
    logit = xs[-1] if tape_hid else xs

    logit4aim = logit[aim_mask] if aim_mask is not None else logit
    if label is not None:
        y4aim = label[aim_mask] if aim_mask is not None else label
    else:
        y4aim = None

    if y4aim is not None and loss_fn is not None:
        loss = loss_fn(logit4aim, y4aim)
    else:
        loss = None
    loss_manager.update(loss, logit4aim.shape[0])
    batch_return = {
        # "logit": logit4aim,
        # "y": y4aim,
        # "aim_mask": aim_mask,
        "loss": loss,
        "batch_size": logit4aim.shape[0],
    }
    batch_metric = metric_manager(logit4aim.detach().detach(), y4aim)

    batch_return.update(batch_metric)
    return batch_return


def mlp_step_for_pyg_graph(
    model: Module,
    data: PyGData,
    loss_fn: Optional[Callable] = None,
    aim_graph_mask: Optional[Union[BoolTensor, LongTensor]] = None,
    batch: Optional[LongTensor] = None,
    metric_manager: "torchmetrics.Metric" = None,
    loss_manager: "LossRecorder" = None,
    stage: str = None,  # train/val/test
    *args,
    **kwargs,
):
    """TODO: implement once we are building the first Graph-level distillation"""
    raise NotImplementedError


def mlp_step_for_pyg(
    model: Module,
    data: PyGData,
    loss_fn: Optional[Callable] = None,
    aim_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    metric_manager: "torchmetrics.Metric" = None,
    loss_manager: "LossRecorder" = None,
    stage: str = None,  # train/val/test
    fabric: Fabric = None,
    *args,
    **kwargs,
):
    if fabric is not None:
        data = fabric.to_device(data)
    batch = data.get("batch", None)
    if batch is not None:  # graph-level GNN with concatenated multiple graphs
        return mlp_step_for_pyg_graph(
            model, data, loss_fn, aim_mask, batch, metric_manager, loss_manager, stage, *args, **kwargs
        )

    else:  # node-level GNN with full graph or pyg neighbor loaders
        return mlp_step_for_pyg_node(
            model, data, loss_fn, aim_mask, metric_manager, loss_manager, stage, *args, **kwargs
        )


def gnn_step_for_pyg(
    model: Module,
    data: PyGData,
    loss_fn: Optional[Callable] = None,
    aim_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    metric_manager: "torchmetrics.Metric" = None,
    loss_manager: "LossRecorder" = None,
    stage: str = None,  # train/val/test
    fabric: Fabric = None,
    *args,
    **kwargs,
):
    if fabric is not None:
        data = fabric.to_device(data)
    batch = data.get("batch", None)
    if batch is not None:  # graph-level GNN with concatenated multiple graphs
        return gnn_step_for_pyg_graph(
            model, data, loss_fn, aim_mask, batch, metric_manager, loss_manager, stage, *args, **kwargs
        )

    else:  # node-level GNN with full graph or pyg neighbor loaders
        return gnn_step_for_pyg_node(
            model, data, loss_fn, aim_mask, metric_manager, loss_manager, stage, *args, **kwargs
        )


train_gnn_step_for_pyg = partial(gnn_step_for_pyg, aim_mask="train")
val_gnn_step_for_pyg = partial(gnn_step_for_pyg, aim_mask="val")
test_gnn_step_for_pyg = partial(gnn_step_for_pyg, aim_mask="test")


def plain_xy_step(
    model,
    batch_data,
    loss_fn: Optional[Callable] = None,
    metric_manager: "torchmetrics.Metric" = None,
    loss_manager: "LossRecorder" = None,
    stage: str = None,  # train/val/test
    *args,
    **kwargs,
):
    x, y = batch_data
    logits = model(x)
    if loss_fn is not None:
        loss = loss_fn(logits, y)
    else:
        loss = None
    loss_manager.update(loss, logits.shape[0])

    batch_return = {
        "batch_size": logits.shape[0],
        # "logit": logits,
        # "y": y, "loss": loss,
    }
    batch_metric = metric_manager(logits, y)
    batch_return.update(batch_metric)
    return batch_return


def configure_step(model, func: Optional[Callable] = None) -> Callable:
    is_compiled = isinstance(model, OptimizedModule)
    ravel_model = model._orig_mod if is_compiled else model
    if func is not None:
        return_func = func
    elif isinstance(ravel_model, GnnGGNNBase):
        return_func = gnn_step_for_pyg
    elif isinstance(ravel_model, MLP):
        return_func = mlp_step_for_pyg
    else:
        return_func = plain_xy_step
    return return_func


def configure_many_step(
    model_list: Union[ModuleList, Sequence[Module]],
    customized_step_func_dict: Optional[Dict[str, Callable]] = None,
) -> List[Callable]:
    step_dict = customized_step_func_dict or {}
    step_impl_list: List[Optional[Callable]] = []
    for i, md in enumerate(model_list):
        func = step_dict.get(md.__class__.__name__, step_dict.get(normalize_md_class_name(md)))
        step_impl_list.append(configure_step(md, func))
    return step_impl_list
