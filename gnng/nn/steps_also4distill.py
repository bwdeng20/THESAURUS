import warnings
from typing import Union, Optional, List, Callable, Dict, Sequence
from torch import Tensor, BoolTensor, LongTensor
from torch.nn import Module, ModuleList
from torch._dynamo import OptimizedModule
from gnng.typing import PyGData, FullChoirOutputElm
from gnng.utils import normalize_md_class_name, index_to_mask
from gnng.nn.models import GnnGGNNBase, MLP


def infer_node_mask(
    data: PyGData, node_mask: Optional[Union[str, BoolTensor, LongTensor]] = None
) -> Optional[BoolTensor]:
    if node_mask is None:
        mask = None
        # mask = torch.ones(datasets.num_nodes, dtype=torch.bool, device=datasets.x.device)
    elif isinstance(node_mask, str):
        mask = data.get(node_mask + "_mask", data.get(node_mask + "_indices", None))
        if mask is None:
            warnings.warn(f"`{node_mask}_mask` or `{node_mask}_indices` not found, return None.")
    elif isinstance(node_mask, LongTensor):
        mask = index_to_mask(node_mask, num_nodes=data.get("num_nodes", data.get("x").shape[0]))
    elif isinstance(node_mask, BoolTensor):
        mask = node_mask
    else:
        raise ValueError
    return mask


def gnn_step4pyg_node(
    model: Module,
    data: PyGData,
    aim_node_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    distill_node_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    tape: Optional[bool] = None,
) -> FullChoirOutputElm:
    aim_mask = infer_node_mask(data, aim_node_mask)
    distill_mask = infer_node_mask(data, distill_node_mask)
    batch_size = data.get("batch_size", None)
    structure = data.get("adj_t", data.get("edge_index", None))
    edge_weight = data.get("edge_weight", None)
    edge_attr = data.get("edge_attr", None)
    label = data.get("y", None)
    tape_hid = tape if tape is not None else getattr(model, "tape", False)
    if batch_size is None:  # Node-level GNN with Full Graph
        xs = model.forward(data.x, structure, edge_weight, edge_attr)
        if tape_hid:
            logit = xs[-1]
            xs4distill = [x[distill_mask] for x in xs]
        else:
            logit = xs
            xs4distill = None
        logit4aim = logit[aim_mask] if aim_mask is not None else logit
        logit4distill = logit[distill_mask] if distill_mask is not None else logit
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
        aim_mask = aim_mask[:batch_size] if aim_mask is not None else None
        distill_mask = distill_mask[:batch_size] if distill_mask is not None else distill_mask

        if tape_hid:
            logit = xs[-1][:batch_size]
            xs4distill = [x[:batch_size][distill_mask] if distill_mask is not None else x[:batch_size] for x in xs]
        else:
            logit = xs[:batch_size]
            xs4distill = None

        logit4aim = logit[aim_mask] if aim_mask is not None else logit
        logit4distill = logit[distill_mask] if distill_mask is not None else logit
        y4aim = None
        if label is not None:
            y4aim = label[:batch_size][aim_mask] if aim_mask is not None else label[:batch_size]
    return logit4aim, y4aim, aim_mask, logit4distill, xs4distill, distill_mask


def gnn_step_wo_distill4pyg_node(
    model: Module, data: PyGData, aim_node_mask: Optional[Union[str, BoolTensor, LongTensor]] = None
):
    aim_mask = infer_node_mask(data, aim_node_mask)
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
    return logit4aim, y4aim, aim_mask


def gnn_step4pyg_graph(
    model: Module,
    data: PyGData,
    aim_graph_mask: Optional[Union[BoolTensor, LongTensor]] = None,
    distill_graph_mask: Optional[Union[BoolTensor, LongTensor]] = None,
    batch: Optional[LongTensor] = None,
    tape: Optional[bool] = None,
) -> FullChoirOutputElm:
    batch = batch if batch is not None else data.get("batch")
    edge_weight = data.get("edge_weight", None)
    edge_attr = data.get("edge_attr", None)
    structure = data.get("adj_t", data.get("edge_index", None))
    label = data.get("y", None)
    tape_hid = tape if tape is not None else getattr(model, "tape", False)
    xs = model(data.x, structure, edge_weight, edge_attr, batch)
    if tape_hid:
        logit = xs[-1]  # B x num_classes
        xs4distill = xs[:-1]  # [total_num_nodes x node hidden dim]
    else:
        logit = xs
        xs4distill = None
    logit4aim = logit[aim_graph_mask]
    logit4distill = logit[distill_graph_mask]
    y4aim = label if label is not None else None
    return logit4aim, y4aim, aim_graph_mask, logit4distill, xs4distill, distill_graph_mask


def gnn_step_wo_distill4pyg_graph(
    model: Module,
    data: PyGData,
    aim_graph_mask: Optional[Union[BoolTensor, LongTensor]] = None,
    batch: Optional[LongTensor] = None,
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
    return logit4aim, y4aim, aim_graph_mask


def gnn_step4pyg(
    model: Module,
    data: PyGData,
    aim_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    distill_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    tape: Optional[bool] = None,
) -> FullChoirOutputElm:
    batch = data.get("batch", None)
    if batch is not None:  # graph-level GNN with concatenated multiple graphs
        return gnn_step4pyg_graph(model, data, aim_mask, distill_mask, batch, tape)

    else:  # node-level GNN with full graph or pyg neighbor loaders
        return gnn_step4pyg_node(model, data, aim_mask, distill_mask, tape)


def gnn_step_wo_distill4pyg(
    model: Module,
    data: PyGData,
    aim_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
):
    batch = data.get("batch", None)
    if batch is not None:  # graph-level GNN with concatenated multiple graphs
        return gnn_step_wo_distill4pyg_graph(model, data, aim_mask, batch)

    else:  # node-level GNN with full graph or pyg neighbor loaders
        return gnn_step_wo_distill4pyg_node(model, data, aim_mask)


def mlp_step4pyg_node(
    model: Module,
    data: PyGData,
    aim_node_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    distill_node_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    tape: Optional[bool] = None,
) -> FullChoirOutputElm:
    aim_mask = infer_node_mask(data, aim_node_mask)
    distill_mask = infer_node_mask(data, distill_node_mask)
    batch_size = data.get("batch_size", None)
    label = data.get("y", None)
    tape_hid = tape if tape is not None else getattr(model, "tape", False)
    if batch_size is None:  # full
        x = data.get("x", data.get("node_fea", data.get("node_features")))
    else:
        aim_mask = aim_mask[:batch_size] if aim_mask is not None else None
        distill_mask = distill_mask[:batch_size] if distill_mask is not None else None
        x = data.x[:batch_size]
        label = label[:batch_size] if label is not None else None

    xs = model.forward(x)
    if tape_hid:
        logit = xs[-1]
        xs4distill: Optional[List[Tensor]] = [x[distill_mask] if distill_mask is not None else x for x in xs]
    else:
        logit = xs
        xs4distill = None

    logit4aim = logit[aim_mask] if aim_mask is not None else logit
    if label is not None:
        y4aim = label[aim_mask] if aim_mask is not None else label
    else:
        y4aim = None
    logit4distill = logit[distill_mask] if distill_mask is not None else logit
    # if logit4aim.numel() == 0:  # in this case only distill with `logit` or/and `xs`
    #     logit4aim = None
    #     y4aim = None
    #
    # if logit4distill.numel() == 0:  # in this case only distill with `logit` or/and `xs`
    #     logit4distill = None
    return logit4aim, y4aim, aim_mask, logit4distill, xs4distill, distill_mask


def mlp_step_wo_distill4pyg_node(
    model: Module,
    data: PyGData,
    aim_node_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
):
    aim_mask = infer_node_mask(data, aim_node_mask)
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
    # if logit4aim.numel() == 0:  # in this case only distill with `logit` or/and `xs`
    #     logit4aim = None
    #     y4aim = None
    #
    # if logit4distill.numel() == 0:  # in this case only distill with `logit` or/and `xs`
    #     logit4distill = None
    return logit4aim, y4aim, aim_mask


def mlp_step4pyg_graph(
    model: Module,
    data: PyGData,
    aim_graph_mask: Optional[Union[BoolTensor, LongTensor]] = None,
    distill_graph_mask: Optional[Union[BoolTensor, LongTensor]] = None,
    batch: Optional[LongTensor] = None,
    tape: Optional[bool] = None,
) -> FullChoirOutputElm:
    """TODO: implement once we are building the first Graph-level distillation"""
    raise NotImplementedError


def mlp_step_wo_distill4pyg_graph(
    model: Module,
    data: PyGData,
    aim_graph_mask: Optional[Union[BoolTensor, LongTensor]] = None,
    batch: Optional[LongTensor] = None,
):
    """TODO: implement once we are building the first Graph-level distillation"""
    raise NotImplementedError


def mlp_step4pyg(
    model: Module,
    data: PyGData,
    aim_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    distill_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    tape: Optional[bool] = None,
) -> FullChoirOutputElm:
    batch = data.get("batch", None)
    if batch is not None:  # graph-level GNN with concatenated multiple graphs
        return mlp_step4pyg_graph(model, data, batch, aim_mask, distill_mask, tape)

    else:  # node-level GNN with full graph or pyg neighbor loaders
        return mlp_step4pyg_node(model, data, aim_mask, distill_mask, tape)


def mlp_step_wo_distill4pyg(
    model: Module,
    data: PyGData,
    aim_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
):
    batch = data.get("batch", None)
    if batch is not None:  # graph-level GNN with concatenated multiple graphs
        return mlp_step_wo_distill4pyg_graph(model, data, aim_mask, batch)

    else:  # node-level GNN with full graph or pyg neighbor loaders
        return mlp_step_wo_distill4pyg_node(model, data, aim_mask)


def configure_step(model, func: Optional[Callable] = None, wo_distill: bool = False) -> Callable:
    is_compiled = isinstance(model, OptimizedModule)
    ravel_model = model._orig_mod if is_compiled else model
    if func is not None:
        return_func = func
    elif isinstance(ravel_model, GnnGGNNBase):
        return_func = gnn_step4pyg if not wo_distill else gnn_step_wo_distill4pyg
    elif isinstance(ravel_model, MLP):
        return_func = mlp_step4pyg if not wo_distill else mlp_step_wo_distill4pyg
    else:
        raise ValueError
    return return_func


def configure_many_step(
    model_list: Union[ModuleList, Sequence[Module]],
    wo_distill: bool = False,
    customized_step_func_dict: Optional[Dict[str, Callable]] = None,
) -> List[Callable]:
    step_dict = customized_step_func_dict or {}
    step_impl_list: List[Optional[Callable]] = []
    for i, md in enumerate(model_list):
        func = step_dict.get(md.__class__.__name__, step_dict.get(normalize_md_class_name(md)))
        step_impl_list.append(configure_step(md, func, wo_distill))
    return step_impl_list
