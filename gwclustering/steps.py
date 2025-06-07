from typing import Union, Optional, List, Callable
from gnng.typing import PyGData
import torch
import torchmetrics
from torch import BoolTensor, LongTensor
from torch._dynamo import OptimizedModule
from lightning.fabric import Fabric

from gnng.training.utils import infer_node_mask
from gnng.training.loss_recorder import LossRecorder
from gnng.training.bo_recorder import BatchOutputRecorder
from gnng.ow.steps import narrow_full_batch_group
from torch_geometric.transforms import ToSparseTensor


# ========================================================== Deving ==============================================
def eval_step_gw_clustering(
    model: "torch.nn.Module",
    data: PyGData,
    loss_fn=None,
    aim_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    metric_manager: "torchmetrics.Metric" = None,
    loss_manager: "LossRecorder" = None,
    batch_out_manager: "BatchOutputRecorder" = None,
    stage: str = None,  # val/test
    fabric: Fabric = None,
    current_epoch: int = None,
    *args,
    **kwargs,
):
    loss_fn.eval()
    if fabric is not None:
        data = fabric.to_device(data)
    x = data.get("x")
    num_nodes = x.size(0)
    mask = infer_node_mask(data, aim_mask)

    batch_size = data.get("batch_size")
    y = data.get("y", None)
    if not isinstance(model, OptimizedModule):  # TODO Trimming is currently not support in JIT mode.
        num_sampled_nodes_per_hop = data.get("num_sampled_nodes_per_hop")
        num_sampled_edges_per_hop = data.get("num_sampled_edges_per_hop")
    else:
        num_sampled_nodes_per_hop = None
        num_sampled_edges_per_hop = None

    # =================== With SparseTensor ======================
    spa_transform = ToSparseTensor()
    dt = spa_transform(data)
    logit, z = model(
        x,
        edge_index=dt.adj_t,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        data_topo=dt.adj_t,
    )

    ret = model.project(
        z,
        edge_index=dt.adj_t,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        data_topo=dt.adj_t,
    )
    pred_h, Ap, wp = ret
    # =================== With SparseTensor ======================

    if batch_size is not None:  # training with mini-batch sampling
        logit = logit[..., :batch_size, :] if logit is not None else None
        pred_h = pred_h[..., :batch_size, :] if pred_h is not None else None
        y = y[:batch_size] if y is not None else None
    else:
        logit = narrow_full_batch_group(logit, mask)
        pred_h = narrow_full_batch_group(pred_h, mask)
        y = narrow_full_batch_group(y, mask)
        batch_size = x.size(0)

    loss_dict = dict()  # empty val/test loss dict if there is no predictor
    bs_dict = dict()
    if (
        getattr(model, "predictor", None) is not None
    ):  # only losses based on single logit/embed, e.g., CE, are available during test
        loss_dict, bs_dict = loss_fn(
            pred_logit=logit,  # If None, then skip
            view_logit=None,
            view_h=None,
            pred_h=pred_h,
            current_epoch=current_epoch,
            return_batch_size=True,
        )
    loss_manager.update(loss_dict, bs_dict)
    batch_return = {"batch_size": batch_size}

    if loss_dict is None:
        to_cache = pred_h.mean(dim=0)
    else:
        to_cache = pred_h.mean(dim=0)

    if loss_dict is not None:
        batch_return.update(loss_dict)
        batch_return.update(bs_dict)
    if batch_out_manager is not None:
        batch_out_manager.update(
            {
                "preds": to_cache.detach(),
                "target": y
            }
        )

    return batch_return


def train_step_gw_clustering(
    model: "torch.nn.Module",
    data: PyGData,
    loss_fn=None,
    aim_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
    metric_manager: "torchmetrics.Metric" = None,
    loss_manager: "LossRecorder" = None,
    batch_out_manager: "BatchOutputRecorder" = None,
    stage: str = "train",  # train
    fabric: Fabric = None,
    current_epoch: int = None,
    augmentors: List[Callable] = None,
    *args,
    **kwargs,
):
    loss_fn.train()
    if fabric is not None:
        data = fabric.to_device(data)
    x = data.get("x")
    num_nodes = x.shape[0]
    mask = infer_node_mask(data, aim_mask)
    batch_size = data.get("batch_size")
    y = data.get("y", None)

    if not isinstance(model, OptimizedModule):  # TODO Trimming is currently not support in JIT mode.
        num_sampled_nodes_per_hop = data.get("num_sampled_nodes_per_hop")
        num_sampled_edges_per_hop = data.get("num_sampled_edges_per_hop")
    else:
        num_sampled_nodes_per_hop = None
        num_sampled_edges_per_hop = None
    if hasattr(model, "normalize_prototypes"):  # UNO head normalization
        model.normalize_prototypes()

    spa_transform = ToSparseTensor(remove_edge_index=True)
    logit1 = logit2 = h1 = h2 = None
    Ap1 = Ap2 = None
    wp1 = wp2 = None
    sp_adj1 = sp_adj2 = None
    if augmentors is not None:
        aug1, aug2 = augmentors
        dt1 = aug1(data)

        # =================== With SparseTensor ======================
        dt1 = spa_transform(dt1)
        logit1, z1 = model(dt1.x, edge_index=dt1.adj_t)
        ret1 = model.project(z1, edge_index=dt1.adj_t)
        sp_adj1 = dt1.adj_t
        # =================== With SparseTensor ======================
        if ret1 is not None:
            h1, Ap1, wp1 = ret1
        del dt1
        torch.cuda.empty_cache()

        # view 2
        dt2 = aug2(data)
        # =================== With SparseTensor ======================
        dt2 = spa_transform(dt2)
        logit2, z2 = model(dt2.x, edge_index=dt2.adj_t)
        ret2 = model.project(z2, edge_index=dt2.adj_t)
        sp_adj2 = dt2.adj_t
        # =================== With SparseTensor ======================
        if ret2 is not None:
            h2, Ap2, wp2 = ret2
        del dt2
        torch.cuda.empty_cache()

        # we compute the SelfConLoss or/and Sup(Con)Loss on the interested node set. So we first get their information
        if batch_size is not None:  # training with mini-batch sampling, interested nodes are always at head
            logit1 = logit1[..., :batch_size, :] if logit1 is not None else None
            logit2 = logit2[..., :batch_size, :] if logit2 is not None else None
            h1 = h1[..., :batch_size, :] if h1 is not None else None
            h2 = h2[..., :batch_size, :] if h2 is not None else None
            sp_adj1 = sp_adj1[:batch_size, :batch_size]
            sp_adj2 = sp_adj2[:batch_size, :batch_size]
        else:  # full-batch training, interested nodes are specified by (aim) `mask`
            logit1 = narrow_full_batch_group(logit1, mask)
            logit2 = narrow_full_batch_group(logit2, mask)
            h1 = narrow_full_batch_group(h1, mask)
            h2 = narrow_full_batch_group(h2, mask)

    # =================== With SparseTensor ======================
    dt = spa_transform(data)
    logit, z = model(
        x,
        edge_index=dt.adj_t,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
    )

    ret = model.project(
        z,
        edge_index=dt.adj_t,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
    )
    pred_h, Ap, wp = ret
    # =================== With SparseTensor ======================

    if batch_size is not None:  # training with mini-batch sampling, interested nodes are always at head
        logit = logit[..., :batch_size, :] if logit is not None else None
        pred_h = pred_h[..., :batch_size, :] if pred_h is not None else None
        z = z[..., :batch_size, :] if z is not None else None
        y = y[:batch_size] if y is not None else None
    else:  # full-batch training, interested nodes are specified by (aim) `mask`
        logit = narrow_full_batch_group(logit, mask)
        pred_h = narrow_full_batch_group(pred_h, mask)
        y = narrow_full_batch_group(y, mask)
        batch_size = y.shape[0]

    view_logit = None
    if logit1 is not None and logit2 is not None:  # (n_view=2, batch_size, n_hid)
        view_logit = torch.stack([logit1, logit2])  # If UNO (n_view=2, n_head,  batch_size, n_hid)
    view_h = None
    if h1 is not None and h2 is not None:
        view_h = torch.stack([h1, h2])

    loss_dict, bs_dict = loss_fn(
        view_logit=view_logit,
        view_h=view_h,
        pred_h=pred_h,
        pred_logit=logit,
        current_epoch=current_epoch,
        return_batch_size=True,
        proto_topo=[Ap1, Ap2],  # !ADDED
        data_topo=[sp_adj1, sp_adj2],
        proto_weight=[wp1, wp2],
    )

    del sp_adj1
    del sp_adj2
    torch.cuda.empty_cache()

    loss_manager.update(loss_dict, bs_dict)

    to_cache = pred_h.mean(dim=0)  # UNO pick or just mean all heads?

    batch_return = {"batch_size": batch_size}
    batch_return.update(loss_dict)
    batch_return.update(bs_dict)

    if batch_out_manager is not None:
        batch_out_manager.update(
            {
                "preds": to_cache.detach(),
                "target": y,
            }
        )
    return batch_return  # somethings to print in terminal
