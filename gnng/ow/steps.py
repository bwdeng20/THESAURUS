import pdb
from typing import Union, Optional, List, Callable

from gnng.ow import PlainGNNGCD
from gnng.typing import PyGData
import torch
import torchmetrics
from torch import BoolTensor, LongTensor
from torch._dynamo import OptimizedModule
from lightning.fabric import Fabric
from gnng.nn.losses.ow_losses import GCDLoss
from gnng.ow.models import NonParametricGCD, ParametricGCD
from gnng.utils import sanitize_dict
from gnng.training.utils import infer_node_mask
from gnng.training.loss_recorder import LossRecorder
from gnng.training.bo_recorder import BatchOutputRecorder
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor


@torch.compile
def narrow_full_batch_group(val, interested_mask):
    if val is not None:  # CAUTION! assume batch dimension the second dim from right side
        if val.ndim == 1:
            val = val[interested_mask] if interested_mask is not None else val
        else:
            val = val[..., interested_mask, :] if interested_mask is not None else val
    return val


def eval_step_gcd(
        model: "torch.nn.Module",
        data: PyGData,
        loss_fn: GCDLoss = None,
        aim_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
        metric_manager: "torchmetrics.Metric" = None,
        loss_manager: "LossRecorder" = None,
        batch_out_manager: "BatchOutputRecorder" = None,
        stage: str = None,  # val/test
        fabric: Fabric = None,
        current_epoch: int = None,
        no_topo: bool = False,
        *args,
        **kwargs,
):
    is_non_parametric_gcd = isinstance(model, NonParametricGCD) or isinstance(model.module, NonParametricGCD)
    is_plain_gnn_gcd = isinstance(model, PlainGNNGCD) or isinstance(model.module, PlainGNNGCD)
    if fabric is not None:
        data = fabric.to_device(data)
    x = data.get("x")
    old_mask = data.get("old_mask")
    mask = infer_node_mask(data, aim_mask)
    stage_mask = infer_node_mask(data, stage)

    batch_size = data.get("batch_size")
    y = data.get("y", None)

    need_tr_set = kwargs.get("need_tr_set", False)
    train_stage_mask = None  #
    if is_non_parametric_gcd or need_tr_set:
        # Since SupKMeans in VanillaGCD requires this information to perform Semi-Kmeans on the whole node set
        train_stage_mask = infer_node_mask(data, "train")

    # if not no_topo:  # using GNN with Graph structure
    #     edge_index = data.get("adj_t", data.get("edge_index", None))
    #     edge_weight = data.get("edge_weight", None)
    #     edge_attr = data.get("edge_attr", None)
    # else:  # using Non-GNN, e.g., MLP, without graph structure
    #     edge_index = edge_weight = edge_attr = None

    if not isinstance(model, OptimizedModule):  # TODO Trimming is currently not support in JIT mode.
        num_sampled_nodes_per_hop = data.get("num_sampled_nodes_per_hop")
        num_sampled_edges_per_hop = data.get("num_sampled_edges_per_hop")
    else:
        num_sampled_nodes_per_hop = None
        num_sampled_edges_per_hop = None

    spa_transform = kwargs.get("spa_transform", ToSparseTensor(remove_edge_index=True))
    dt = spa_transform(data)
    if no_topo:
        dt.adj_t = SparseTensor.eye(dt.x.shape[0], device=dt.x.device)

    # logit (None), encoder output
    out, z = model(
        x,
        edge_index=dt.adj_t,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
    )

    if batch_size is not None:  # with mini-batch sampling
        out = out[..., :batch_size, :] if out is not None else None
        z = z[..., :batch_size, :] if z is not None else None
        y = y[:batch_size] if y is not None else None
        old_mask = old_mask[:batch_size] if old_mask is not None else None
        stage_mask = stage_mask[:batch_size] if stage_mask is not None else None
        train_stage_mask = train_stage_mask[:batch_size] if train_stage_mask is not None else None
    else:
        out = narrow_full_batch_group(out, mask)
        z = narrow_full_batch_group(z, mask)
        y = narrow_full_batch_group(y, mask)
        old_mask = narrow_full_batch_group(old_mask, mask)
        stage_mask = narrow_full_batch_group(stage_mask, mask)
        train_stage_mask = narrow_full_batch_group(train_stage_mask, mask)
        batch_size = x.size(0)

    if out is None and is_plain_gnn_gcd:
        out = z  # since z is the output of GNN encoder output, and no projector here

    train_labelled_mask = None  # by default, no training sample information is used during evaluation. But GCD needs it
    if old_mask is not None:  # GCD tasks.
        if stage_mask is not None:  # only `old_mask & valid_mask` nodes have available y for computing loss
            labelled_mask = torch.logical_and(old_mask, stage_mask)
        else:  # all `old_mask` nodes have available y for computing losses here (NOT metrics)
            labelled_mask = old_mask
        if train_stage_mask is not None:
            train_labelled_mask = torch.logical_and(old_mask, train_stage_mask)

    else:  # Non-GCD tasks. Self-supervised (and supervised) graph contrastive learning
        if stage_mask is not None:
            labelled_mask = stage_mask
        else:
            labelled_mask = None  # all nodes have available y for this stage
        if train_stage_mask is not None:
            train_labelled_mask = train_stage_mask

    view_logit = None
    loss_dict = dict()  # empty val/test loss dict if there is no predictor
    bs_dict = dict()
    if (
            view_logit is not None and getattr(model, "predictor", None) is not None
    ):  # only losses based on single logit/embed, e.g., CE, are available during val and test
        loss_dict, bs_dict = loss_fn(
            pred_logit=out,  # If None, then skip
            view_logit=None,
            view_h=None,
            y=y,
            labelled_mask=labelled_mask,
            current_epoch=current_epoch,
            return_batch_size=True,
        )

    if getattr(loss_fn, "loss_fn_uno", None) is not None:
        if loss_dict is not None and "uno_head_loss_cache" in loss_dict:
            uno_head_loss = loss_dict.pop("uno_head_loss_cache")
            best_head_idx = uno_head_loss.argmin()
            out = out[best_head_idx]
        else:
            out = out.mean(0)  # mean over head dimension

    loss_manager.update(loss_dict, bs_dict)
    # Both labelled and unlabeled are involved, we take the total number of them as bs
    batch_return = {"batch_size": batch_size}
    if loss_dict is not None:
        batch_return.update(loss_dict)
        batch_return.update(bs_dict)

    # cache `logit` as `preds` for parametric models; `z` as `preds` for non-parametric ones
    z = z.detach() if z is not None else None
    if batch_out_manager is not None:
        batch_stuff2store = {
            "labelled_mask": labelled_mask,
            "train_labelled_mask": train_labelled_mask,  # in eval_step (val/test), these two masks are not identical
            "old_mask": old_mask,
            "z": z,
            "target": y,
            "stage_mask": stage_mask,
        }
        if is_non_parametric_gcd:
            spc = {"preds": batch_stuff2store["z"]}  # embeddings to perform semi-kmeans in VanillaGCD
            batch_stuff2store.update(spc)
        else:
            if out is not None:
                spc = {"preds": out.detach()}
                batch_stuff2store.update(spc)
        batch_out_manager.update(sanitize_dict(batch_stuff2store))

    if metric_manager is not None and out is not None:  # Parametric Head can make predictions every batch
        out2mt = out[stage_mask]
        y2mt = y[stage_mask]
        z2mt = z[stage_mask] if z is not None else None

        if old_mask is not None:
            old_mask2mt = old_mask[stage_mask]
        else:  # all nodes are from known-classes
            old_mask2mt = torch.ones(out2mt.shape[0], dtype=torch.bool, device=out2mt.device)
        metric_manager.update(out2mt.detach(), y2mt, old_mask=old_mask2mt, x=z2mt)
    return batch_return


def train_step_gcd(
        model: "torch.nn.Module",
        data: PyGData,
        loss_fn: GCDLoss = None,
        aim_mask: Optional[Union[str, BoolTensor, LongTensor]] = None,
        metric_manager: "torchmetrics.Metric" = None,
        loss_manager: "LossRecorder" = None,
        batch_out_manager: "BatchOutputRecorder" = None,
        stage: str = "train",  # train
        fabric: Fabric = None,
        current_epoch: int = None,
        augmentors: List[Callable] = None,
        no_topo: bool = False,
        *args,
        **kwargs,
):
    # x -> backbone (GNN) -> z| -> projector (GNN or MLP) -> h: for supvervised or self contrastive learning
    #                         | -> predictor (GNN or MLP) -> logit: for classification-based gwclustering
    is_non_parametric_gcd = isinstance(model, NonParametricGCD) or isinstance(model.module, NonParametricGCD)
    is_plain_gnn_gcd = isinstance(model, PlainGNNGCD) or isinstance(model.module, PlainGNNGCD)
    if fabric is not None:
        data = fabric.to_device(data)
    x = data.get("x")
    old_mask = data.get("old_mask")
    mask = infer_node_mask(data, aim_mask)
    stage_mask = infer_node_mask(data, stage)
    batch_size = data.get("batch_size")
    y = data.get("y", None)
    # if not no_topo:  # using GNN with Graph structure
    #     edge_index = data.get("adj_t", data.get("edge_index", None))
    #     edge_weight = data.get("edge_weight", None)
    #     edge_attr = data.get("edge_attr", None)
    # else:  # using Non-GNN, e.g., MLP, without graph structure
    #     edge_index = edge_weight = edge_attr = None

    need_tr_set = kwargs.get("need_tr_set", False)

    if not isinstance(model, OptimizedModule):  # TODO Trimming is currently not support in JIT mode.
        num_sampled_nodes_per_hop = data.get("num_sampled_nodes_per_hop")
        num_sampled_edges_per_hop = data.get("num_sampled_edges_per_hop")
    else:
        num_sampled_nodes_per_hop = None
        num_sampled_edges_per_hop = None
    if hasattr(model, "normalize_prototypes"):  # UNO head normalization
        model.normalize_prototypes()

    spa_transform = kwargs.get("spa_transform", ToSparseTensor(remove_edge_index=True))

    logit1 = logit2 = h1 = h2 = None
    if augmentors is not None:
        aug1, aug2 = augmentors
        if aug1 is not None and aug2 is not None:  # in case (None,None)
            dt1 = aug1(data)
            dt1 = spa_transform(dt1)
            if no_topo:
                dt1.adj_t = SparseTensor.eye(dt1.x.shape[0], device=dt1.x.device)
            logit1, z1 = model(dt1.x, edge_index=dt1.adj_t)
            h1 = model.project(z1, edge_index=dt1.adj_t)

            del dt1
            torch.cuda.empty_cache()

            dt2 = aug2(data)

            dt2 = spa_transform(dt2)
            if no_topo:
                dt2.adj_t = SparseTensor.eye(dt2.x.shape[0], device=dt2.x.device)
            logit2, z2 = model(dt2.x, edge_index=dt2.adj_t)
            h2 = model.project(z2, edge_index=dt2.adj_t)

            del dt2
            torch.cuda.empty_cache()

            # we compute the SelfConLoss or/and Sup(Con)Loss on the interested node set. So we first get their information
            if batch_size is not None:  # training with mini-batch sampling, interested nodes are always at head
                logit1 = logit1[..., :batch_size, :] if logit1 is not None else None
                logit2 = logit2[..., :batch_size, :] if logit2 is not None else None
                h1 = h1[..., :batch_size, :] if h1 is not None else None
                h2 = h2[..., :batch_size, :] if h2 is not None else None
            else:  # full-batch training, interested nodes are specified by (aim) `mask`
                logit1 = narrow_full_batch_group(logit1, mask)
                logit2 = narrow_full_batch_group(logit2, mask)
                h1 = narrow_full_batch_group(h1, mask)
                h2 = narrow_full_batch_group(h2, mask)

    dt = spa_transform(data)
    if no_topo:
        dt.adj_t = SparseTensor.eye(dt.x.shape[0], device=dt.x.device)

    logit, z = model(
        x,
        edge_index=dt.adj_t,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
    )

    if batch_size is not None:  # training with mini-batch sampling, interested nodes are always at head
        logit = logit[..., :batch_size, :] if logit is not None else None
        z = z[..., :batch_size, :] if z is not None else None
        y = y[:batch_size] if y is not None else None
        old_mask = old_mask[:batch_size] if old_mask is not None else None
        stage_mask = stage_mask[:batch_size] if stage_mask is not None else None
    else:  # full-batch training, interested nodes are specified by (aim) `mask`
        logit = narrow_full_batch_group(logit, mask)
        z = narrow_full_batch_group(z, mask)
        y = narrow_full_batch_group(y, mask)
        old_mask = narrow_full_batch_group(old_mask, mask)
        stage_mask = narrow_full_batch_group(stage_mask, mask)
        batch_size = y.shape[0]

    # `(aim) mask` where to SelfConLoss or/and Sup(Con)Loss, the interested node set
    # `stage_mask` where the current stage nodes. e.g., training set mask
    # `old_mask`   where the labels are old known ones. ONLY for GCD tasks
    if old_mask is not None:  # GCD tasks.
        if stage_mask is not None:  # only `old_mask & train_mask` nodes have available y for train stage
            labelled_mask = torch.logical_and(old_mask, stage_mask)
        else:  # all `old_mask` nodes have available y for this stage
            labelled_mask = old_mask
    else:  # Non-GCD tasks. Self-supervised (and supervised) graph contrastive learning
        if stage_mask is not None:
            labelled_mask = stage_mask
        else:
            labelled_mask = None  # all `aim_mask` nodes have available y for this stage
    view_logit = None
    if logit1 is not None and logit2 is not None:  # (n_view=2, batch_size, n_hid)
        view_logit = torch.stack([logit1, logit2])  # If UNO (n_view=2, n_head,  batch_size, n_hid)
    view_h = None
    if h1 is not None and h2 is not None:
        view_h = torch.stack([h1, h2])
   # pdb.set_trace()
    if logit is None and is_plain_gnn_gcd:
        logit = z  # since z is the output of GNN encoder output, and no projector here
    loss_dict, bs_dict = loss_fn(
        view_logit=view_logit,
        view_h=view_h,
        pred_logit=logit,
        y=y,
        labelled_mask=labelled_mask,
        current_epoch=current_epoch,
        return_batch_size=True,
    )

    if "uno_head_loss_cache" in loss_dict:  # view_logit=(n_view, n_head, n_batch, n_fea)
        uno_head = loss_dict.pop("uno_head_loss_cache")
        best_head_idx = uno_head.argmin()
        logit = logit[best_head_idx]

    loss_manager.update(loss_dict, bs_dict)
    # Both labelled and unlabeled are involved, we take the total number of them as bs
    batch_return = {"batch_size": batch_size}
    batch_return.update(loss_dict)
    batch_return.update(bs_dict)

    # cache `logit` as `preds` for parametric models; `z` as `preds` for non-parametric ones

    z = z.detach() if z is not None else None
    if batch_out_manager is not None:
        batch_stuff2store = {
            "labelled_mask": labelled_mask,
            "train_labelled_mask": labelled_mask,  # in train_step, these two masks are identical
            "old_mask": old_mask,
            "z": z,
            "target": y,
            "stage_mask": stage_mask,
        }
        if is_non_parametric_gcd:  # e.g., VanillaGCD
            spc = {"preds": batch_stuff2store["z"]}
            batch_stuff2store.update(spc)
        else:  # e.g., SimGCD,
            if logit is not None:
                spc = {"preds": logit.detach()}
                batch_stuff2store.update(spc)
        batch_out_manager.update(sanitize_dict(batch_stuff2store))

    if metric_manager is not None and logit is not None:  # Parametric GCD
        logit2mt = logit[stage_mask]
        y2mt = y[stage_mask]
        z2mt = z[stage_mask] if z is not None else None

        if old_mask is not None:
            old_mask2mt = old_mask[stage_mask]
        else:  # all nodes are from known-classes
            old_mask2mt = torch.ones(logit2mt.shape[0], dtype=torch.bool, device=logit2mt.device)
        metric_manager.update(logit2mt.detach(), y2mt.detach(), old_mask=old_mask2mt, x=z2mt)

    return batch_return  # some things to print in terminal
