from typing import List, Any, Tuple, Dict
import torch
import numpy as np
from torch import Tensor
from torchmetrics import Metric
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.clustering.normalized_mutual_info_score import normalized_mutual_info_score


def clustering_accuracy(y_pred, y_true, num_all_classes=None):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    assert y_pred.shape[0] == y_true.shape[0]
    if num_all_classes is None:
        num_all_classes = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((num_all_classes, num_all_classes), dtype=np.int64)

    np.add.at(w, (y_pred, y_true), 1)

    ind = linear_sum_assignment(w.max() - w)

    ind = np.vstack(ind).T

    # ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) / y_true.shape[0]
    return total_acc


def clustering_accuracy_subset(y_pred, y_true, subset=None, num_all_classes=None):
    if subset is None or torch.all(subset):  # actually on all samples
        do_all = True
        old_classes_gt = None
    else:
        old_classes_gt = set(y_true[subset].cpu().tolist())
        do_all = False
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    assert y_pred.shape[0] == y_true.shape[0]
    if num_all_classes is None:
        num_all_classes = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((num_all_classes, num_all_classes), dtype=np.int64)

    np.add.at(w, (y_pred, y_true), 1)

    ind = linear_sum_assignment(w.max() - w)

    ind = np.vstack(ind).T
    if do_all:
        acc = sum([w[i, j] for i, j in ind]) / y_true.shape[0]
    else:
        ind_map = {j: i for i, j in ind}
        old_correct = 0
        total_old_instances = 0
        for i in old_classes_gt:
            old_correct += w[ind_map[i], i]
            total_old_instances += sum(w[:, i])
        acc = old_correct / total_old_instances
    return acc


def clustering_accuracy_summary(y_pred, y_true, old_mask=None, num_all_classes=None, rectify=True, x=None):
    if old_mask is None or torch.all(old_mask):  # all samples have known classes
        old_classes = y_true.unique().cpu().tolist()
        new_classes = None
    else:
        old_classes = y_true[old_mask].unique().cpu().tolist()
        new_classes = y_true[~old_mask].unique().cpu().tolist()
        #print(f"Summary =============================== {old_classes}")
        #print(f"Summary =============================== {new_classes}")
    res_dict = {"NMI": normalized_mutual_info_score(y_pred, y_true)}

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    old_mask = np.asarray(old_mask)

    assert y_pred.shape[0] == y_true.shape[0]
    if num_all_classes is None:
        num_all_classes = max(np.max(y_pred), np.max(y_true)) + 1

    # gwclustering accuracy, w is the transpose of confusion matrix: row is with predicted class
    w = np.zeros((num_all_classes, num_all_classes), dtype=np.int64)
    np.add.at(w, (y_pred, y_true), 1)
    # the mapping from predicted class to ground-truth class
    ind = linear_sum_assignment(w.max() - w)

    ind = np.vstack(ind).T

    cluster_acc = np.sum([w[i, j] for i, j in ind]) / y_true.shape[0]
    res_dict["ClusteringAccuracy"] = torch.tensor(cluster_acc)

    ind_map = {j: i for i, j in ind}  # j: pred_class --> i: true_class
    acc_old = None
    if old_classes is not None:
        # classical accuracy on samples from old_classes
        acc_old = np.sum(y_pred[old_mask] == y_true[old_mask]) / old_mask.sum()
        res_dict["AccuracyOld"] = torch.tensor(acc_old)

        # gwclustering accuracy on samples from old_classes
        old_correct = 0
        total_old_instances = 0
        for i in old_classes:
            old_correct += w[ind_map[i], i]
            total_old_instances += np.sum(w[:, i])
        res_dict["ClusteringAccuracyOld"] = torch.tensor(old_correct / total_old_instances)
    else:  # all samples are from old_classes
        # classical accuracy on samples from old_classes
        acc_old = np.sum(y_pred == y_true) / old_mask.sum()
        res_dict["AccuracyOld"] = torch.tensor(acc_old)

    if new_classes is not None:  # gwclustering accuracy on samples from new_classes
        new_correct = 0
        total_new_instances = 0
        for i in new_classes:
            new_correct += w[ind_map[i], i]
            total_new_instances += np.sum(w[:, i])
        res_dict["ClusteringAccuracyNew"] = torch.tensor(new_correct / total_new_instances)

    # rectified
    if rectify:  # ClusteringAccuracy after linear assignment within UKnown Classes
        new_classes = list(new_classes)
        w4new = w[new_classes, :][:, new_classes]
        ind4new = linear_sum_assignment(w4new.max() - w4new)
        ind4new = np.vstack(ind4new).T

        ind_map4new = {j: i for i, j in ind4new}
        new_correct = 0
        total_new_instances = 0
        for i, new_clss in enumerate(new_classes):
            new_correct += w4new[ind_map4new[i], i]
            total_new_instances += np.sum(w[:, new_clss])

        cluster_acc_new = new_correct / total_new_instances

        # new_correct2 = sum([w4new[i, j] for i, j in ind4new])
        # total_new_instances2 = np.sum(w[:, new_classes])
        # cluster_acc2 = new_correct2 / total_new_instances2
        # assert np.abs(cluster_acc2 - cluster_acc) < 1e-12, f"{cluster_acc},{cluster_acc2}"

        res_dict["ClusteringAccuracyNewRectified"] = torch.tensor(cluster_acc_new)
        if acc_old is not None:
            res_dict["HRScore"] = torch.tensor(harmonic_mean([cluster_acc_new, acc_old]))
    # # Remove debug statements
    # for k, v in res_dict.items():
    #     assert not torch.isnan(v), f"`{k}` is NaN"

    w_trans = w.T
    detect_cm = class_cm2sup_cm(w_trans, [old_classes, new_classes])
    res_dict["RejectAccuracy"] = torch.tensor(np.trace(detect_cm) / detect_cm.sum())

    # These two are only logged into wandb UI as wandb.Table plots and SHOULD be popped from `res_dict`
    # before Trainer tracks the scalar metrics.
    res_dict["ConfusionMatrix"] = torch.from_numpy(w_trans)
    res_dict["DetectConfusionMatrix"] = torch.from_numpy(detect_cm)
    res_dict["old_classes"] = torch.tensor(old_classes)
    res_dict["new_classes"] = torch.tensor(new_classes)
    # if x is not None and x.numel() > 0:
    #     x_np = x.cpu().numpy()
    #     y_unique = np.unique(y_pred)
    #     if y_unique.size < 2:  # to prevent some cases where all samples are predicted as the same class
    #         y_pred[-1] = old_classes[0] if y_unique[0] != old_classes[0] else old_classes[-1]
    #     silhouette_score = metrics.silhouette_score(x_np, y_pred, metric="euclidean")
    #     normalized_silhouette_score = (silhouette_score + 1) / 2.0
    #     res_dict["SilhouetteScore"] = torch.tensor(silhouette_score)
    #     res_dict["NormalizedSilhouetteScore"] = torch.tensor(normalized_silhouette_score)
    #     if acc_old is not None:
    #         res_dict["NSAScore"] = torch.tensor(harmonic_mean([normalized_silhouette_score, acc_old]))
    return res_dict


def harmonic_mean(arr):
    arr = np.asarray(arr)
    assert np.all(arr >= 0.0)
    if 0.0 in arr:
        return 0.0
    else:
        return arr.shape[0] / np.sum(1.0 / arr)


def class_cm2sup_cm(class_cm, super_classes):
    num_super_classes = len(super_classes)
    sup_cm = np.zeros((num_super_classes, num_super_classes), dtype=int)
    for i in range(num_super_classes):
        for j in range(num_super_classes):
            sup_cm[i, j] = np.sum(class_cm[super_classes[i], :][:, super_classes[j]])
    return sup_cm


class GCDAccuracy(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]
    old_mask: List[Tensor]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("old_mask", default=[], dist_reduce_fx="cat")
        self.add_state("x", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor, old_mask: Tensor, x: Tensor = None) -> None:
        """Update state with predictions and targets."""
        self.preds.append(preds.detach())
        self.target.append(target.detach())
        self.old_mask.append(old_mask.detach())
        self.x.append(x.detach() if x is not None else torch.tensor([]))

    def on_compute_start(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        preds = dim_zero_cat(self.preds).cpu()
        if preds.dim() > 1:
            preds = preds.argmax(dim=-1)
        target: Tensor = dim_zero_cat(self.target).cpu()
        if target.dim() > 1:  # in case not one-hot label
            target = target.argmax(dim=1)
        old_mask: Tensor = dim_zero_cat(self.old_mask).cpu()
        x: Tensor = dim_zero_cat(self.x).cpu()
        return preds, target, old_mask, x

    def compute(self) -> Dict[str, Tensor]:
        preds, target, old_mask, x = self.on_compute_start()
        return clustering_accuracy_summary(preds, target, old_mask, x=x)


class GCDSummary(GCDAccuracy):
    pass
