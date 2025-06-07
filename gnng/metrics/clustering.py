from typing import List, Any, Tuple, Dict
import torch
import numpy as np
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import accuracy_score, f1_score


def clustering_summary(y_true, y_pred):
    """
    evaluate the gwclustering performance.
    args:
        y_true: ground truth
        y_pred: prediction
    returns:
        acc: accuracy
        nmi: normalized mutual information
        ari: adjust rand index
        f1: f1 score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)

    l1 = np.unique(y_true).tolist()
    num_class1 = len(l1)
    l2 = np.unique(y_pred).tolist()
    num_class2 = len(l2)

    num_all_classes = max(num_class1, num_class2)
    w = np.zeros((num_all_classes, num_all_classes), dtype=int)
    np.add.at(w, (y_pred, y_true), 1)

    # the mapping from predicted class to ground-truth class
    ind = linear_sum_assignment(w.max() - w)
    ind = np.vstack(ind).T
    ind_map = {j: i for i, j in ind}  # gt_class j --> pred_class i

    new_predict = -np.ones(y_pred.shape[0],dtype=np.int64)
    for true_c, pred_c in ind_map.items():
        new_predict[y_pred == pred_c] = true_c

    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average="macro")
    res = {
        "ClusteringAccuracy": torch.tensor(acc),
        "NMI": torch.tensor(nmi),
        "ARI": torch.tensor(ari),
        "ClusteringF1": torch.tensor(f1),
        "ConfusionMatrix": torch.tensor(w.T),
        "y_pred_lsa": torch.tensor(new_predict),
        "assign_map": torch.tensor(ind),
        "y_true": torch.tensor(y_true),
    }
    return res


class ClusteringSummary(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.append(preds.detach())
        self.target.append(target.detach())

    def on_compute_start(self) -> Tuple[Tensor, Tensor]:
        preds = dim_zero_cat(self.preds).cpu()
        if preds.dim() > 1:
            preds = preds.argmax(dim=-1)
        target: Tensor = dim_zero_cat(self.target).cpu()
        return preds, target

    def compute(self) -> Dict[str, Tensor]:
        preds, target = self.on_compute_start()
        return clustering_summary(target, preds)
