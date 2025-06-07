import torch
from torch import Tensor
from numpy import ndarray
from typing import Union, Optional
import gnng
from gnng.clustering.base import KMeansBase
from gnng.clustering.fit_functional import fit_predict_once_semi, fit_predict_once


class TorchSemiKMeans(KMeansBase):
    def __init__(self,
                 n_clusters: int = None,
                 metric: str = 'euclidean',
                 init: Union[str, Tensor, ndarray] = 'k-means++',
                 random_state: int = 0,
                 n_init: Union[str, int] = "auto",
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 verbose: bool = True,
                 k: int = None,
                 atomic_op: Union[bool, str] = True):
        super().__init__(n_clusters=n_clusters,
                         init=init,
                         random_state=random_state,
                         n_init=n_init,
                         max_iter=max_iter,
                         metric=metric,
                         tol=tol,
                         verbose=verbose,
                         k=k,
                         atomic_op=atomic_op)

    def fit_predict_unsup(self, X: Tensor,
                          adaptive_tol: bool = True,
                          return_centroids: bool = False,
                          atomic_op: Optional[Union[bool, str]] = None,
                          batch_size=-1,
                          recover_label: bool = True):
        atomic_op = atomic_op if atomic_op is not None else self.atomic_op
        return self.fit_predict(Xu=X, adaptive_tol=adaptive_tol, return_centroids=return_centroids,
                                atomic_op=atomic_op, batch_size=batch_size, recover_label=recover_label)

    def fit_predict(self,
                    Xu: Tensor,
                    Xl: Tensor = None,
                    yl: Tensor = None,
                    adaptive_tol: bool = True,
                    return_centroids: bool = False,
                    atomic_op: Optional[Union[bool, str]] = None,
                    batch_size=-1,
                    recover_label: bool = True):
        atomic_op = atomic_op if atomic_op is not None else self.atomic_op

        if adaptive_tol:
            tol = torch.mean(torch.var(Xu, dim=0)) * self.tol
        else:
            tol = self.tol

        random_states = torch.arange(self.n_init) + self.random_state
        # g = torch.Generator()
        # g.manual_seed(self.random_state)
        # random_states = torch.randperm(10000, generator=g)[:self.n_init * self.world_size]
        # random_states = random_states[self.rank:self.n_init * self.world_size:self.world_size]
        self.stats = {'centroids': [], 'inertia': [], 'label': []}
        for run_id in range(self.n_init):  # we can uncover this loop, at cost of more Memory
            random_state = int(random_states[run_id])
            if yl is not None:
                label, centroids, inertia = fit_predict_once_semi(Xu=Xu,
                                                                  Xl=Xl, yl=yl,
                                                                  n_clusters=self.n_clusters,
                                                                  init=self.init,
                                                                  max_iter=self.max_iter,
                                                                  distance_metric=self.distance_metric,
                                                                  verbose=self.verbose,
                                                                  batch_size=batch_size,
                                                                  tol=tol,
                                                                  generator=random_state,
                                                                  run_id=run_id,
                                                                  atomic_op=atomic_op,
                                                                  recover_label=recover_label)
            else:
                label, centroids, inertia = fit_predict_once(X=Xu,
                                                             n_clusters=self.n_clusters,
                                                             init=self.init,
                                                             max_iter=self.max_iter,
                                                             distance_metric=self.distance_metric,
                                                             verbose=self.verbose,
                                                             batch_size=batch_size,
                                                             tol=tol,
                                                             generator=random_state,
                                                             run_id=run_id,
                                                             atomic_op=atomic_op)

            self.stats['centroids'].append(centroids)
            self.stats['inertia'].append(inertia)
            self.stats['label'].append(label)

        # self.stats['centroids'] = torch.stack(self.stats['centroids'])
        # self.stats['label'] = torch.stack(self.stats['label'])
        self.stats['inertia'] = torch.stack(self.stats['inertia'])
        best_inertia, best_run_id = torch.min(self.stats['inertia'], dim=0)
        best_run_id = best_run_id.item()
        best_centroids = self.stats['centroids'][best_run_id]
        best_labels = self.stats['label'][best_run_id]

        if self.verbose:
            print(f"Final min inertia {best_inertia.item()}.")

        self.cluster_centers_ = best_centroids
        if return_centroids:
            return best_labels, best_centroids
        return best_labels


def main_test():
    import time
    import numpy as np
    from sklearn.datasets import make_blobs
    from gnng.metrics.clustering import ClusteringSummary
    from pprint import pprint
    Supervised = True

    gnng.TH_COMPILE = False

    num_clusters = 10
    X, y = make_blobs(
        n_samples=500,
        n_features=32,
        centers=num_clusters,
        cluster_std=1,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=1,
    )  # For reproducibility

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    km = TorchSemiKMeans(
        n_clusters=num_clusters,
        init="k-means++",
        random_state=1,
        n_init=1,
        atomic_op=False,
    )
    print(km)

    if Supervised:  # permutate such that labelled samples are at head
        y = np.array(y)
        labelled_mask = y > num_clusters // 2

        l_targets = y[labelled_mask]
        l_feats = X[labelled_mask]
        u_feats = X[~labelled_mask]

        cat_feats = np.concatenate((l_feats, u_feats))
        # y = np.concatenate((y[labelled_mask], y[~labelled_mask]))
        u_feats = torch.from_numpy(u_feats).to(device)
        l_feats = torch.from_numpy(l_feats).to(device)
        l_targets = torch.from_numpy(l_targets).to(device)

        start_at = time.time()
        pred, centers = km.fit_predict(u_feats, l_feats, l_targets, return_centroids=True)
        end_at = time.time()
        assert torch.all(pred[:l_feats.size(0)] == l_targets)

        ori_nid = torch.arange(X.shape[0], device=device)
        nid = torch.cat([ori_nid[labelled_mask], ori_nid[~labelled_mask]], dim=0)
        pred[nid] = pred.clone()
        assert torch.all(pred != -1)

    else:  # unsupervised
        start_at = time.time()
        pred, centers = km.fit_predict(Xu=torch.from_numpy(X).to(device), return_centroids=True)
        end_at = time.time()

    elapsed = end_at - start_at

    evaluator = ClusteringSummary()
    res = evaluator(pred, torch.from_numpy(y))
    pprint(res)
    # print(
    #     f"Clustering finished in {elapsed:.5f} secs! \n"
    #     f"nmi             : {nmi_score(pred, y):.5f}\n"
    #     f"silhouette_score: {silhouette_score(X, y):.5f}"
    # )

    # Plotting starts here
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import style

    style.use("ggplot")
    colors = list(mcolors.TABLEAU_COLORS)
    X = np.asarray(X)
    centers = np.asarray(centers.cpu())
    for i in range(len(X)):
        x = X[i]
        plt.scatter(x[0], x[1], color=colors[pred[i]], s=10)

    for i in range(num_clusters):
        plt.scatter(centers[i][0], centers[i][1], s=130, marker="*", color="m", edgecolors="k")
    plt.show()


if __name__ == "__main__":
    main_test()
