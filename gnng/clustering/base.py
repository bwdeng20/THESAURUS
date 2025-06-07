import torch.distributed as dist
from .distance import *
from functools import partial
import numpy as np
from gnng.clustering.utils import index_op_deterministic
from typing import Union


def update_centers(X, labels, n_clusters, state, atomic_op: Union[bool, str] = "deterministic"):
    counts = torch.bincount(labels, minlength=n_clusters)
    if atomic_op == True:
        state.zero_()
        fac = counts.detach().clone()
        fac[fac == 0] = 1  # empty cluster
        state.index_add_(0, labels, X)
        state = state / fac.view(-1, 1)

    elif atomic_op == "deterministic":
        # torch.use_deterministic_algorithms(True)
        state.zero_()
        fac = counts.detach().clone()
        fac[fac == 0] = 1  # empty cluster
        state.index_add_(0, labels, X)
        state = state / fac.view(-1, 1)
        # torch.use_deterministic_algorithms(False)

    else:  # deterministic alternative operator via Sparse matrix multiplication, torch_sparse.SparseTensor
        state = index_op_deterministic(labels, X, shape=(n_clusters, X.size(0)),reduce="mean")
    return state, counts


def predict(X: torch.Tensor, cluster_centers_, distance_metric, batch_size=-1):
    if batch_size == -1:
        d = distance_metric(X, cluster_centers_)
        inertia, pred_labels = d.min(dim=1)
        inertia = inertia.sum()
    else:
        split_size = min(batch_size, X.size(0))
        inertia, pred_labels = 0., []
        for f in X.split(split_size, dim=0):  # batched prediction
            d = distance_metric(f, cluster_centers_)
            inertia_, labels_ = d.min(dim=1)
            inertia += inertia_.sum()
            pred_labels.append(labels_)
        pred_labels = torch.cat(pred_labels, dim=0)
    return pred_labels, inertia


class KMeansBase(torch.nn.Module):
    """
    A basic clustering algorithm (such as K-Means) for grouping data into clusters.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form. Must be greater than 0.

    init : {'k-means++', 'random', callable, or array-like of shape (n_clusters, n_features)}, default='k-means++'
        Method for initialization:
        - 'k-means++': Selects initial cluster centers for k-means clustering in a smart way to speed up convergence.
        - 'random': Chooses `n_clusters` observations at random from the data for the initial centroids.
        - callable: A function that takes X, n_clusters, and a random_state as input and returns the initial centroids.
        - array-like of shape (n_clusters, n_features): A fixed set of initial centroids.

    n_init : int, default=10
        The number of times the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output in terms of inertia from the n_init runs.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic. For reproducibility, pass a RandomState instance or None.

    max_iter : int, default=300
        The maximum number of iterations of the k-means algorithm for a single run.

    tol : float, default=1e-4
        Tolerance for convergence. If the change in inertia is less than this value, the algorithm will stop.

    verbose : bool, default=False
        Verbosity mode. If greater than 0, the algorithm will print progress information.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features), optional
        The final cluster centers of the fitted model.

    Notes
    -----
    The algorithm follows the standard K-Means method (or any chosen variant)
    for clustering. Convergence is determined by the inertia (the sum of squared
    distances from each point to its assigned cluster center).
    """

    def __init__(self,
                 n_clusters: int,
                 init: str = 'k-means++',
                 n_init: Union[str, int] = "auto",
                 random_state=0,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 metric='euclidean',
                 verbose: bool = False,
                 k: int = None,
                 atomic_op: Union[bool, str] = True,
                 ):
        super().__init__()
        self.n_clusters = n_clusters if k is None else k
        if k is not None and n_clusters is not None:
            assert k == n_clusters, f"k: {k} != n_clusters: {n_clusters}"

        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.init = init
        self.random_state = random_state
        self.verbose = verbose  # Assuming a default value for verbosity

        self.n_init = 1
        self.parse_n_init(n_init)
        self.distance_metric = self.parse_distance_func(metric)

        # runtime cache
        self.stats = None

        # atomic_op style, .e.g, torch.index_add
        self.atomic_op = atomic_op

    def fit_predict_once(self, X, *args, **kwargs):
        """ run with one seed and initialization"""
        raise NotImplementedError

    def fit_predict(self, X, *args, **kwargs):
        """ run with multiple seeds and initializations"""
        raise NotImplementedError

    def parse_n_init(self, n_init):
        if n_init == "auto":
            if isinstance(self.init, (np.ndarray, torch.Tensor)) or self.init.lower() == "k-means++":
                self.n_init = 1
        elif isinstance(n_init, int):
            self.n_init = n_init
        else:
            self.n_init = 10  # by default 10 runs with different seeds
        return self.n_init

    def reset_parameters(self):
        return

    @staticmethod
    def parse_distance_func(dist_name):
        if dist_name == 'euclidean' or dist_name == 2:
            func = pairwise_euclidean
        elif dist_name == 'cosine':
            func = pairwise_cosine
        elif isinstance(dist_name, int):  # p-norm, p in [1,\inf]
            func = partial(pairwise_p, p=dist_name)
        else:
            raise NotImplementedError
        return func

    # def predict(self, X: torch.Tensor, cluster_centers_=None, batch_size=-1):
    #     if cluster_centers_ is None:
    #         cluster_centers_ = self.cluster_centers_
    #     return predict(X=X, cluster_centers_=cluster_centers_,
    #                    distance_metric=self.distance_metric,
    #                    batch_size=batch_size)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"n_clusters={self.n_clusters}, "
                f"metric='{self.distance_metric}', "
                f"init={self.init}, "
                f"random_state={self.random_state}, "
                f"n_init={self.n_init}, "
                f"max_iter={self.max_iter}, "
                f"tol={self.tol}, "
                f"verbose={self.verbose})"
                )


class DistClustering(KMeansBase):
    def __init__(self,
                 n_clusters,
                 init='k-means++',
                 n_init=10,
                 random_state=0,
                 max_iter=300,
                 tol=1e-4,
                 distributed=True,
                 verbose=True):
        super(DistClustering, self).__init__(n_clusters, init, n_init, random_state, max_iter, tol)

        self.is_root_worker = True if not dist.is_initialized() else (dist.get_rank() == 0)
        self.verbose = verbose and self.is_root_worker
        self.distributed = distributed and dist.is_initialized()
        if verbose and self.distributed and self.is_root_worker:
            print('Perform K-means in distributed mode.')
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0

    def distributed_sync(self, tensor):
        tensors_gather = [-torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.stack(tensors_gather)
        return output

    def fit_predict(self, X):
        raise NotImplementedError
