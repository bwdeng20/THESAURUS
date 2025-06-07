"""
The code is substantially from
https://github.com/sgvaze/generalized-category-discovery/blob/main/methods/clustering/faster_mix_KMeans_pytorch.py
"""

import time
from typing import Union

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import check_random_state
import torch
from tqdm.auto import trange


class InvalidDataError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def pairwise_distance(data1, data2, batch_size=None):
    r"""
    using broadcast mechanism to calculate pairwise Euclidian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    """
    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    if batch_size is None:
        dis = (A - B) ** 2
        # return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0], device=data1.device)
        while i < data1.shape[0]:
            if i + batch_size < data1.shape[0]:
                dis_batch = (A[i: i + batch_size] - B) ** 2
                dis_batch = dis_batch.sum(dim=-1)
                dis[i: i + batch_size] = dis_batch
                i = i + batch_size
                #  torch.cuda.empty_cache()
            elif i + batch_size >= data1.shape[0]:
                dis_final = (A[i:] - B) ** 2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis


class KMeans:
    def __init__(
            self,
            n_clusters: int = None,
            tol: float = 1e-5,
            max_iter: int = 100,
            init: str = "k-means++",
            n_init: int = 1,
            random_state: Union[int, "np.random.RandomState"] = 0,
            n_jobs: int = None,
            pairwise_batch_size: int = None,
            force_on_cpu: bool = True,  # cpu is faster than cuda if num_nodes ~< 50000
            device: Union[str, int, torch.device] = "cuda",
            impl="native",
            verbose: bool = True,
            save_the_last_run: bool = False,
            max_points_per_centroid: int = None,  # only for FAISS impl
            # legacy args
            k: int = None,
            tolerance: float = None,
            max_iterations: int = None,
    ):
        self.k = int(k) if k is not None else int(n_clusters)
        self.tolerance = tolerance if tolerance is not None else tol
        self.max_iterations = max_iterations if max_iterations is not None else max_iter
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pairwise_batch_size = pairwise_batch_size
        self.force_on_cpu = force_on_cpu
        self.verbose = verbose
        self.save_the_last_run = save_the_last_run
        self.max_points_per_centroid = max_points_per_centroid
        self.impl = impl
        if force_on_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def split_for_val(self, l_feats, l_targets, val_prop=0.2):
        np.random.seed(0)

        # Reserve some labelled examples for validation
        num_val_instances = int(val_prop * l_targets.shape[0])
        val_idxs = np.random.choice(range(l_targets.shape[0]), size=(num_val_instances,), replace=False)
        val_idxs.sort()
        remaining_idxs = list(set(range(l_targets.shape[0])) - set(val_idxs.tolist()))
        remaining_idxs.sort()
        remaining_idxs = np.array(remaining_idxs)

        val_l_targets = l_targets[val_idxs]
        val_l_feats = l_feats[val_idxs]

        remaining_l_targets = l_targets[remaining_idxs]
        remaining_l_feats = l_feats[remaining_idxs]

        return remaining_l_feats, remaining_l_targets, val_l_feats, val_l_targets

    def kpp(self, X, pre_centers=None, k=None, random_state=None):
        random_state = check_random_state(random_state)
        if pre_centers is not None:
            C = pre_centers
        else:  # randomly pick one sample as one center
            C = X[random_state.randint(0, X.shape[0])]

        C = C.view(-1, X.shape[1])

        while C.shape[0] < k:
            dist = pairwise_distance(X, C, self.pairwise_batch_size)
            dist = dist.view(-1, C.shape[0])
            d2, _ = torch.min(dist, dim=1)
            prob = d2 / d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = random_state.rand()

            if (cum_prob >= r).nonzero().shape[0] == 0:
                debug = 0  # all samples are almost identical so that no gwclustering is needed.
                # pdb.set_trace()
                raise InvalidDataError(f"All {X.shape} samples are almost identical")
            else:
                ind = (cum_prob >= r).nonzero()[0][0]
            C = torch.cat((C, X[ind].view(1, -1)), dim=0)

        return C

    def fit_once(self, X, random_state):
        centers = torch.zeros(self.k, X.shape[1]).type_as(X)
        labels = -torch.ones(X.shape[0], device=X.device)
        # initialize the centers, the first 'k' elements in the dataset will be our initial centers

        if self.init == "k-means++":
            centers = self.kpp(X, k=self.k, random_state=random_state)

        elif self.init == "random":
            random_state = check_random_state(self.random_state)
            idx = random_state.choice(X.shape[0], self.k, replace=False)
            for i in range(self.k):
                centers[i] = X[idx[i]]

        else:
            for i in range(self.k):
                centers[i] = X[i]

        # begin iterations

        best_labels, best_inertia, best_centers = None, None, None
        if self.verbose:
            iterbar = trange(self.max_iterations, desc="Kmeans one trial")
        else:
            iterbar = range(self.max_iterations)
        for i in iterbar:
            centers_old = centers.clone()
            dist = pairwise_distance(X, centers, self.pairwise_batch_size)
            mindist, labels = torch.min(dist, dim=1)
            inertia = mindist.sum()

            for idx in range(self.k):
                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(X, 0, selected)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            if center_shift ** 2 < self.tolerance:
                # break out of the main loop if the results are optimal, ie.
                # the centers don't change their positions much (more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1

    def fit_mix_once(self, u_feats, l_feats, l_targets, random_state):
        def supp_idxs(c):
            return l_targets.eq(c).nonzero().squeeze(1)

        l_classes = torch.unique(l_targets)
        support_idxs = list(map(supp_idxs, l_classes))
        l_centers = torch.stack([l_feats[idx_list].mean(0) for idx_list in support_idxs])
        cat_feats = torch.cat((l_feats, u_feats))

        centers = torch.zeros([self.k, cat_feats.shape[1]]).type_as(cat_feats)
        centers[: l_classes.shape[0]] = l_centers

        labels = -torch.ones(cat_feats.shape[0]).type_as(cat_feats).long()

        l_classes = l_classes.cpu().long().numpy()
        l_targets = l_targets.cpu().long().numpy()
        l_num = l_targets.shape[0]
        cid2ncid = {cid: ncid for ncid, cid in enumerate(l_classes)}  # Create the mapping table for New cid (ncid)
        for i in range(l_num):
            labels[i] = cid2ncid[l_targets[i]]

        # initialize the centers, the first 'k' elements in the dataset will be our initial centers
        centers = self.kpp(u_feats, l_centers, k=self.k, random_state=random_state)

        # Begin iterations
        best_labels, best_inertia, best_centers = None, None, None
        iter_bar = trange(self.max_iterations, desc="K-means iteration") if self.verbose else range(self.max_iterations)
        for it in iter_bar:
            centers_old = centers.clone()
            dist = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
            u_mindist, u_labels = torch.min(dist, dim=1)
            u_inertia = u_mindist.sum()
            l_mindist = torch.sum((l_feats - centers[labels[:l_num]]) ** 2, dim=1)
            l_inertia = l_mindist.sum()
            inertia = u_inertia + l_inertia
            labels[l_num:] = u_labels

            for idx in range(self.k):
                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(cat_feats, 0, selected)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))

            if center_shift ** 2 < self.tolerance:
                # break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1

    def fit(self, u_feats):  # legacy for clustering code
        self.before_fit(u_feats)
        pred_labels, cluster_centers, inertia, n_iter = self._fit_(u_feats)
        self.post_fit([pred_labels, cluster_centers, inertia, n_iter])
        return pred_labels, cluster_centers, inertia, n_iter

    def post_fit(self, return_list):
        if not self.save_the_last_run:
            self.labels_ = None
            self.inertia_ = None
            self.cluster_centers_ = None
            self.n_iter_ = None
        return_list = [out.to(self.ori_dv) if isinstance(out, torch.Tensor) else out for out in return_list]
        return return_list

    def before_fit(self, features):
        self.ori_dv = features.device

    def _fit_(self, X, use_impl=None):
        random_state = check_random_state(self.random_state)
        best_inertia = None
        use_impl = use_impl or self.impl

        if use_impl == "native":
            X = X.to(self.device)
            if effective_n_jobs(self.n_jobs) == 1:
                for it in range(self.n_init):
                    labels, inertia, centers, n_iters = self.fit_once(X, random_state)
                    if best_inertia is None or inertia < best_inertia:
                        self.labels_ = labels.clone()
                        self.cluster_centers_ = centers.clone()
                        best_inertia = inertia
                        self.inertia_ = inertia
                        self.n_iter_ = n_iters
            else:
                # parallelisation of k-means runs
                seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
                results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.fit_once)(X, seed) for seed in seeds)
                # Get results with the lowest inertia
                labels, inertia, centers, n_iters = zip(*results)
                best = np.argmin(inertia)
                self.labels_ = labels[best]
                self.inertia_ = inertia[best]
                self.cluster_centers_ = centers[best]
                self.n_iter_ = n_iters[best]

        elif use_impl == "faiss":
            import faiss
            X = X.cpu()  # numpy into faiss, numpy out faiss
            max_per_centroid = self.max_points_per_centroid or X.shape[0]
            faiss_kmeans = faiss.Kmeans(
                d=X.shape[-1],
                k=self.k,
                niter=self.max_iterations,
                nredo=self.n_init,
                spherical=False,
                verbose=self.verbose,
                seed=self.random_state,
                max_points_per_centroid=max_per_centroid,
                gpu=1 if self.device.type == "cuda" else False,
            )
            faiss_kmeans.train(X)
            dis2centers, closest_center_index = faiss_kmeans.index.search(X, 1)
            self.cluster_centers_ = torch.from_numpy(faiss_kmeans.centroids)
            self.labels_ = torch.from_numpy(np.ravel(closest_center_index[:, 0]))
            self.inertia_ = faiss_kmeans.obj[-1]
            self.n_iter_ = self.max_iterations  # faiss kmeans no early stopping
        else:
            raise ValueError(f"Kmeans impl can be from ['faiss', 'native'], not {use_impl}!")
        return self.labels_, self.cluster_centers_, self.inertia_, self.n_iter_

    def _fit_mix_(self, u_feats, l_feats, l_targets):
        u_feats = u_feats.to(self.device, non_blocking=True)
        l_feats = l_feats.to(self.device, non_blocking=True)
        l_targets = l_targets.to(self.device, non_blocking=True)
        random_state = check_random_state(self.random_state)
        best_inertia = None
        fit_func = self.fit_mix_once

        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):
                labels, inertia, centers, n_iters = fit_func(u_feats, l_feats, l_targets, random_state)

                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters

        else:
            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(fit_func)(u_feats, l_feats, l_targets, seed) for seed in seeds
            )
            # Get results with the lowest inertia

            labels, inertia, centers, n_iters = zip(*results)
            best = torch.argmin(torch.stack(inertia))
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]

        return self.labels_, self.cluster_centers_, self.inertia_, self.n_iter_

    def fit_predict_unsup(self, X, return_centroids=False):
        self.before_fit(X)
        pred_labels, cluster_centers, inertia, n_iter = self._fit_(X)
        self.post_fit([pred_labels, cluster_centers, inertia, n_iter])
        if return_centroids:
            return pred_labels, cluster_centers
        return pred_labels

    def fit_predict(self, u_feats, l_feats, l_targets, return_centroids=False):
        self.before_fit(u_feats)
        pred_labels, cluster_centers, inertia, n_iter = self._fit_mix_(u_feats, l_feats, l_targets)
        self.post_fit([pred_labels, cluster_centers, inertia, n_iter])
        if return_centroids:
            return pred_labels, cluster_centers
        return pred_labels

    def __repr__(self):
        desc = (
            f"{self.__class__.__name__}(k={self.k}, init={self.init}, max_iterations={self.max_iterations}, "
            f"n_init={self.n_init}, tolerance={self.tolerance}, pairwise_batch_size={self.pairwise_batch_size}, "
            f"random_state={self.random_state}, force_on_cpu={self.force_on_cpu})"
        )
        return desc


class SupKMeans(KMeans):
    def fit(self, X):
        raise NotImplementedError


def main_test():
    from sklearn.datasets import make_blobs
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
    from sklearn.metrics import silhouette_score

    Supervised = True

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
    km = KMeans(
        k=num_clusters,
        init="k-means++",
        random_state=1,
        n_jobs=None,
        pairwise_batch_size=100,
        force_on_cpu=True,
        save_the_last_run=True,
    )

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
        ori_nid = torch.arange(X.shape[0], device=device)

        start_at = time.time()
        km.fit_mix(u_feats, l_feats, l_targets)
        end_at = time.time()

        nid = torch.cat([ori_nid[labelled_mask], ori_nid[~labelled_mask]], dim=0).cpu()
        centers = km.cluster_centers_.cpu()
        # pred = -torch.ones(y.shape[0], dtype=torch.long)
        #
        # pred[nid] = km.labels_.cpu()  # inverse permutate

        pred = km.labels_.cpu()
        pred[nid] = pred.clone()
        assert torch.all(pred != -1)

    else:  # unsupervised
        X = torch.from_numpy(X).to(device)
        start_at = time.time()
        km.fit(X)
        end_at = time.time()
        X = X.cpu()

        centers = km.cluster_centers_.cpu()
        pred = km.labels_.cpu()

    elapsed = end_at - start_at

    print(
        f"Clustering finished in {elapsed:.5f} secs! \n"
        f"nmi             : {nmi_score(pred, y):.5f}\n"
        f"silhouette_score: {silhouette_score(X, y):.5f}"
    )

    # Plotting starts here
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import style

    style.use("ggplot")
    colors = list(mcolors.TABLEAU_COLORS)
    for i in range(len(X)):
        x = X[i]
        plt.scatter(x[0], x[1], color=colors[pred[i]], s=10)

    for i in range(num_clusters):
        plt.scatter(centers[i][0], centers[i][1], s=130, marker="*", color="m", edgecolors="k")
    plt.show()


if __name__ == "__main__":
    main_test()
