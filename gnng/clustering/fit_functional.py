import torch
import tqdm
from typing import Optional, Union
from torch import Tensor
from gnng.clustering.base import predict, update_centers
from gnng.clustering.utils import parse_th_generator
from gnng.clustering.center_init import initialize_centroids
from gnng.clustering.post_assign import reassign_centroids_th


@torch.no_grad()
def fit_predict_once(X: torch.Tensor,
                     n_clusters: int,
                     init="k-means++",
                     tol=1e-4,
                     generator=None,
                     run_id=None,
                     atomic_op: Optional[Union[bool, str]] = None,
                     max_iter=100,
                     distance_metric=None,
                     verbose=False,
                     batch_size=-1):
    min_inertia, best_centroids, best_labels = float('Inf'), None, None
    rgen = parse_th_generator(generator, device=X.device)

    old_state = initialize_centroids(X, n_clusters=n_clusters, init=init,
                                     generator=rgen,
                                     distance_metric=distance_metric)
    old_labels, _ = predict(X, old_state, distance_metric=distance_metric, batch_size=batch_size)
    labels = old_labels

    # pre allocate centroids (state) and histgram (counts)
    state = torch.zeros(n_clusters, X.size(1), dtype=X.dtype, device=X.device)
    progress_bar = tqdm.tqdm(total=max_iter, disable=not verbose)
    for n_iter in range(max_iter):
        # reinit the centroids and histgram to fill in
        # update centroids in a group parallel way
        #   https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/7
        # index_add & scatter_add NOT deterministic on CUDA as explained
        #   https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/7
        state, counts = update_centers(X=X, labels=labels,
                                       n_clusters=n_clusters,
                                       state=state, atomic_op=atomic_op)

        # tackle empty cluster like FAISS, change counts and state inplace
        reassign_centroids_th(counts, state, generator=rgen)

        # compute the distance and inertia with updated centroids
        labels, inertia = predict(X, state, distance_metric=distance_metric,
                                  batch_size=batch_size)

        if inertia < min_inertia:
            min_inertia = inertia
            best_centroids, best_labels = state, labels
            if verbose:
                print(f"Iter: {n_iter}, {min_inertia} --> {inertia}")

        if verbose:
            progress_bar.set_description(
                f'Redo [{run_id + 1}], iteration {n_iter:03d} with inertia {inertia:.2f}')
            progress_bar.update(n=1)

        # convergence check
        center_shift = distance_metric(old_state, state, pairwise=False)
        # if torch.equal(labels, old_labels):
        #     # First check the labels for strict convergence.
        #     if verbose:
        #         print(
        #             f"run {run_id + 1} converged at iter {n_iter}/{max_iter} strict convergence.")
        #     break
        # else:
        # No strict convergence, check for tol based convergence.
        # center_shift_tot = (center_shift ** 2).sum()
        center_shift_tot = center_shift.sum()
        if center_shift_tot <= tol:
            if verbose:
                print(
                    f"run {run_id + 1} converged at iter {n_iter}/{max_iter} center shift "
                    f"{center_shift_tot} within tolerance {tol} "
                    f"and min inertia {min_inertia.item()}."
                )
            break
        old_labels[:] = labels
        old_state = state
    progress_bar.close()
    return best_labels, best_centroids, min_inertia


@torch.no_grad()
def fit_predict_once_semi(Xu: Tensor,
                          Xl: Tensor,
                          yl: Tensor,
                          n_clusters: int,
                          init="k-means++",
                          tol=1e-4,
                          generator=None,
                          run_id=None,
                          atomic_op: Optional[Union[bool, str]] = None,
                          max_iter=100,
                          distance_metric=None,
                          verbose=False,
                          batch_size=-1,
                          l_classes=None,
                          recover_label=True):
    dv = Xu.device
    rgen = parse_th_generator(generator, device=dv)

    num_u = Xu.size(0)
    num_l = Xl.size(0)
    d = Xl.size(-1)
    num_all = num_l + num_u

    X = torch.cat([Xl, Xu])

    all_labels = torch.empty(num_all, dtype=torch.long, device=dv).fill_(-1)
    l_classes = torch.unique(yl) if l_classes is None else l_classes  # eg [3,5,6], n_clusters=8
    num_l_class = l_classes.size(0)
    raw2new_lbl = torch.empty(n_clusters, dtype=torch.long,
                              device=dv).fill_(-1)
    raw2new_lbl[l_classes] = torch.arange(num_l_class, device=dv, dtype=torch.long)  # eg [-1,-1,0,-1,1,2,-1,-1]
    yl_new = raw2new_lbl[yl]
    all_labels[:num_l] = yl_new # reorder the labels such that the known labels are put in the first `num_l_class` eles

    l_centers = torch.full((num_l_class, d), -1, dtype=X.dtype, device=dv)

    l_centers, l_counts = update_centers(Xl, yl_new, n_clusters=num_l_class,
                                         state=l_centers, atomic_op=atomic_op)

    old_state = initialize_centroids(Xu, n_clusters=n_clusters, init=init,
                                     generator=rgen,
                                     distance_metric=distance_metric,
                                     pre_init_centers=l_centers)

    state = old_state.detach().clone()

    labels_u, inertia_u = predict(Xu, state, distance_metric=distance_metric, batch_size=batch_size)
    all_labels[num_l:] = labels_u
    min_inertia, best_centroids, best_labels = float('Inf'), old_state, all_labels
    progress_bar = tqdm.tqdm(total=max_iter, disable=not verbose)
    for n_iter in range(max_iter):
        # compute the distance and inertia with updated centroids
        labels_u, inertia_u = predict(Xu, state, distance_metric=distance_metric, batch_size=batch_size)

        inertia_l = distance_metric(Xl, state[yl_new], pairwise=False).sum()
        inertia = inertia_u + inertia_l
        all_labels[num_l:] = labels_u

        state, counts = update_centers(X, all_labels, n_clusters, state, atomic_op)
        # tackle empty cluster like FAISS, change counts and state inplace
        reassign_centroids_th(counts, state, generator=rgen)

        if inertia < min_inertia:
            min_inertia = inertia
            best_centroids, best_labels = state, all_labels
            if verbose:
                print(f"Iter: {n_iter}, {min_inertia} --> {inertia}")

        if verbose:
            progress_bar.set_description(
                f'Redo [{run_id + 1}], iteration {n_iter:03d} with inertia {inertia:.2f}')
            progress_bar.update(n=1)

        # convergence check
        center_shift = distance_metric(old_state, state, pairwise=False)

        # center_shift = self.distance_metric(old_state, state).diag().sum()
        # No strict convergence, check for tol based convergence.
        # center_shift_tot = (center_shift ** 2).sum()
        center_shift_tot = center_shift.sum()
        if center_shift_tot <= tol:
            if verbose:
                print(
                    f"run {run_id + 1} converged at iter {n_iter}/{max_iter} center shift "
                    f"{center_shift_tot} within tolerance {tol} "
                    f"and min inertia {min_inertia.item()}."
                )
            break
        old_state[...] = state
    progress_bar.close()
    if recover_label:
        # eg, raw2new_lbl = [-1,-1,0,-1,1,2,-1,-1], n_clusters=8
        new2raw_lbl = torch.full_like(raw2new_lbl, -1)
        new2raw_lbl[:num_l_class] = l_classes  # eg [3,5,6], n_clusters=8

        all_classes = torch.arange(n_clusters, device=dv)
        # eg, new2raw_lbl = [3,5,6] + [0, 1, 2, 4, 7]
        new2raw_lbl[num_l_class:] = all_classes[~torch.isin(all_classes, l_classes)]

        best_labels = new2raw_lbl[best_labels]
        best_centroids = best_centroids[new2raw_lbl]  # right?

    assert best_centroids is not None
    return best_labels, best_centroids, min_inertia
