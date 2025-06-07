import torch
import numpy as np
from .utils import stable_cumsum, parse_th_generator
from .distance import pairwise_euclidean
from gnng._compile import cd_compile

@cd_compile
def kmeans_plus_plus(X,
                     n_clusters,
                     generator,
                     pairwise_distance,
                     pre_init_centers=None,
                     n_local_trials=None):

    n_samples, n_features = X.shape
    num_pre = pre_init_centers.size(0) if pre_init_centers is not None else 0

    rgen = parse_th_generator(generator, device=X.device)
    centers = torch.empty((n_clusters, n_features), dtype=X.dtype, device=X.device).fill_(0.)
    if num_pre > 0:
        centers[:num_pre] = pre_init_centers
    indices = torch.full((n_clusters,), -1, dtype=torch.int, device=X.device)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    if num_pre == 0:
        # Pick first center randomly and track index of point
        #     center_id = random_state.randint(n_samples)
        center_id = torch.randint(n_samples, (1,), generator=rgen, device=X.device)
        centers[0] = X[center_id]
        indices[0] = center_id

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = pairwise_distance(centers[0, None], X)
        current_pot = closest_dist_sq.sum()
    else:
        closest_dist_sq = pairwise_distance(centers[num_pre - 1, None], X)
        current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(num_pre, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        #         rand_vals = random_state.random_sample(n_local_trials) * current_pot
        rand_vals = torch.rand(n_local_trials, generator=rgen, device=X.device) * current_pot

        candidate_ids = torch.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        torch.clip(candidate_ids, None, closest_dist_sq.numel() - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = pairwise_distance(
            X[candidate_ids], X)

        # update the closest distances squared and potential for each candidate
        torch.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(dim=1)

        # Decide which candidate is the best
        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        indices[c] = best_candidate
        centers[c] = X[best_candidate]

    return centers, indices, num_pre

@cd_compile
def pick_random_centers(X,
                        n_clusters,
                        pre_init_centers=None,
                        generator=None):
    rgen = parse_th_generator(generator, device=X.device)

    num_pre = pre_init_centers.size(0) if pre_init_centers is not None else 0
    num_to_pick = n_clusters - num_pre

    indices = torch.randperm(X.size(0), generator=rgen)[:num_to_pick]
    centers = torch.empty(n_clusters, X.size(1), dtype=X.dtype, device=X.device)
    if num_pre > 0:
        centers[:num_pre] = pre_init_centers
    centers[num_pre:] = X[indices]
    return centers, indices, num_pre


def initialize_centroids(X: torch.Tensor,
                         n_clusters,
                         init='k-means++',
                         pre_init_centers=None,
                         distance_metric=None,
                         generator=None):
    distance_metric = pairwise_euclidean if distance_metric is None else distance_metric
    if isinstance(init, str):
        rgen = parse_th_generator(generator, device=X.device)
        if init == 'random':
            centers, indices, num_pre = pick_random_centers(X, n_clusters, pre_init_centers, generator)
        elif init == 'k-means++':
            centers, indices, num_pre = kmeans_plus_plus(X,
                                                         generator=rgen,
                                                         n_clusters=n_clusters,
                                                         pre_init_centers=pre_init_centers,
                                                         pairwise_distance=distance_metric)
        else:
            raise NotImplementedError
    elif isinstance(init, (np.ndarray, torch.Tensor)):
        centers = torch.as_tensor(init, device=X.device)
        if pre_init_centers is not None:
            raise RuntimeError('`pre_init_centers` is overwritten by `init`')
    else:
        raise NotImplementedError
    return centers
