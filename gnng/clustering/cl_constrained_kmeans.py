import torch
import tqdm
from typing import Union
from gnng.clustering.base import KMeansBase, predict, update_centers
from gnng.clustering.utils import get_group_union_find, index_op_deterministic, parse_th_generator
from gnng.clustering.post_assign import reassign_centroids_th
from gnng.clustering.center_init import initialize_centroids


def check_ml_cl_constraints(assignment, constraints, constraint_type='must_link',
                            return_mask=False):
    if constraints is None:
        return None
    src_lb = assignment[constraints[:, 0]]
    dst_lb = assignment[constraints[:, 1]]
    if constraint_type == 'must_link':
        violate_mask = src_lb != dst_lb
    elif constraint_type == 'cannot_link':
        violate_mask = src_lb == dst_lb
    else:
        raise ValueError("constraint_type must be either 'must_link' or 'cannot_link'")

    if return_mask:
        return violate_mask
    else:
        return constraints[violate_mask, :]


def handle_ml_sn(must_constraints, num_nodes, sn_index=None):
    if must_constraints is None:  # every node is a single SuperNode/Group
        return torch.arange(num_nodes), torch.ones(num_nodes), False

    if sn_index is None:
        # sn_index=[0,0,1,2,2,...] means groups {0,1},{2}, {3,4}...
        sn_index, sn_sizes = get_group_union_find(num_nodes, must_constraints.cpu().numpy())
        sn_sizes = torch.from_numpy(sn_sizes)
        sn_index = torch.from_numpy(sn_index)
    else:  # assuming sn_index has consecutive values, like [3,1,2,0] for 4 SuperNode/Group
        sn_sizes = torch.bincount(sn_index)
    return sn_index, sn_sizes, True


def refine_cannot_link_cop(sn2c_assign_descend, sn2c_dist_descend,
                           sn_cannot_constraints, sn_sizes):
    raise NotImplementedError


def refine_cannot_link_style1(sn2c_assign_descend, sn2c_dist_descend,
                              sn_cannot_constraints, sn_sizes):
    num_sn, n_clusters = sn2c_assign_descend.shape
    sn_assign_ptr = torch.zeros(num_sn, device=sn2c_dist_descend.device,
                                dtype=torch.long, )
    sn_assign = sn2c_assign_descend.gather(1, sn_assign_ptr.unsqueeze(1)).squeeze(1)
    cur_cost_sn = sn2c_dist_descend.gather(1, sn_assign_ptr.unsqueeze(1)).squeeze(1)
    num_violated = 0
    if sn_cannot_constraints is not None:  # otherwise skip the Cannot-Link Refine
        sn_violated_pair = check_ml_cl_constraints(sn_assign, sn_cannot_constraints, constraint_type='cannot_link',
                                                   return_mask=False)
        num_violated = num_violated_vanilla = sn_violated_pair.size(0)

        for it in tqdm.trange(num_violated_vanilla, disable=True):
            if num_violated == 0:
                print(f"Finish CL refine at internal iter {it}/{num_violated_vanilla}.")
                break
            # while num_violated != 0 and reassign_times < num_violated_vanilla:  # change num_violated times can ensure?
            sn_violated, sn_violated_counts = torch.unique(sn_violated_pair, return_counts=True)

            sn_vio2c_dist_descend = sn2c_dist_descend[sn_violated]

            sn_vio_assign_ptr_vio = sn_assign_ptr[sn_violated]
            cur_cost_sn_vio = cur_cost_sn[sn_violated]

            next_c_ptr = torch.clamp(sn_vio_assign_ptr_vio + 1, 0, n_clusters - 1)
            next_cost = sn_vio2c_dist_descend.gather(1, next_c_ptr.unsqueeze(1)).squeeze(1)
            # sn_vio2c_assign_descend = sn2c_assign_descend[sn_violated]
            # next_c = sn_vio2c_assign_descend.gather(1, next_c_ptr.unsqueeze(1)).squeeze(1)
            #
            # # 1. one not-convergent strategy
            # increased_cost_amortized = (next_cost - cur_cost_sn_vio) * sn_sizes[sn_violated] / sn_violated_counts
            # sn2re = sn_violated[torch.argmin(increased_cost_amortized)]

            # 2. also not-convergent but better than 1.
            # sn2re = sn_violated[torch.argmax(sn_violated_counts)]

            # # 3.
            cost_amortized = next_cost * sn_sizes[sn_violated] / sn_violated_counts
            sn2re = sn_violated[torch.argmin(cost_amortized)]

            new_pt = sn_assign_ptr[sn2re] + 1
            sn_assign_ptr[sn2re] = torch.clamp(new_pt, 0, n_clusters - 1)  # assign sn to the next closest center

            sn_assign = sn2c_assign_descend.gather(1, sn_assign_ptr.unsqueeze(1)).squeeze(1)

            sn_violated_pair = check_ml_cl_constraints(assignment=sn_assign,
                                                       constraints=sn_cannot_constraints,
                                                       constraint_type='cannot_link',
                                                       return_mask=False)
            num_violated = sn_violated_pair.size(0)

    return sn_assign, num_violated


def fit_predict_once_ml_cl_style1(X: torch.Tensor,
                                  n_clusters: int,
                                  must_constraints=None,
                                  cannot_constraints=None,
                                  init="k-means++",
                                  tol=1e-4,
                                  generator=None,
                                  run_id=None,
                                  atomic_op: bool = False,
                                  sn_index=None,
                                  sn_sizes=None,
                                  max_iter=None,
                                  distance_metric=None,
                                  verbose=False,
                                  batch_size=-1):
    num_nodes = X.size(0)
    num_features = X.size(1)
    rgen = parse_th_generator(generator, device=X.device)

    # 1 Cope with ML and CL Constraints
    ## 1.1 handle MustLink as SuperNode/group,
    if sn_index is None:
        sn_index, sn_sizes, has_ml = handle_ml_sn(must_constraints, num_nodes, sn_index)

    num_sn = sn_sizes.size(0)
    ## 1.2 DonNOT need translate node-level CannotConstraints into group-level cannot_constraints
    if cannot_constraints is not None:
        sn_cannot_constraints = torch.unique(sn_index[cannot_constraints], dim=0)
    else:
        sn_cannot_constraints = None
    # 2 Preparation
    min_inertia, best_centroids, best_labels, best_sn_assign = float('Inf'), None, None, None

    old_state = initialize_centroids(X, n_clusters=n_clusters, init=init, generator=rgen,
                                     distance_metric=distance_metric)

    state = torch.zeros(n_clusters, num_features, dtype=X.dtype, device=X.device)
    sn2c_dist = torch.zeros(num_sn, n_clusters, device=X.device, dtype=X.dtype)
    # 3 Main SuperNode/Group clustering Iteration
    progress_bar = tqdm.tqdm(total=max_iter, disable=not verbose)
    for n_iter in range(max_iter):
        # compute the distance and inertia with updated centroids Without ML and CL Constraints
        n2c_dist = distance_metric(X, old_state)  # (N,C) node2center distances
        if atomic_op:
            sn2c_dist.zero_()
            sn2c_dist.index_add_(0, sn_index, n2c_dist)  # (M,C) super_node2center distances
            sn2c_dist = sn2c_dist / sn_sizes.view(-1, 1)
        else:
            sn2c_dist = index_op_deterministic(sn_index, n2c_dist, reduce="mean")

        sn2c_dist_descend, sn2c_assign_descend = torch.sort(sn2c_dist, dim=1, descending=False)
        sn_assign, num_violated = refine_cannot_link_style1(
            sn2c_assign_descend=sn2c_assign_descend,
            sn2c_dist_descend=sn2c_dist_descend,
            sn_cannot_constraints=sn_cannot_constraints,
            sn_sizes=sn_sizes)
        # check if refine success!
        if num_violated > 0:
            raise RuntimeError("Cannot-Link Constraints Violated!")

        # recover from sn_assign to node_assign
        labels = sn_assign[sn_index]

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
            progress_bar.set_description(
                f'Redo [{run_id + 1}], iteration {n_iter:03d} with inertia {inertia:.2f}')
            progress_bar.update(n=1)

        # convergence check
        center_shift = distance_metric(old_state, state, pairwise=False)

        center_shift_tot = center_shift.sum()
        if center_shift_tot <= tol:
            if verbose:
                print(
                    f"run {run_id + 1} converged at iter {n_iter}/{max_iter} center shift "
                    f"{center_shift_tot} within tolerance {tol} "
                    f"and min inertia {min_inertia.item()}."
                )
            break
        old_state = state
    progress_bar.close()
    return best_labels, best_centroids, min_inertia


def fit_predict_once_ml_cl_style2(X: torch.Tensor,
                                  n_clusters: int,
                                  must_constraints=None,
                                  cannot_constraints=None,
                                  tol=1e-4,
                                  generator=None,
                                  run_id=None,
                                  atomic_op: bool = False,
                                  sn_index=None,
                                  sn_sizes=None,
                                  max_iter=None,
                                  distance_metric=None,
                                  verbose=False,
                                  batch_size=-1):
    raise NotImplementedError


def parse_fit_predict(cl_method="style1"):
    if cl_method == "style1":
        return fit_predict_once_ml_cl_style1


class TorchCLKmeans(KMeansBase):
    """
    A heuristic constrained Kmeans
    """

    def __init__(self,
                 n_clusters: int = None,
                 link_constraint_method: str = "style1",
                 init: str = 'k-means++',
                 n_init: Union[str, int] = "auto",
                 random_state=0,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 metric='euclidean',
                 verbose: bool = False,
                 k: int = None):
        super().__init__(n_clusters=n_clusters,
                         init=init,
                         n_init=n_init,
                         random_state=random_state,
                         max_iter=max_iter,
                         tol=tol,
                         metric=metric,
                         verbose=verbose,
                         k=k)
        self.link_constraint_method = link_constraint_method

    def fit_predict(self, X: torch.Tensor,
                    must_constraints=None,
                    cannot_constraints=None,
                    adaptive_tol: bool = True,
                    return_centroids: bool = False,
                    atomic_op: bool = False,
                    sn_index=None,
                    batch_size=-1):

        if adaptive_tol:
            tol = torch.mean(torch.var(X, dim=0)).item() * self.tol
        else:
            tol = self.tol

        num_nodes = X.size(0)
        # preprocess ML and CL constraints
        sn_index, sn_sizes, has_ml = handle_ml_sn(must_constraints, num_nodes, sn_index)
        num_sn = sn_sizes.size(0)
        sn_index = sn_index.to(X.device)
        sn_sizes = sn_sizes.to(X.device)

        if num_sn < self.n_clusters:
            raise ValueError(f"{num_sn} grouped MustLink SuperNodes cannot be divided into "
                             f"a smaller number ({self.n_clusters}) of clusters.")

        random_states = torch.arange(self.n_init) + self.random_state
        # g = torch.Generator()
        # g.manual_seed(self.random_state)
        # random_states = torch.randperm(10000, generator=g)[:self.n_init * self.world_size]
        # random_states = random_states[self.rank:self.n_init * self.world_size:self.world_size]
        self.stats = {'centroids': [], 'inertia': [], 'label': []}
        for run_id in range(self.n_init):  # we can uncover this loop in the future, at cost of more Memory
            random_state = int(random_states[run_id])

            if self.link_constraint_method == 'style1':
                label, centroids, inertia = fit_predict_once_ml_cl_style1(X=X,
                                                                          n_clusters=self.n_clusters,
                                                                          init=self.init,
                                                                          tol=tol,
                                                                          generator=random_state,
                                                                          sn_index=sn_index,
                                                                          sn_sizes=sn_sizes,
                                                                          must_constraints=None,
                                                                          cannot_constraints=cannot_constraints,
                                                                          run_id=run_id,
                                                                          atomic_op=atomic_op,
                                                                          max_iter=self.max_iter,
                                                                          distance_metric=self.distance_metric,
                                                                          verbose=self.verbose,
                                                                          batch_size=batch_size
                                                                          )
            else:
                raise NotImplementedError

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


if __name__ == '__main__':
    from gnng.clustering.data import generate_simulated_clustering_data

    num_clusters = 4
    points_per_cluster = 5000
    dimensions = 2
    cluster_std = 2
    must_link_ratio = 0.0001  # 每个簇内 5% 的可能 must_link 对
    cannot_link_ratio = 0.000001  # 所有不同簇之间 5% 的可能 cannot_link 对
    random_seed = 123

    X_data, y_label, must_link, cannot_link = generate_simulated_clustering_data(
        num_clusters=num_clusters,
        points_per_cluster=points_per_cluster,
        dimensions=dimensions,
        cluster_std=cluster_std,
        must_link_ratio=must_link_ratio,
        cannot_link_ratio=cannot_link_ratio,
        random_seed=random_seed
    )

    must_link = torch.unique(must_link, dim=0)
    cannot_link = torch.unique(cannot_link, dim=0)
    print(f"生成的数据点数量: {X_data.shape[0]}")
    print(f"数据维度: {X_data.shape[1]}")
    print(f"must_link关系数量: {must_link.shape[0]}")
    print(f"cannot_link关系数量: {cannot_link.shape[0]}")
    ml_vio = check_ml_cl_constraints(y_label, constraints=must_link, constraint_type="must_link")
    assert ml_vio.numel() == 0

    cl_vio = check_ml_cl_constraints(y_label, constraints=cannot_link, constraint_type="cannot_link")
    assert cl_vio.numel() == 0

    clustering_model = TorchCLKmeans(metric='euclidean',
                                     init='k-means++',
                                     random_state=0,
                                     n_clusters=4,
                                     n_init="auto",
                                     max_iter=300,
                                     tol=1e-4,
                                     verbose=True)
    pred, centers = clustering_model.fit_predict(X_data.cuda(),
                                                 # must_constraints=must_link,
                                                 cannot_constraints=cannot_link,
                                                 adaptive_tol=False, atomic_op=False,
                                                 return_centroids=True)

    from gnng.metrics.clustering import ClusteringSummary
    from gnng.visualization import plot_graph
    from pprint import pprint

    evaluator = ClusteringSummary()
    res = evaluator(pred, y_label)
    pprint(res)

    plot_graph(X=X_data.cpu().numpy(),
               labels=pred.cpu().numpy(),
               cluster_centers=centers.cpu().numpy(),
               display=True)
