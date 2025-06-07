import torch
from gnng.clustering.base import KMeansBase
from gnng.clustering.fit_functional import fit_predict_once


class TorchKMeans(KMeansBase):
    def __init__(self,
                 n_clusters=None,
                 metric='euclidean',
                 init='k-means++',
                 random_state=0,
                 n_init="auto",
                 max_iter=100,
                 tol=1e-4,
                 verbose=True,
                 k: int = None):
        super().__init__(n_clusters=n_clusters,
                         init=init,
                         random_state=random_state,
                         n_init=n_init,
                         max_iter=max_iter,
                         metric=metric,
                         tol=tol,
                         verbose=verbose,
                         k=k)

    def fit_predict(self, X: torch.Tensor,
                    adaptive_tol: bool = True,
                    return_centroids: bool = False,
                    atomic_op: bool = False,
                    batch_size=-1):

        if adaptive_tol:
            tol = torch.mean(torch.var(X, dim=0)) * self.tol
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
            label, centroids, inertia = fit_predict_once(X=X,
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


if __name__ == '__main__':
    torch.cuda.set_device(0)
    from torch_clustering.kmeans import PyTorchKMeans as OldKmeans

    g = torch.Generator()
    g.manual_seed(2025)
    X = torch.randn(10000, 256, generator=g).cuda()

    # old_model = OldKmeans(metric='cosine',
    #                       init='k-means++',
    #                       random_state=0,
    #                       n_clusters=1000,
    #                       n_init=10,
    #                       max_iter=300,
    #                       tol=1e-4,
    #                       verbose=True)
    # old_model.fit_predict(X)

    clustering_model = TorchKMeans(metric='euclidean',
                                   init='k-means++',
                                   random_state=0,
                                   n_clusters=500,
                                   n_init=2,
                                   max_iter=300,
                                   tol=1e-4,
                                   verbose=True)
    pred1 = clustering_model.fit_predict(X, adaptive_tol=True, atomic_op=True)
    # print(torch.bincount(pred1))

    pred2 = clustering_model.fit_predict(X, adaptive_tol=False, atomic_op=False)

    assert torch.equal(pred1, pred2)
