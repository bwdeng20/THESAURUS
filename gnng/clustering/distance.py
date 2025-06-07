import torch
import torch.nn.functional as F


def pairwise_cosine(x1: torch.Tensor, x2: torch.Tensor, pairwise=True):
    if not pairwise:
        return 1 - F.cosine_similarity(x1, x2)
    else:
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        return 1 - x1 @ x2.T


def pairwise_p(x1: torch.Tensor, x2: torch.Tensor, pairwise=True, p=2, sqrt=True):
    if not pairwise:
        res = torch.norm(x1 - x2, p=p, dim=-1)
    else:
        res = torch.cdist(x1, x2, p=p)
    return res if sqrt else torch.pow(res, p)


def pairwise_euclidean(x1: torch.Tensor, x2: torch.Tensor, pairwise=True, sqrt=True):
    if not pairwise:
        res = torch.norm(x1 - x2, p=2, dim=-1)
    else:
        res = torch.cdist(x1, x2, p=2.)
    return res if sqrt else torch.pow(res, 2)
