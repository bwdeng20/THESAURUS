import torch

def reassign_centroids_th(hassign, centroids, generator=None):
    """Reassign centroids when some of them collapse, optimized for PyTorch (CUDA) tensors."""
    device = centroids.device

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)  # For reproducibility in tests

    k, d = centroids.shape
    nsplit = 0

    empty_cents = torch.nonzero(hassign == 0).ravel()
    if empty_cents.numel() == 0:
        return nsplit

    fac = torch.ones(d, device=device)
    fac[::2] += 1 / 1024.
    fac[1::2] -= 1 / 1024.

    while empty_cents.numel() > 0:
        probas = (hassign.float() - 1).clamp(min=0)
        probas_sum = probas.sum()
        if probas_sum == 0:
            break
        probas /= probas_sum

        nnz = (probas > 0).sum().item()
        nreplace = min(nnz, empty_cents.numel())
        # print(f"torch nreplace : {nreplace}")
        cjs = torch.multinomial(probas,
                                nreplace, replacement=False, generator=generator)

        ci = empty_cents[:nreplace]
        cj = cjs

        c = centroids[cj]
        centroids[ci] = c * fac
        centroids[cj] = c / fac

        hassign_new = (hassign[cj] // 2)
        hassign[ci] = hassign_new
        hassign[cj] -= hassign_new
        nsplit += nreplace

        empty_cents = empty_cents[nreplace:]

    return nsplit
