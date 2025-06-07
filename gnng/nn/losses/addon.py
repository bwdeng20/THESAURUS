import torch


# jensen-shannon-divergence, not metric, [0, log2] -->usually [0,1]
class JSD(torch.nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


# jensen-shannon-distance=sqrt(JSD), valid metric, [0, sqrt(log2)] -->usually [0,1]
class JSMetric(JSD):
    def forward(self, p: torch.tensor, q: torch.tensor):
        jensen_shannon_divergence = super().forward(p, q)
        return torch.sqrt(jensen_shannon_divergence)
