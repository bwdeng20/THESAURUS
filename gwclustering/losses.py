import pdb
import warnings
import torch
import numpy as np
from gnng.pyg_ext import power_ew
from gnng.nn.losses.base import GnnGLoss
from typing import Optional, Callable
import torch.nn.functional as F
from gnng.utils import no_bracket_name

class KLRegularizedFGW(torch.nn.Module):
    def __init__(
            self,
            num_iters: int = 5,
            alpha: float = 0.5,  # weight of topology
            epsilon: float = 0.01,
            sinkhorn_num_iters: int = 5,
            rel_epsilon: bool = True,
            cache_transport: bool = False,
            cache_sinkhorn: bool = False,
            use_primal_sinkhorn: bool = False,
            use_KL_regularization: bool = True,
            verbose: bool = False,
    ):
        """
        `cache_transport` and `cache_sinkhorn` are not supported for batched input !!!
        """
        super().__init__()
        assert 0 <= alpha < 1, "alpha can be 0.9999999 for numerical stability."
        self.num_iters = num_iters
        self.sinkhorn_num_iters = sinkhorn_num_iters
        self.alpha = alpha
        self.epsilon = epsilon
        self.rel_epsilon = rel_epsilon
        self.cache_transport = cache_transport
        self.cache_sinkhorn = cache_sinkhorn
        self.use_primal_sinkhorn = use_primal_sinkhorn
        self.use_KL_regularization = use_KL_regularization

        self.Q = None
        self.u = None
        self.v = None

        self.verbose = verbose

    @torch.no_grad()
    def sinkhorn_KL_dual(self, a, b, M, Q0, u=None, v=None):
        """
        Args:
            a: Tensor [num_heads, num_prototypes]
                The marginal of prototypes
            b: Tensor [batch_size]
                The marginal of samples
            M: Tensor [num_heads, num_prototypes, batch_size]
                The gradient of FGW target functions
            Q0: Tensor [num_heads, num_prototypes, batch_size]
                The reference point in KL divergence
            u: Tensor [num_heads, num_prototypes]
                The itermediate vector of the sinkhorn algorithm
            v: Tensor [num_heads, batch_size]
                The itermediate vector of the sinkhorn algorithm
        Returns:
            Q: Tensor [num_heads, num_prototypes, batch_size]
                The transportation Q that **minimizes** <M, Q>
        """
        H, K = a.shape
        (B,) = b.shape

        if self.rel_epsilon:
            mx = M[M != torch.inf].abs().max()
            eps = mx * self.epsilon  # 2-20
        else:
            eps = self.epsilon

        if u is None:
            u = torch.ones(H, K, dtype=M.dtype, device=M.device) / K
        if v is None:
            v = torch.ones(H, B, dtype=M.dtype, device=M.device) / B

        if self.use_KL_regularization:
            Ker = torch.exp(M / (-eps)) * Q0
        else:
            Ker = torch.exp(M / (-eps))

        # Kp = Ker / a[:, :, None]
        Kp = Ker / b[None, None, :]
        # u:HK = a:HK / (K:HKB @ v:HB)
        # v:HB = b:B / (K:HKB.T @ u:HK)

        err = 1
        for ii in range(self.sinkhorn_num_iters):
            # pdb.set_trace()
            uprev = u
            vprev = v
            # KtransposeU = (u[:, None, :] @ Ker).squeeze(dim=-2)
            KV = (Ker @ v[:, :, None]).squeeze(dim=-1)
            # v = b[None, :] / KtransposeU
            u = a / KV
            # u = 1.0 / (Kp @ v[:, :, None]).squeeze(dim=-1)
            v = 1 / (u[:, None, :] @ Kp).squeeze(dim=-2)
            if (
                    # ((KtransposeU.isinf()).any())
                    ((KV.isinf()).any())
                    # or ((KtransposeU.isnan()).any())
                    or ((KV.isnan()).any())
                    or u.isnan().any()
                    or v.isnan().any()
                    or u.isinf().any()
                    or v.isinf().any()
            ):
                # pdb.set_trace()
                u = uprev
                v = vprev
                break
        return u[:, :, None] * Ker * v[:, None, :], u, v

    @torch.no_grad()
    def sinkhorn_KL_primal(self, a, b, M, Q0, u=None, v=None):
        """
        Args:
            a: Tensor [num_heads, num_prototypes]
                The marginal of prototypes
            b: Tensor [batch_size]
                The marginal of samples
            M: Tensor [num_heads, num_prototypes, batch_size]
                The gradient of FGW target functions
            Q0: Tensor [num_heads, num_prototypes, batch_size]
                The reference point in KL divergence
            u: Tensor [num_heads, num_prototypes]
                The itermediate vector of the sinkhorn algorithm
            v: Tensor [num_heads, batch_size]
                The itermediate vector of the sinkhorn algorithm
        Returns:
            Q: Tensor [num_heads, num_prototypes, batch_size]
                The transportation Q that **minimizes** <M, Q>
        """
        H, K = a.shape
        (B,) = b.shape

        if self.rel_epsilon:
            mx = M[M != torch.inf].abs().max()
            eps = mx * self.epsilon
        else:
            eps = self.epsilon

        if self.use_KL_regularization:
            Ker = torch.exp(M / (-eps)) * Q0
        else:
            Ker = torch.exp(M / (-eps))

        sum_Ker = torch.sum(Ker)
        Ker /= sum_Ker

        tol = 1e-6
        for it in range(self.sinkhorn_num_iters):
            Ker_prev = Ker
            # normalize each row: total weight per prototype must be 1/K
            Ker /= torch.sum(Ker, dim=-1, keepdim=True)
            Ker = Ker * a[:, :, None]

            # normalize each column: total weight per sample must be 1/B
            Ker /= torch.sum(Ker, dim=-2, keepdim=True)
            Ker = Ker * b[None, None, :]

            if Ker.isinf().any() or Ker.isnan().any():
                warnings.warn("Iterative Bregman Projection Unstable")
                # if self.training:
                #     pdb.set_trace()
                # else:
                #     Ker = Q0
                Ker = Ker_prev
                break

            if it % 10 == 0:
                err = torch.norm(Ker_prev - Ker)
                if err <= tol:
                    if self.verbose:
                        print(
                            f"[{it}/{self.sinkhorn_num_iters}] GW INNER sinkhorn loop: "
                            f"{err:.7f}<= tol = {tol:.7f}, early stopping."
                        )
                    break

        return Ker

    @torch.no_grad()
    def forward(self, cost, adj_proto, adj_sample, proto_weight=None, sample_weight=None):
        if isinstance(adj_sample, torch.Tensor):
            return self.forward_with_torch_native(cost, adj_proto, adj_sample, proto_weight, sample_weight)
        else:
            return self.forward_with_torch_sparse(cost, adj_proto, adj_sample, proto_weight, sample_weight)

    @torch.no_grad()
    def forward_with_torch_native(self, cost, adj_proto, adj_sample, proto_weight=None, sample_weight=None):
        """
        Args:
            cost: Tensor [num_heads, num_prototypes, batch_size]
            adj_proto: Tensor [num_heads, num_prototypes, num_prototypes]
            adj_sample: (Sparse) Tensor [batch_size, batch_size]
            proto_weight: Tensor [num_heads, num_prototypes]
            sample_weight: Tensor [batch_size]

        Returns:
            Q: Tensor [num_heads, num_prototypes, batch_size]
                The pseudo label based on FGW optimal transport,
                    that is, for each head, each sample, Q sum to 1 along the prototypes
        """
        if adj_proto.dim() == 2:
            adj_proto = adj_proto[None, :]
        if cost.dim() == 2:
            cost = cost[None, :]

        C = cost.transpose(-1, -2)
        H, K, B = C.shape

        if proto_weight is None:
            proto_weight = torch.ones(H, K, device=C.device) / K
        if sample_weight is None:
            sample_weight = torch.ones(B, device=C.device) / B

        if self.training and self.cache_transport and self.Q is not None:
            Q = self.Q
        else:
            Q = proto_weight[:, :, None] * sample_weight[None, None, :]

        const_grad_1 = adj_proto ** 2 @ proto_weight[:, :, None]
        const_grad_2 = (sample_weight @ adj_sample ** 2)[None, None, :]

        if self.training and self.cache_sinkhorn and self.u is not None:
            sinkhorn_u = self.u
        else:
            sinkhorn_u = torch.ones(H, K, dtype=C.dtype, device=C.device) / K

        if self.training and self.cache_sinkhorn and self.v is not None:
            sinkhorn_v = self.v
        else:
            sinkhorn_v = torch.ones(H, B, dtype=C.dtype, device=C.device) / B

        cpt = 0
        tol = 1e-6
        err = 1e15
        while cpt < self.num_iters and err > tol:
            tens = (
                    2 * self.alpha * (-2 * (adj_proto @ Q @ adj_sample) + const_grad_1 + const_grad_2)
                    + (1 - self.alpha) * C
            )
            Qprev = Q

            if self.use_primal_sinkhorn:
                Q = self.sinkhorn_KL_primal(proto_weight, sample_weight, tens, Q, sinkhorn_u, sinkhorn_v)
            else:
                Q, sinkhorn_u, sinkhorn_v = self.sinkhorn_KL_dual(
                    proto_weight, sample_weight, tens, Q, sinkhorn_u, sinkhorn_v
                )
            cpt += 1
            if cpt % 10 == 0:
                rel = (Q - Qprev) / Q
                rel = torch.nan_to_num(rel, 0.0, 0.0, 0.0)
                err = torch.abs(rel).mean()
                if err <= tol:
                    if self.verbose:
                        print(f"[{cpt}/{self.num_iters}]GW OUTER-loop: {err:.7f}<= tol = {tol:.7f}, early stopping.")
                    break

        if self.training and self.cache_transport:
            self.Q = Q
        # if self.training and self.cache_sinkhorn:
        #     self.u = sinkhorn_u
        #     self.v = sinkhorn_v

        err = torch.abs(torch.sum(Q, dim=(-1, -2)) - 1)
        if (err > 1e-5).any():
            msg = (
                f"Not converged! error: {err}. You might want to increase the "
                f"regularization parameter `epsilon` {self.epsilon}."
            )
            raise RuntimeError(msg)
        # pdb.set_trace()
        return Q.transpose(-1, -2) / sample_weight.unsqueeze(-1)

    @torch.no_grad()
    def forward_with_torch_sparse(self, cost, adj_proto, adj_sample, proto_weight=None, sample_weight=None):
        """
        Args:
            cost: Tensor [num_heads, num_prototypes, batch_size]
            adj_proto: Tensor [num_heads, num_prototypes, num_prototypes]
            adj_sample: SparseTensor [batch_size, batch_size]
            proto_weight: Tensor [num_heads, num_prototypes]
            sample_weight: Tensor [batch_size]

        Returns:
            Q: Tensor [num_heads, num_prototypes, batch_size]
                The pseudo label based on FGW optimal transport,
                    that is, for each head, each sample, Q sum to 1 along the prototypes
        """
        if adj_proto.dim() == 2:
            adj_proto = adj_proto[None, :]
        if cost.dim() == 2:
            cost = cost[None, :]

        C = cost.transpose(-1, -2)
        H, K, B = C.shape

        if proto_weight is None:
            proto_weight = torch.ones(H, K, device=C.device) / K
        if sample_weight is None:
            sample_weight = torch.ones(B, device=C.device) / B

        if self.training and self.cache_transport and self.Q is not None:
            Q = self.Q
        else:
            Q = proto_weight[:, :, None] * sample_weight[None, None, :]

        const_grad_1 = adj_proto ** 2 @ proto_weight[:, :, None]

        # const_grad_2 = (sample_weight @ adj_sample **2 )[None, None, :]
        try:
            const_grad_2 = (power_ew(adj_sample, 2) @ sample_weight[None, :, None]).reshape(1, 1, B)
        except Exception:
            pdb.set_trace()
        if self.training and self.cache_sinkhorn and self.u is not None:
            sinkhorn_u = self.u
        else:
            sinkhorn_u = torch.ones(H, K, dtype=C.dtype, device=C.device) / K

        if self.training and self.cache_sinkhorn and self.v is not None:
            sinkhorn_v = self.v
        else:
            sinkhorn_v = torch.ones(H, B, dtype=C.dtype, device=C.device) / B

        cpt = 0
        tol = 1e-6
        err = 1e15
        while cpt < self.num_iters and err > tol:
            tens = (
                    2
                    * self.alpha
                    * (
                            -2 * (adj_proto @ (adj_sample @ Q.transpose(-1, -2)).transpose(-1, -2))
                            + const_grad_1
                            + const_grad_2
                    )
                    + (1 - self.alpha) * C
            )
            Qprev = Q

            if self.use_primal_sinkhorn:
                Q = self.sinkhorn_KL_primal(proto_weight, sample_weight, tens, Q, sinkhorn_u, sinkhorn_v)
            else:
                Q, sinkhorn_u, sinkhorn_v = self.sinkhorn_KL_dual(
                    proto_weight, sample_weight, tens, Q, sinkhorn_u, sinkhorn_v
                )
            cpt += 1
            if cpt % 10 == 0:
                rel = (Q - Qprev) / Q
                rel = torch.nan_to_num(rel, 0.0, 0.0, 0.0)
                err = torch.abs(rel).mean()
                if err <= tol:
                    if self.verbose:
                        print(f"[{cpt}/{self.num_iters}]GW OUTER-loop: {err:.7f}<= tol = {tol:.7f}, early stopping.")
                    break

        if self.training and self.cache_transport:
            self.Q = Q
        # if self.training and self.cache_sinkhorn:
        #     self.u = sinkhorn_u
        #     self.v = sinkhorn_v

        err = torch.abs(torch.sum(Q, dim=(-1, -2)) - 1)
        if (err > 1e-5).any():
            msg = (
                f"Not converged! error: {err}. You might want to increase the "
                f"regularization parameter `epsilon` {self.epsilon}."
            )
            raise RuntimeError(msg)
        # pdb.set_trace()
        return Q.transpose(-1, -2) / sample_weight.unsqueeze(-1)

    def clear_cache(self, clear_sinkhorn=True, clear_transport=True):
        if clear_sinkhorn:
            self.u = None
            self.v = None
        if clear_transport:
            self.Q = None


# TODO remove
def check_dict_nan(loss_dict):
    not_valid_flag = torch.zeros(len(loss_dict), dtype=torch.bool)
    loss_names = list(loss_dict.keys())
    for i, name in enumerate(loss_names):
        v = loss_dict[name]
        if not isinstance(v, float):
            if torch.any(torch.isnan(v)):
                not_valid_flag[i] = True
        else:
            if np.any(np.isnan(v)):
                not_valid_flag[i] = True
    if not_valid_flag.any():
        msg = ""
        for i in not_valid_flag.nonzero().ravel().tolist():
            name = loss_names[i]
            msg += f"{name} is {loss_dict[name]}"
        raise ValueError(msg)


class GWPseudoLabelLossv2(GnnGLoss):
    def __init__(
            self,
            temp: float = 0.1,
            gw_num_iters: int = 5,
            gw_alpha: float = 0.5,  # (0.1,1,0.1)
            gw_epsilon: float = 0.01,
            rel_epsilon: bool = False,
            sinkhorn_num_iters: int = 5,
            use_primal_sinkhorn: bool = True,
            use_KL_regularization: bool = True,
    ):
        super().__init__()

        self.temp = temp
        self.fgw = KLRegularizedFGW(
            num_iters=gw_num_iters,
            alpha=gw_alpha,
            epsilon=gw_epsilon,
            sinkhorn_num_iters=sinkhorn_num_iters,
            rel_epsilon=rel_epsilon,
            use_primal_sinkhorn=use_primal_sinkhorn,
            use_KL_regularization=use_KL_regularization,
            cache_transport=False,
            cache_sinkhorn=False,
            verbose=False,
        )

    def swapped_prediction(self, logits, targets):
        n_view = logits.size(0)
        loss = 0.0
        for view in range(n_view):
            for other_view in range(n_view):
                if view != other_view:
                    loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (n_view * (n_view - 1))  # (n_head,)

    def cross_entropy_loss(self, preds, targets):  # [n_head, B, K]
        preds = F.log_softmax(preds / self.temp, dim=-1)
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

    def forward(
            self,
            view_head_logits,
            proto_topo=None,
            data_topo=None,
            proto_weight=None,
            out_head_loss_cache=None,
            *args,
            **kwargs,
    ):
        n_view, n_head, bs, n_classes = view_head_logits.shape
        cost = -view_head_logits.clone().detach().requires_grad_(False)
        targets = torch.zeros_like(cost, requires_grad=False)
        for v in range(n_view):
            targets[v, ...] = self.fgw(cost[v, ...], proto_topo[v], data_topo[v], proto_weight[v]).type_as(targets)

        loss_head = self.swapped_prediction(view_head_logits, targets)
        out_head_loss_cache[...] = loss_head.detach().clone()
        return loss_head.mean()


class GWClusteringLossv2(GnnGLoss):
    def __init__(self, sup_weight: float = 0.35,
                 loss_fn_gw_pseudo: Optional[Callable] = None):
        super().__init__(reduction="none")
        self.sup_weight = sup_weight
        self.loss_fn_gw_pseudo = loss_fn_gw_pseudo

    def forward(
            self,
            view_logit=None,  # n_view, batch_size, ...
            view_h=None,
            pred_logit=None,
            pred_h=None,
            proto_topo=None,
            data_topo=None,
            proto_weight=None,
            current_epoch=None,
            return_batch_size=False,
    ):
        if pred_logit is None and view_logit is None and view_h is None:
            if return_batch_size:
                return None, None
            return None

        num_all = pred_h.size(-2)

        bs_dict = {"batch_size": num_all}
        loss_dict = {}
        if view_h is not None:
            view_h = torch.nn.functional.normalize(view_h, dim=-1)

        ##########################################################
        # !ADDED GW
        gw_pseudo_loss = 0.0
        if (
            self.loss_fn_gw_pseudo is not None
            and view_h is not None
            and proto_topo is not None
            and data_topo is not None
        ):
            if view_h.ndim == 3:
                view_h4gw = view_h.unsqueeze(1)
            elif view_h.ndim != 4:
                raise RuntimeError(f"`(n_view, n_head, n_batch, n_fea)` tensor is expected")
            else:
                view_h4gw = view_h
            out_head_loss_cache = -torch.ones(view_h4gw.size(1))
            gw_pseudo_loss = self.loss_fn_gw_pseudo(
                view_h4gw,
                proto_topo,
                data_topo,
                proto_weight,
                current_epoch=current_epoch,
                out_head_loss_cache=out_head_loss_cache,
            )

            loss_cls_name = no_bracket_name(self.loss_fn_gw_pseudo)
            loss_dict[loss_cls_name] = gw_pseudo_loss
            loss_dict["gw_head_loss_cache"] = out_head_loss_cache

        loss = (1 - self.sup_weight) * (gw_pseudo_loss)
        loss_dict["loss"] = loss
        loss_dict["sup_weight"] = self.sup_weight

        check_dict_nan(loss_dict)
        if return_batch_size:
            return loss_dict, bs_dict
        return loss_dict
