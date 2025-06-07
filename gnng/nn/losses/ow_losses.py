import math
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Union

from torch.nn import CrossEntropyLoss

from gnng.nn.losses.base import GnnGLoss
from gnng.utils import no_bracket_name


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters: int = 3, epsilon: float = 0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits):
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


class G2CrossEntropyLoss(CrossEntropyLoss):
    def __init__(
            self,
            weight: Optional[Tensor] = None,
            size_average=None,
            ignore_index: int = -100,
            reduce=None,
            reduction: str = "mean",
            label_smoothing: float = 0.0,
            temp: float = 1.0,  # by default for GCD methods!
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.temp = temp

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.temp != 1.0:
            input = input / self.temp
        return F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

    def __repr__(self) -> str:
        return (
            f"G2CrossEntropyLoss(weight={self.weight}, "
            f"ignore_index={self.ignore_index}, "
            f"label_smoothing={self.label_smoothing}, "
            f"temp={self.temp}, "
            f"reduction='{self.reduction}')"
        )


class G2MSELoss(torch.nn.MSELoss):

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean", num_classes: int = -1) -> None:
        super().__init__(size_average, reduce, reduction)
        self.num_classes = num_classes

    def forward(self, input, target, num_classes=-1):
        if target.dim() == 1:
            if num_classes == -1:
                num_classes = self.num_classes
            target = torch.nn.functional.one_hot(target, num_classes=num_classes).to(input.dtype)
        return F.mse_loss(input, target.to(input.device), reduction=self.reduction)


class G2MAELoss(torch.nn.L1Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', num_classes: int = -1) -> None:
        super().__init__(size_average, reduce, reduction)
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor, num_classes=-1) -> Tensor:
        if target.dim() == 1:
            if num_classes == -1:
                num_classes = self.num_classes
            target = torch.nn.functional.one_hot(target, num_classes=num_classes).to(input.dtype)
        return F.l1_loss(input, target, reduction=self.reduction)


class SelfDistillLoss(GnnGLoss):
    def __init__(
            self,
            warmup_teacher_temp_epochs: int,
            warmup_teacher_temp: float = 0.07,
            teacher_temp: float = 0.04,
            student_temp: float = 0.1,
            reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp_schedule_warmup = np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs)

    def get_teacher_temp(self, current_epoch: int):
        if current_epoch < self.warmup_teacher_temp_epochs:
            return self.teacher_temp_schedule_warmup[current_epoch]
        else:
            return self.teacher_temp

    def get_full_teacher_temps(self, max_epochs: int):
        return np.concatenate(
            (
                np.linspace(self.warmup_teacher_temp, self.teacher_temp, self.warmup_teacher_temp_epochs),
                np.ones(max_epochs - self.warmup_teacher_temp_epochs) * self.teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between temperatured-softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.get_teacher_temp(epoch)
        teacher_out = F.softmax(teacher_output.detach() / temp, dim=-1)
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        res = self.reduce_output(loss)
        return res


class SelfLabelLoss(GnnGLoss):
    def __init__(self, teacher_temp: float = 0.04, student_temp: float = 0.1, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between temperatured-softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        teacher_out = F.softmax(teacher_output.detach() / self.teacher_temp, dim=-1)
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        res = self.reduce_output(loss)
        return res


class UNOLoss(GnnGLoss):
    def __init__(
            self, known_classes: Union[int, float] = 0.5, temp: float = 0.1, sk_num_iters: int = 3,
            sk_epsilon: float = 0.05
    ):
        super().__init__()
        self.known_classes = known_classes
        self.temp = temp
        self.sk = SinkhornKnopp(sk_num_iters, sk_epsilon)

    def swapped_prediction(self, logits, targets):
        n_view = logits.size(0)
        loss = 0.0
        for view in range(n_view):
            for other_view in range(n_view):
                if view != other_view:
                    loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (n_view * (n_view - 1))

    def cross_entropy_loss(self, preds, targets):  # [n_head, B, K]
        preds = F.log_softmax(preds / self.temp, dim=-1)
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

    def forward(self, view_head_logits, labels, know_mask, out_head_loss_cache=None, nlc=None):
        n_view, n_head, bs, n_classes = view_head_logits.shape
        nlc = nlc or self.known_classes
        if isinstance(nlc, float):
            nlc = int(self.known_classes * n_classes)
        logits_unlab = view_head_logits  # logits_unlab = view_head_logits[..., nlc:]
        # create targets
        targets_lab = F.one_hot(labels[know_mask], num_classes=nlc).to(view_head_logits.device, view_head_logits.dtype)

        targets = torch.zeros_like(view_head_logits, requires_grad=False)
        # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        for v in range(n_view):
            for h in range(n_head):
                targets[v, h, know_mask, :nlc] = targets_lab.type_as(targets)
                # targets[v, h, ~know_mask, nlc:] = self.sk(logits_unlab[v, h, ~know_mask]).type_as(targets)
                targets[v, h, ~know_mask, :] = self.sk(logits_unlab[v, h, ~know_mask]).type_as(targets)

        loss_cluster = self.swapped_prediction(view_head_logits, targets)
        out_head_loss_cache[...] = loss_cluster.detach().clone()
        return loss_cluster.mean()


class SimSiamLoss(GnnGLoss):

    def forward(self, p1, p2, z1, z2):
        loss_b1 = F.cosine_similarity(p1, z2).mean()
        loss_b2 = F.cosine_similarity(p2, z1).mean()
        loss = -(loss_b1 + loss_b2) / 2
        return loss


class EntropyRegLoss(GnnGLoss):
    def __init__(self, reg_temp: float = 0.1, reg_loss_weight: float = 2, plain: bool = False):
        super().__init__(reduction="none")
        self.reg_temp = reg_temp
        self.reg_loss_weight = reg_loss_weight
        self.plain = plain

    def forward(self, h1, h2=None):
        avg_probs = (h1 / self.reg_temp).softmax(dim=1).mean(dim=0)
        if h2 is not None:  # reg on the average of two views
            avg_probs2 = (h2 / self.reg_temp).softmax(dim=1).mean(dim=0)
            avg_probs = (avg_probs + avg_probs2) / 2
        if self.plain:
            me_max_loss = -torch.sum((-avg_probs) * torch.log(avg_probs)) + math.log(avg_probs.shape[0])
        else:
            me_max_loss = -torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(avg_probs.shape[0])
        return me_max_loss * self.reg_loss_weight


class InfoNCELoss(GnnGLoss):
    def __init__(self, tau=0.5, intra_view: bool = True, reduction="mean"):
        super().__init__(reduction=reduction)
        self.intra_view = intra_view
        self.tau = tau

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        return torch.mm(z1, z2.t())

    def loss_single_branch(self, h1, h2):
        bs = h1.shape[0]
        if self.intra_view:
            contrast_feature = torch.cat([h1, h2], dim=0)
        else:
            contrast_feature = h2

        t_sim = torch.div(torch.matmul(h1, contrast_feature.T), self.tau)
        logits = t_sim
        # for numerical stability
        # logits_max, _ = torch.max(t_sim, dim=1, keepdim=True)
        # logits = t_sim - logits_max.detach()

        if self.intra_view:  # mask-out self-contrast cases
            logits.fill_diagonal_(0.0)
            # compute log_prob
            log_prob = logits.diagonal(offset=bs) - torch.logsumexp(logits, dim=-1)
        else:
            log_prob = logits.diagonal() - torch.logsumexp(logits, dim=-1)

        return -log_prob

    def forward(self, h1, h2):
        l1 = self.loss_single_branch(h1, h2)
        l2 = self.loss_single_branch(h2, h1)
        loss = (l1 + l2) * 0.5
        res = self.reduce_output(loss)
        return res


class GeneralSupConLoss(GnnGLoss):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temp=0.07, contrast_mode="all", base_temp=0.07, reduction="mean"):
        super().__init__(reduction=reduction)
        self.temp = temp
        self.contrast_mode = contrast_mode
        self.base_temp = base_temp

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [n_views, bsz, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [n_views, bsz,...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[1]
        if labels is None and mask is not None:
            pass
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, device=device)
        elif labels is not None:
            if labels.ndim > 1: # in case not class index label, but label distribution
                labels =labels.argmax(dim=-1)
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(f"Num of labels {labels.shape[0]} does not match num of features {batch_size}")
            mask = torch.eq(labels, labels.T).to(device)
        else:  # labels is not None and mask is not None
            raise ValueError("Cannot define both `labels` and `mask`")

        contrast_count = features.shape[0]
        contrast_feature = features.view(batch_size * contrast_count, -1)
        if self.contrast_mode == "one":
            anchor_feature = features[0, ...]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temp)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        mask.fill_diagonal_(0.0)
        logits.fill_diagonal_(0.0)
        # compute log_prob
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # compute mean of log-likelihood over all positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temp / self.base_temp) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)
        loss = self.reduce_output(loss)
        return loss


class SemiSupConLoss(GeneralSupConLoss):
    def forward(self, features, y=None, labelled_mask=None):
        # construct extra supervision signals with only training samples, but negative samples can be from unlabelled
        pos_mask = torch.eq(y, y.unsqueeze(dim=1))
        no_avi_label_mask = ~labelled_mask
        pos_mask[no_avi_label_mask] = False
        pos_mask[:, no_avi_label_mask] = False
        pos_mask.fill_diagonal_(True)

        loss = super().forward(features, mask=pos_mask, labels=None)
        return loss


class SupConLoss(GeneralSupConLoss):
    def forward(self, features, y=None, labelled_mask=None):
        # only operate on labelled train nodes specified by labelled_mask
        if labelled_mask is not None:  # (n_view, batch_size,) ... -->  (n_view, labelled_batch_size,)
            features = features[:, labelled_mask, ...]
            y = y[labelled_mask]
        loss = super().forward(features, mask=None, labels=y)
        return loss


class GCDLoss(GnnGLoss):
    def __init__(
            self,
            sup_weight: float = 0.35,
            loss_fn_pred_sup_cls: Optional[Callable] = None,
            loss_fn_view_sup_cls: Optional[Callable] = None,
            loss_fn_self_label: Optional[Callable] = None,  # not UNO+
            loss_fn_self_distill: Optional[Callable] = None,
            loss_fn_uno: Optional[Callable] = None,  # UNO+
            loss_fn_view_entropy_reg: Optional[Callable] = None,
            loss_fn_pred_entropy_reg: Optional[Callable] = None,
            loss_fn_self_con: Optional[Callable] = None,
            loss_fn_sup_con: Optional[Callable] = None,
            loss_fn_simsiam: Optional[Callable] = None,
            loss_fn_entropy_reg: Optional[Callable] = None,  # for legacy Entropy Regularization
            loss_fn_normal_sup_cls: Optional[Callable] = None,  # for legacy Supervised Loss on Prediction
    ):
        super().__init__(reduction="none")
        assert 0.0 <= sup_weight <= 1.0
        self.sup_weight = sup_weight
        self.loss_fn_pred_sup_cls = loss_fn_pred_sup_cls
        if self.loss_fn_pred_sup_cls is None:
            self.loss_fn_pred_sup_cls = loss_fn_normal_sup_cls

        self.loss_fn_view_sup_cls = loss_fn_view_sup_cls
        self.loss_fn_pred_entropy_reg = loss_fn_pred_entropy_reg  # ER on pred
        # only for two-view cases
        self.loss_fn_view_entropy_reg = loss_fn_view_entropy_reg  # ER on multiple views
        if self.loss_fn_view_entropy_reg is None:
            self.loss_fn_view_entropy_reg = loss_fn_entropy_reg
        self.loss_fn_uno = loss_fn_uno
        self.loss_fn_self_label = loss_fn_self_label
        self.loss_fn_self_distill = loss_fn_self_distill
        self.loss_fn_self_con = loss_fn_self_con
        self.loss_fn_sup_con = loss_fn_sup_con
        self.loss_fn_simsiam = loss_fn_simsiam

    def __repr__(self):
        desc = (
            f"{self.__class__.__name__}(sup_weight={self.sup_weight},SupCls={self.loss_fn_pred_sup_cls},"
            f"SupViewCls={self.loss_fn_view_sup_cls},SelfDistill={self.loss_fn_self_distill},"
            f"SelfLabel={self.loss_fn_self_label},SelfLabelUNO={self.loss_fn_uno},"
            f"EntropyReg={self.loss_fn_view_entropy_reg},EntropyRegPred={self.loss_fn_pred_entropy_reg},"
            f"SupContrastive={self.loss_fn_self_con},"
            f"SelfContrastive={self.loss_fn_sup_con},SimSiam={self.loss_fn_simsiam})"
        )
        return desc

    def forward(
            self,
            view_logit=None,  # n_view, batch_size, ...
            view_h=None,
            pred_logit=None,
            y=None,
            labelled_mask=None,
            current_epoch=None,
            return_batch_size=False,
            **kwargs
    ):
        """

        Args:
            view_logit: Tensor [n_view, batch_size, ...]
                The predicted (unnormalized) logits of view augmentations in contrastive learning
            view_h: Tensor [n_view, batch_size, ...]
                The projected hidden features of view augmentations in contrastive learning
            pred_logit:  Tensor [batch_size, ...]
                The predicted (unnormalized) logits of the true input data
            y:  Tensor [batch_size, ...]
                The labels
            labelled_mask: BoolTensor
                indicating the samples with available labels to compute the loss within current stage (train/valid/test)
            current_epoch:  int
            return_batch_size: bool

        Returns:
            Union[Dict[str,Any], Tuple[Dict[str,Any],Dict[str,Any]]]
        """
        if pred_logit is None and view_logit is None and view_h is None:
            if return_batch_size:
                return None, None
            return None

        if labelled_mask is None:
            labelled_mask = Ellipsis

        has_labelled = True
        num_all = y.shape[0]
        if labelled_mask is Ellipsis:
            num_labelled = view_h.shape[1]
        elif labelled_mask.any():
            num_labelled = labelled_mask.sum()
        else:
            has_labelled = False
            num_labelled = 0

        bs_dict = {"batch_size": num_all}
        loss_dict = {}
        if view_logit is not None:
            logit1, logit2 = view_logit.unbind(0)
        else:
            logit1 = logit2 = None

        unnormalized_projected = view_h
        if view_h is not None:
            view_h = torch.nn.functional.normalize(view_h, dim=-1)  # The projected hidden features --> unit sphere
            h1, h2 = view_h.unbind(0)
        else:
            h1 = h2 = None

        # normal supervised training signals
        normal_sup_cls_loss = 0.0
        if (
                self.sup_weight != 0
                and self.loss_fn_pred_sup_cls is not None
                and has_labelled
                and (pred_logit is not None)
                and (y is not None)
        ):
            normal_sup_cls_loss = self.loss_fn_pred_sup_cls(pred_logit[labelled_mask], y[labelled_mask])

            loss_cls_name = no_bracket_name(self.loss_fn_pred_sup_cls)
            loss_dict[loss_cls_name] = normal_sup_cls_loss
            bs_dict[f"{loss_cls_name}_batch_size"] = num_labelled

        # gwclustering, sup
        view_sup_cls_loss = 0.0
        if (
                self.sup_weight != 0
                and self.loss_fn_view_sup_cls is not None
                and has_labelled
                and (logit1 is not None)
                and (y is not None)
        ):
            # sup_logits = torch.cat([logit1[labelled_mask], logit2[labelled_mask]], dim=0) / 0.1
            # sup_labels = y[labelled_mask].tile(1, 2).squeeze(0)  # [y, y]
            # view_sup_cls_loss = self.loss_fn_view_sup_cls(sup_logits, sup_labels)

            sup_logits1 = logit1[labelled_mask]
            yk = y[labelled_mask]
            view_sup_cls_loss = self.loss_fn_view_sup_cls(sup_logits1, yk)
            if logit2 is not None:
                sup_logits2 = logit2[labelled_mask]
                view_sup_cls_loss2 = self.loss_fn_view_sup_cls(sup_logits2, yk)
                view_sup_cls_loss = (view_sup_cls_loss + view_sup_cls_loss2) / 2.0
            loss_cls_name = no_bracket_name(self.loss_fn_view_sup_cls)
            loss_dict[loss_cls_name] = view_sup_cls_loss
            bs_dict[f"{loss_cls_name}_batch_size"] = num_labelled

        # gwclustering, entropy regularization
        me_max_loss = 0.0
        if self.loss_fn_view_entropy_reg is not None and (logit1 is not None):
            me_max_loss = self.loss_fn_view_entropy_reg(logit1, logit2)
            loss_cls_name = no_bracket_name(self.loss_fn_view_entropy_reg)
            loss_dict[loss_cls_name] = me_max_loss

        pred_entropy_loss = 0.0
        if self.loss_fn_pred_entropy_reg is not None and (pred_logit is not None):
            pred_entropy_loss = self.loss_fn_pred_entropy_reg(pred_logit, None)
            loss_cls_name = no_bracket_name(self.loss_fn_pred_entropy_reg)
            loss_dict[loss_cls_name] = pred_entropy_loss

        # The bellowing losses are for two-view contrastive training process and disabled during testing.
        # gwclustering, unsup | Self-Distill proposed in SimGCD
        self_distill_loss = 0.0
        if self.loss_fn_self_distill is not None and (logit1 is not None) and (logit2 is not None):
            cluster_loss2to1 = self.loss_fn_self_distill(logit1, logit2.detach(), current_epoch)
            cluster_loss1to2 = self.loss_fn_self_distill(logit2, logit1.detach(), current_epoch)
            self_distill_loss = (cluster_loss2to1 + cluster_loss1to2) / 2.0
            loss_cls_name = no_bracket_name(self.loss_fn_self_distill)
            loss_dict[loss_cls_name] = self_distill_loss

        self_label_loss = 0.0
        if self.loss_fn_self_label is not None and (logit1 is not None) and (logit2 is not None):
            loss2to1 = self.loss_fn_self_label(logit1, logit2.detach(), current_epoch)
            loss1to2 = self.loss_fn_self_label(logit2, logit1.detach(), current_epoch)
            self_label_loss = (loss2to1 + loss1to2) / 2.0
            loss_cls_name = no_bracket_name(self.loss_fn_self_label)
            loss_dict[loss_cls_name] = self_label_loss
        # gwclustering, unsup | Sinkhorn pseudo labelling (aka, self-label) in SimGCD
        uno_loss = 0.0
        if self.loss_fn_uno is not None and (view_logit is not None):  # need (n_view, n_head, n_batch, n_fea)
            if view_logit.ndim == 3:
                view_logit4uno = view_logit.unsqueeze(1)
            elif view_logit.ndim != 4:
                raise RuntimeError(f"`(n_view, n_head, n_batch, n_fea)` tensor is expected")
            else:
                view_logit4uno = view_logit
            out_head_loss_cache = -torch.ones(view_logit4uno.size(1))
            uno_loss = self.loss_fn_uno(view_logit4uno, y, labelled_mask, out_head_loss_cache=out_head_loss_cache)
            loss_cls_name = no_bracket_name(self.loss_fn_uno)
            loss_dict[loss_cls_name] = uno_loss
            loss_dict["uno_head_loss_cache"] = out_head_loss_cache

        # simsiam contrastive learning
        simsiam_loss = 0.0
        if self.loss_fn_simsiam is not None and (logit1 is not None) and (unnormalized_projected is not None):
            u_h1, u_h2 = unnormalized_projected.detach().unbind(0)
            simsiam_loss = self.loss_fn_simsiam(
                logit1,
                logit2,
                u_h1,
                u_h2,
            )
            loss_cls_name = no_bracket_name(self.loss_fn_simsiam)
            loss_dict[loss_cls_name] = simsiam_loss

        # contrastive learning, unsup
        self_con_loss = 0.0
        if self.loss_fn_self_con is not None and (h2 is not None):  # InfoNCE
            self_con_loss = self.loss_fn_self_con(h1, h2)
            loss_cls_name = no_bracket_name(self.loss_fn_self_con)
            loss_dict[loss_cls_name] = self_con_loss

        # contrastive learning, sup
        sup_con_loss = 0.0  # SupCon or SemiSupCon
        if (
                self.sup_weight != 0
                and self.loss_fn_sup_con is not None
                and has_labelled
                and (h2 is not None)
                and (y is not None)
        ):
            sup_con_loss = self.loss_fn_sup_con(view_h, y=y, labelled_mask=labelled_mask)
            loss_cls_name = no_bracket_name(self.loss_fn_sup_con)

            loss_dict[loss_cls_name] = sup_con_loss
            if isinstance(self.loss_fn_sup_con, SupConLoss):
                bs_dict[f"{loss_cls_name}_batch_size"] = num_labelled

        loss = (1 - self.sup_weight) * (
                uno_loss + self_distill_loss + me_max_loss + self_con_loss + simsiam_loss + pred_entropy_loss
        ) + self.sup_weight * (normal_sup_cls_loss + view_sup_cls_loss + sup_con_loss + self_label_loss)

        loss_dict["loss"] = loss
        loss_dict["sup_weight"] = self.sup_weight
        if return_batch_size:
            return loss_dict, bs_dict
        return loss_dict

    def brief_repr(self):
        desc_list = []
        if self.loss_fn_pred_sup_cls:
            desc_list.append("PredCE")
        if self.loss_fn_view_sup_cls:
            desc_list.append("ViewCE")
        if self.loss_fn_self_distill:
            desc_list.append("SD")
        if self.loss_fn_view_entropy_reg:
            desc_list.append("ViewER")
        if self.loss_fn_pred_entropy_reg:
            desc_list.append("PredER")
        if self.loss_fn_self_label:
            desc_list.append("SL")
        if self.loss_fn_self_con:
            desc_list.append("SSC")
        if self.loss_fn_sup_con:
            desc_list.append("SC")
        if self.loss_fn_uno:
            desc_list.append("UNO")
        if self.loss_fn_simsiam:
            desc_list.append("SSiam")
        desc = "_".join(desc_list)
        return desc
