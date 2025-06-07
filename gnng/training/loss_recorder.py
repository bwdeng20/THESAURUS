from typing import List, Any, Dict, Union, Optional
import torch
from copy import deepcopy
from torch import Tensor


class LossRecorder:
    def __init__(
        self,
        fabric_instance=None,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        mode="mean",
        compute_on_cpu: bool = False,
    ) -> None:
        self.mode = mode
        self.compute_on_cpu = compute_on_cpu
        self.fabric_instance = fabric_instance
        self.prefix = self._check_arg(prefix, "prefix")
        self.postfix = self._check_arg(postfix, "postfix")
        self.loss_dict: Dict[str, List[Tensor]] = {
            "loss": [],
        }
        self.bs_dict: Dict[str, List[Tensor]] = {
            "loss": [],
        }

    def update(
        self, loss_dict: Union[Tensor, float, Dict[str, Any], None], bs_dict: Union[int, Tensor, Dict[str, Any]] = None
    ) -> None:

        if not loss_dict:  # None or empty dict
            return

        if isinstance(bs_dict, (int, Tensor)):
            batch_size_c = torch.as_tensor(bs_dict)
        elif isinstance(bs_dict, Dict) and bs_dict:  # common batch size for most losses in `loss_dict`
            batch_size_c = torch.as_tensor(bs_dict.get("batch_size"))
        elif bs_dict is None:
            batch_size_c = torch.as_tensor(loss_dict.pop("batch_size"))
        else:
            raise TypeError
        if self.compute_on_cpu:
            batch_size_c = batch_size_c.cpu()

        # single loss
        if isinstance(loss_dict, (Tensor, float)):
            to_record = loss_dict * batch_size_c if self.mode == "mean" else loss_dict
            to_record = torch.as_tensor(to_record).detach()
            if self.compute_on_cpu:
                to_record = to_record.cpu()
            self.loss_dict["loss"].append(to_record)
            self.bs_dict["loss"].append(batch_size_c)
            return
        # multiple losses
        for loss_name, avg_b_loss in loss_dict.items():
            if "loss" not in loss_name.lower() or avg_b_loss is None:
                continue  # in case loss_dict has other non-losses, just skip
            avg_b_loss = torch.as_tensor(avg_b_loss).detach()
            # try to find the batch size of `loss_name(_batch_size)`; otherwise `common batch size`
            bs = bs_dict.get(loss_name, bs_dict.get(f"{loss_name}_batch_size", batch_size_c))
            bs = torch.as_tensor(bs)
            if self.compute_on_cpu:
                avg_b_loss = avg_b_loss.cpu()
                bs = bs.cpu()
            to_record = avg_b_loss * bs if self.mode == "mean" else avg_b_loss
            if loss_name not in self.loss_dict:
                self.loss_dict[loss_name] = [to_record]
                self.bs_dict[loss_name] = [bs]
            else:
                self.loss_dict[loss_name].append(to_record)
                self.bs_dict[loss_name].append(bs)

    def compute(self):
        loss_dict = {}
        num_sample_dict = {}
        for loss_name, avg_b_loss_list in self.loss_dict.items():
            bs_list = self.bs_dict[loss_name]
            if len(bs_list) == 0:
                continue
            assert len(avg_b_loss_list) == len(bs_list)
            loss_accum = torch.stack(avg_b_loss_list).sum()
            bs_accum = torch.stack(bs_list).sum()

            if self.fabric_instance is not None:
                epoch_loss_sum = self.fabric_instance.all_reduce(loss_accum, reduce_op="sum")
                epoch_bs_sum = self.fabric_instance.all_reduce(bs_accum, reduce_op="sum")
                epoch_avg_loss = epoch_loss_sum / epoch_bs_sum
                bs_accum = epoch_bs_sum
            else:
                epoch_avg_loss = loss_accum / bs_accum

            loss_dict[loss_name] = epoch_avg_loss
            num_sample_dict[loss_name] = bs_accum

        loss_dict = {self._set_name(k): v for k, v in loss_dict.items()}
        num_sample_dict = {self._set_name(f"num_{k}"): v for k, v in num_sample_dict.items()}
        return loss_dict, num_sample_dict

    def _set_name(self, base: str) -> str:
        """Adjust name of metric with both prefix and postfix."""
        name = base if self.prefix is None else self.prefix + base
        return name if self.postfix is None else name + self.postfix

    @staticmethod
    def _check_arg(arg: Optional[str], name: str) -> Optional[str]:
        if arg is None or isinstance(arg, str):
            return arg
        raise ValueError(f"Expected input `{name}` to be a string, but got {type(arg)}")

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None):
        mc = deepcopy(self)
        if prefix:
            mc.prefix = self._check_arg(prefix, "prefix")
        if postfix:
            mc.postfix = self._check_arg(postfix, "postfix")
        return mc

    def reset(self):
        self.loss_dict.clear()
        self.loss_dict["loss"] = []
        self.bs_dict.clear()
        self.bs_dict["loss"] = []

    def __repr__(self) -> str:
        """Return the representation of the metric collection including all metrics in the collection."""
        repr_str = f"{self.__class__.__name__}(\n{'_'.join(list(self.loss_dict.keys()))}"
        if self.prefix:
            repr_str += f",\n  prefix={self.prefix}{',' if self.postfix else ''}"
        if self.postfix:
            repr_str += f"{',' if not self.prefix else ''}\n  postfix={self.postfix}"
        return repr_str + "\n)"


if __name__ == "__main__":
    num_batch = 10
    device = torch.device("cuda")
    losses = torch.rand(num_batch, device=device) * 10
    bses = torch.tensor([5] * (num_batch - 1) + [2], device=device)

    recorder = LossRecorder(prefix="train/", postfix="_hhh", compute_on_cpu=False)

    loss_accum = 0.0
    num_accum = 0
    for i in range(num_batch):
        loss = losses[i] * bses[i]
        loss_accum = loss_accum + loss
        num_accum = num_accum + bses[i]

        recorder.update(losses[i], bses[i])

    print(f"{loss_accum:.5f}, {num_accum: 5d}, {loss_accum /num_accum: .5f}")
    print(f"{recorder.compute()}")
