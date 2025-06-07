from typing import List, Any, Dict, Optional, Iterable
import torch
from torchmetrics.utilities import dim_zero_cat
from copy import deepcopy


class BatchOutputRecorder:
    def __init__(
        self,
        fabric_instance=None,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        store_on_cpu: bool = False,
    ) -> None:
        self.store_on_cpu = store_on_cpu
        self.fabric_instance = fabric_instance
        self.prefix = self._check_arg(prefix, "prefix")
        self.postfix = self._check_arg(postfix, "postfix")
        self.output_dict: Dict[str, List[Any]] = {}

    def update(
        self,
        batch_return: Dict[str, Any],
    ) -> None:
        if not batch_return:  # None or empty dict
            return

        for name, val in batch_return.items():
            if isinstance(val, torch.Tensor) and self.store_on_cpu:
                val = val.to("cpu")
            val = val.detach()
            if name not in self.output_dict:
                self.output_dict[name] = [val]
            else:
                self.output_dict[name].append(val)

    def compute(self, query=None, set_fix=True):
        out_dict = {}
        query = query if query is not None else self.output_dict.keys()
        if isinstance(query, str):
            query = [query]
        queried_dict = {k: self.output_dict[k] for k in query}

        if self.fabric_instance is not None:
            gather_dict = self.fabric_instance.all_gather(queried_dict)
        else:
            gather_dict = queried_dict

        for name, bo_list in gather_dict.items():
            if len(bo_list) == 0:
                continue
            try:
                bo_cat = dim_zero_cat(bo_list)
            except Exception:
                bo_cat = bo_list
            out_dict[name] = bo_cat

        if set_fix:
            return_dict = {self._set_name(k): v for k, v in out_dict.items()}
        else:
            return_dict = out_dict
        return return_dict

    def _set_name(self, base: str) -> str:
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
        self.output_dict.clear()

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}(\n{'|'.join(list(self.output_dict.keys()))}"
        if self.prefix:
            repr_str += f",\n  prefix={self.prefix}{',' if self.postfix else ''}"
        if self.postfix:
            repr_str += f"{',' if not self.prefix else ''}\n  postfix={self.postfix}"
        return repr_str + "\n)"


if __name__ == "__main__":
    num_batch = 3
    recorder = BatchOutputRecorder(prefix="val/", postfix="_ttt", store_on_cpu=False)

    for i in range(num_batch):
        fake_bo = {
            "pred": torch.rand(i + 1, 2),
            "labelled_mask": torch.rand(i + 1) > 0.0,
            "name": f"nn{i}nn",
        }
        recorder.update(fake_bo)
    print(f"{recorder.compute()}")
