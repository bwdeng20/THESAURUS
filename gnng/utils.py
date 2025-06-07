from typing import Optional, Union, Dict, Any
from pathlib import Path
import copy
import numpy as np
import torch
import fsspec
from fsspec.core import url_to_fs
from torch import Tensor
from torch.nn.utils import weight_norm, remove_weight_norm

PathStr = Union[Path, str]


def describe_np_array(arr):
    # 计算常见的统计信息
    count = arr.size  # 元素个数
    mean = np.mean(arr)  # 均值
    std_dev = np.std(arr)  # 标准差
    var = np.var(arr)  # 方差
    min_val = np.min(arr)  # 最小值
    max_val = np.max(arr)  # 最大值
    sum_val = np.sum(arr)  # 总和
    median = np.median(arr)  # 中位数
    percentile_25 = np.percentile(arr, 25)  # 25百分位数
    percentile_75 = np.percentile(arr, 75)  # 75百分位数

    # 将统计信息格式化为字符串
    result = (f"count: {count}\n"
              f"mean: {mean}\n"
              f"std: {std_dev}\n"
              f"var: {var}\n"
              f"min: {min_val}\n"
              f"25%: {percentile_25}\n"
              f"50% (median): {median}\n"
              f"75%: {percentile_75}\n"
              f"max: {max_val}\n"
              f"sum: {sum_val}\n")

    return result


def are3masks_disjoint(mask1, mask2, mask3):
    assert are_masks_disjoint(mask1, mask2)
    assert are_masks_disjoint(mask1, mask3)
    assert are_masks_disjoint(mask2, mask3)
    return True


def are_masks_disjoint(mask1: Tensor, mask2: Tensor):
    assert mask1.dtype == mask2.dtype == torch.bool
    m1_or_m2 = torch.logical_or(mask1, mask2)
    return torch.all(torch.logical_xor(mask1[m1_or_m2], mask2[m1_or_m2]))


def index2mask(index, num_nodes: Optional[int] = None):
    num_nodes = index.max() + 1 if num_nodes is None else num_nodes

    if isinstance(index, Tensor):
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=index.device)
    else:
        mask = np.zeros(num_nodes, dtype=bool)
    mask[index] = True
    return mask


def to_mask(split, num_nodes: Optional[int] = None):
    if isinstance(split, torch.BoolTensor):
        return split
    elif isinstance(split, torch.LongTensor):
        return index2mask(split, num_nodes)
    else:
        raise TypeError


def mask2index(mask):
    return mask.nonzero().ravel()


def is_sparse_arr_symmetric(spm):
    return (abs(spm - spm.T) > 1e-12).nnz == 0


def normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def normalize_md_class_name(model):
    return normalize_string(model.__class__.__name__)


def parse_compile_kwargs(compile_kwargs: Optional[Union[Dict[str, Any], bool]] = None):
    if compile_kwargs is None:
        return False, {}
    elif isinstance(compile_kwargs, bool):
        return compile_kwargs or False, {}
    else:
        compile_kwargs = compile_kwargs or {}
        return len(compile_kwargs) > 0, compile_kwargs


def _is_local_file_protocol(path: PathStr) -> bool:
    return fsspec.utils.get_protocol(str(path)) == "file"


def get_filesystem(path: PathStr, **kwargs: Any):
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs


def update_dict_no_collision(dict1, dict2):
    for k, v in dict2.items():
        if k in dict1:
            raise KeyError(f"{k} already in the first dict")
        else:
            dict1[k] = v
    return dict1


def nested_any_dict2nested_str_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            nested_any_dict2nested_str_dict(v)
        elif isinstance(v, str):
            pass
        else:  # classes
            d[k] = str(v)
    return d


def scalar_tensor2pynum(maybe_tensor):
    if isinstance(maybe_tensor, Tensor) and maybe_tensor.numel() == 1:
        return maybe_tensor.item()
    else:
        return maybe_tensor


def report_df2dict(df, decimal_precision=2, exclude=("loss_", "seed", "time_fit", "time_test")):
    string_cols = df.select_dtypes(include=["object"]).columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

    def asset_priority(name, priority_keys=("test", "repr")):
        priority = 0
        for pk in priority_keys:
            if pk in name:
                priority += 1
        return priority

    numeric_cols = sorted(numeric_cols, key=asset_priority, reverse=True)
    result = {}
    cached = {}
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        # always cache others for later collection
        cached[f"mean_{col}"] = mean
        cached[f"std_{col}"] = std
        if col.lower() in exclude:
            scale = 1.0
        else:
            scale = 100.0

        repr_mean = np.around(mean * scale, decimals=decimal_precision).astype(str)
        repr_std = np.around(std * scale, decimals=decimal_precision).astype(str)
        # always put repr at first
        result[f"repr1_{col}"] = f"{repr_mean} ({repr_std})"
        result[f"repr2_{col}"] = f"{repr_mean}±{repr_std}"
    result.update(dict(df[string_cols].to_dict("list")))
    result.update(cached)
    return result


def no_bracket_name(loss_instance):
    name = loss_instance.__class__.__name__
    return name.replace("()", "")


def sanitize_dict(dd, warning=False):
    new_dict = {}
    for k, v in dd.items():
        if v is None:
            if warning:
                print(f"{k} is None")
        else:
            new_dict[k] = v
    return new_dict


def is_weight_norm_applied2thislevel(module):
    p_weight_g = getattr(module, "weight_g", None)
    p_weight_v = getattr(module, "weight_v", None)
    if p_weight_g is not None and p_weight_v is not None:
        print(f"Layer {module} has weight normalization.")
        return True
    return False


def deepcopy_wn(model):
    """
    Deepcopy a model, removing and then reapplying weight normalization.

    Args:
        model (nn.Module): The original model with weight normalization.

    Returns:
        nn.Module: A deep copy of the model with weight normalization reapplied.
    """
    # 记录应用了weight norm的子模块名称和其requires_grad属性
    weight_norm_modules = {}

    # 遍历模型中的每个子模块
    for name, layer in model.named_modules():
        if is_weight_norm_applied2thislevel(layer):
            # 记录子模块的名称、weight_g和weight_v的requires_grad属性
            weight_norm_modules[name] = {
                "layer": layer,
                "weight_g_requires_grad": layer.weight_g.requires_grad,
                "weight_v_requires_grad": layer.weight_v.requires_grad,
                "weight_g_data": layer.weight_g.data,
                "weight_v_data": layer.weight_v.data,
            }
            remove_weight_norm(layer)  # 移除权重归一化

    # 深拷贝模型
    new_model = copy.deepcopy(model)

    # 在新模型中重新应用权重归一化，并恢复requires_grad属性
    for name, layer in new_model.named_modules():
        if name in weight_norm_modules:
            layer = weight_norm(layer)  # 重新应用weight norm
            # 设置requires_grad属性
            layer.weight_g.requires_grad = weight_norm_modules[name]["weight_g_requires_grad"]
            layer.weight_v.requires_grad = weight_norm_modules[name]["weight_v_requires_grad"]
            layer.weight_g.data = weight_norm_modules[name]["weight_g_data"]
            layer.weight_v.data = weight_norm_modules[name]["weight_v_data"]
    return new_model


def set_weight_g_fixed(model):
    """
    将模型中所有被weight norm应用的层的weight_g设置为不可学习，且值全为1。

    Args:
        model (nn.Module): 包含weight norm的模型。
    """
    for layer in model.modules():
        if hasattr(layer, "weight_g"):
            # 设置weight_g为不可学习
            layer.weight_g.requires_grad = False
            # 将weight_g的值设置为1
            layer.weight_g.data.fill_(1)


index_to_mask = index2mask
mask_to_index = mask2index
