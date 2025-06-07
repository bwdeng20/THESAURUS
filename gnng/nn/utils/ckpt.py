import warnings
import torch
from typing import Dict, Optional
from torch.nn import Module
from gnng.typing import ParsableCkpt


def load_th_or_pl_ckpt2md(model: Module, ckpt: Optional[ParsableCkpt] = None, map_location: Optional[str] = None):
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt is None:
        warnings.warn(f"No checkpoint is loaded for {type(model)}!")
        return model.to(map_location)
    if not isinstance(ckpt, Dict):
        loaded = torch.load(ckpt, map_location=map_location)
    else:
        loaded = ckpt

    model_state_dict = loaded["state_dict"]
    is_ckpt_lit = "pytorch-lightning_version" in loaded
    if is_ckpt_lit:
        state_dict_wo_prefix = {k.split(".", 1)[1]: v for k, v in model_state_dict.items()}
    else:
        state_dict_wo_prefix = model_state_dict
    assert isinstance(state_dict_wo_prefix, Dict)
    model.load_state_dict(state_dict_wo_prefix)
    return model
