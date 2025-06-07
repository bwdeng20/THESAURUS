import torch
from gnng._config import get_torch_compile_trigger


# Define a conditional compile decorator based on the global flag
def cd_compile(fn):
    if get_torch_compile_trigger():
        return torch.compile(fn)  # Apply @torch.compile if the flag is True
    else:
        return fn  # Otherwise, return the original function