import torch


class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        pass

    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        return x
