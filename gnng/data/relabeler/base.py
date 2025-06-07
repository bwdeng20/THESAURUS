from __future__ import annotations

from abc import ABC, abstractmethod
from gnng.typing import PyGData


class LabelTransform(ABC):
    """Base class for graph augmentors."""

    def __init__(self, copy=True):
        self.copy = copy

    @abstractmethod
    def transform_label(self, g: PyGData) -> PyGData:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(self, g) -> PyGData:
        if self.copy:
            new_g = g.detach().clone()
        else:
            new_g = g
        return self.transform_label(new_g)