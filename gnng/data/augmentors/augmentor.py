from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import List
from gnng.typing import PyGData


class Augmentor(ABC):
    """Base class for graph augmentors."""

    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: PyGData) -> PyGData:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(self, g) -> PyGData:
        return self.augment(g)


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: PyGData) -> PyGData:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


class RandomChoice(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int):
        super(RandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, g: PyGData) -> PyGData:
        num_augmentors = len(self.augmentors)
        perm = torch.randperm(num_augmentors)
        idx = perm[: self.num_choices]
        for i in idx:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g
