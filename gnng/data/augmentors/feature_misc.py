from gnng.typing import PyGData
from gnng.data.augmentors.augmentor import Augmentor
from gnng.data.augmentors.functional import (
    add_gaussian_noise,
    feature_perturbation,
    contrastive_stretching,
)


class FeatureGaussianNoise(Augmentor):
    def __init__(self, noise_std: float):
        super(FeatureGaussianNoise, self).__init__()
        if noise_std < 0.0:
            raise ValueError(f"The standard deviation of added noise should > =0, but got {noise_std}")
        self.noise_std = noise_std

    def augment(self, g: PyGData) -> PyGData:
        new_g = g.detach().clone()
        if self.noise_std == 0.0:
            return new_g
        new_g.x = add_gaussian_noise(new_g.x, self.noise_std)
        return new_g


class FeaturePerturbation(Augmentor):
    def __init__(self, perturbation_factor: float):
        super(FeaturePerturbation, self).__init__()
        if perturbation_factor < 0.0:
            raise ValueError(f"The perturbation factor should >= 0, but got {perturbation_factor}")
        self.perturbation_factor = perturbation_factor

    def augment(self, g: PyGData) -> PyGData:
        new_g = g.detach().clone()
        if self.perturbation_factor == 0.0:
            return new_g
        new_g.x = feature_perturbation(new_g.x, self.perturbation_factor)
        return new_g


class FeatureStretching(Augmentor):
    def __init__(self, stretch_factor: float):
        super(FeatureStretching, self).__init__()
        if stretch_factor < 0.0:
            raise ValueError(f"The stretch factor should >= 0, but got {stretch_factor}")
        self.stretch_factor = stretch_factor

    def augment(self, g: PyGData) -> PyGData:
        new_g = g.detach().clone()
        if self.stretch_factor == 0.0:
            return new_g
        new_g.x = contrastive_stretching(new_g.x, self.stretch_factor)
        return new_g
