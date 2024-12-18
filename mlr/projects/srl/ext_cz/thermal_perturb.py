from czone.types import BasePostTransform
import numpy as np

class ThermalPerturbation(BasePostTransform):

    def __init__(self, dwf, rng=None):
        """
        Args:
            dwf: float, Debye-Waller factor as root-mean-squared displacement (Angstroms)
        """

        self.dwf = dwf
        self.rng = np.random.default_rng() if rng is None else rng

    @property
    def dwf(self):
        return self._dwf
    
    @dwf.setter
    def dwf(self, val):
        v = float(val)

        if v < 0:
            raise ValueError(f"Debye-Waller factor should be non-negative, but is {val:0.4f}.")
        
        self._dwf = v

    def apply_function(self, points: np.ndarray, species: np.ndarray, **kwargs):

        perturbations = self.dwf * self.rng.normal(size=points.shape)

        return points + perturbations, species

