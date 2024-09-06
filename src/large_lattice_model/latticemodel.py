import numpy as np
from scipy.constants import c, h, k


class LatticeModel:
    """A class tha model the motional spectra of atoms trapped in a one-dimensional optical lattice."""

    def __init__(self, w=60e-6):
        # Yb
        self.m = 2.83846417e-25  # atomic mass
        self.fL = 394798e9  # lattice frequency
        self.wc = 518295836590863.0  # clock frequency

        self.w = w  # lattice waist

        self._set_scale()

    def _set_scale(self):
        self.lL = c / self.fL  # lattice wavelength
        self.Er = h**2 / (2 * self.m * self.lL**2)  # recoil energy
        self.vrec = h / (2 * self.m * self.lL**2)  # recoil frequency

        # problem scale
        self.k = 2 * np.pi / self.lL
        self.kappa = np.sqrt(2) / self.w

        self.lc = c / self.wc
        self.kc = 2 * np.pi / self.lc
