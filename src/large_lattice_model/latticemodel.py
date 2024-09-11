import numpy as np
from scipy.constants import c, h, k

import large_lattice_model.settings as settings


def set_atom(atom):
    """Set the scale parameters of the lattice model (recoil energy, lattice wavevector, clock wavevector).
    By default the settings are for 171Yb.

    Parameters
    ----------
    atom : str,
        One of '171Yb', '87Sr', '88Sr'

    Notes:
    ------
    If your atom is not in the list use [](set_atomic_properties) instead.

    """
    settings.Er, settings.k, settings.kc = settings.scale_parameters_from_atom(atom)
    return settings.Er, settings.k, settings.kc


def set_atomic_properties(lattice_frequency, clock_frequency, atomic_mass):
    """Set custom scale parameters of the lattice model (recoil energy, lattice wavevector, clock wavevector).

    Parameters
    ----------
    lattice_frequency : float
        lattice frequency in Hz
    clock_frequency : float
        clock frequency in Hz
    atomic_mass : float
        atomic mass in atomic mass constants
    """
    settings.Er, settings.k, settings.kc = settings.scale_parameters(lattice_frequency, clock_frequency, atomic_mass)
    return settings.Er, settings.k, settings.kc
