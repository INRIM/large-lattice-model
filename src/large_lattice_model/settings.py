"""This submodule store the scale parameters of the lattice model to be used by other functions.
It define Er, k and kc (recoil energy, lattice wavevector and clock wavevector).
"""

import numpy as np
from scipy.constants import c, h, physical_constants

amu = physical_constants["atomic mass constant"][0]


def scale_parameters(lattice_frequency, clock_frequency, atomic_mass):
    """Return recoil energy, lattice wavevector and clock wavevector from lattice frequency, clock frequency and atomic mass.

    Parameters
    ----------
    lattice_frequency : float
        lattice frequency in Hz
    clock_frequency : float
        clock frequency in Hz
    atomic_mass : float
        atomic mass in atomic mass constants

    Returns
    -------
    (float, float, float)
        recoil energy, lattice wavevector and clock wavevector in SI units
    """
    lattice_wavelength = c / lattice_frequency
    recoil_energy = h**2 / (2 * atomic_mass * amu * lattice_wavelength**2)
    lattice_k = 2 * np.pi / lattice_wavelength
    clock_wavelength = c / clock_frequency
    clock_k = 2 * np.pi / clock_wavelength

    return recoil_energy, lattice_k, clock_k


_atomic_parameters = {
    "171Yb": scale_parameters(394798.267e9, 518295836590863.0, 170.936323),
    "88Sr": scale_parameters(368554e9, 429228066418007.0, 87.9056121),
    "87Sr": scale_parameters(368554e9, 429228004229873.0, 86.90887750),
}


def scale_parameters_from_atom(atom=None):
    """Return recoil energy, lattice wavevector and clock wavevector for commonly used atoms in optical lattice clocks.

    Parameters
    ----------
    atom : str, optional
        one of 171Yb, 88Sr, 87Sr, by default None

    Returns
    -------
    (float, float, float)
        recoil energy, lattice wavevector and clock wavevector in SI units

    """
    if (atom is None) or (atom == "Yb"):
        return _atomic_parameters["171Yb"]
    if atom in _atomic_parameters.keys():
        return _atomic_parameters[atom]
    else:
        raise ValueError(f"Please specify an atom in {_atomic_parameters.keys}")


Er, k, kc = scale_parameters_from_atom("171Yb")
