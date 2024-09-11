import numpy as np
from scipy.constants import c, h, physical_constants
from scipy.constants import k as kB

amu = physical_constants["atomic mass constant"][0]


def scale_parameters(lattice_frequency, clock_frequency, atomic_mass):
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
    if (atom is None) or (atom == "Yb"):
        return _atomic_parameters["171Yb"]
    if atom in _atomic_parameters.keys():
        return _atomic_parameters[atom]
    else:
        raise ValueError(f"Please specify an atom in {_atomic_parameters.keys}")


Er, k, kc = scale_parameters_from_atom("171Yb")
