"""
This submodule implements the sideband function in the traditional approximation (Blatt et al., 2009).
"""

import numpy as np
from scipy.constants import h
from scipy.constants import k as kB

from large_lattice_model import settings


def E(D, nz):
    """Simple expression for the energy levels of a lattice in 1D"""
    vrec = settings.Er / h
    vz = 2 * vrec * np.sqrt(D)
    return vz * h * (nz + 0.5) - vrec * h * (nz**2 + nz + 0.5) / 2


def max_nz(D):
    """Return the maximum nz for a given depth

    Parameters
    ----------
    D : float
        depth of the lattice in Er

    Returns
    -------
    int
        maximum nz for a lattice trap with depth D
    """
    # ansatz
    max_n = 2 * np.sqrt(D)
    test = np.arange(max_n)
    return np.amax(np.where(E(D, test) < D * settings.Er))


def sigma_blatt_nz(x, D, nz, Tr):
    """Sideband shape for atoms in the nz level. Blatt 2009 Eq. 10"""
    vrec = settings.Er / h
    vz = 2 * vrec * np.sqrt(D)
    gamma = vz - vrec * (nz + 1)
    alpha = gamma * h * vz / (vrec * kB * Tr)
    return (
        alpha**2
        / gamma
        * (1.0 - abs(x) / gamma)
        * np.exp(-alpha * (1.0 - abs(x) / gamma))
        * -(np.sign(abs(x) - gamma) - 1.0)
        / 2.0
    )


def blue_sideband(x, D, Tz, Tr, b=1):
    """Return the blue lattice sidebands in the traditional approximation (Blatt et al., 2009)..

    Parameters
    ----------
    x : 1D array, float
        frequency in Hz
    D : float
        depth of the lattice in Er
    Tz : float
        longitudinal temperature in K
    Tr : float
        radial temperature in K
    b : float
        amplitude scaling of the blue sideband, by default 1

    Returns
    -------
    array_like
        Value of the blue sidebands calculated for frequency x


    """
    res = 0.0
    norm = 0.0
    Nmax = max_nz(D)
    for nz in np.arange(Nmax):
        res += np.exp(-E(D, nz) / (kB * Tz)) * sigma_blatt_nz(x, D, nz, Tr)
        norm += np.exp(-E(D, nz) / (kB * Tz))
    return b * res / norm


def red_sideband(x, D, Tz, Tr, r=1):
    """Return the red lattice sidebands in the traditional approximation (Blatt et al., 2009)..

    Parameters
    ----------
    x : 1D array, float
        frequency in Hz
    D : float
        depth of the lattice in Er
    Tz : float
        longitudinal temperature in K
    Tr : float
        radial temperature in K
    r : float
        amplitude scaling of the blue sideband, by default 1

    Returns
    -------
    array_like
        Value of the red sidebands calculated for frequency x


    """

    res = 0.0
    norm = 0.0
    Nmax = max_nz(D)
    for nz in np.arange(Nmax + 1):
        res += np.exp(-E(D, nz + 1) / (kB * Tz)) * sigma_blatt_nz(x, D, nz, Tr)
        norm += np.exp(-E(D, nz + 1) / (kB * Tz))
    return r * res / norm
