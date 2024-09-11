from functools import lru_cache

import numpy as np
import scipy.integrate as integ
from numba import float64, vectorize
from scipy.constants import k as kB

import large_lattice_model.settings as settings
from large_lattice_model.latticemodel import R, U, max_nz

lru_maxsize = 65536


@lru_cache(maxsize=lru_maxsize)
def Zf(rho, z, D, nz):
    """Return the longitudinal wavefunction in the lattice (Beloy 2020 Eq. D2)

    Inputs
    ------
    rho : radial coordinates in units of kappa**-1
    z : longitudinal coordinate (normal units)
    D : depth of the lattice in Er
    nz : longitudinal quantum number

    Returns
    -------
    Z : longitudinal wavefunction
    """

    Drho = D * np.exp(-(rho**2))
    return np.sqrt(2 * settings.k / np.pi) * sf.mathieu_se(nz + 1, Drho / 4, (settings.k * z + np.pi / 2))


@lru_cache(maxsize=lru_maxsize)
def beloy_x(rho, D, nz):
    """Beloy2020 z_nz(rho) function (page 6)

    Inputs
    ------
    rho : radial coordinates in units of kappa**-1
    D : depth of the lattice in Er
    nz : longitudinal quantum number

    Returns
    -------
    z_nz :
    """

    lim = np.pi / (2 * settings.k)

    res2 = integ.quad(lambda z: Zf(rho, z, D, nz) ** 2 * np.cos(settings.k * z) ** 2, 0, lim)
    # integral is even
    return 2 * abs(res2[0]) * np.exp(-(rho**2))


@lru_cache(maxsize=lru_maxsize)
def beloy_y(rho, D, nz):
    """Beloy2020 z_nz(rho) function (page 6)

    Inputs
    ------
    rho : radial coordinates in units of kappa**-1
    D : depth of the lattice in Er
    nz : longitudinal quantum number

    Returns
    -------
    z_nz :
    """

    lim = np.pi / (2 * settings.k)

    res2 = integ.quad(lambda z: Zf(rho, z, D, nz) ** 2 * np.sin(settings.k * z) ** 2, 0, lim)
    # integral is even
    return 2 * abs(res2[0]) * np.exp(-(rho**2))


@lru_cache(maxsize=lru_maxsize)
def beloy_z(rho, D, nz):
    """Beloy2020 z_nz(rho) function (page 6)

    Inputs
    ------
    rho : radial coordinates in units of kappa**-1
    D : depth of the lattice in Er
    nz : longitudinal quantum number

    Returns
    -------
    z_nz :
    """

    lim = np.pi / (2 * settings.k)

    res2 = integ.quad(lambda z: Zf(rho, z, D, nz) ** 2 * np.cos(settings.k * z) ** 4, 0, lim)
    # integral is even
    return 2 * abs(res2[0]) * np.exp(-2 * rho**2)


@vectorize([(float64, float64, float64)(float64, float64, float64)])
def beloy_XYZ(D, Tz, Tr):
    beta_r = settings.Er / (kB * Tr)
    beta_z = settings.Er / (kB * Tz)

    Nz = max_nz(D)
    nnz = np.arange(0, Nz + 1)
    Qnz = np.exp(U(0, D, nnz) * (beta_r - beta_z))

    numx = 0.0
    numy = 0.0
    numz = 0.0
    den = 0.0

    for nz in nnz:
        R0 = R(0, D, nz)
        # x
        resx = integ.quad(lambda rho: beloy_x(rho, D, nz) * rho * (np.exp(-U(rho, D, nz) * beta_r) - 1), 0, R0)
        numx += resx[0] * Qnz[nz]

        # y = exp(-rho^2) - x
        resy = integ.quad(lambda rho: np.exp(-(rho**2)) * rho * (np.exp(-U(rho, D, nz) * beta_r) - 1), 0, R0)
        numy += (resy[0] - resx[0]) * Qnz[nz]

        # z
        res = integ.quad(lambda rho: beloy_z(rho, D, nz) * rho * (np.exp(-U(rho, D, nz) * beta_r) - 1), 0, R0)
        numz += res[0] * Qnz[nz]

        res = integ.quad(lambda rho: rho * (np.exp(-U(rho, D, nz) * beta_r) - 1), 0, R0)
        den += res[0] * Qnz[nz]

    return numx / den, numy / den, numz / den
