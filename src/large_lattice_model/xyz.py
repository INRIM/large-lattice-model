from functools import lru_cache

import numpy as np
import scipy.integrate as integrate
from scipy.constants import k as kB

import large_lattice_model.settings as settings
from large_lattice_model.latticemodel import R, U, max_nz
from large_lattice_model.mathieu import mathieu_se

lru_maxsize = 65536


@lru_cache(maxsize=lru_maxsize)
def Zf(rho, z, D, nz):
    """Return the longitudinal wavefunction in the lattice (Beloy 2020 Eq. D2)

    Inputs
    ------
    rho : float
        radial coordinates in units of kappa**-1
    z : float
        longitudinal coordinate in m
    D : float
        depth of the lattice in Er
    nz : int
        longitudinal quantum number

    Returns
    -------
    float
        longitudinal wavefunction
    """

    Drho = D * np.exp(-(rho**2))
    return np.sqrt(2 * settings.k / np.pi) * mathieu_se(nz + 1, Drho / 4, (settings.k * z + np.pi / 2))


@lru_cache(maxsize=lru_maxsize)
def beloy_xn(rho, D, nz):
    """Beloy2020 :math:`x_{nz}(\rho)` function (page 6)

    Parameters
    ----------
    rho : float
        radial coordinates in units of kappa**-1
    D : float
        depth of the lattice in Er
    nz : int
        longitudinal quantum number

    Returns
    -------
    float
        :math:`x_{nz}(\rho)`
    """

    lim = np.pi / (2 * settings.k)

    res2 = integrate.quad(lambda z: Zf(rho, z, D, nz) ** 2 * np.cos(settings.k * z) ** 2, 0, lim)
    # integral is even
    return 2 * abs(res2[0]) * np.exp(-(rho**2))


@lru_cache(maxsize=lru_maxsize)
def beloy_yn(rho, D, nz):
    """Beloy2020 :math:`y_{nz}(\rho)` function (page 6)

    Parameters
    ----------
    rho : float
        radial coordinates in units of kappa**-1
    D : float
        depth of the lattice in Er
    nz : int
        longitudinal quantum number

    Returns
    -------
    float
        :math:`y_{nz}(\rho)`
    """

    lim = np.pi / (2 * settings.k)

    res2 = integrate.quad(lambda z: Zf(rho, z, D, nz) ** 2 * np.sin(settings.k * z) ** 2, 0, lim)
    # integral is even
    return 2 * abs(res2[0]) * np.exp(-(rho**2))


@lru_cache(maxsize=lru_maxsize)
def beloy_zn(rho, D, nz):
    """Beloy2020 :math:`z_{nz}(\rho)` function (page 6)

    Parameters
    ----------
    rho : float
        radial coordinates in units of kappa**-1
    D : float
        depth of the lattice in Er
    nz : int
        longitudinal quantum number

    Returns
    -------
    float
        :math:`z_{nz}(\rho)`
    """

    lim = np.pi / (2 * settings.k)

    res2 = integrate.quad(lambda z: Zf(rho, z, D, nz) ** 2 * np.cos(settings.k * z) ** 4, 0, lim)
    # integral is even
    return 2 * abs(res2[0]) * np.exp(-2 * rho**2)


@np.vectorize
def beloy_XYZ(D, Tz, Tr):
    """Return the effective trap depths X, Y and Z from the Born-Oppenheimer model (Beloy2020 eq. 19)

    Parameters
    ----------
    D : array_like
        depth of the lattice in Er
    Tz : array_like
        longitudinal temperature in K
    Tz : array_like
        radial temperature in K

    Returns
    -------
    (array_like, array_like, array_like)
        effective trap depths X, Y and Z
    """
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
        resx = integrate.quad(lambda rho: beloy_xn(rho, D, nz) * rho * (np.exp(-U(rho, D, nz) * beta_r) - 1), 0, R0)
        numx += resx[0] * Qnz[nz]

        # y = exp(-rho^2) - x
        resy = integrate.quad(lambda rho: np.exp(-(rho**2)) * rho * (np.exp(-U(rho, D, nz) * beta_r) - 1), 0, R0)
        numy += (resy[0] - resx[0]) * Qnz[nz]

        # z
        res = integrate.quad(lambda rho: beloy_zn(rho, D, nz) * rho * (np.exp(-U(rho, D, nz) * beta_r) - 1), 0, R0)
        numz += res[0] * Qnz[nz]

        res = integrate.quad(lambda rho: rho * (np.exp(-U(rho, D, nz) * beta_r) - 1), 0, R0)
        den += res[0] * Qnz[nz]

    return numx / den, numy / den, numz / den


# def modified_ushijima_zeta(D, Tr, j):
#     return (1 + j*(kB*Tr)/(D*settings.Er))**-1


# @np.vectorize
# def  modified_ushijima_XYZ(D, Tr, nz):
#     """Return the effective trap depths Xn, Yn and Zn from the modified Ushijima model (Beloy2020 eq. 23)

#     Parameters
#     ----------
#     D : array_like
#         depth of the lattice in Er
#     Tz : array_like
#         longitudinal temperature in K
#     Tz : array_like
#         radial temperature in K

#     Returns
#     -------
#     (array_like, array_like, array_like)
#         effective trap depths X, Y and Z
#     """

#     Xn = modified_ushijima_zeta(D, Tr, 1) - (nz+0.5)*modified_ushijima_zeta(D, Tr, 0.5)*D**-0.5
#     Yn = (nz+0.5)*modified_ushijima_zeta(D, Tr, 0.5)*D**-0.5
#     Zn = modified_ushijima_zeta(D, Tr, 2) - 2*(nz+0.5)*modified_ushijima_zeta(D, Tr, 1.5)*D**-0.5 + 1.5*(nz**2 + nz + 0.5)* modified_ushijima_zeta(D, Tr, 1)*D**-1
