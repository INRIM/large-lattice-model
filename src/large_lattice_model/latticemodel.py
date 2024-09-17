"""
This submodule implements the main lattice model functions in the Born-Oppenheimer approximation from Beloy 2020.
When possible, functions are cached with `lru_cache` from `functools` to improve performances over repeated calls.
"""

from functools import lru_cache  # with maxsize > ~16k reduces computation time from 70 s to 10 s

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.constants import k as kB
from scipy.special import eval_genlaguerre, factorial

import large_lattice_model.settings as settings
from large_lattice_model.mathieu import mathieu_b, mathieu_se

lru_maxsize = 65536


def set_atom(atom):
    """Set the scale parameters of the lattice model (recoil energy, lattice wavevector, clock wavevector).
    By default the settings are for 171Yb.

    Parameters
    ----------
    atom : str,
        One of '171Yb', '87Sr', '88Sr'

    Notes
    -----
    If your atom is not in the list use :func:`set_atomic_properties` instead.

    """
    settings.Er, settings.k, settings.kc = settings.scale_parameters_from_atom(atom)
    return settings.Er, settings.k, settings.kc


def set_properties(lattice_frequency, clock_frequency, atomic_mass):
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


def U(rho, D, nz):
    """Return the lattice potential surfaces (Beloy2020 Eq. D2).

    Parameters
    ----------
    rho : array_like
        radial coordinates in units of kappa**-1
    D : array_like
        depth of the lattice in Er
    nz : array_like
        longitudinal quantum number

    Returns
    -------
    U : lattice potential surface in Er
    """
    Drho = D * np.exp(-(rho**2))
    return mathieu_b(nz + 1, Drho / 4) - Drho / 2


@np.vectorize
@lru_cache(maxsize=lru_maxsize)
def R(E, D, nz):
    """Inverse of the lattice potential surface U(rho) (Beloy2020 pag. 4)

    Parameters
    ----------
    E : array_like
        potential energy in Er
    D : array_like
        depth of the lattice in Er
    nz : array_like
        longitudinal quantum number

    Returns
    -------
    array_like
        radial coordinates in units of kappa**-1

    Notes
    -----
    This is calculated numerically using Brent’s method (`brentq` option in `scipy.optimize.root_scalar`).

    """

    min_eps = 0.0
    max_eps = 2.7  # max radius for nz = 0 and D=1500
    try:
        res = opt.root_scalar(
            lambda x, *args: U(x, *args) - E, args=(D, nz), bracket=[min_eps, max_eps], method="brentq"
        )
    except ValueError:
        return 0.0

    return res.root


def DeltaU(rho, D, nz, dn=1):
    """Return the difference between lattice potential surfaces :math:`U(rho, D, nz+dn) - U(rho, D, nz)`.

    Parameters
    ----------
    rho : array_like
        radial coordinates in units of kappa**-1
    D : array_like
        depth of the lattice in Er
    nz : array_like
        longitudinal quantum number
    dn : array_like, optional
        longitudinal quantum number jump (default: 1), by default 1

    Returns
    -------
    _type_
        _description_
    """

    return U(rho, D, nz + dn) - U(rho, D, nz)


# Harmonic Oscillator
def rabi_ho(rho, D, nz, dn=1):
    """Normalized Rabi frequency for the harmonic oscillator (Wineland1979 eq. 31)

    Parameters
    ----------
    rho : array_like
        radial coordinates in units of kappa**-1
    D : array_like
        depth of the lattice in Er
    nz : array_like
        longitudinal quantum number
    dn : array_like, optional
        longitudinal quantum number jump (default: 1), by default 1

    Returns
    -------
    array_like
        normalized Rabi frequency  (between 0 and 1)

    """

    Drho = D * np.exp(-(rho**2))
    eta = settings.kc / settings.k / np.sqrt(2) / Drho**0.25

    return (
        np.exp(-(eta**2) / 2)
        * np.sqrt(factorial(nz) / factorial(nz + dn))
        * eta ** (dn)
        * eval_genlaguerre(nz, dn, eta**2)
    )


# Using Mathieu Functions
@np.vectorize
def rabi_bo(rho, D, nz, dn=1):
    """Normalized Rabi frequency for the Born-Oppenheimer wavefunctions calculated using Mathieu functions (Beloy2020 appendix)

    Parameters
    ----------
    rho : array_like
        radial coordinates in units of kappa**-1
    D : array_like
        depth of the lattice in Er
    nz : array_like
        longitudinal quantum number
    dn : array_like, optional
        longitudinal quantum number jump (default: 1), by default 1

    Returns
    -------
    array_like
        normalized Rabi frequency  (between 0 and 1)

    """
    Drho = D * np.exp(-(rho**2))
    k = settings.k
    kc = settings.kc

    lim = np.pi / (2 * k)
    if dn % 2:
        # fmt: off
        def integrand(z):
            return 2 * k / np.pi * mathieu_se(nz + 1, Drho / 4, k * z + np.pi / 2) * np.sin(kc * z) * mathieu_se(nz + 1 + dn, Drho / 4, k * z + np.pi / 2)
        # fmt: on

    else:
        # fmt: off
        def integrand(z):
            return 2 * k / np.pi * mathieu_se(nz + 1, Drho / 4, k * z + np.pi / 2) * np.cos(kc * z) * mathieu_se(nz + 1 + dn, Drho / 4, k * z + np.pi / 2)
        # fmt: on

    # integral is even
    res2 = integrate.quad(integrand, 0, lim)
    return 2 * abs(res2[0])


def Gn(E, D, nz):
    """Density of states for the lattic trap (Beloy2020 eq. 11)

    Parameters
    ----------
    E : array_like
        potential energy in Er
    D : array_like
         depth of the lattice in Er
    nz : array_like
        longitudinal quantum number of the starting level

    Returns
    -------
    array_like
        density of states at energy E (same units of Beloy2020 fig. 3)
    """
    return R(E, D, nz) ** 2 * np.pi / (2 * settings.k)


def Gr(rc, D, nz):
    """Density of states for the lattic trap at a given radius (Beloy2020 eq. 11)

    Parameters
    ----------
    rc : array_like
        Condon point at energy E
    D : array_like
         depth of the lattice in Er
    nz : array_like
        longitudinal quantum number of the starting level

    Returns
    -------
    array_like
        density of states at energy E (same units of Beloy2020 fig. 3)
    """

    return rc**2 * np.pi / (2 * settings.k)


@lru_cache(maxsize=lru_maxsize)
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
    # ansatz 	twice the harmonic oscillators levels
    max_n = int(-U(0, D, 0) / np.sqrt(D))
    test = np.arange(max_n)
    return np.amax(np.where(U(0, D, test) < 0))


def two_temp_dist(E, E_min, Tz, Tr):
    """Two temperature distribution of atoms in the lattice based on Beloy2020 eq. 24.
    Not normalized.

    Parameters
    ----------
    E : array_like
        atom energy level in Er
    E_min : float
        bottom of the trap in Er
    Tz : float
        longitudinal temperature in K
    Tr : float
        radial temperature in K

    Returns
    -------
    float or ndarray
        not normalized temperature distribution as :math:`e^{-(E-E_{min})/(k T_r)} e^{-E_{min}/(k T_z)}`
    """
    beta_r = settings.Er / (kB * Tr)
    beta_z = settings.Er / (kB * Tz)
    return np.exp(-(E - E_min) * beta_r) * np.exp(-E_min * beta_z)


def lorentzian(x, x0, w):
    """Simple lorentzian with center x0 , half-width at half-maximum w,  and peak 0.5.

    Parameters
    ----------
    x : array_like
        points where to calculate the function
    x0 : array_like
        center of the curve
    w : array_like
        half-width at half-maximum of the curve

    Returns
    -------
    array_like
        value of the function
    """
    den = 1 + 1 / w**2 * (x - x0) ** 2
    return 0.5 / den
