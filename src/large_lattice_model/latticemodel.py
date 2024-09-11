from functools import lru_cache  # with maxsize > ~16k reduces computation time from 70 s to 10 s

import numpy as np
import scipy.integrate as integ
import scipy.optimize as opt
from numba import float64, int64, njit, vectorize
from scipy.constants import c, h, hbar
from scipy.constants import k as kB
from scipy.special import eval_genlaguerre, factorial

import large_lattice_model.settings as settings
from large_lattice_model.mathieu import mathieu_b

lru_maxsize = 65536


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


def U(rho, D, nz):
    """Return the lattice potential surfaces (Beloy2020 Eq. D2).

    Inputs
    ------
    rho : radial coordinates in units of kappa**-1
    D : depth of the lattice in Er
    nz : longitudinal quantum number

    Returns
    -------
    U : lattice potential surface in Er
    """
    Drho = D * np.exp(-(rho**2))
    return mathieu_b(nz + 1, Drho / 4) - Drho / 2


@vectorize([float64(float64, float64, int64)])
@lru_cache(maxsize=lru_maxsize)
def R(E, D, nz):
    """Inverse of the lattice potential surface U(rho) (Beloy2020 pag. 4)

    Inputs
    ------
    E : potential energy in Er
    D : depth of the lattice in Er
    nz : longitudinal quantum number

    Returns
    -------
    r : radial coordinates in units of kappa**-1
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
    """Return the difference between lattice potential surfaces U(rho, D, nz+dn) - U(rho, D, nz).

    Inputs
    ------
    rho : radial coordinates in units of kappa**-1
    D : depth of the lattice in Er
    nz : longitudinal quantum number of the starting level
    dn : longitudinal quantum number jump (default: 1)

    Returns
    -------
    DeltaU : difference U(rho, nz+dn) - U(rho, nz)
    """

    return U(rho, D, nz + dn) - U(rho, D, nz)


# Harmonic Oscillator
def Omega(rho, D, nz, dn=1):
    """Normalized Rabi frequency for the harmonic oscillator (Wineland1979 eq. 31)

    Inputs
    ------
    rho : radial coordinates in units of kappa**-1
    D : depth of the lattice in Er
    nz : longitudinal quantum number of the starting level
    dn : longitudinal quantum number jump (default: 1)

    Returns
    -------
    Omega : Normalized Rabi frequency  (between 0 and 1)
    """

    Drho = D * np.exp(-(rho**2))
    eta = settings.kc / settings.k / np.sqrt(2) / Drho**0.25

    return (
        np.exp(-(eta**2) / 2)
        * np.sqrt(factorial(nz) / factorial(nz + dn))
        * eta ** (dn)
        * eval_genlaguerre(nz, dn, eta**2)
    )


# TODO: add mathieu_se to mathieu
# # Using Mathieu Functions
# @vectorize([float64(float64, float64, int64, int64)])
# def OmegaMat(rho, D, nz, dn=1):
# 	"""Normalized Rabi frequency for Mathieu wavefunctions (Beloy2020 appendix)

# 	Inputs
# 	------
# 	rho : radial coordinates in units of kappa**-1
# 	D : depth of the lattice in Er
# 	nz : longitudinal quantum number of the starting level
# 	dn : longitudinal quantum number jump (default: 1)

# 	Returns
# 	-------
# 	Omega : Normalized Rabi frequency (between 0 and 1)
# 	"""

# 	Drho = D*np.exp(-rho**2)
# 	k = settings.k
# 	kc = settings.kc

# 	lim = np.pi/(2*k)
# 	if dn % 2:
# 		res2 = integ.quad(lambda z: 2*k/np.pi * sf.mathieu_se(nz+1,  Drho/4, (k*z + np.pi/2)) * np.sin(kc*z) * sf.mathieu_se(nz+1+dn, Drho/4, (k*z + np.pi/2)), 0,lim) # pygsl -- slower but no bugs!
# 	else:
# 		res2 = integ.quad(lambda z: 2*k/np.pi * sf.mathieu_se(nz+1,  Drho/4, (k*z + np.pi/2)) * np.cos(kc*z) * sf.mathieu_se(nz+1+dn, Drho/4, (k*z + np.pi/2)), 0,lim) # pygsl -- slower but no bugs!

# 	# integral is even
# 	return 2*abs(res2[0])


def Gn(E, D, nz):
    """Density of states for the lattic trap (Beloy2020 eq. 11)

    Inputs
    ------
    E : potential energy in Er
    D : depth of the lattice in Er
    nz : longitudinal quantum number of the starting level

    Returns
    -------
    G : density of states at energy E (same units of Beloy2020 fig. 3)

    """

    return R(E, D, nz) ** 2 * np.pi / (2 * settings.k)


def Gr(rc, D, nz):
    """Density of states for the lattic trap at a given radius (Beloy2020 eq. 11)

    Inputs
    ------
    rc : Condon point at energy E
    D : depth of the lattice in Er
    nz : longitudinal quantum number of the starting level

    Returns
    -------
    G : density of states at energy E (same units of Beloy2020 fig. 3)

    """

    return rc**2 * np.pi / (2 * settings.k)


@lru_cache(maxsize=lru_maxsize)
def max_nz(D):
    """Return the maximum nz for a given depth"""
    # ansatz 	twice the harmonic oscillators levels
    max_n = int(-U(0, D, 0) / np.sqrt(D))
    test = np.arange(max_n)
    return np.amax(np.where(U(0, D, test) < 0))


# lorentzian
def lor(x, x0, w):
    """Simple lorentzian with center x0 and HWHM w peak 0.5"""
    den = 1 + 1 / w**2 * (x - x0) ** 2
    return 0.5 / den


# both sideband
# it is faster to calculate sidebands at the same time
def sidebands(x, D, Tz, Tr, b, r, wc, dn=1, E_max=0.0, fac=10):
    """Lattice sidebands as a sum of lorentzian.

    Inputs
    ------
    x : frequency in Hz
    D : depth of the lattice in Er
    Tz : longitudinal temperature in K
    Tr : radial temperature in K
    b : blue sidebands scaling
    r : red sidebands scaling
    wc : carrier half width half maximum
    dn : order of the sideband (default: 1)
    E_max : max energy levels populated (default: 0)
    fac : control the number of lorentzian to be used in the sum
    higher number will give smoother sidebands but take more computational time
    (default: 10)

    Returns
    -------
    Both sidebands as excitation.

    """
    Nz = int(max_nz(D) * 1.0 + 0.5)
    beta_r = settings.Er / (kB * Tr)
    beta_z = settings.Er / (kB * Tz)

    # simple exp factor for a given Tz
    # will be just used computationally to reduce the number of lorentzian used at high nz
    # r = exp(-betaz)

    tot = np.zeros(x.shape)
    total_norm = 0

    for nz in np.arange(Nz + 1):
        E_min = U(0, D, nz)

        # this just save computational time
        # use less samples for high levels
        # formula with r is normalized exponential distribution (from geometric series)
        # high dn use higher number because it has sharper lorentzian
        # N = int(Natoms*r**nz/(1-r**Nz)*(1-r)+2.)*dn

        # method to calculate number of lorentzian function to sum
        N = int(DeltaU(0, D, nz, dn) * fac * (nz + 1) ** -0.5)

        # Uniform sampling in E
        EE = np.linspace(E_min, E_max, N)[:, np.newaxis]
        rc = R(EE, D, nz)
        # dE = (E_max - E_min)/N

        # calc normalization
        pp = Gr(rc, D, nz) * np.exp(-(EE - E_min) * beta_r) * np.exp(-E_min * beta_z)
        total_norm += np.trapz(
            pp, EE, axis=0
        )  # sum(pp, axis=0) *dE #trapz(pp, EE, axis=0) #trapz is a bit slower, but handles better different Ns

        # blue
        x0 = DeltaU(rc, D, nz, dn) * settings.Er / h
        ff = Omega(rc, D, nz, dn) * wc

        # sum lorentzian for blue sideband - note cutoff on energy
        blue = pp * lor(x, x0, ff) * (U(rc, D, nz + dn) < E_max)

        res = b * np.trapz(blue, EE, axis=0)  # sum(yy, axis=0)*dE #trapz(yy, EE, axis=0) # speed sum > trapz > simps

        tot += res

        # red
        if nz >= dn:
            # rc = R(EE, D, nz)  # same as blue
            x0 = DeltaU(rc, D, nz, -dn) * settings.Er / h
            ff = Omega(rc, D, nz - dn, dn) * wc

            # sum lorentzian on red sideband
            red = pp * lor(x, x0, ff)

            res = r * np.trapz(red, EE, axis=0)  # trapz(yy, EE, axis=0) # sum(yy, axis=0)*median(diff(EE, axis=0)) #

            tot += res

    return tot / total_norm
