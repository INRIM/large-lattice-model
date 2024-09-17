"""
This submodule implements the main sideband function in the Born-Oppenheimer approximation.
"""

import numpy as np
from scipy.constants import h
from scipy.constants import k as kB

from large_lattice_model import settings
from large_lattice_model.latticemodel import DeltaU, Gr, R, U, lorentzian, max_nz, rabi_ho, two_temp_dist


def sidebands(x, D, Tz, Tr, b, r, wc, dn=1, E_max=0.0, fac=10):
    """Return lattice sidebands in the Born-Oppenheimer model as a finite sum of lorentzian functions.

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
        amplitude scaling of the blue sideband
    r : float
        amplitude scaling of the red sideband
    wc : float
        carrier half-width half-maximum in Hz
    dn : int, optional
        order of the sideband, by default 1
    E_max : float, optional
        max energy level in Er, by default 0.0
    fac : float, optional
        parameter controlling the number of lorentzian functions used to calculate the sideband shape, higher number
        will give smoother sidebands at the expense of more computational time, by default 10

    Returns
    -------
    array_like
        Value of the sidebands, both red and blue, calculated for frequency x


    Notes
    -----
    The sideband shape is calculated numerically as the finite sum of lorentzian functions.
    The number of functions used in the calculation is proportional to the energy gap :math:`U_{n_z'}(0) - U_{n_z}(0)`
    (equivalent to summing a uniform distribution of Lorentzian functions in energy) and it is suppressed by
    a scaling :math:`\propto 1/\sqrt{n_z}`, to save computational time on the scarcely populated high longitudinal states.

    Notes
    -----
    It is numerically faster to calculate both red and blue sideband at the same time.

    """
    Nz = int(max_nz(D) * 1.0 + 0.5)

    tot = np.zeros(x.shape)
    total_norm = 0

    for nz in np.arange(Nz + 1):
        E_min = U(0, D, nz)

        # method to calculate number of lorentzian function to sum
        # this just save computational time
        # use less samples for high levels
        N = int(DeltaU(0, D, nz, dn) * fac * (nz + 1) ** -0.5)

        # Uniform sampling in E, as a *vertical* array
        EE = np.linspace(E_min, E_max, N)[:, np.newaxis]
        rc = R(EE, D, nz)
        # dE = (E_max - E_min)/N

        # calc normalization
        pp = Gr(rc, D, nz) * two_temp_dist(EE, E_min, Tz, Tr)
        total_norm += np.trapz(pp, EE, axis=0)

        # blue
        x0 = DeltaU(rc, D, nz, dn) * settings.Er / h
        ff = rabi_ho(rc, D, nz, dn) * wc

        # sum lorentzian for blue sideband - note cutoff on energy
        blue = pp * lorentzian(x, x0, ff) * (U(rc, D, nz + dn) < E_max)

        res = b * np.trapz(blue, EE, axis=0)  # sum(yy, axis=0)*dE #trapz(yy, EE, axis=0) # speed sum > trapz > simps

        tot += res

        # red
        if nz >= dn:
            # rc = R(EE, D, nz)  # same as blue
            x0 = DeltaU(rc, D, nz, -dn) * settings.Er / h
            ff = rabi_ho(rc, D, nz - dn, dn) * wc

            # sum lorentzian on red sideband
            red = pp * lorentzian(x, x0, ff)

            res = r * np.trapz(red, EE, axis=0)  # trapz(yy, EE, axis=0) # sum(yy, axis=0)*median(diff(EE, axis=0)) #

            tot += res

    return tot / total_norm
