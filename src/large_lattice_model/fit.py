"""
This submodule implements functions useful for fitting sidebands.
"""

from large_lattice_model.latticemodel import lorentzian
from large_lattice_model.sidebands import sidebands


def fit_lorentzian(x, A, x0, w):
    """Lorentzian with center x0 , half-width at half-maximum w,  and peak A.
    Used to fit the carrier.

    Parameters
    ----------
    x : array_like
        points where to calculate the function
        A : array_like
        amplitude of the function
    x0 : array_like
        center of the curve
    w : array_like
        half-width at half-maximum of the curve

    Returns
    -------
    array_like
        value of the function
    """
    return A * lorentzian(x, x0, w)


def get_fit_sidebands(w0, dn=1, E_max=0, fac=10):
    """Return a simplified sideband function that depends on only 4 parameters for easy fittings.

    Parameters
    ----------
    w0 : float
        linewidth of the carrier in Hz
    dn : int, optional
        order of the sideband, by default 1
    E_max : float, optional
        max energy level in Er, by default 0.0
    fac : float, optional
        parameter controlling the number of lorentzian functions used to calculate the sideband shape, higher number
        will give smoother sidebands at the expense of more computational time, by default 10

    Returns
    -------
    callable
        sideband function that depends only on A, D, Tz and Tr

    """

    def fit_sidebands(x, A, D, Tz, Tr):
        """Simplified sideband function for fitting w that depends on only 4 parameters.

        Parameters
        ----------
        x : array_like
            frequency value in Hz
        A : float
            scaling factor of the sidebands amplitude
        D : float
            trap depth in Er
        Tz : float
            longitudinal temperature in K
        Tr : float
            radial temperature in K

        Returns
        -------
        float or ndarray
            value of the sidebands calculated at frequency x, with fixed carrier width.
        """
        return sidebands(x, D, Tz, Tr, A, A, w0, dn=dn, E_max=E_max, fac=fac)

    return fit_sidebands
