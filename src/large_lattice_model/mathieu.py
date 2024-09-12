import ctypes
from ctypes.util import find_library

import numpy as np
import scipy.special
from numba import float64, int64, njit, vectorize

# scipy Mathieu special functions are currently bugged (they have some discontinuities)
# see https://github.com/scipy/scipy/pull/14577
# in the past I moved to the Mathieu special functions implemented in pygsl
# installed by
# apt-get install libgsl-dev
# pip3 instal pygsl
# and found under testing
# this no longer works (see https://github.com/pygsl/pygsl/issues/55) and pygsl is hard to install
# this file use ctypes to load the GSL Mathieu function directly (apt-get install libgsl-dev)
# and use numba for a fast vectorization
# Finally this file fallback to scipy if GSL is not found and provide an analitycal approximation made fast using numba

scipy_mathieu_b = scipy.special.mathieu_b


def _load_lib(libname):
    lib_path = find_library(libname)
    if lib_path is None:
        return None
    else:
        try:
            lib = ctypes.CDLL(lib_path)
        except OSError:
            return None

    return lib


gsl = _load_lib("gsl")

if gsl:
    gsl.gsl_sf_mathieu_b.restype = ctypes.c_double
    gsl.gsl_sf_mathieu_b.argtypes = [ctypes.c_int, ctypes.c_double]

    gsl_sf_mathieu_b = gsl.gsl_sf_mathieu_b

    # this is fast!
    @vectorize([float64(int64, float64)], nopython=True)
    def gsl_mathieu_b(n, q):
        if n < 1:
            return np.nan
        return gsl_sf_mathieu_b(n, q)

    mathieu_b = gsl_mathieu_b

else:
    # If the GSL library is not found, fallback to scipy
    mathieu_b = scipy_mathieu_b


def scipy_mathieu_se(m, q, x):
    """Odd Mathieu function from scipy implementation, with input in radians and dropping the derivative"""
    func, deriv = scipy.special.mathieu_sem(m, q, x * 180.0 / np.pi)
    return func


if gsl:
    gsl.gsl_sf_mathieu_se.restype = ctypes.c_double
    gsl.gsl_sf_mathieu_se.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double]

    gsl_sf_mathieu_se = gsl.gsl_sf_mathieu_se

    # this is fast!
    @vectorize([float64(int64, float64, float64)], nopython=True)
    def gsl_mathieu_se(n, q, x):
        return gsl_sf_mathieu_se(n, q, x)

    mathieu_se = gsl_mathieu_se

else:
    # If the GSL library is not found, fallback to scipy
    mathieu_se = scipy_mathieu_se


@vectorize([float64(int64, float64)], nopython=True)
def mathieu_b_asymptotic(m, q):
    """Asymptotic approximation of the characteristic value of odd Mathieu functions

    See https://dlmf.nist.gov/28
    """

    s = 2 * (m - 1) + 1
    h = q**0.5

    # Convert each polynomial in s to Horner's form
    term8 = -1.0 / (2**25) * (((527 * s + 15617) * s + 69001) * s + 41607) * s
    term7 = -1.0 / (2**20) * (((63 * s + 1260) * s + 2943) * s + 486)
    term6 = -1.0 / (2**17) * (((33 * s + 410) * s + 405) * s)
    term5 = -1.0 / (2**12) * ((5 * s + 34) * s + 9) * s**2
    term4 = -1.0 / (2**7) * ((s + 3) * s**2)
    term3 = -1.0 / 8 * (s**2 + 1)
    #    term2 = 2.0 * s * h
    #    term1 = -2.0 * h**2

    result = (2.0 * s - 2.0 * h) * h + term3 + (term4 + (term5 + (term6 + (term7 + term8 / h) / h) / h) / h) / h

    return result
