import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.special import mathieu_b as scipy_mathieu_b

from large_lattice_model.mathieu import gsl, gsl_mathieu_b, mathieu_b_asymptotic

gsl_available = gsl is not None


@pytest.mark.skipif(not gsl_available, reason="GSL library is not available, skipping gsl_mathieu_b tests.")
def test_gsl_mathieu_b():
    n, x = 5, 2.0
    gsl_result = gsl_mathieu_b(n, x)
    assert np.isfinite(gsl_result)


# scipy is taken as a benchmark only for q<30
# as this region is bug-free in its implementation


@pytest.mark.skipif(not gsl_available, reason="GSL library is not available, skipping gsl_mathieu_b tests.")
@given(
    n=st.integers(min_value=1, max_value=25),
    x=st.floats(min_value=0.001, max_value=30),
)
def test_gsl_mathieu_b_vs_scipy(n, x):
    gsl_result = gsl_mathieu_b(n, x)
    scipy_result = scipy_mathieu_b(n, x)

    assert gsl_result == pytest.approx(scipy_result)


@pytest.mark.skipif(not gsl_available, reason="GSL library is not available, skipping gsl_mathieu_b tests.")
@given(
    n=arrays(np.int32, 10, elements=st.integers(min_value=1, max_value=35)),
    x=arrays(np.float32, 10, elements=st.floats(min_value=0.001, max_value=30, allow_nan=False, allow_infinity=False)),
)
def test_gsl_mathieu_b_vs_scipy_numpy(n, x):
    gsl_result = gsl_mathieu_b(n, x)
    scipy_result = scipy_mathieu_b(n, x)

    assert gsl_result == pytest.approx(scipy_result)


# the asymptotic function is only valid for high depth and low n (the higher the depth the higher the acceptable n)
# here is only an easy test


@pytest.mark.skipif(not gsl_available, reason="GSL library is not available, skipping mathieu_b_asymptotic tests.")
@given(
    n=st.integers(min_value=1, max_value=8),
    x=st.floats(min_value=150, max_value=500, allow_nan=False, allow_infinity=False),
)
def test_mathieu_b_asymptotic_vs_gsl(n, x):
    asymptotic_result = mathieu_b_asymptotic(n, x)
    gsl_result = gsl_mathieu_b(n, x)

    assert asymptotic_result == pytest.approx(gsl_result, rel=1e-1)
