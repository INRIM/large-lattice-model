import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

plt.close("all")
plt.ion()

from large_lattice_model.mathieu import gsl_mathieu_b, mathieu_b_asymptotic, scipy_mathieu_b

N = 1000
n = np.arange(1, 25)
D = np.linspace(1, 1500, N)
q = D / 4


legend_lines = custom_lines = [Line2D([0], [0], color="C0", lw=1), Line2D([0], [0], color="C1", lw=1)]

plt.figure()
plt.title("GSL vs Scipy")
for i in n:
    a = scipy_mathieu_b(i, q)
    b = gsl_mathieu_b(i, q)

    plt.plot(D, a, ":", color="C0")
    plt.plot(D, b, "-", color="C1")

plt.legend(legend_lines, ["Scipy", "GSL"])
plt.xlabel("D")
plt.ylabel("Mathieu_b")


plt.figure()
plt.title("GSL vs Asymptotic")
for i in n:
    a = mathieu_b_asymptotic(i, q)
    b = gsl_mathieu_b(i, q)

    plt.plot(D, a, ":", color="C0")
    plt.plot(D, b, "-", color="C1")

plt.ylim(-500, 1000)
plt.legend(legend_lines, ["Asymptotic", "GSL"])
plt.xlabel("D")
plt.ylabel("Mathieu_b")
plt.show()
