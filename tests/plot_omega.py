import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from large_lattice_model.latticemodel import Omega, OmegaMat, OmegaMat2, R, max_nz

plt.close("all")
plt.ion()

D = 200
Nz = max_nz(D)
rr = np.linspace(1e-5, 4, 100)


plt.figure()
plt.title("n -> n+1")
for nz in np.arange(Nz):
    # rr = arange(0, maxr[n], 0.1/kappa) # cut plot at trap edge
    max_r = R(0, D, nz)
    max_n = np.amax(np.where(rr < max_r))

    rr_down = rr[:max_n]
    rr_up = rr[max_n:]

    plt.plot(rr_down, Omega(rr_down, D, nz) ** 2, color="C0")
    plt.plot(rr_down, OmegaMat(rr_down, D, nz) ** 2, color="C1")


plt.ylim(0, 0.4)

plt.xlabel("r*kappa")
plt.ylabel("Omega2")
