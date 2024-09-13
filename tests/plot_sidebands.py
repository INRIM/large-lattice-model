import numpy as np
from matplotlib import pyplot as plt

from large_lattice_model.latticemodel import lorentzian
from large_lattice_model.sidebands import sidebands

plt.close("all")
plt.ion()

D = 100
Tz = 2e-6
Tr = 1e-6
wc = 2e3


nu = np.linspace(-50e3, 50e3, 200)
out = sidebands(nu, D, Tz, Tr, 3, 3, wc) + lorentzian(nu, 0, wc)


plt.figure()
plt.plot(nu / 1e3, out, "-")
plt.xlabel("Detuning /kHz")
plt.ylabel("Signal")
