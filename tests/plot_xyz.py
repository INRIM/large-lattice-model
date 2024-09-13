import numpy as np
from matplotlib import pyplot as plt

from large_lattice_model.xyz import beloy_XYZ

plt.close("all")
plt.ion()

D = np.linspace(50, 500, 20)
Tz = D * 1e-6 / 50
Tr = D * 1e-6 / 50


X, Y, Z = beloy_XYZ(D, Tz, Tr)


plt.figure()
plt.plot(D, X, label="X")
plt.plot(D, Y, label="Y")
plt.plot(D, Z, label="Z")
plt.xlabel("Trap depth /Er")
plt.ylabel("X, Y, Z")
plt.legend(loc=0)
