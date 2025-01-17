import numpy as np
from large_lattice_model.xyz import beloy_XYZ, modified_ushijima_XYZ_nz
from matplotlib import pyplot as plt

plt.close("all")
plt.ion()

D = np.linspace(50, 500, 20)
Tz = D * 1e-6 / 50
Tr = D * 1e-6 / 50


X, Y, Z = beloy_XYZ(D, Tz, Tr)


plt.figure()
plt.title("Beloy_XYZ")
plt.plot(D, X, label="X")
plt.plot(D, Y, label="Y")
plt.plot(D, Z, label="Z")
plt.xlabel("Trap depth /Er")
plt.ylabel("X, Y, Z")
plt.legend(loc=0)


X, Y, Z = modified_ushijima_XYZ_nz(D, Tr, nz=0)


plt.figure()
plt.title("Modified Ushijima XYZ_nz")
plt.plot(D, X, label="X")
plt.plot(D, Y, label="Y")
plt.plot(D, Z, label="Z")
plt.xlabel("Trap depth /Er")
plt.ylabel("X, Y, Z")
plt.legend(loc=0)
