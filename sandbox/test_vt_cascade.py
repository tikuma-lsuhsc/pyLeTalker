import control as ct
import numpy as np
from matplotlib import pyplot as plt

from letalker.constants import c, vocaltract_areas
from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import segments

areas = vocaltract_areas["aa"]
fs = 44100
sample_kws = {"method": "bilinear"}

z1 = segments.DefaultLosslessPropagationTF()
z2 = segments.DefaultViscousLossTF(drop_reactive=False)
z3 = segments.DefaultLaminarResistance()
s = segments.SeriesNetwork(z1, z2, z3)
sys = s(
    areas[0], length, _use_improper=False
)  # , fs=fs, sample_kws=sample_kws, sample_last=True)
print(sys)

ct.bode_plot(sys, omega_limits=(10, 20000))
plt.show()
exit()

f = np.arange(10, 5000, 1)
omega = 2 * np.pi * f
omega = 2 * np.pi * f
domega = omega[1] - omega[0]
resp = ct.frequency_response(sys, omega=omega)

tau = length / c
print(tau)
sys2 = ct.tf([1, 0], [tau, 1])
r2 = ct.frequency_response(sys2, omega=omega)
print(sys2)

fig, ax = plt.subplots(3, 1)
ax[0].plot(f, 20 * np.log10(resp.magnitude.reshape(4, -1).T))
ax[0].plot(f, 20 * np.log10(r2.magnitude.reshape(1, -1).T))
ax[1].plot(f, resp.phase.reshape(4, -1).T)
ax[2].plot(
    f[:-1],
    -np.diff(resp.phase.reshape(4, -1).T, axis=0) / domega,
    f[:-1],
    -np.diff(r2.phase.reshape(1, -1).T, axis=0) / domega,
)
ax[2].axhline(tau, ls=":", c="k")
plt.legend(["1", "2", "3", "4"])
plt.show()
