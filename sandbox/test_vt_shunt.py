import control as ct
import numpy as np
from matplotlib import pyplot as plt

from letalker.constants import vocaltract_areas
from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import segments

areas = vocaltract_areas["aa"]
fs = 44100
sample_kws = {"method": "bilinear"}


z = segments.DefaultYieldingWall()
z1 = segments.DefaultHeatLossGain()
s = segments.ShuntNetwork(z)
sys = s(
    areas[0], length, _use_improper=True
)  # , fs=fs, sample_kws=sample_kws, sample_last=True)
print(sys)

ct.bode_plot(sys)
plt.show()
exit()

f = np.arange(10, 5000, 1)
omega = 2 * np.pi * f
domega = omega[1] - omega[0]
resp = ct.frequency_response(sys, omega=omega)

fig, ax = plt.subplots(3, 1)
ax[0].plot(f, 20 * np.log10(resp.magnitude.reshape(4, -1).T))
ax[1].plot(f, resp.phase.reshape(4, -1).T)
ax[2].plot(
    f[:-1],
    -np.diff(resp.phase.reshape(4, -1).T, axis=0) / domega,
)
plt.legend(["1", "2", "3", "4"])
plt.show()
