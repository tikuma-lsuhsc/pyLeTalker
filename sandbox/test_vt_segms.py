import numpy as np
from matplotlib import pyplot as plt

from letalker.constants import vocaltract_areas
from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import segments

areas = vocaltract_areas["aa"]
fs = 44100
sample_kws = {"method": "bilinear"}


series = segments.SeriesNetwork(
    segments.DefaultInertance(),
    segments.DefaultViscousLossTF(resistive_only=False),
    segments.DefaultLaminarResistance(),
)
shunt = segments.ShuntNetwork(
    segments.DefaultYieldingWall(),
    segments.DefaultHeatLossGain(),
)

segm = segments.TSegment(series, shunt)

sys = segm(areas[0], length * 44)
dsys = sys.sample(1 / fs, **sample_kws)
print(dsys)

tf = sys.to_tf()
dtf = dsys.to_tf()

f = np.arange(10, 5000, 1)
omega = 2 * np.pi * f

from scipy.signal import freqs, freqz

omega, H = freqs(tf.num[0][0], tf.den[0][0], omega)
omega, Hz = freqz(dtf.num[0][0], dtf.den[0][0], f, fs)
domega = np.diff(omega)

fig, ax = plt.subplots(2, 1)
ax[0].plot(f, 20 * np.log10(np.abs(H)))
ax[0].plot(f, 20 * np.log10(np.abs(Hz)))
ax[1].plot(f[:-1], -np.diff(np.angle(H[1:] / H[:-1]), axis=0) / domega)
ax[1].plot(f[:-1], -np.diff(np.angle(Hz[1:] / Hz[:-1]), axis=0) / domega)
plt.show()
