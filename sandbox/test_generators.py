from matplotlib import pyplot as plt
import numpy as np
from math import ceil, log2
from scipy.signal import get_window

from letalker.function_generators import SineGenerator, ModulatedSineGenerator
from letalker.constants import fs

N = 20000

t = np.arange(N).reshape((-1, 1)) / fs

fo = 101

m_freq = 1 / 2
extent = 0.2

sine = SineGenerator(fo)
am_sine = ModulatedSineGenerator(fo, m_freq, am_extent=extent)
fm_sine = ModulatedSineGenerator(fo, m_freq, fm_extent=extent)
amfm_sine = ModulatedSineGenerator(fo, m_freq, am_extent=extent, fm_extent=extent)

x = sine(N)
y = am_sine(N)
z = fm_sine(N)
w = amfm_sine(N)

plt.figure()
plt.plot(t, x, t, y, t, z, t, w)
plt.figure()
nfft = 2 ** ceil(log2(N * 4))
win = get_window('hamming',N)
plt.magnitude_spectrum(x, fs, pad_to=nfft, scale="dB",window=win)
plt.magnitude_spectrum(y, fs, pad_to=nfft, scale="dB",window=win)
plt.magnitude_spectrum(z, fs, pad_to=nfft, scale="dB",window=win)
plt.magnitude_spectrum(w, fs, pad_to=nfft, scale="dB",window=win)
plt.xlim([0, 1000])
plt.show()
