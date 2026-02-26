import numpy as np
import numba as nb
from matplotlib import pyplot as plt
from scipy.signal import welch, periodogram

from letalker.elements import LeTalkerAspirationNoise, KlattAspirationNoise
from letalker.constants import fs

N = 1000
t = np.arange(N)

# nfsrc = LeTalkerAspirationNoise()
nfsrc = KlattAspirationNoise()

# f, Sxx = nfsrc.noise_source.psd()
# plt.plot(f, 10 * np.log10(Sxx))
# plt.axhline(10*np.log10(25))
# # plt.plot(f, Sxx**0.5)
# plt.grid()
# plt.show()
# exit()

L = 0.8

x = nfsrc.generate_noise(11000)
# plt.plot(x)
# plt.show()
# exit()

f, Pxx = welch(x[1000:], fs, nperseg=1000, noverlap=750, nfft=1024 * 8)
# f, Pxx = periodogram(x[1000:], fs)
plt.plot(f, 10 * np.log10(Pxx * 1000))
plt.show()
exit()

runner = nfsrc.create_runner(N, length=L)
ug = np.maximum(np.random.randn(N) + 200 * np.sin(np.arange(N) * 650 / fs), 0)


@nb.njit
def sim(runner, ug):
    v = np.empty_like(ug)
    for i, xn in enumerate(ug):
        v[i] = runner.step(i, xn)

    return v


x = sim(runner, ug)
res = nfsrc.create_result(runner)
plt.subplot(1, 2, 1)
# plt.plot(res)
plt.plot(res.src_noise)
plt.plot(res.unoise)
# plt.axhline(nf.RE2b)
plt.subplot(1, 2, 2)
plt.magnitude_spectrum(res.unoise, fs, pad_to=1024 * 8, scale="linear")
plt.show()
