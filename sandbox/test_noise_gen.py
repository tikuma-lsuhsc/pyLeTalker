from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import get_window, freqz, sos2tf, butter, welch
from scipy.fft import rfft

from letalker import ColoredNoiseGenerator
from letalker.constants import fs

N = 200000

t = np.arange(N) / fs

fo = 101

m_freq = 1 / 2
extent = 0.2

# b,a = butter(2, [300,3000],'bandpass',fs=fs)
# b, a = LeTalkerAspirationNoise.cb, LeTalkerAspirationNoise.ca
b = [1.25]
a = [1, -0.75]
# f, H = freqz(b, a, fs=fs)
# print(np.abs(H).max())
# b *= 1/np.abs(H).max()
# print(b)

generator = ColoredNoiseGenerator(b, a)
x = generator.generate(N)[0]

nwin = N // 100
win = get_window("boxcar", nwin)

S = win.sum()  # rectangular window
# x = np.random.randn(N)*fs**0.5
x = x.reshape((N // nwin, nwin))

nfft = 1024*8
X = np.mean(np.abs(rfft(x.reshape(N // nwin, nwin) * win, nfft, axis=-1)) ** 2, 0) / S  # / fs
f = np.arange(nfft // 2 + 1) * fs / nfft

print(x.var() / fs)
print(X.mean())

plt.subplots(3, 1, sharex=True, sharey=True)
plt.subplot(3, 1, 1)
plt.plot(f, X)
plt.subplot(3, 1, 2)
f, H = welch(
    x.reshape(-1),
    fs,
    nperseg=nwin,
    noverlap=nwin * 3 // 4,
    nfft=nfft,
    scaling="density",
    detrend=False,
)
plt.plot(f, H*fs/2)

plt.subplot(3, 1, 3)
b, a = sos2tf(generator.sos)
f, H = freqz(b, a, fs=fs)
plt.plot(f, np.abs(H) ** 2)
plt.show()
