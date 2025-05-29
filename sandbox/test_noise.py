"""Test Aspiration Noise Elements, excerpted from Elements_Basics.ipynb"""

from matplotlib import pyplot as plt
import numpy as np
import letalker as lt

from scipy.signal import welch
from scipy.fft import next_fast_len

# create aspiration noise sources
lt_noise = lt.LeTalkerAspirationNoise()
kl_noise = lt.KlattAspirationNoise()

# create kinematic vocal fold models at fo=100Hz and assign the noise objects to them
kwargs = {
    "upstream": lt.constants.vocaltract_areas["trach"][-1],
    "downstream": lt.constants.vocaltract_areas["aa"][0],
    "length": lt.constants.male_vf_params["L0"],
}

ug_gen = lt.RosenbergGenerator(100, 600)

vf_lt = lt.VocalFoldsUg(ug_gen, aspiration_noise=lt_noise, **kwargs)
vf_kl = lt.VocalFoldsUg(ug_gen, aspiration_noise=kl_noise, **kwargs)

# create their simulation runner objects
n = round(vf_lt.fs)  # number of samples to simulate
lt_runner = vf_lt.create_runner(n)
kl_runner = vf_kl.create_runner(n)

# preallocate forward pressure output array (ignore backward outputs for now)
Fout = np.empty((n, 2))

# run simulation
b = [0, 0]
for i in range(n):
    Fout[i, 0], _ = lt_runner.step(i, 0.0, 0.0)
    Fout[i, 1], _ = kl_runner.step(i, 0.0, 0.0)

# gather the results
lt_res = vf_lt.create_result(lt_runner)
kl_res = vf_kl.create_result(kl_runner)
t = lt_res.ts

fs = lt.constants.fs
N = round(lt_res.fs * 0.1)
nfft = nfft = next_fast_len(fs)

# PSD estimation
f, Pxx = welch(
    [lt_res.ug, kl_res.ug], nperseg=N, noverlap=N // 2, nfft=nfft, axis=-1, fs=fs
)
_, Pyy = welch(Fout, nperseg=N, noverlap=N // 2, nfft=nfft, axis=0, fs=fs)
_, Pzz = welch(
    [lt_res.aspiration_noise.unoise, kl_res.aspiration_noise.unoise],
    nperseg=N,
    noverlap=N // 2,
    nfft=nfft,
    axis=-1,
    fs=fs,
)
PyydB = 10 * np.log10(Pyy / 2)

# plot PSDs
fig, axes = plt.subplots(2, 3, figsize=[14, 8], sharex="col", sharey="col")
axes[0, 0].plot(f, PyydB[:, 0])
axes[0, 0].grid()
axes[0, 0].set_xlim((0, 5000))
axes[0, 0].set_ylim((0, 80))
axes[0, 0].set_title("$F_{out}$")
axes[0, 0].set_ylabel("LeTalkerAspirationNoise")
axes[1, 0].plot(f, PyydB[:, 1])
axes[1, 0].grid()
axes[1, 0].set_xlabel("frequency (Hz)")
axes[1, 0].set_ylabel("KlattAspirationNoise")

axes[0, 1].plot(f, 10 * np.log10(Pxx[0, :] / 2), label="$U_g$")
axes[0, 1].plot(f, 10 * np.log10(Pzz[0, :] / 2), label="$U_{noise}$")
axes[0, 1].grid()
axes[0, 1].legend()
axes[0, 1].set_xlim((0, 5000))
axes[0, 1].set_ylim((-50, 40))
axes[0, 1].set_title("$U_g$/$U_{noise}$")
axes[1, 1].plot(f, 10 * np.log10(Pxx[1, :] / 2), label="$U_g$")
axes[1, 1].plot(f, 10 * np.log10(Pzz[1, :] / 2), label="$U_{noise}$")
axes[1, 1].grid()
axes[1, 1].set_xlabel("frequency (Hz)")

axes[0, 2].plot(t, lt_res.aspiration_noise.re2**0.5, label="$R_e$")
axes[0, 2].axhline(lt_res.aspiration_noise.re2b**0.5, c="C1", label="$R_{e,c}$")
axes[0, 2].grid()
axes[0, 2].set_title("$R_e$")
axes[0, 2].legend()
axes[1, 2].plot(t, kl_res.aspiration_noise.re2**0.5, label="$R_e$")
axes[1, 2].axhline(kl_res.aspiration_noise.re2b**0.5, c="C1", label="$R_{e,c}$")
axes[1, 2].grid()
axes[1, 2].set_xlim((0, 0.1))
axes[1, 2].set_xlabel("time (s)")

plt.show()
