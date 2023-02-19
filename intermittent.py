import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import ffmpegio as ff

from src_kinematic import KinematicSource, TWO_PI


def smoothstep(x):
    s = np.zeros_like(x, dtype=float)
    tf = x >= 1
    s[tf] = 1.0
    tf = ~tf & (x > 0)
    s[tf] = 3 * x[tf] ** 2 - 2 * x[tf] ** 3
    return s


def add_transitions(x, nd0, nd1):
    s = smoothstep(np.arange(nd0) / nd0)
    x[:nd0] = s * x[:nd0]

    s = smoothstep(np.arange(nd1) / nd1)
    x[-nd1:] = (1 - s) * x[-nd1:]

    return x


if __name__ == "__main__":
    F0 = 100  # Fund. freq.
    x02 = 0.1  # Adduction setting - dist. of superior part of fold from midline
    xib = 0.1  # bulging set usually to something between 0.0 and 0.15
    npr = 0.7  # nodal point ratio
    PL = 8000  # resp pressure
    L0 = 1.6  # prephontory VF length
    T0 = 0.3  # prephontory VF thickness
    F0n = 125  # F0 for normalization

    # Mesh discretization
    Ny = 21
    Nz = 15
    # theta = 0

    Tk = np.concatenate(
        [
            [0.1 + 0.01 * np.random.randn()],
            1 + 0.1 * np.random.randn(3),
            [0.1 + 0.01 * np.random.randn()],
        ],
    )  # silence|reg|irr|reg|silence

    Delta = 0.02 + 0.001 * np.random.randn(4)  # transition times

    Fs = 44100
    A = 0.2

    phi0 = TWO_PI * np.random.rand()

    Nk = (Tk * Fs).astype(int)  # Total number of samples in simulation
    nk = np.cumsum(Nk)  # starting time of each segment
    t = np.arange(Nk.sum()) / Fs

    nDelta = (Delta * Fs / 2).astype(int)  # half-transition durations

    eth = np.zeros_like(t, dtype=complex)
    phi = TWO_PI * F0 * t + phi0  # base phase function

    # first regular region
    idx = np.arange(nk[0] - nDelta[0], nk[1] + nDelta[1])
    eth[idx] = add_transitions(np.exp(1j * phi[idx]), *(2 * nDelta[:2]))

    # second regular region (irregularity)
    idx = np.arange(nk[1] - nDelta[1], nk[2] + nDelta[2])
    am = A * np.cos(phi[idx] / 2) + (1 - A)  # period-doubling amplitude modulation
    eth[idx] += add_transitions(np.exp(1j * phi[idx]) * am, *(2 * nDelta[1:3]))

    idx = np.arange(nk[2] - nDelta[2], nk[3] + nDelta[3])
    eth[idx] += add_transitions(np.exp(1j * phi[idx]), *(2 * nDelta[2:4]))

    ga, xi, xipos, xi0, y, z, ca, npout, xtest = KinematicSource(
        eth, F0, L0, T0, PL, F0n, x02, npr, xib, Ny, Nz
    )

    ff.audio.write("intermittent.wav", Fs, ga, overwrite=True)

    NFFT = round(0.1 * Fs // 4)
    noverlap = NFFT // 4 * 3

    # plt.subplot(2, 1, 1)
    # plt.plot(t, ga)
    # plt.legend(["1:1", "2:2"])
    # plt.xlabel("time (s)")
    # plt.ylabel("glottal area")
    # plt.xlim(0, t[-1])
    # plt.subplot(2, 1, 2)
    plt.specgram(
        signal.resample_poly(ga, 1, 4),
        NFFT,
        Fs / 4,
        detrend="none",
        noverlap=noverlap,
        scale="dB",
        pad_to=8192 * 4,
    )
    plt.ylim(0, 1000)
    # plt.clim(-150, 0)
    plt.xlabel("time (s)")
    plt.ylabel("frequency (Hz)")
    # plt.ylabel("spectrum (dB)")
    plt.savefig("intermittent.png")
