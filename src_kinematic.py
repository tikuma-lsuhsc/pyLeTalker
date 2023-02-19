import numpy as np

TWO_PI = 2 * np.pi


def KinematicSource(eth, F0, L0, T0, PL, F0n, x02, npr, xib, Ny, Nz):
    goffset = 0.0001
    freq = F0
    L = 0.56 * L0 * np.sqrt((F0 / F0n))
    eps = (L - L0) / L0
    T = T0 / (1 + 0.8 * eps)
    zn = npr * T
    pth = 10000 * (0.14 + 0.06 * (F0 / F0n) ** 2)

    if PL > pth:
        xm = 0.05 * L0 * np.sqrt((PL - pth) / pth)
    else:
        xm = 0

    c = np.pi * F0 * T

    xcv = (4.0 - 4.4 * zn / T) / 10

    pgap = 0
    x01 = xcv + x02

    # attenuation with abduction - ad hoc rule
    xm = 0.1
    xm = xm * max(0, 0.25 * (np.cos(x02 * 3.5) + 1) ** 2)

    omega = TWO_PI * freq

    ny = Ny
    nz = Nz

    dydz = (L / (ny - 1)) * (T / (nz - 1))

    y = np.arange(ny).reshape(1, -1, 1) * L / (ny - 1)
    z = np.arange(nz).reshape(1, 1, -1) * T / (nz - 1)

    eth = eth.reshape(-1, 1, 1)
    cth = eth.real
    sth = eth.imag
    lengtri = 1 - y / L
    amplsin = xm * np.sin(np.pi * y / L)
    xi0 = lengtri * (x02 + (x01 - x02 - 4 * xib * z / T) * (1 - z / T)) + goffset
    xi1 = amplsin * (sth - omega * (z - zn) * cth / c)  # linearized
    npout = np.argmin(np.abs(z - zn))

    # -------------------

    xi = xi0 + xi1
    xipos = np.maximum(0, xi)
    xtest = xi1[:, 9, 7]

    # glottal area calculation
    g = xipos[:, :-1, :] + xipos[:, 1:, :]  # initial area estimate
    ga = np.sum(np.amin(g, axis=2), axis=1) * L / (ny - 1) + pgap

    # Contact area
    # simply add up the number of  elements that have crossed the midline
    tf = (xi < 0).astype(int)
    xminus = np.sum(
        tf[:, :-1, :-1] + tf[:, 1:, :-1] + tf[:, :-1, 1:] + tf[:, 1:, 1:], axis=(1, 2)
    )
    ca = dydz / 4 * xminus

    return (ga, xi, xipos, xi0, y, z, ca, npout, xtest)


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

    Fs = 44100
    N = 18000

    t = np.arange(N) / Fs

    # analytic glottal width waveform at "th"
    phi0 = 0
    A = 0.05
    phi = TWO_PI * F0 * t.reshape(-1, 1, 1) + phi0
    am = A * np.cos(phi / 2) + (1 - A)  # period-doubling amplitude modulation
    eth = np.exp(1j * phi)

    ga, xi, xipos, xi0, y, z, ca, npout, xtest = KinematicSource(
        eth * am, F0, L0, T0, PL, F0n, x02, npr, xib, Ny, Nz
    )
    ga1, xi, xipos, xi0, y, z, ca, npout, xtest = KinematicSource(
        eth, F0, L0, T0, PL, F0n, x02, npr, xib, Ny, Nz
    )

    from matplotlib import pyplot as plt

    plt.subplot(2, 1, 1)
    plt.plot(t, ga)
    plt.plot(t, ga1)
    plt.legend(["1:1", "2:2"])
    plt.xlabel("time (s)")
    plt.ylabel("glottal area")
    plt.xlim(0, t[-1])
    plt.subplot(2, 1, 2)
    plt.magnitude_spectrum(ga, Fs, scale="dB", pad_to=8192 * 16)
    plt.magnitude_spectrum(ga1, Fs, scale="dB", pad_to=8192 * 16)
    plt.xlim(0, 1000)
    plt.ylim(-150, 0)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("spectrum (dB)")
    plt.show()
