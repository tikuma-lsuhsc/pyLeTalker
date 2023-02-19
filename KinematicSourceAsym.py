import numpy as np


def KinematicSourceAsym(F0r, F0l, x02r, x02l, xibr, xibl, phi0r, phi0l, iter, dt):
    # ----Right Vocal fold------------------
    freqr = F0r
    F0n = 125
    T0 = 0.5
    # freqp = 105
    L0 = 1.6
    L = 0.56 * L0 * np.sqrt(((F0r + F0l) / (2 * F0n)))
    # T = 0.5

    strain = (L - L0) / L0
    T = T0 / (1 + 0.8 * strain)
    znr = 0.3 * T

    pth = 10000 * (0.14 + 0.06 * (F0r / F0n) ^ 2)
    xmr = 0.05 * L0 * np.sqrt((8000 - pth) / pth)
    cr = np.pi * F0r * T
    # xcv = 4.0-4.4*zn/T #????
    xcvr = (4.0 - 4.4 * znr / T) / 10
    # xcvr = 0.4
    # phi0 = 0

    # x02 = 0.4
    pgap = 0
    x01r = xcvr + x02r

    xi02r = x02r
    xi01r = x01r

    # ----Left Vocal fold------------------
    freql = F0l
    F0n = 125
    T0 = 0.5
    # freqp = 105
    L0 = 1.6
    L = 0.56 * L0 * np.sqrt(((F0r + F0l) / (2 * F0n)))
    # T = 0.5

    strain = (L - L0) / L0
    T = T0 / (1 + 0.8 * strain)
    znl = 0.3 * T

    pth = 10000 * (0.14 + 0.06 * (F0l / F0n) ^ 2)
    xml = 0.05 * L0 * np.sqrt((8000 - pth) / pth)
    cl = np.pi * F0l * T
    # xcv = 4.0-4.4*zn/T #????
    xcvl = (4.0 - 4.4 * znl / T) / 10
    # xcvl = 0.4
    # phi0 = 0

    # x02 = 0.4
    pgap = 0
    x01l = xcvl + x02l

    xi02l = x02l
    xi01l = x01l
    # L
    # T
    # zn
    # pth
    # xm
    # c
    # xcv
    # x01

    # ----Right Fold-----------

    t = iter * dt
    omega = 2 * np.pi * freqr
    # t = theta/omega

    ny = 21
    nz = 15
    dydz = (L / (ny - 1)) * (T / (nz - 1))
    ga = 0
    ca = 0

    y = np.arange(ny) * L / (ny - 1)
    z = np.arange(nz) * T / (nz - 1)
    xi0 = np.zeros(ny, nz)
    xi1 = np.zeros(ny, nz)
    xi = np.zeros(ny, nz)
    xiR = np.zeros(ny, nz)
    xiL = np.zeros(ny, nz)
    xiposr = np.zeros(ny, nz)
    xiposl = np.zeros(ny, nz)
    xineg = np.zeros(ny, nz)

    sth = np.sin(omega * t + phi0r)
    cth = np.cos(omega * t + phi0r)

    for i in range(ny):
        lengtri = 1 - y(i) / L
        amplsin = xmr * np.sin(np.pi * y(i) / L)

        for j in range(nz):
            xi0[i, j] = lengtri * (
                x02r + (x01r - x02r - 4 * xibr * z(j) / T) * (1 - z(j) / T)
            )
            # xi1[i,j] = amplsin*np.sin(omega*(t-z(j)/c) + phi0)
            xi1[i, j] = amplsin * (sth - omega * (z(j) - znr) * cth / cr)  # linearized
            # version

            xi[i, j] = xi0[i, j] + xi1[i, j]
            xiR[i, j] = xi[i, j]
            xiposr[i, j] = max(0, xi[i, j])
            xineg[i, j] = min(0, xi[i, j])

    # ----Left Fold-----------

    t = iter * dt
    omega = 2 * np.pi * freql
    # t = theta/omega

    ny = 21
    nz = 15
    dydz = (L / (ny - 1)) * (T / (nz - 1))
    ga = 0
    ca = 0

    y = np.arange(ny) * L / (ny - 1)
    z = np.arange(nz) * T / (nz - 1)
    xi1 = np.zeros(ny, nz)
    xi = np.zeros(ny, nz)
    xipos = np.zeros(ny, nz)
    xineg = np.zeros(ny, nz)

    sth = np.sin(omega * t + phi0l)
    cth = np.cos(omega * t + phi0l)

    for i in range(ny):
        lengtri = 1 - y(i) / L
        amplsin = xml * np.sin(np.pi * y(i) / L)

        for j in range(nz):
            xi0[i, j] = lengtri * (
                x02l + (x01l - x02l - 4 * xibl * z(j) / T) * (1 - z(j) / T)
            )
            # xi1[i,j] = amplsin*np.sin(omega*(t-z(j)/c) + phi0)
            xi1[i, j] = amplsin * (sth - omega * (z(j) - znl) * cth / cl)  # linearized
            # version

            xi[i, j] = -xi0[i, j] - xi1[i, j]
            xiL[i, j] = xi[i, j]
            xiposl[i, j] = max(0, xi[i, j])
            xineg[i, j] = min(0, xi[i, j])

    dL = L / (ny - 1)

    for i in range(ny - 1):
        tmp1 = min(xiR[i, 0] - xiL[i, 0], xiR[i + 1, 0] - xiL[i + 1, 0])
        tmp2 = max(xiR[i, 0] - xiL[i, 0], xiR[i + 1, 0] - xiL[i + 1, 0])
        A1 = tmp1 * dL
        B1 = 0.5 * (max(xiL[i, 0], xiL[i + 1, 0]) - min(xiL[i, 0], xiL[i + 1, 0])) * dL
        C1 = 0.5 * (max(xiR[i, 0], xiR[i + 1, 0]) - min(xiR[i, 0], xiR[i + 1, 0])) * dL

        g = A1 + B1 + C1  # initial area estimate
        g = max(0, g)

        for j in range(nz - 1):
            tmp1 = min(
                xiR[i, j + 1] - xiL[i, j + 1], xiR[i + 1, j + 1] - xiL[i + 1, j + 1]
            )
            tmp2 = max(
                xiR[i, j + 1] - xiL[i, j + 1], xiR[i + 1, j + 1] - xiL[i + 1, j + 1]
            )
            A1 = tmp1 * dL
            B1 = (
                0.5
                * (
                    max(xiL[i, j + 1], xiL[i + 1, j + 1])
                    - min(xiL[i, j + 1], xiL[i + 1, j + 1])
                )
                * dL
            )
            C1 = (
                0.5
                * (
                    max(xiR[i, j + 1], xiR[i + 1, j + 1])
                    - min(xiR[i, j + 1], xiR[i + 1, j + 1])
                )
                * dL
            )
            gtmp = max(0, A1 + B1 + C1)
            g = min(g, gtmp)

        ga = ga + g

    for i in range(ny - 1):
        xminus = 0
        xmag = 0

        for j in range(nz - 1):
            # Attempt at Druker's version
            # xminus = xminus + -(min(0,xi[i,j]) + min(0,xi(i+1,j)) + min(0,xi(i,j+1)) + min(0,xi(i+1,j+1)))
            # xmag = xmag + abs(xi[i,j]) + abs(xi(i+1,j)) + abs(xi(i,j+1)) + abs(xi(i+1,j+1))

            # simply add up the number of  elements that have crossed the
            # midline
            xminus = xminus + (
                (min(0, xi[i, j]) < 0)
                + (min(0, xi[i + 1, j]) < 0)
                + (min(0, xi[i, j + 1]) < 0)
                + (min(0, xi[i + 1, j + 1]) < 0)
            )

        # divide by 4 because each set of four elements constructs a square
        # it is the number of squares that needs to be multiplied by dydz
        ca = ca + dydz * xminus / 4

    return [ga, xiR, xiL, y, z]


if __name__ == "__main__":
    F0l = 110  # Fund. freq.
    x02l = 0.1  # Adduction setting - dist. of superior part of fold from midline
    xibl = 0.25  # bulging set to either 0.0 or 0.15
    F0r = 110  # Fund. freq.
    x02r = 0.4  # Adduction setting - dist. of superior part of fold from midline
    xibr = 0.0  # bulging set to either 0.0 or 0.15
    phi0l = 0
    phi0r = 0
    Fs = 44100
    N = 2000

    dt = 1 / Fs

    ga = np.zeros(N)

    for iter in range(N):
        ga[iter], xir, xil, y, z = KinematicSourceAsym(
            F0r, F0l, x02r, x02l, xibr, xibl, phi0r, phi0l, iter, dt
        )
