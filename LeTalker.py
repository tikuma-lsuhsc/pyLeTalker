from math import atan, pi, sqrt, cos
import numpy as np
from scipy.io.matlab import loadmat
from dataclasses import dataclass

from rules_consconv import rules_consconv
from calc_pressures import calc_pressures
from rk3m import rk3m
from calcflow import calcflow
from zerocross import zerocross


@dataclass
class Outcome:
    x1: np.array  # lower mass disp
    x2: np.array  # upper mass disp
    xb: np.array  # body mass disp
    po: np.array  # output pressure
    ug: np.array  # glottal flow
    ga: np.array  # glottal area
    theta: np.array
    x: np.array
    xb: np.array
    nois: np.array
    ad: np.array
    ps: np.array
    pi: np.array
    f1: np.array
    f2: np.array
    f0 = None

    def __init__(self, N):
        self.x1 = np.zeros(N)
        self.x2 = np.zeros(N)
        self.xb = np.zeros(N)
        self.po = np.zeros(N)
        self.ug = np.zeros(N)
        self.ga = np.zeros(N)
        self.theta = np.zeros(N)
        self.x = np.zeros(N)
        self.xb = np.zeros(N)
        self.nois = np.zeros(N)
        self.ad = np.zeros(N)
        self.ps = np.zeros(N)
        self.pi = np.zeros(N)
        self.f1 = np.zeros(N)
        self.f2 = np.zeros(N)
        self.f0 = None


def LeTalker(p, c, N, Fs, ctvect, tavect, plvect):
    # LeTalker version 1.23
    # LeTalker - means "Lumped element Talker" and is the control code
    # for running the three-mass model of Story and Titze (1995) and Titze and Story (2002) with sub and supra glottal
    # airways
    # Author: Brad Story, Univ. of Arizona
    # Date: 06.18.2014

    p = rules_consconv(p, c)

    # load filter coefficients for band pass filtering glottal noise
    fcoeffs = loadmat("filter_coeffs")
    ca = fcoeffs["ca"].reshape(-1)
    cb = fcoeffs["cb"].reshape(-1)

    # If SUB=1 the subglottal system is included SUB=0 will take it out.
    SUB = p.SUB
    if SUB == 0:
        p.ar[-1] = 1000000

    # If SUPRA=1 the supraglottal system is included SUPRA=0 will take it out.
    SUPRA = p.SUPRA
    if SUPRA == 0:
        p.ar[0] = 1000000

    neq = 6
    F = np.zeros(neq)

    r = Outcome(N)

    p.tan_th0 = p.xc / p.T
    p.th0 = atan(p.tan_th0)

    F = np.zeros(6)
    # For use with Titze 1 all zeros
    # For use with Titze 2 below-----
    # F[0] = p.x01
    # F[1] = p.x02
    # F[2] = p.xb0

    # -----
    a = p.ar
    Nsect = len(a)
    f = np.zeros(Nsect)
    b = np.zeros(Nsect)
    r1 = np.zeros(Nsect)
    r2 = np.zeros(Nsect)
    # aprev = np.zeros(jmoth)
    # atot = np.zeros(jmoth)

    # ---Initialize vectors for noise generation--
    xnois = np.zeros(5)
    ynois = np.zeros(5)
    Unois = 0

    # vocal tract attenuation - same everywhere
    # alpha = .98*ones(1eom_3m,Nsect)
    csnd = 35000
    rho = 0.00114
    rhoc = rho * csnd
    mu = 0.000186
    PI = pi
    # PI2 = 2 * PI

    bprev = 0
    fprev = 0
    Pox = 0.0

    dt = 1 / Fs
    tme = 0.0
    R = 128.0 / (9.0 * pi**2)

    # --initialize----------
    p.a1 = 2 * p.L * F[0]
    p.a2 = 2 * p.L * F[1]
    p.ga = max(0, min(p.a1, p.a2))

    # vocal tract attenuation - based on xsect area
    alpha = 1 - p.vtatten / np.sqrt(a)

    # posterior glottal gap
    # pgap = 0
    p.psg = p.ps

    # ------

    tcos = 0.002

    ieplx = 0  # larynx+
    jmoth = 43  # mouth
    itrch = jmoth + 1  # lung
    jtrch = Nsect - 1  # larynx-

    for n in range(N):
        t = n * dt

        # -- check for time-varying muscle activation and pressure ramp
        if len(tavect):
            p.ata = tavect[n]

        if len(ctvect):
            p.act = ctvect[n]

        if len(plvect):
            p.ps = plvect[n]
            if t <= tcos:  # ramp the driving pressure
                fcos = 1 / (2 * tcos)
                p.ps = (
                    p.ps * (cos(2 * pi * fcos * t - pi) + 1) / 2
                )  # subglottal pressure

        else:
            if t <= tcos:  # ramp the driving pressure
                fcos = 1 / (2 * tcos)
                p.ps = (
                    p.PL * (cos(2 * pi * fcos * t - pi) + 1) / 2
                )  # subglottal pressure
            else:
                p.ps = p.PL

        # -------------------------

        p = rules_consconv(p, c)

        # This is internally necessary because p.ps was changed to p.psg in calc_pressures.m to accomodate the trachea wave
        # propagation for other implementations
        p.psg = p.ps

        # ================Titze 1 ==========================

        znot = p.zn / p.T
        p.xn = (1 - znot) * (p.x01 + F[0]) + znot * (p.x02 + F[1])
        p.tangent = (p.x01 - p.x02 + 2 * (F[0] - F[1])) / p.T
        p.x1 = p.xn - (-p.zn) * p.tangent
        p.x2 = p.xn - (p.T - p.zn) * p.tangent
        # ====================================================

        # ================Titze 2 ==========================
        #  p.tan_th0 = (p.x01-p.x02)/p.T
        #  p.th0 = atan(p.tan_th0)
        #  p.tangent = tan(p.th0 + F[0])
        #  p.xn = p.x02 + (p.T - p.zn)*p.tan_th0 + F[1]
        #  p.x1 = p.xn - ( -p.zn)*p.tangent
        #  p.x2 = p.xn - (p.T - p.zn)*p.tangent
        # ====================================================

        # Vocal fold area calculations
        p.zc = min(p.T, max(0.0, p.zn + p.xn / p.tangent))
        p.a1 = max(p.delta, 2 * p.L * p.x1)
        p.a2 = max(p.delta, 2 * p.L * p.x2)
        p.an = max(p.delta, 2 * p.L * p.xn)
        p.zd = min(p.T, max(0, -0.2 * p.x1 / p.tangent))
        p.ad = min(p.a2, 1.2 * p.a1)

        # contact area calculations
        if p.a1 > p.delta and p.a2 <= p.delta:
            p.ac = p.L * (p.T - p.zc)
        elif p.a1 <= p.delta and p.a2 > p.delta:
            p.ac = p.L * p.zc
        elif p.a1 <= p.delta and p.a2 <= p.delta:
            p.ac = p.L * p.T

        # Calculate driving pressures
        p.pe = f[ieplx] + b[ieplx]
        p = calc_pressures(p)

        # Integrate EOMs with Runge-Kutta
        F = rk3m(F, dt, p)

        # Glottal area calculation
        p.ga = max(0, min(p.a1, p.a2))

        # Calculate the glottal flow
        ug = calcflow(p.ga, f[jtrch], b[ieplx], a, csnd, rhoc)

        # =========noise generator====================
        # Added on 10.04.12 - Reynolds number calculation and option for adding
        # noise to simulate turbulence-generated sound, based on Samlan and
        # Story (2011) JSLHR
        if p.Noise:
            RE2 = (ug * rho / (mu * p.L)) ** 2
            RE2b = 1440000  # Corresponds to RE = 1200
            Anois = 0.000004 * (RE2 - RE2b)
            Unois = 0

            tmp1 = np.sum(cb * xnois[-1::-1])
            tmp2 = np.sum(cb[1:] * ynois[-2::-1])
            ynois[-1] = tmp1 - tmp2

            if Anois > 0:
                Unois = Anois * ynois[-1]

            ug = ug + Unois

            xnois[:-1] = xnois[1:]
            ynois[:-1] = ynois[1:]
            xnois[-1] = np.random.rand() - 0.5

        #   ============== wave propagation in vocal tract ======================= */

        D = a[:jtrch] + a[1 : jtrch + 1]
        r1 = (a[:jtrch] - a[1 : jtrch + 1]) / D
        r2 = -r1

        # ---even sections in trachea--- */

        # f[ieplx] = alpha*b[ieplx] + p.u[n]*(rhoc/a[ieplx])
        f[ieplx] = alpha[ieplx] * b[ieplx] + ug * (rhoc / a[ieplx])

        if SUPRA == 0:
            f[ieplx] = 0

        if SUB == 1:
            p.psg = 2 * f[jtrch] * alpha[jtrch] - ug * (rhoc / a[jtrch])
            f[itrch] = 0.9 * p.ps - 0.8 * b[itrch] * alpha[itrch]
        else:
            p.psg = p.ps
            f[jtrch] = p.ps

        f = alpha * f
        b = alpha * b

        # ---even sections in trachea--- */
        if SUB == 1:
            Psi = (
                f[itrch + 1 : jtrch - 1 : 2] * r1[itrch + 1 : jtrch - 1 : 2]
                + b[itrch + 2 : jtrch : 2] * r2[itrch + 1 : jtrch - 1 : 2]
            )
            b[itrch + 1 : jtrch - 1 : 2] = b[itrch + 2 : jtrch : 2] + Psi
            f[itrch + 2 : jtrch : 2] = f[itrch + 1 : jtrch - 1 : 2] + Psi
            b[jtrch] = f[jtrch] * alpha[jtrch] - ug * rhoc / a[jtrch]

        # ---even sections in supraglottal--- */

        Psi = (
            f[ieplx + 1 : jmoth - 1 : 2] * r1[ieplx + 1 : jmoth - 1 : 2]
            + b[ieplx + 2 : jmoth : 2] * r2[ieplx + 1 : jmoth - 1 : 2]
        )
        b[ieplx + 1 : jmoth - 1 : 2] = b[ieplx + 2 : jmoth : 2] + Psi
        f[ieplx + 2 : jmoth : 2] = f[ieplx + 1 : jmoth - 1 : 2] + Psi

        # odd sections */

        if SUB == 1:
            Psi = (
                f[itrch:jtrch:2] * r1[itrch:jtrch:2]
                + b[itrch + 1 : jtrch + 1 : 2] * r2[itrch:jtrch:2]
            )
            b[itrch:jtrch:2] = b[itrch + 1 : jtrch + 1 : 2] + Psi
            f[itrch + 1 : jtrch + 1 : 2] = f[itrch:jtrch:2] + Psi

        Psi = (
            f[ieplx:jmoth:2] * r1[ieplx:jmoth:2]
            + b[ieplx + 1 : jmoth + 1 : 2] * r2[ieplx:jmoth:2]
        )
        b[ieplx:jmoth:2] = b[ieplx + 1 : jmoth + 1 : 2] + Psi
        f[ieplx + 1 : jmoth + 1 : 2] = f[ieplx:jmoth:2] + Psi

        # ------------- Lip Radiation -----------  */
        am = sqrt(a[jmoth] / PI)
        L = (2.0 / dt) * 8.0 * am / (3.0 * PI * csnd)
        a2 = -R - L + R * L
        a1 = -R + L - R * L
        b2 = R + L + R * L
        b1 = -R + L + R * L
        b[jmoth] = (1 / b2) * (f[jmoth] * a2 + fprev * a1 + bprev * b1)
        Pout = (1 / b2) * (Pox * b1 + f[jmoth] * (b2 + a2) + fprev * (a1 - b1))
        Pox = Pout

        bprev = b[jmoth]
        fprev = f[jmoth]

        # --Complete reflection at lips------
        # b[jmoth] = -f[jmoth]
        # Pout = b[jmoth]+f[jmoth]
        # ---------------------------------------- */

        tme = tme + dt

        # Assign output signals to structure--------
        r.po[n] = Pout
        r.ug[n] = ug
        r.nois[n] = Unois
        r.ga[n] = p.ga
        r.ad[n] = p.ad
        r.x1[n] = p.x1
        r.x2[n] = p.x2
        r.xb[n] = F[2]
        r.ps[n] = p.psg
        r.pi[n] = p.pe
        r.f1[n] = p.f1
        r.f2[n] = p.f2

        if np.isnan(r.x1).any() or np.isnan(r.x2).any() or np.isnan(r.xb).any():
            r.x1 = np.zeros(N)
            r.x2 = np.zeros(N)
            break
        elif np.isinf(r.x1).any() or np.isinf(r.x2).any() or np.isinf(r.xb).any():
            r.x1 = np.zeros(N)
            r.x2 = np.zeros(N)
            break

    tmp = r.x1
    n1 = round(0.4 * len(tmp))
    n2 = len(tmp) - round(0.1 * len(tmp))
    tmp = tmp[n1:n2]
    tmp = tmp
    if min(tmp) > 0:
        tmp = tmp - 0.1 * max(tmp)

    f0, tme = zerocross(tmp, Fs)

    r.f0 = round(np.mean(f0))

    return p, r


if __name__ == "__main__":
    from InitializeLeTalker import Constants, Properties
    from PlotLeTalkerWaveforms import PlotLeTalkerWaveforms

    Fs = 44100
    N = 4000
    c = Constants()
    p = Properties()
    p.Noise = True
    [p, r] = LeTalker(p, c, N, Fs, [], [], [])
    PlotLeTalkerWaveforms(r, p, Fs, 1, 2)
