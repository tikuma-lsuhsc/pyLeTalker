from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import pi
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..__util import format_parameter
from .._backend import PyRunnerBase, RunnerBase
from ..function_generators import Constant
from ..function_generators.abc import SampleGenerator
from .abc import AspirationNoise, Element, VocalTract
from .VocalFoldsBase import VocalFoldsBase


def calc_stress(eps: float) -> tuple[float, float, float]:
    """Stress calculations for the three-mass model based on Titze and Story

    :param eps: strain
    :return sigmuc: _description_
    :return sigl:  passive ligament stress
    :return sigp: passive muscle stress
    """

    ep1 = np.full(3, -0.5)
    ep2 = np.array([-0.35, 0.0, -0.05])
    sig0 = np.array([5000, 4000, 10000])
    sig2 = np.array([300000, 13930, 15000])
    Ce = np.array([4.4, 17, 6.5])

    strs = (-sig0 / ep1) * (eps - ep1)
    tf = eps > ep2
    strs[tf] += sig2[tf] * (
        np.exp(Ce[tf] * (eps - ep2[tf])) - 1 - Ce[tf] * (eps - ep2[tf])
    )

    return strs


def eom_mats(
    m: NDArray, k: NDArray, kc: NDArray, b: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Equations of motion for (linear) Three-Mass model

    :param yo: displacements and velocities of the 3 masses (lower, upper, body)
    :param tt: time
    :param f1: lower force (driving pressure)
    :param f2: upper force (driving pressure)
    :param m: masses (lower, upper, body)
    :param b: damping constants (lower, upper, body)
    :param k: spring constant (lower, upper, body, shear-coupling)
    :return K: the derivative of yo
    :return B:
    :return w: reciprocal
    """
    # dyo4 = (
    #     f1 / m1
    #     - (k1 + kc) / m1 * yo1
    #     + kc / m1 * yo2
    #     + k1 / m1 * yo3
    #     - b1 / m1 * yo4
    #     + b1 / m1 * yo6
    # )
    # dyo5 = (
    #     f2 / m2
    #     + kc / m2 * yo1
    #     - (k2 + kc) / m2 * yo2
    #     + k2 / m2 * yo3
    #     - b2 / m2 * yo5
    #     + b2 / m2 * yo6
    # )
    # dyo6 = (
    #     k1 / M * yo1
    #     + k2 / M * yo2
    #     - (k1 + k2 + K) / M * yo3
    #     + b1 / M * yo4
    #     + b2 / M * yo5
    #     - (b1 + b2 + B) / M * yo6
    # )

    # K = np.array([[-k1 - kc, kc, k1], [kc, -k2 - kc, k2], [k1, k2, -k1 - k2 - K]]) / m
    K = (
        np.reshape(
            np.dot(
                k,
                [
                    [-1, 0, 1, 0, 0, 0, 1, 0, -1],
                    [0, 0, 0, 0, -1, 1, 0, 1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, -1],
                ],
            )
            + np.dot(kc[..., np.newaxis], [[-1, 1, 0, 1, -1, 0, 0, 0, 0]]),
            (-1, 3, 3),
        )
        / m[..., np.newaxis]
    )
    # B = np.array([[-b1, 0, b1], [0, -b2, b2], [b1, b2, -b1 - b2 - B]]) / m
    B = (
        np.reshape(
            np.dot(
                b,
                [
                    [-1, 0, 1, 0, 0, 0, 1, 0, -1],
                    [0, 0, 0, 0, -1, 1, 0, 1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, -1],
                ],
            ),
            (-1, 3, 3),
        )
        / m[..., np.newaxis]
    )

    return K, B, 1 / m


class LeTalkerVocalFolds(VocalFoldsBase):
    """Self-oscillating 3-mass model (Titze and Story, 2002)

    [1] B. H. Story and I. R. Titze, “Voice simulation with a body‐cover model of the vocal folds,” Journal of the Acoustical Society of America, vol. 97, no. 2, pp. 1249–1260, Feb. 1995, doi: 10.1121/1.412234.

    """

    delta: float = 1e-7  # glottal area to be considered closed

    G: float = 0.2  # gain of elongation
    R: float = 3.0  # torque ratio
    H: float = 0.2  # adductory strain factor
    sigam: float = 1050000  # max active muscle stress
    mub: float = 5000
    muc: float = 5000  # shear modulus in body
    rho_tissue: float = 1.04  # tissue density

    act: SampleGenerator = Constant(0.2)
    ata: SampleGenerator = Constant(0.2)
    alc: SampleGenerator = Constant(0.5)
    x0: SampleGenerator = Constant([0.011, 0.01])
    zeta: NDArray = np.array([0.1, 0.6, 0.1])

    Lo: float = 1.6
    To: float = 0.3
    Dmo: float = 0.4  # resting depth of muscle
    Dlo: float = 0.2  # resting depth of deep layer
    Dco: float = 0.2  # resting depth of the cover

    def __init__(
        self,
        *,
        act: float | NDArray | SampleGenerator | None = None,
        ata: float | NDArray | SampleGenerator | None = None,
        alc: float | NDArray | SampleGenerator | None = None,
        x0: NDArray | SampleGenerator | None = None,
        Lo: float | None = None,
        To: float | None = None,
        Dmo: float | None = None,  # resting depth of muscle
        Dlo: float | None = None,  # resting depth of deep layer
        Dco: float | None = None,  # resting depth of the cover
        zeta: NDArray | SampleGenerator | None = None,
        upstream: VocalTract | None = None,
        downstream: VocalTract | None = None,
        aspiration_noise: bool | dict | AspirationNoise | None = None,
    ):
        """LeTalker 3-mass vocal fold model with muscle activity inputs

        :param act: cricothyroid activity, can range from 0 to 1, defaults to 0.25
        :param ata: thyroarytenoid activity, can range from 0 to 1, defaults to 0.25
        :param alf: lca activity, defaults to 0.5
        :param x0: prephonatory displacement (lower, upper)
        :param Lo: vocal fold length at rest
        :param To: vocal fold thickness at rest
        :param zeta: damping ratios [lower, upper, body], defaults to np.array([0.1, 0.6, 0.1])
        :param upstream: connected subglottal tract, defaults to None
        :param downstream: connected supraglottal tract, defaults to None
        :param aspiration_noise: True to use aspiration noise model, defaults to None
        """

        super().__init__(upstream, downstream, aspiration_noise)

        if Lo is not None:
            self.Lo = float(Lo)
        if To is not None:
            self.To = float(To)
        if Dmo is not None:
            self.Dmo = float(Dmo)
        if Dlo is not None:
            self.Dlo = float(Dlo)
        if Dco is not None:
            self.Dco = float(Dco)
        if act is not None:
            self.act = format_parameter(act, 0)
        if ata is not None:
            self.ata = format_parameter(ata, 0)
        if alc is not None:
            self.alc = format_parameter(alc, 0)
        if x0 is not None:
            self.x0 = format_parameter(x0, shape=(2,))
        if zeta is not None:
            self.zeta = np.ascontiguousarray(zeta)
            if self.zeta.shape != (3,):
                raise TypeError(f"{zeta=} must be a 3-element array of damping ratios")

        # saved last results of geometry calculation
        self._last_timing = None
        self._L = None
        self._T = None
        self._zn = None
        self._Db = None
        self._Dc = None
        self._eps = None

    def generate_sim_params(self, n: int, n0: int = 0) -> tuple[NDArray, ...]:
        """Generate parameter arrays"""

        L, T, zn, m, k, kc, b = self.rules_consconv(n, n0)

        K, B, w = eom_mats(m, k, kc, b)

        x0 = np.empty((n, 3))
        x0[:, :2] = self.x0(n, n0)
        x0[:, 2] = 0.0

        subglottal_area, epiglottal_area = self.areas_of_connected_tracts(n, n0)

        # if not (np.isfinite(epiglottal_area) and np.isfinite(subglottal_area)):
        #     raise RuntimeError(
        #         f"Both area_epiglottal and area_subglottal must be set to a finite value."
        #     )

        return (
            L,
            T,
            K,
            B,
            w,
            x0,
            zn,
            epiglottal_area,
            subglottal_area,
            self.delta,
            self.rho,
            self.c,
            1 / self.fs,
        )

    @property
    def _nb_states(self) -> int:
        """number of states"""
        return 6

    def set_geometry(
        self, n: int, n0: int = 0
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """calculate vocal fold geometry

        :param n: simulation length (number of samples)
        :param n0: starting time sample index, defaults to 0
        :return L: vocal fold vibrating length
        :return T: vocal fold vibrating thickness
        :return zn: nodal point
        :return eps: strain
        """
        self._set_geometry(n, n0)

    def _set_geometry(
        self,
        n: int,
        n0: int = 0,
        act: NDArray | None = None,
        ata: NDArray | None = None,
        alc: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """calculate vocal fold geometry

        :param n: simulation length (number of samples)
        :param n0: starting time sample index, defaults to 0
        :param act: CT muscle activation
        :param ata: TA muscle activation
        :param alc: LC muscle activation
        :return L: vocal fold vibrating length
        :return T: vocal fold vibrating thickness
        :return zn: nodal point
        :return eps: strain
        """
        # Rules based on Titze and Story 2002, but with constant convergence ("near
        # rectangular" prephonatory glottal configuration).
        # Author: Brad Story

        if self._last_timing == (n, n0):
            return self._L, self._T, self._zn, self._eps

        if act is None:
            act = self.act(n, n0)
        if ata is None:
            ata = self.ata(n, n0)
        if alc is None:
            alc = self.alc(n, n0)

        eps = self.G * (self.R * act - ata) - self.H * alc  # strain (55)
        L = self.Lo * (1 + eps)  # effective vibrating length
        T = self.To / (1 + 0.8 * eps)  # thickness (59)

        zn = T * (1 + ata) / 3  # Nodal Point Rule (58)

        # ----as written in the paper-----
        Db = (ata * self.Dmo + 0.5 * self.Dlo) / (1 + 0.2 * eps)  # Depth (60)
        Dc = (self.Dco + 0.5 * self.Dlo) / (1 + 0.2 * eps)  # Depth (61)

        self._last_timing = (n, n0)
        self._L = L
        self._T = T
        self._zn = zn
        self._Db = Db
        self._Dc = Dc
        self._eps = eps

        return L, T, zn, Db, Dc, eps

    def rules_consconv(
        self, n: int, n0: int = 0
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """calculate physical vocal fold parameters

        :param n: simulation length (number of samples)
        :param n0: starting time sample index, defaults to 0
        :return L: vocal fold vibrating length
        :return T: vocal fold vibrating thickness
        :return zn: nodal point
        :return m: masses [m1,m2,M]
        :return k: stiffnesses [k1,k2,K]
        :return kc: coupled stiffness
        :return b: damping [b1,b2,B]
        """
        # Rules based on Titze and Story 2002, but with constant convergence ("near
        # rectangular" prephonatory glottal configuration).
        # Author: Brad Story

        act = self.act(n, n0)
        ata = self.ata(n, n0)
        alc = self.alc(n, n0)

        L, T, zn, Db, Dc, eps = self._set_geometry(n, n0, act, ata, alc)

        sigmuc, sigl, sigp = calc_stress(eps)
        sigm = ata * self.sigam * max(0, 1 - 1.07 * (eps - 0.4) ** 2) + sigp

        # ----------As in the paper-------------------------------
        sigb = (0.5 * sigl * self.Dlo + sigm * self.Dmo) / Db  # %body fiber stress
        sigc = (sigmuc * self.Dco + 0.5 * sigl * self.Dlo) / Dc

        muc = self.muc

        # bar-plate model, not used for three-mass model
        rho = self.rho_tissue

        area = L * T

        m = rho * area * Dc  # (30)

        pi2 = pi**2

        k = 2 * muc * area / Dc  # (25)
        a = zn / T
        b = 1 - a

        k_ = k + (pi2 * sigc * (Dc / L)) * T

        k1 = k_ * a  # (44)
        k2 = k_ * b  # (45)
        K = 2 * self.mub * area / Db + pi2 * sigb * Db * T / L  # stiffness of body (49)
        kmat = np.stack([k1, k2, K], -1)

        m1 = m * a  # (47) using (30) for m
        m2 = m * b  # (48)
        M = rho * area * Db  # mass of body (50)
        mmat = np.stack([m1, m2, M], -1)

        kappa = 0.5 * muc * L * Dc / T  # (29)
        kc = (kappa / (1 / 3 - a * b) - k) * a * b  # (46)

        bmat = 2 * self.zeta * (mmat * kmat) ** 0.5

        return L, T, zn, mmat, kmat, kc, bmat

    class Runner(PyRunnerBase):
        n: int
        L: NDArray  # vf vibrating length
        T: NDArray  # total vf thickness
        K: NDArray  # stiffness matrix
        B: NDArray  # damping matrix
        w: NDArray  # reciprocal of masses [btm,top,body]
        x0: NDArray  # prephonatory displacement [btm,top,body]
        zn: NDArray  # dampling ratio [btm,top,body]
        As: NDArray
        Ae: NDArray
        ug: NDArray  # glottal area
        psg: NDArray  # subglottal pressure
        peplx: NDArray  # supraglottal pressure

        y: np.ndarray  # EoM state
        f: np.ndarray  # force vector

        delta: float
        c: float
        rhoc: float
        rhoc2: float
        dt: float

        def __init__(
            self,
            nb_steps: int,
            s_in: NDArray,
            anoise: RunnerBase,
            L: NDArray,
            T: NDArray,
            K: NDArray,  # stiffness matrix
            B: NDArray,  # damping matrix
            w: NDArray,  # reciprocal of masses [m1,m2,mbase]
            x0: NDArray,
            zn: NDArray,
            Ae: NDArray,
            As: NDArray,
            delta: float,
            rho: float,
            c: float,
            dt: float,
        ):
            super().__init__()

            self.n = nb_steps

            # flow to pressure conversion
            self.rhoc = rho * c
            self.rhoc2 = rho * c**2
            self.c = c
            self.dt = dt

            # equations of motions
            self.K = K
            self.B = B
            self.w = w

            # pressure force calculation
            self.Ae = Ae
            self.As = As
            self.zn = zn
            self.L = L
            self.T = T
            self.x0 = x0
            self.delta = delta

            # states
            self.y = s_in
            self.f_ = np.zeros(3)  # internal var to avoid reallocation

            # output
            x0 = self.x0[0 if self.x0.shape[0] > 1 else -1]
            self.ag = np.empty(nb_steps + 1)  # area
            self.ag[0] = 2 * L[0 if self.L.shape[0] > 1 else -1] * x0[:2].min()
            self.ug = np.empty(nb_steps + 1)  # flow
            self.psg = np.empty(nb_steps + 1)  # subglottal pressure
            self.peplx = np.empty(nb_steps + 1)  # supraglottal pressure
            self.f = np.empty((nb_steps + 1, 2))  # exerted forces [bottom,upper]
            self.f[0] = 0
            self.x = np.empty((nb_steps + 1, 3))  # displacements [bottom,upper,body]
            self.x[0] = x0

            self._anoise = anoise

        def step(self, i: int, fsg: float, beplx: float) -> tuple[float, float]:

            rhoc = self.rhoc
            rhoc2 = self.rhoc2
            c = self.c
            delta = self.delta
            dt = self.dt

            Ae = self.Ae[i if self.Ae.shape[0] > 1 else -1]
            As = self.As[i if self.Ae.shape[0] > 1 else -1]
            zn = self.zn[i if self.zn.shape[0] > 1 else -1]
            L = self.L[i if self.L.shape[0] > 1 else -1]
            T = self.T[i if self.T.shape[0] > 1 else -1]
            x0 = self.x0[i if self.x0.shape[0] > 1 else -1]

            K = self.K[i if self.K.shape[0] > 2 else -1]
            B = self.B[i if self.B.shape[0] > 2 else -1]
            w = self.w[i if self.w.shape[0] > 1 else -1]

            psg = self.psg[i]
            pe = self.peplx[i]
            y = self.y
            f = self.f_

            x = self.x[i]

            # compute the forces generated by the glottal flow
            f[0], f[1], a1, a2 = self.calc_forces(
                psg, pe, x[0], x[1], Ae, zn, L, T, delta
            )

            # update the vocal fold displacements
            self.y = self.rk3m(y, i, f, K, B, w, dt)

            ga = max(0, min(a1, a2))

            self.ag[i + 1] = ga

            # compute the glottal flow
            ug = self.calc_flow(fsg, beplx, ga, As, Ae, c, rhoc2)

            # inject turbulent noise
            ug += self._anoise.step(i, ug)

            # compute the forward pressure in epilarynx
            feplx = beplx + ug * rhoc / Ae

            # compute the backward pressure in subglottis
            bsg = fsg - ug * rhoc / As

            self.f[i + 1] = f[:2]
            self.x[i + 1] = x0 + y[:3]
            self.ug[i + 1] = ug
            self.psg[i + 1] = fsg + bsg
            self.peplx[i + 1] = feplx + beplx

            return feplx, bsg

        @staticmethod
        def calc_flow(
            fsg: float,
            beplx: float,
            ga: float,
            As: float,
            Ae: float,
            c: float,
            rhoc2: float,
        ) -> float:
            """calculate glottal flow

            :param fsg: forward subglottal pressure
            :param beplx: backward epiglottal pressure
            :param ga: glottal area
            :param As: subglottal area
            :param Ae: epiglottal area
            :param c: speed of sound
            :param rhoc2: air density times the square of speed of sound
            :return ug: glottal flow
            """
            # Ishizaka-Flanagan pressure recovery coefficient
            ke = 2 * ga / Ae * (1 - ga / Ae)

            # kc = 1 # kinetic pressure loss coefficient

            # transglottal pressure coefficient (kc-ke)
            kt = 1 - ke

            # Astar = (As * Ae) / (As + Ae)
            if np.isfinite(As) and np.isfinite(Ae):
                Astar = (As * Ae) / (As + Ae)
            else:
                Astar = np.inf

            R = ga / Astar
            Qa = 4.0 * kt / rhoc2
            a = -ga * c / kt

            Q = Qa * (fsg - beplx)
            if fsg < beplx:
                # depends on whether ug is positive or negative
                R = -R
                Q = -Q
            ug = a * (R - (R**2 + Q) ** 0.5)
            return ug

        @staticmethod
        def calc_forces(
            psg: float,
            pe: float,
            xi1: float,
            xi2: float,
            Ae: float,
            zn: float,
            L: float,
            T: float,
            delta: float,
        ) -> tuple[float, float, float, float]:
            """calculate exerted forces

            :param psg: subglottal pressure
            :param pe: supraglottal pressure
            :param xi1: lower displacement (absolute)
            :param xi2: upper displacement (absolute)
            :param Ae: supraglottal crosssectional area
            :param zn: nodal point
            :param L: length of vocal fold
            :param T: thickness of vocal fold
            :param delta: minimum allowable area for glottal flow
            :return f1: lower force
            :return f2: upper force
            :return a1: lower glottal area
            :return a2: upper glottal
            """
            # Driving pressure calculations for the three-mass model based on Titze 2002
            # I. R. Titze, “Regulating glottal airflow in phonation: Application of the maximum power transfer theorem
            # to a low dimensional phonation model,” Journal of the Acoustical Society of America, vol. 111, no. 1,
            # pp. 367–376, Jan. 2002, doi: 10.1121/1.1417526.
            # Appendix

            znot = zn / T
            xin = (1 - znot) * xi1 + znot * xi2  # nodal point displacement
            tangent = (xi1 - xi2) / T
            # tangent = (x01 - x02 + 2 * (x1 - x2)) / T
            # xi1 = xin - (-zn) * tangent
            # xi2 = xin - (T - zn) * tangent
            # xi1 = (x01 + x1) + zn/T * (x1 - x2)

            zd = min(T, max(0.0, -0.2 * xi1 / tangent))  # (A28) flow detachment point
            zc = min(T, max(0.0, zn + xin / tangent))  # (A44) contact point
            a1 = max(delta, 2 * L * xi1)  # (A5) lower area
            a2 = max(delta, 2 * L * xi2)  # (A6) upper area
            an = max(delta, 2 * L * xin)  # (A27) nodal point area
            ad = min(a2, 1.2 * a1)  # (A26) area at flow detachment point

            # pressure recovery coefficients
            ke = 2 * ad / Ae * (1 - ad / Ae)  # (A23)
            pkd = (psg - pe) / (1 - ke)  # (A25)
            ph = (psg + pe) / 2  # (A29)

            if a1 > delta and a2 <= delta:
                # upper contact only
                if zc >= zn:  # (A31, A32)
                    f1 = L * zn * psg
                    f2 = L * ((zc - zn) * psg + (T - zc) * ph)
                else:  # (A33, A34)
                    f1 = L * (zc * psg + (zn - zc) * ph)
                    f2 = L * (T - zn) * ph
            elif a1 <= delta and a2 > delta:
                # lower contact only
                if zc < zn:  # (A35, A36)
                    f1 = L * (zc * ph + (zn - zc) * pe)
                    f2 = L * (T - zn) * pe
                if zc >= zn:  # ( A37, A38)
                    f1 = L * zn * ph
                    f2 = L * ((zc - zn) * ph + (T - zc) * pe)
            elif a1 <= delta and a2 <= delta:
                # contact at both bottom and top
                f1 = L * zn * ph
                f2 = L * (T - zn) * ph
            else:  # if a1 > delta and a2 > delta:
                # no contact
                if a1 < a2:  # divergent glottis
                    if zd <= zn:  # detachment point above nodal point (A14 & A15)
                        f1 = L * (zn * psg - (zn - zd + (ad / a1) * zd) * pkd)
                        f2 = L * (T - zn) * (psg - pkd)
                    else:  # detachment point below nodal point (A16 and A17)
                        f1 = L * zn * (psg - (ad**2 / (an * a1)) * pkd)
                        f2 = L * (
                            (T - zn) * psg - ((T - zd) + (ad / an) * (zd - zn)) * pkd
                        )
                else:  # convergent glottis (A18 & A19)
                    f1 = L * zn * (psg - (a2**2 / (an * a1)) * pkd)
                    f2 = L * (T - zn) * (psg - (a2 / an) * pkd)

            return f1, f2, a1, a2

        @staticmethod
        def rk3m(y, time, f, K, B, w, dt):

            #
            # Fourth order Runge-Kutta algorithm for Three-mass model */
            # Brad H. Story    6-1-98            */

            tt = time + 0.5

            eom_3m = LeTalkerVocalFolds.Runner.eom_3m
            k1 = dt * eom_3m(y, time, f, K, B, w)
            k2 = dt * eom_3m(y + k1 / 2.0, tt, f, K, B, w)
            k3 = dt * eom_3m(y + k2 / 2.0, tt, f, K, B, w)
            k4 = dt * eom_3m(y + k3, time + 1, f, K, B, w)

            y = y + k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0

            return y

        @staticmethod
        def eom_3m(
            yo: NDArray, tt: float, f: NDArray, K: NDArray, B: NDArray, w: NDArray
        ) -> NDArray:
            """Equations of motion for (linear) Three-Mass model

            :param yo: displacements and velocities of the 3 masses (lower, upper, body)
            :param tt: time
            :param f1: lower force (driving pressure)
            :param f2: upper force (driving pressure)
            :param m: masses (lower, upper, body)
            :param b: damping constants (lower, upper, body)
            :param k: spring constant (lower, upper, body, shear-coupling)
            :return dyo: the derivative of yo
            """
            #
            # %Author: Brad Story
            # %
            # %x1=yo(1) x2=yo(2) xb=yo(3)
            # %x1dot=yo(4) x2dot=yo(5) xbdot=yo(6)
            # % f1,f2: forces
            # % b1,b2,B: damping constant (mass 1, mass 2, body mass)
            # % k1,k2,kc,K: spring constant (mass1, mass2, shear-coupling, body mass

            # dyo(1) = yo(4)  # velocity of mass 1
            # dyo(2) = yo(5)  # v of mass 2
            # dyo(3) = yo(6)  # v of body mass
            # dyo(4) = (f1 - b1*(yo(4)-yo(6)) - k1*((yo(1))-(yo(3))) - kc*((yo(1))-(yo(2))) )/m1
            # dyo(5) = (f2 - b2*(yo(5)-yo(6)) - k2*((yo(2))-(yo(3))) - kc*((yo(2))-(yo(1))) )/m2
            # dyo(6) = (k1*((yo(1))-(yo(3))) + k2*((yo(2))-(yo(3))) + b1*(yo(4)-yo(6))+b2*(yo(5)-yo(6)) - K*(yo(3))-B*yo(6))/M

            dyo = np.empty(6)
            dyo[:3] = yo[3:]
            dyo[3:] = K @ yo[:3] + B @ yo[3:] + f * w

            return dyo

    @property
    def known_length(self) -> bool:
        """True if vocal fold length is available pre-simulation"""
        return True

    def length(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:
        """glottal length

        Parameters
        ----------
        nb_samples, optional
            number of samples, required if dynamic, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            glottal length in cm. 1D array if fo or L0 is dynamic otherwise a scalar value.
        """

        return self._retrieve_geometry_samples("_L", nb_samples, n0)

    def thickness(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:
        """glottal thickness

        Parameters
        ----------
        nb_samples, optional
            number of samples, required if dynamic, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            glottal thickness in cm. 1D array if fo or L0 is dynamic otherwise a scalar value.
        """

        return self._retrieve_geometry_samples("_T", nb_samples, n0)

    def nodal_point(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:

        return self._retrieve_geometry_samples("_zn", nb_samples, n0)

    def depth_body(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:

        return self._retrieve_geometry_samples("_Db", nb_samples, n0)

    def depth_cover(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:

        return self._retrieve_geometry_samples("_Dc", nb_samples, n0)

    def _retrieve_geometry_samples(self, name, nb_samples, n0):

        val = getattr(self, name)

        exists = val is not None
        if exists:
            if val.ndim > 1:  # dynamic
                n_, n0_ = self._last_timing
                n1_ = n0_ + n_
                n1 = n0 + nb_samples
                if n0 >= n0_ and n0 < n1_ and n1 > n0_ and n1 <= n1_:
                    val = val[n0 - n0_ : n1 - n0_]
                else:
                    exists = False

        if not exists:
            self.rules_consconv(nb_samples, n0)
            val = getattr(self, name)

        if nb_samples == 1 and val.ndim:
            val = val[0]
        return val

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n", "y", "ag", "ug", "psg", "peplx", "x", "f"]

    @dataclass
    class Results(Element.Results):
        y: NDArray
        ag: NDArray
        ug: NDArray
        psg: NDArray
        peplx: NDArray
        x: NDArray
        f: NDArray
        aspiration_noise: Element.Results | None

        @cached_property
        def final_states(self) -> NDArray:
            return self.y

        @cached_property
        def length(self) -> NDArray:
            element = cast(LeTalkerVocalFolds, self.element)
            return element.length(self.nb_samples, self.n0)

        @cached_property
        def thickness(self) -> NDArray:
            element = cast(LeTalkerVocalFolds, self.element)
            thickness = element.thickness(self.nb_samples, self.n0)
            return thickness

        @cached_property
        def x0(self) -> NDArray:
            element = cast(LeTalkerVocalFolds, self.element)
            return element.x0(self.nb_samples, self.n0)

        @cached_property
        def glottal_area(self) -> NDArray:
            return self.ag[:-1]

        @cached_property
        def glottal_flow(self) -> NDArray:
            return self.ug[:-1]

        @cached_property
        def subglottal_pressure(self) -> NDArray:
            return self.psg[:-1]

        @cached_property
        def intraglottal_pressure(self) -> NDArray:
            return self.f[:-1, :2].mean(-1) * 2 * self.thickness * self.length

        @cached_property
        def displacements(self) -> NDArray:
            return self.x[:-1]

    # override result class
    _ResultsClass = Results

    def draw3d(
        self,
        n: int,
        x: ArrayLike,
        *,
        axes: "Axes3D" | None = None,
        update: bool | None = None,
        units: Literal["cm", "mm"] = "cm",
        xb0: float | None = None,
        right_only: bool = False,
        obj_kws: (
            dict[
                Literal[
                    "rlower",
                    "llower",
                    "rupper",
                    "lupper",
                    "rbody",
                    "lbody",
                    "lower",
                    "upper",
                    "cover",
                    "body",
                ],
                dict,
            ]
            | None
        ) = None,
    ):
        """visualize vocal fold movement"""

        # get data

        L = self.length(1, n)  # length of masses
        T = self.thickness(1, n)  # base mass thickness
        T1 = self.nodal_point(1, n)  # bottom mass thickness
        Db = self.depth_cover(1, n)
        Dc = self.depth_body(1, n)

        if xb0 is None:
            xb0 = self.x0(1, n) + Dc
        else:
            xb0 = np.full(2, xb0)

        if units == "mm":
            L *= 10
            T *= 10
            T1 *= 10
            Db *= 10
            Dc *= 10
            xb0 *= 10

        xpos = np.empty(4)
        xpos[:2] = np.maximum(x[:2], 0)
        xpos[2:] = x[2] + xb0

        gid0 = "letalker.ltvf"

        if update:
            from matplotlib.collections import PolyCollection

            for h in axes.findobj(
                match=lambda artist: (artist.get_gid() or "").startswith(gid0),
                include_self=False,
            ):
                name = h.get_gid().rsplit(".", 1)[1]
                if name[1:] == "lower":
                    x1 = xpos[0]
                    x2 = xpos[0] + Dc
                    z1 = 0
                    z2 = T1
                elif name[1:] == "upper":
                    x1 = xpos[1]
                    x2 = xpos[1] + Dc
                    z1 = T1
                    z2 = T
                else:  # 'body'
                    x1 = xpos[2]
                    x2 = xpos[2] + Db
                    z1 = 0
                    z2 = T

                if name[0] == "l":
                    x1 = -x1
                    x2 = -x2

                data = h._vec
                xs = data[0]
                data[0] = np.where(xs == xs[0], x1, x2)
                zs = data[2]
                data[2] = np.where(zs == zs[2], z1, z2)
                PolyCollection.set_verts(h, [], False)
        else:
            # gather custom properties
            obj_kws = obj_kws or {}
            for s in "lr":
                for i, c in enumerate(("lower", "upper")):
                    fname = s + c
                    obj_kws[fname] = {
                        "color": f"C{i}",
                        **obj_kws.get("cover", {}),
                        **obj_kws.get(c, {}),
                        **obj_kws.get(fname, {}),
                    }

                fname = f"{s}body"
                obj_kws[fname] = {
                    "color": "C2",
                    **obj_kws.get("body", {}),
                    **obj_kws.get(fname, {}),
                }

            T2 = T - T1  # top mass thickness
            for name, x0, z0, dx, dz, c in zip(
                ("lower", "upper", "body"),
                xpos,
                [0, T1, 0, T1],
                [Dc, Dc, Db, Db],
                [T1, T2, T1, T2],
                ["C0", "C1", "C2", "C2"],
            ):
                axes.bar3d(x0, 0, z0, dx, L, dz, color=c, gid=f"{gid0}.r{name}")
                if not right_only:
                    axes.bar3d(-x0, 0, z0, -dx, L, dz, color=c, gid=f"{gid0}.l{name}")
