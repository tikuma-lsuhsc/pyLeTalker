"""internal utility functions"""

from __future__ import annotations

from typing_extensions import Literal, Callable, get_args
from numpy.typing import NDArray

import numpy as np

from ..constants import (
    smb_xlar_default,
    smb_Alar_default,
    smb_nlarp_default,
    smb_xp_default,
    smb_xa_coefs,
)


pi = np.pi


def smb_vt_area_fun(
    Ap: float,
    npc: float,
    xc: float,
    Ac: float,
    nca: float,
    Aa: float,
    nain: float,
    xin: float,
    Ain: float,
    xlip: float,
    Alip: float,
    xlar: float | None = None,
    Alar: float | None = None,
    nlarp: float | None = None,
    xp: float | None = None,
    xa: float | None = None,
) -> Callable[[NDArray], NDArray]:
    """Stone-Marxen-Birkholz 1D vocal tract model area function

    Args:
        Ap: Cross-sectional area in cm2 at the posterior control point
        npc: Warping exponent of segment between the posterior control point and
              constriction
        xc: Position of the constriction in cm
        Ac: Cross-sectional area in cm2 at the constriction
        nca: Warping exponent of segment between the constriction and the anterior
             control point.
        Aa: Cross-sectional area in cm2 at the anterior control point
        nain: Warping exponent of segment between the anterior control point and
              the incisors
        xin: Position of the incisors in cm
        Ain: Cross-sectional area in cm2 at the incisors
        xlip: Vocal tract length in cm
        Alip: Cross-sectional area in cm2 at the lips
        xlar: Length of the larynx tube in cm. Defaults to None.
        Alar: Cross-sectional area in cm2 of the larynx tube. Defaults to None.
        nlarp: Warping exponent of segment between the larynx and the posterior
               control point. Defaults to None.
        xp: Position of the posterior control point in cm. Defaults to None.
        xa: Position of the anterior control point in cm. Defaults to None.

    Returns:
        function to generate cross-sectional area profile as a function of given
        a vector of position along the center of vocal tract in cm

    Reference
    ---------

    [1] S. Stone, M. Marxen, and P. Birkholz, “Construction and evaluation of a
        parametric one-dimensional vocal tract model,” IEEE/ACM Trans. Audio, Speech,
        Language Process., vol. 26, no. 8, pp. 1381–1392, Aug. 2018,
        doi: 10.1109/TASLP.2018.2825601.

    """

    # set defaults
    if xlar is None:
        xlar = smb_xlar_default
    if Alar is None:
        Alar = smb_Alar_default
    if nlarp is None:
        nlarp = smb_nlarp_default
    if xp is None:
        xp = smb_xp_default
    if xa is None:
        xa = np.dot(smb_xa_coefs, (1.0, xc, nca, Aa, xin, xc * nca))

    n = [nlarp, npc, nca, nain]

    Acp = np.array([Alar, Ap, Ac, Aa, Ain])
    a = (Acp[1:] + Acp[:-1]) / 2
    b = np.diff(Acp) / 2

    xcp = np.array([xlar, xp, xc, xa, xin])
    dxcp = np.diff(xcp)

    def func(x: NDArray) -> NDArray:

        i = np.searchsorted(xcp, x)

        A = np.empty_like(x)

        j0 = (np.argmax(x >= 0)) if x[0] < 0 else 0
        A[:j0] = np.nan

        j1 = np.argmax(i > 0)
        A[j0:j1] = Alar

        for k, (ak, bk, nk, xck, yck) in enumerate(zip(a, b, n, xcp[1:], dxcp)):
            inext = k + 1
            if i[j1] > inext:
                # not sampled in this segment
                continue
            j0 = j1
            j1 = np.argmax(i[j0:] > inext) + j0
            ak = ak + bk * np.cos(pi * ((xck - x[j0:j1]) / yck) ** nk)

            # zero threshold negative area
            ak[ak < 0] = 0

            A[j0:j1] = ak

        j0 = j1
        j1 = (np.argmax(x[j0] > xlip) + j0) if x[-1] > xlip else x.size
        A[j0:j1] = Alip
        A[j1:] = np.nan

        return A

    return func


TongueHornModel = Literal["fant", "stevens", "atal", "ishizaka", "lin", "lin2"]


def fant_horn_vt_area(
    dx: float,
    Amin: float,
    Xmin: float,
    lhorn: float | tuple[float, float] | None = None,
    Amax: float | tuple[float, float] | None = None,
    l: float | None = None,
    A1: float | None = None,
    l1: float | None = None,
    Alar: float | None = None,
    llar: float | None = None,
    model: TongueHornModel | None = None,
    hold_max: bool | None = None,
):
    """Simplified vocal tract models with horn-shaped tongue constriction

    :param Amin: main tongue body constriction
    :param Xmin: its axial location
    :param lhorn: tongue constriction width (one-sided), defaults to None
    :param Amax: maximum tube area, defaults to None
    :param l: total length, defaults to None
    :param A1: area between lips, defaults to None
    :param l1: lip thickness, defaults to None
    :param Alar: laryngeal pipe area, defaults to None
    :param llar: laryngeal pipe length, defaults to None

    Reference
    ---------

    [1] G. Fant, The Acoustic Theory of Speech Production. The Hague, the
        Netherlands: Mouton, 1960.
    [2] K. N. Stevens and A. S. House, “Development of a quantitative description
        of vowel articulation,” J. Acoust. Soc. Am., vol. 27, no. 3, pp. 484–493,
        May 1955, doi: 10.1121/1.1907943.
    [3] B. S. Atal, J. J. Chang, M. V. Mathews, and J. W. Tukey, “Inversion
        of articulatory-to-acoustic transformation in the vocal tract by a
        computer-sorting technique,” J. Acoust. Soc. Am., vol. 63, no. 5,
        pp. 1535–1555, May 1978, doi: 10.1121/1.381848.
    [4] Q. Lin, “Speech production theory and articulatory speech synthesis,”
        PhD Thesis, Royal Inst. of Technology (KTH), Stockholm, Sweden, 1990.

    """

    if model is None:
        model = "fant"
    else:
        assert model in get_args(TongueHornModel)

    hold_max = (
        ([True, False] if model == "stevens" else [model in ["fant", "lin", "lin2"]])
        if hold_max is None
        else np.atleast_1d(hold_max)
    )

    if l is None:
        l = 17.5
    if A1 is None:
        A1 = 4.0
    if l1 is None:
        l1 = 1.0
    if Alar is None:
        Alar = 0.5
    if llar is None:
        llar = 2.5

    if lhorn is not None:
        lhorn = np.atleast_1d(lhorn)
    if Amax is not None:
        Amax = np.atleast_1d(Amax)

    if model == "fant":
        if lhorn is None:
            lhorn = np.array([4.75])
        if Amax is None:
            Amax = np.array([8.0])
        h = lhorn / np.acosh((Amax / Amin) ** 0.5)
        pb = [h[0]]
        pf = [h[-1]]
        fnc = lambda x, h: Amin * np.cosh((x - Xmin) / h) ** 2
    elif model == "stevens":
        if lhorn is None:
            lhorn = np.array([4.0])
        if Amax is None:
            Amax = np.array([np.pi * 1.6**2])

        rmin = Amin**0.5
        rmax = Amax**0.5
        c = (lhorn**-2) * (rmax - rmin)
        pb = [c[0]]
        pf = [c[-1]]
        fnc = lambda x, c: (c * (x - Xmin) ** 2 + rmin) ** 2
    elif model == "atal":
        if lhorn is None:
            lhorn = np.array([np.pi / 0.30])
        if Amax is None:
            Amax = np.array([9 + Amin])

        a0 = (Amax - Amin) / 2
        a1 = a0 - Amin
        ca = pi / lhorn
        pb = [a0[0], a1[0], ca[0]]
        pf = [a0[-1], a1[-1], ca[-1]]

        fnc = lambda x, a0, a1, ca: a0 - a1 * np.cos(ca * (x - Xmin))
    elif model == "ishizaka":
        if lhorn is None:
            lhorn = np.array([8, 7]) * (l / 17)
        if Amax is None:
            Amax = np.array([8.0])
        a0 = (Amax + Amin) / 2
        a1 = (Amax - Amin) / 2
        pb = [a0[0], a1[0], 1, 0, lhorn[0]]
        pf = [a0[-1], a1[-1], 0.4, 0.6, lhorn[-1]]
        fnc = lambda x, a0, a1, c1, c2, l: a0 - a1 * np.cos(
            np.pi * (c1 + c2 * (x - Xmin) / l) * (x - Xmin) / l
        )
    else:
        if lhorn is None:
            lhorn = np.array([4.0])
        if Amax is None:
            Amax = np.array([8.0])
        a = (Amax - Amin) / 2  # missing H
        c = np.pi / lhorn / 1  # missing R
        if model == "lin":
            fnc = lambda x, a, c: Amin + a * (1 - np.cos(c * (x - Xmin)))
        elif model == "lin2":
            a = a / 2
            fnc = lambda x, a, c: Amin + a * (1 - np.cos(c * (x - Xmin))) ** 2
        else:
            raise ValueError(f"unknown tongue horn {model=}")

        pb = [a[0], c[0]]
        pf = [a[-1], c[-1]]

    k = round(l / dx / 2) * 2  # number of segments (must be even)
    x = (
        np.arange(k) * dx + dx / 2
    )  # evaluate at the center of the spatial sample duration

    xcp = np.array([llar, Xmin - lhorn[0], Xmin, Xmin + lhorn[-1], l - l1])
    if xcp[1] < xcp[0]:
        xcp[1] = xcp[0]
    if xcp[-2] > xcp[-1]:
        xcp[-2] = xcp[-1]
    i = np.searchsorted(x, xcp)

    A = np.zeros_like(x)

    A[: i[0]] = Alar
    A[i[0] : i[1]] = Amax[0] if hold_max[0] else fnc(x[i[0] : i[1]], *pb)
    A[i[1] : i[2]] = fnc(x[i[1] : i[2]], *pb)
    A[i[2] : i[3]] = fnc(x[i[2] : i[3]], *pf)
    A[i[3] : i[4]] = Amax[-1] if hold_max[-1] else fnc(x[i[3] : i[4]], *pf)
    A[i[4] :] = A1

    return A
