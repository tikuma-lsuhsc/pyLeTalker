"""internal utility functions"""

from typing import Callable
from numpy.typing import NDArray

import numpy as np

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
        xlar = 2.0
    if Alar is None:
        Alar = 1.5125
    if nlarp is None:
        nlarp = 1.0
    if xp is None:
        xp = 3.0948
    if xa is None:
        xa = (
            -2.347
            - 0.061 * xc
            - 2.052 * nca
            - 0.159 * Aa
            + 1.161 * xin
            + 0.143 * xc * nca
        )

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
