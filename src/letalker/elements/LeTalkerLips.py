from __future__ import annotations
from typing import Iterator
from numpy.typing import NDArray, ArrayLike

from dataclasses import dataclass
from math import pi
import numpy as np
import scipy.signal as sps

from ..constants import c as c_default
from ..__util import format_parameter
from ..function_generators.abc import SampleGenerator
from .abc import Element, Lips, VocalTract

from ..core import has_numba
from ..function_generators import Constant

if has_numba:
    from numba.types import intc, float64, CPointer
    import numba as nb


def ishizaka_flanagan_lip_model(
    area: float | NDArray, c: float = c_default
) -> tuple[float, float]:
    """Ishizaka-Flanagan Lips RL Model

    Parameters
    ----------
    area
        lip opening area in cm²
    c
        speed of sound in cm/s

    Returns
    -------
        R - acoustic resistance
        L - acoustic inductance

    References:

        [1] K. Ishizaka and J. L. Flanagan, “Synthesis of voiced sounds from a two-mass model of
            the vocal cords,” The Bell System Technical Journal, vol. 51, no. 6, pp. 1233–1268,
            1972, doi: 10.1002/j.1538-7305.1972.tb02651.x.

    """
    R = 128.0 / (9.0 * pi**2)
    L = 8.0 * (area / pi) ** 0.5 / (3.0 * pi * c)
    return R, L


class LeTalkerLips(Lips):
    """Ishizaka-Flanagan """
    @dataclass
    class Results(Element.Results):
        final_states: NDArray
        pout: NDArray  # radiated acoustic pressure
        uout: NDArray  # radiated flow

    # override result class
    _ResultsClass = Results

    RunnerSpec = (
        [
            ("n", nb.int64),
            ("A", nb.float64[:, ::1, ::1]),
            ("b", nb.float64[:, :]),
            ("c", nb.float64[:]),
            ("s", nb.float64[::1]),
            ("pout", nb.float64[:]),
            ("uout", nb.float64[:]),
        ]
        if has_numba
        else []
    )

    class Runner:
        n: int
        A: np.ndarray
        b: np.ndarray
        c: np.ndarray
        s: np.ndarray
        pout: np.ndarray
        uout: np.ndarray

        def __init__(
            self, nb_steps: int, s: NDArray, A: NDArray, b: NDArray, c: NDArray
        ):
            self.n = nb_steps
            self.A = np.ascontiguousarray(A)
            self.b = b
            self.c = c
            self.s = np.ascontiguousarray(s)
            self.pout = np.empty(nb_steps)
            self.uout = np.empty(nb_steps)

        def step(self, i: int, flip: float) -> float:

            Ai = self.A[i if i < self.A.shape[0] else -1]
            bi = self.b[i if i < self.b.shape[0] else -1]
            c = self.c[i if i < self.c.shape[0] else -1]
            s = self.s

            # states: Po_prev, b_prev, f_prev
            s[:2] = s @ Ai + bi * flip
            s[-1] = flip

            self.pout[i] = s[0]
            self.uout[i] = c * (flip - s[1])

            return s[1]

    _area: SampleGenerator = Constant(1.0)

    def __init__(self, upstream: VocalTract | float | None = None):

        super().__init__(upstream)

    @property
    def pout_index(self) -> int:
        """column index of the resultant NDArray with radiated pressure output"""
        return 0

    @property
    def runner_info(self) -> tuple[type, list[tuple[str, type]]]:
        return LeTalkerLips.Runner, LeTalkerLips.RunnerSpec

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n", "s", "pout", "uout"]

    def generate_sim_params(
        self, n: int, n0: int = 0
    ) -> tuple[NDArray, NDArray, NDArray | float]:
        """Returns simulation parameters

        Parameters
        ----------
        n
            number of simulation samples
        n0, optional
            starting simulation time in sample index, by default 0

        Returns
        -------
            A   3x2 or nx3x2 array
            b   length-2 or nx2 array
            c   float or length-n array
        """

        if self._area is None:
            raise RuntimeError("Lips opening area has not been set!")

        area = (
            self.upstream.output_area(n, n0)
            if isinstance(self.upstream, VocalTract)
            else np.array(self.upstream or np.inf)
        )
        n1 = area.shape[0] if area.ndim else 1

        R, L = ishizaka_flanagan_lip_model(area, self.c)

        # bilinear transformation
        Ldt = 2.0 * self.fs * L
        a2 = -R - Ldt + R * Ldt
        a1 = -R + Ldt - R * Ldt
        b2 = R + Ldt + R * Ldt
        b1 = -R + Ldt + R * Ldt

        c1 = b1 / b2
        c2 = a1 / b2
        c3 = a2 / b2

        # states: Poprev, bprev, fprev
        A = np.zeros((n1, 3, 2))
        b = np.empty((n1, 2))
        A[:, 0, 0] = A[:, 1, 1] = c1
        A[:, 2, 0] = c2 - c1
        A[:, 2, 1] = c2
        b[:, 0] = 1 + c3
        b[:, 1] = c3

        # b(jmoth) = (f(jmoth)*c3 + fprev*c2 + bprev*c1);

        c = np.atleast_1d(area / (self.rho * self.c))

        return A, b, c

    @property
    def nb_states(self) -> int:
        """number of states"""
        return 3

    def iter_tf(
        self, n: int | None = None, n0: int = 0
    ) -> Iterator[tuple[NDArray, NDArray]]:
        """Iterate transfer function over time

        Parameters
        ----------
        n, optional
            number of samples, by default None
        n0, optional
            starting sample index, by default 0

        Yields
        ------
            b
                numerator coefficients
            a
                denominator coefficients
        """
        if n is None:
            if self.is_dynamic:
                raise ValueError("Must specify n (and n0) for a dynamic lips")
            else:
                n = 1

        A_, B_, C_ = self.generate_sim_params(n, n0)

        A = np.zeros((3, 3))
        B = np.zeros((3, 1))
        B[-1, 0] = 1.0
        C = np.eye(2, 3)

        for a, b, c in zip(A_, B_, C_):
            A[:-1, :] = a.T
            B[:-1, 0] = b

            yield sps.ss2tf(A, B, c @ A, c @ B)

    def iter_freqz(
        self,
        n: int | None = None,
        n0: int = 0,
        forN: ArrayLike | int = 512,
        whole: bool = False,
        include_nyquist: bool = False,
    ) -> Iterator[tuple[NDArray, NDArray]]:
        """Iteratively compute the frequency response of the model over time

        Parameters
        ----------
        n, optional
            number of samples, by default None
        n0, optional
            starting sample index, by default 0
        forN, optional
            If a single integer, then compute at that many frequencies (default
            is N=512). This is a convenient alternative to:

            `np.linspace(0, fs if whole else fs/2, N, endpoint=include_nyquist)`

            If an array_like, compute the response at the frequencies given in Hz.
        whole, optional
            Normally, frequencies are computed from 0 to the Nyquist frequency,
            fs/2 (upper-half of unit-circle). If whole is True, compute frequencies
            from 0 to fs. Ignored if forN is array_like.
        include_nyquist, optional
            If whole is False and forN is an integer, setting include_nyquist to
            True will include the last frequency (Nyquist frequency) and is
            otherwise ignored (default).

        Returns
        -------
        f
            The frequencies at which h was computed, in the same units as fs.
        h
            The frequency responses, as complex numbers. If the lips is not
            dynamic or only requested at 1 time index, the shape of h is (2,N).
            Otherwise, h.shape is (2,N,n), or the time axis is appended as the
            third dimension.

        """

        nmat = (n or 1) if self.is_dynamic else 1
        bmat = np.empty((4, nmat, 2, 1) if nmat > 1 else (4, 1, 2, 1))
        amat = np.empty((4, nmat, 1, 1) if nmat > 1 else (4, 1, 1, 1))

        for i, (b, a) in enumerate(self.iter_tf(n, n0)):
            yield sps.freqz(
                b.T[..., np.newaxis],
                a,
                worN=forN,
                whole=whole,
                fs=self.fs,
                include_nyquist=include_nyquist,
            )
