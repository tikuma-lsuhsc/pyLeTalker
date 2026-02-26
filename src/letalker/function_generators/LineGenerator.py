from __future__ import annotations

from typing import Literal, overload, TYPE_CHECKING, Sequence
from numpy.typing import NDArray

import logging

logger = logging.getLogger("letalker")

import numpy as np
from numpy.polynomial import polynomial as npp


from .abc import FunctionGenerator
from .mixins import TaperMixin
from . import _functions as _f

if TYPE_CHECKING:
    from .StepGenerator import StepTypeLiteral


class BasePolyGenerator(FunctionGenerator):
    @overload
    def __init__(
        self,
        c: Sequence[NDArray],
        tb: NDArray | None,
        **kwargs,
    ):
        """(Contiguous or Piecewise) Polynomial Generator

        Parameters
        ----------
        c
            sequence of n sets of polynomial coefficients if piecewise, 1 set of coefficients if contiguous
        tb
            boundary time between successive polynomials, None to use contiguous single polynomial.
        """

    def __init__(
        self,
        c: Sequence[NDArray],
        tb: NDArray | None,
        **kwargs,
    ):

        nsegs = len(c)
        if not nsegs:
            raise ValueError("c cannot be empty. No polynomial coefficients given.")
        is_contiguous = nsegs == 1
        if (nsegs > 1 and (tb is None or len(tb) != nsegs - 1)) or (
            is_contiguous and (tb is not None and len(tb) != 0)
        ):
            raise TypeError(f"mismatched dimensions between {c=} and {tb=}")

        c = [np.asarray(ci) for ci in c]

        shape = c[0].shape[1:]
        if not all(ci.shape[1:] == shape for ci in c[1:]):
            raise ValueError("not all items of c have the same shape.")

        super().__init__(**kwargs)

        self._func = _f.poly if is_contiguous else _f.pwpoly
        self._args = (c[0],) if is_contiguous else (c, tb)
        self._shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        """shape of each sample (empty if scalar)"""
        return self._shape

    @property
    def ndim(self) -> int:
        """number of dimensions of each sample (0 if scalar)"""
        return len(self._shape)

    def __call__(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        **kwargs,
    ) -> NDArray:
        """generate waveform

        Parameters
        ----------
        n
            number of samples to generate

        n0
            initial sample index, defaults to 0

        """

        ts_args = (n, n0, sample_edges, upsample_factor)
        t = self.ts(*ts_args).flatten()
        return self._func(*self._args, t)

    def derivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        ts_args = (n, n0, sample_edges, upsample_factor)
        t = self.ts(*ts_args).flatten()
        return self._func.derivative(*self._args, t, nu)

    def antiderivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        ts_args = (n, n0, sample_edges, upsample_factor)
        t = self.ts(*ts_args).flatten()
        return self._func.antiderivative(*self._args, t, nu)

    # @property
    # def order_of_vanishing(self) -> int:
    #     if self._taper:
    #         raise NotImplementedError(
    #             "derivative with transitions is not currently implemented"
    #         )

    #     return None if self._taper else 2

    # @property
    # def has_closed_form(self) -> bool:
    #     return True


class LineGenerator(TaperMixin, BasePolyGenerator):
    """Line generator"""

    @overload
    def __init__(
        self,
        tp: tuple[float, float],
        xp: tuple[float | NDArray, float | NDArray],
        *,
        outside_values: Literal["hold", "extend"] | None = None,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.002,
        transition_initial_on: bool = False,
    ):
        """Linear 1D Interpolator

        Parameters
        ----------
        tp
            The initial and final time stamps.
        xp
            The initial and final levels
        outside_values, optional
            'hold' (default) to maintain the end-point levels, or
            'extend' to extend the linear change indefinitely.
        transition_time, optional
            transition time or times if multiple steps, by default None (no transitions)
        transition_type, optional
            transition type or types if multiple steps with different types, by default 'raised_cos'
        transition_time_constant, optional
            transition time constant/duration or constants/durations if multiple steps with different settings, by default 0.002
        transition_initial_on, optional
            initial signal level, by default 0.0

        Note: If `transition_type` or `transition_time_constant` is shorter than `transition_time`, their values are cycled.
        """

    def __init__(
        self,
        tp: tuple[float, float],
        xp: tuple[float | NDArray, float | NDArray] | NDArray,
        *,
        outside_values: Literal["hold", "extend"] | None = None,
        **kwargs,
    ):

        xp = np.asarray(xp)

        if tp[0] == tp[1] and all(np.isfinite(t) for t in tp):
            raise ValueError(f"The time points ({tp=}) must be finite and unique.")

        if tp[0] > tp[1]:
            xp[1], xp[0] = xp
            tp[1], tp[0] = tp

        m = (xp[1] - xp[0]) / (tp[1] - tp[0])
        c = np.array([xp[0] - m * tp[0], m])

        extend = outside_values == "extend"

        super().__init__(
            c=[c] if extend else [[xp[0]], c, [xp[1]]],
            tb=None if extend else tp,
            TaperBaseClass=BasePolyGenerator,
            **kwargs,
        )

    @property
    def extrapolate(self) -> str:
        return "extend" if len(self._args[0]) == 1 else "hold"

    def __repr__(self):
        mode = self.extrapolate
        c = self._args[0][0 if mode == "extend" else 1]
        return f"LineGenerator[{c[0]:0.2f}+{c[1]:0.2f}t:{mode}]"

    def __call__(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        **kwargs,
    ) -> NDArray:
        """generate waveform

        Parameters
        ----------
        n
            number of samples to generate

        n0
            initial sample index, defaults to 0

        """

        return self._taper_call(n, n0, sample_edges, upsample_factor, **kwargs)

    def derivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        return self._taper_derivative(
            n, n0, sample_edges, upsample_factor, nu=nu, **kwargs
        )

    def antiderivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        return self._taper_antiderivative(
            n, n0, sample_edges, upsample_factor, nu=nu, **kwargs
        )

    # @property
    # def order_of_vanishing(self) -> int:
    #     return None if self._taper else 2

    # @property
    # def has_closed_form(self) -> bool:
    #     return False if self._taper else True
