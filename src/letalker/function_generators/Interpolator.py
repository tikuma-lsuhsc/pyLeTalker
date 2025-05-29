from __future__ import annotations

from typing_extensions import Literal, Unpack, overload
from collections.abc import Sequence
from numpy.typing import NDArray, ArrayLike

import scipy.interpolate as spi
import scipy.signal as sps
import numpy as np
from math import ceil

from .abc import FunctionGenerator, AnalyticFunctionGenerator
from .mixins import TaperMixin
from .StepGenerator import StepTypeLiteral


class BaseInterpolator(FunctionGenerator):
    def __init__(
        self,
        tp_or_fs: ArrayLike | float,
        xp: ArrayLike,
        *,
        degree: Literal["linear", "quadratic", "cubic"] | int | type = 3,
        interp_kws: dict | None = None,
        **kwargs: Unpack[dict],
    ):
        """`1-D B-Spline Interpolator

        Parameters
        ----------
        tp_or_fs
            The time of the data points, must be 1D and strictly increasing. If a single
            value is given, it is taken as the sampling rate of xp, which is sampled
            at the interval of `1/tp_or_fs`, starting at time 0.
        xp
            The value of the data points. It can have arbitrary number of dimensions,
            but the length along `axis` (see below) must match the length of `t`.
            Values must be finite.
        degree
            B-spline degree specified either by the literal strings or a positive
            integer. By default, 'cubic' or 3 is used.

            Alternately, an interpolator class which conforms to `scipy.interpolate.PPoly`
            class (e.g., `scipy.interpolate.CubicSpline`) may be specified to replace
            the default B-Spline interpolator.
        interp_kws
            Optional keyword arguments to construct B-splines using
            `scipy.interpolate.make_interp_spline()` or specified interpolation class as
            `degree`. For B-spline interpolation, available keywords are `bc_type`,
            `check_finite`, and `axis`. See `scipy`'s documentation for their usages.
        """

        if not np.asarray(tp_or_fs).ndim:
            tp_or_fs = np.arange(len(xp)) / tp_or_fs

        super().__init__(**kwargs)

        self._int = (
            spi.make_interp_spline(
                tp_or_fs,
                xp,
                {"linear": 1, "quadratic": 2, "cubic": 3}.get(degree, degree),
                axis=0,
                **interp_kws or {},
            )
            if isinstance(degree, (str, int))
            else degree(tp_or_fs, xp, axis=0, **interp_kws or {})
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """shape of each sample (empty if scalar)"""
        return self._int.c.shape[1:]

    @property
    def ndim(self) -> int:
        """number of dimensions of each sample (0 if scalar)"""
        return self._int.c.ndim - 1

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

        return self._int(self.ts(*ts_args).ravel())

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
        return self._int.derivative(nu)(self.ts(*ts_args).ravel())

    # @property
    # def order_of_vanishing(self) -> int | None:
    #     return self._int.c.shape[0] + 1

    def antiderivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        if self._taper:
            raise NotImplementedError(
                "derivative with transitions is not currently implemented"
            )

        ts_args = (n, n0, sample_edges, upsample_factor)
        return self._int.antiderivative(nu)(self.ts(*ts_args).ravel())


class Interpolator(TaperMixin, BaseInterpolator):
    """Interpolator"""

    @classmethod
    def from_dict(
        points: dict[float, float | ArrayLike],
        **kwargs: Unpack[dict],
    ):
        """construct 1D B-Spline Interpolator from x-y data in dict

        Parameters
        ----------
        points
            Each data point as a dict item, of which key is an independent variable value (x)
            and value is the dependent variable(s) (y). The value may be 1D (`float`) or multi-
            dimensional (`ArrayLike`).
        IntCls
            cubic
            Interpolator class which conforms to `scipy.interpolate.PPoly` class
            (e.g., `scipy.interpolate.CubicSpline`) or `'linear'` (default) to
            use the linear B-spline interpolation. This class will be instantiated
            with `tp` and `xp` as its first two arguments.
        interp_kws
            Keyword argument to construct `IntCls` object
        """
        tp = sorted(points)
        xp = [points[t] for t in tp]
        return Interpolator(tp, xp, **kwargs)

    @overload
    def __init__(
        self,
        tp_or_fs: ArrayLike,
        xp: ArrayLike,
        *,
        degree: Literal["linear", "quadratic", "cubic"] | int | type = 3,
        interp_kws: dict | None = None,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.002,
        transition_initial_on: bool = False,
        **kwargs: Unpack[dict],
    ):
        """`1-D B-Spline Interpolator

        Parameters
        ----------
        tp_or_fs
            The time of the data points, must be 1D and strictly increasing. If a single
            value is given, it is taken as the sampling rate of xp, which is sampled
            at the interval of `1/tp_or_fs`, starting at time 0.
        xp
            The value of the data points. It can have arbitrary number of dimensions,
            but the length along `axis` (see below) must match the length of `t`.
            Values must be finite.
        degree, optional
            B-spline degree specified either by the literal strings or a positive
            integer. By default, 'cubic' or 3 is used.

            Alternately, an interpolator class which conforms to `scipy.interpolate.PPoly`
            class (e.g., `scipy.interpolate.CubicSpline`) may be specified to replace
            the default B-Spline interpolator.
        interp_kws, optional
            Optional keyword arguments to construct B-splines using
            `scipy.interpolate.make_interp_spline()` or specified interpolation class as
            `degree`. For B-spline interpolation, available keywords are `bc_type`,
            `check_finite`, and `axis`. See `scipy`'s documentation for their usages.
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

    def __init__(self, tp_or_fs: ArrayLike | float, xp: ArrayLike, **kwargs):
        super().__init__(
            tp_or_fs=tp_or_fs, xp=xp, TaperBaseClass=BaseInterpolator, **kwargs
        )

    def __repr__(self):
        return f"Interpolator[{self._int}]"

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


class ClampedInterpolator(Interpolator):
    """Clamped interpolator"""

    @overload
    def __init__(
        self,
        tp_or_fs: ArrayLike | float,
        xp: ArrayLike,
        *,
        degree: Literal["linear", "quadratic", "cubic"] | int | type = 3,
        interp_kws: dict | None = None,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.002,
        transition_initial_on: bool = False,
        **kwargs: Unpack[dict],
    ):
        """`1-D B-Spline Interpolator with clamped extrapolation

        Parameters
        ----------
        tp_or_fs
            The time of the data points, must be 1D and strictly increasing. If a single
            value is given, it is taken as the sampling rate of xp, which is sampled
            at the interval of `1/tp_or_fs`, starting at time 0.
        xp
            The value of the data points. It can have arbitrary number of dimensions,
            but the length along `axis` (see below) must match the length of `t`.
            Values must be finite.
        degree, optional
            B-spline degree specified either by the literal strings or a positive
            integer. By default, 'cubic' or 3 is used.

            Alternately, an interpolator class which conforms to `scipy.interpolate.PPoly`
            class (e.g., `scipy.interpolate.CubicSpline`) may be specified to replace
            the default B-Spline interpolator. The specified class must support
            `bc_type='clamped'` keyword argument.
        interp_kws, optional
            Optional keyword arguments to construct B-splines using
            `scipy.interpolate.make_interp_spline()` or specified interpolation class as
            `degree`. For B-spline interpolation, available keywords are `check_finite`
            and `axis`. See `scipy`'s documentation for their usages.
        transition_time, optional
            transition time or times if multiple steps, by default None (no transitions)
        transition_type, optional
            transition type or types if multiple steps with different types, by default
            'raised_cos'
        transition_time_constant, optional
            transition time constant/duration or constants/durations if multiple steps
            with different settings, by default None 0.002
        transition_initial_on, optional
            initial signal level, by default 0.0

        Note: If `transition_type` or `transition_time_constant` is shorter than `transition_time`, their values are cycled.
        """

    def __init__(
        self,
        tp_or_fs: ArrayLike,
        xp: ArrayLike,
        **kwargs,
    ):
        int_kws = kwargs.get("interp_kws", {})
        if "bc_type" not in int_kws:
            kwargs["interp_kws"] = {**int_kws, "bc_type": "clamped"}

        super().__init__(tp_or_fs=tp_or_fs, xp=xp, **kwargs)

        given_tp = np.asarray(tp_or_fs).ndim
        self._np0 = ceil(tp_or_fs[0] * self.fs) if given_tp else 0
        self._np1 = ceil(tp_or_fs[-1] * self.fs) if given_tp else len(xp)
        self._xp0 = xp[0]
        self._xp1 = xp[-1]

    def __repr__(self):
        return f"ClampedInterpolator[{self._int}]"

    def _clamp_prep(self, n, n0):

        x = np.empty((n, *self.shape))

        np0 = max(n0, self._np0)
        np1 = min(n0 + n, self._np1)
        x[: np0 - n0] = self._xp0
        x[np1 - n0 :] = self._xp1
        return x, np0, np1

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

        x, np0, np1 = self._clamp_prep(n, n0)
        x[np0 - n0 : np1 - n0] = self._taper_call(
            np1 - np0, np0, sample_edges, upsample_factor, **kwargs
        )
        return x

    def derivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        x, np0, np1 = self._clamp_prep(n, n0)
        x[np0 - n0 : np1 - n0] = self._taper_derivative(
            np1 - np0, np0, sample_edges, upsample_factor, nu=nu, **kwargs
        )
        return x

    def antiderivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        x, np0, np1 = self._clamp_prep(n, n0)
        x[np0 - n0 : np1 - n0] = self._taper_antiderivative(
            np1 - np0, np0, sample_edges, upsample_factor, nu=nu, **kwargs
        )
        return x


class PeriodicInterpolator(Interpolator, AnalyticFunctionGenerator):
    """Periodic interpolator"""

    @overload
    def __init__(
        self,
        tp_or_fs: ArrayLike | float,
        xp: ArrayLike,
        *,
        closed: bool = True,
        degree: Literal["linear", "quadratic", "cubic"] | int | type = 3,
        interp_kws: dict | None = None,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.002,
        transition_initial_on: bool = False,
        **kwargs: Unpack[dict],
    ):
        """`1-D B-Spline Interpolator

        Parameters
        ----------
        tp_or_fs
            The time of the data points, must be 1D and strictly increasing. If a single
            value is given, it is taken as the sampling rate of xp, which is sampled
            at the interval of `1/tp_or_fs`, starting at time 0.
        xp
            The value of the data points. It can have arbitrary number of dimensions,
            but the length along `axis` (see below) must match the length of `t`.
            Values must be finite.
        closed
            `True` (default) if `tp[-1]-tp[0]` is the fundamental period of the periodic
            function. If `False` and the sampling rate is given in `tp_or_fs`, the
            closure point is automatically added as `((len(xp)+1)/fs, xp[0])`. If `False`
            and specific time points are given in `tp_or_fs`, it raises a `TypeError`
            exception.
        degree
            B-spline degree specified either by the literal strings or a positive
            integer. By default, 'cubic' or 3 is used.

            Alternately, an interpolator class which conforms to `scipy.interpolate.PPoly`
            class (e.g., `scipy.interpolate.CubicSpline`) may be specified to replace
            the default B-Spline interpolator. The specified class must support
            `by_type='periodic'` keyword argument.
        interp_kws
            Optional keyword arguments to construct B-splines using
            `scipy.interpolate.make_interp_spline()` or specified interpolation class as
            `degree`. For B-spline interpolation, available keywords are `check_finite`
            and `axis`. See `scipy`'s documentation for their usages. Note `bc_type` is
            forced to 'periodic' for this class object.
        transition_time, optional
            transition time or times if multiple steps, by default None (no transitions)
        transition_type, optional
            transition type or types if multiple steps with different types, by default
            'raised_cos'
        transition_time_constant, optional
            transition time constant/duration or constants/durations if multiple steps
            with different settings, by default 0.002
        transition_initial_on, optional
            initial signal level, by default 0.0

        Note: If `transition_type` or `transition_time_constant` is shorter than
        `transition_time`, their values are cycled.
        """

    def __init__(
        self,
        tp_or_fs: ArrayLike,
        xp: ArrayLike,
        *,
        closed: bool = True,
        **kwargs,
    ):

        if not np.asarray(tp_or_fs).ndim:
            if not closed:
                xp = [*xp, xp[0]]  # repeat the last sample
        elif not closed:
            raise TypeError(
                f"{closed=} and time points are given in tp_or_fs. The period of the function cannot be deduced."
            )

        int_kws = kwargs.get("interp_kws", {})
        if "bc_type" not in int_kws:
            kwargs["interp_kws"] = {**int_kws, "bc_type": "periodic"}

        super().__init__(
            tp_or_fs=tp_or_fs,
            xp=xp,
            fo=1 / (tp_or_fs[-1] - tp_or_fs[0]),
            **kwargs,
        )

    def __repr__(self):
        return f"PeriodicInterpolator[{self._int}]"

    def __call__(self, *args, analytic: bool = False, **kwargs) -> NDArray:
        """generate waveform

        Parameters
        ----------
        n
            number of samples to generate

        n0
            initial sample index, defaults to 0

        analytic
            True to return analytic waveform (i.e., cos + j sin), defaults to False

        """

        x = super().__call__(*args, **kwargs)
        if analytic:
            x = sps.hilbert(x, axis=0)
        return x

    def derivative(self, *args, analytic: bool = False, **kwargs) -> NDArray:

        x = super().antiderivative(*args, **kwargs)
        if analytic:
            x = sps.hilbert(x, axis=0)
        return x

    def antiderivative(self, *args, analytic: bool = False, **kwargs) -> NDArray:

        x = super().antiderivative(*args, **kwargs)
        if analytic:
            x = sps.hilbert(x, axis=0)
        return x
