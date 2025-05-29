from __future__ import annotations

from typing import Sequence, overload, Literal
from numpy.typing import NDArray

import numpy as np

from .abc import FunctionGenerator
from .mixins import TaperMixin, StepTypeLiteral
from . import _functions as _f


class BaseConstant(FunctionGenerator):

    def __init__(self, level: float | NDArray, **kwargs):
        super().__init__(**kwargs)
        self._c = np.asarray([level])

    @property
    def level(self) -> float:
        return self._c[0]

    @property
    def shape(self) -> tuple[int]:
        """shape of each sample (empty if scalar)"""
        return self._c.shape[1:]

    @property
    def ndim(self) -> int:
        """number of dimensions of each sample (0 if scalar)"""
        return self._c.ndim - 1

    @property
    def is_fixed(self) -> bool:
        """True if truly constant (i.e., without any transition)"""
        return True

    def __call__(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        *,
        force_time_axis: bool | Literal["axis_only", "tile_data"] = True,
        **kwargs,
    ) -> NDArray | float:
        """generate samples

        Parameters
        ----------
        n
            number of samples to generate
        n0, optional
            starting time index (not used), by default 0
        sample_edges, optional
            not used, by default False
        upsample_factor, optional
            not used, by default 0
        force_time_axis, optional
            True or 'axis_only' to prepend time axis. If 'tile_data',
            the time axis is expanded to size n with the constant value
            repeated, by default True

        Returns
        -------
            constant value
        """
        c = self.level
        if force_time_axis:
            c = c[np.newaxis, ...]
            if force_time_axis == "tile_data":
                c = np.repeat(c, n, 0)
        return c

    def derivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        *,
        force_time_axis: bool | Literal["once", "tile"] = True,
        **kwargs,
    ) -> NDArray | float:

        c = np.zeros(self.shape)
        if force_time_axis:
            c = c[np.newaxis, ...]
            if force_time_axis == "tile_data":
                c = np.repeat(c, n, 0)

    def antiderivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        # numerical approximation
        args = (n, n0, sample_edges, upsample_factor)
        return _f.poly_antiderivative(self._c, self.ts(*args).ravel(), nu)

    # @property
    # def order_of_vanishing(self) -> int:
    #     return 1


class Constant(TaperMixin, BaseConstant):
    """Constant function generator"""

    @overload
    def __init__(
        self,
        level: float | Sequence[float],
        *,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.002,
        transition_initial_on: bool = False,
        **kwargs,
    ):
        """constant function generator

        Parameters
        ----------
        level
            signal level
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

    def __init__(self, level: float | NDArray, **kwargs):
        super().__init__(level=level, TaperBaseClass=BaseConstant, **kwargs)

    @property
    def is_fixed(self) -> bool:
        """True if truly constant (i.e., without any transition)"""
        return self._taper is None

    def __call__(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        *,
        force_time_axis: bool | Literal["once", "tile_data"] = True,
        **kwargs,
    ) -> NDArray | float:
        """generate waveform

        Parameters
        ----------
        n
            number of samples to generate
        n0
            initial sample index, defaults to 0
        force_time_axis, optional
            True or 'axis_only' to prepend time axis. If 'tile_data',
            the time axis is expanded to size n with the constant value
            repeated, by default True

        """

        ts_args = (n, n0, sample_edges, upsample_factor)

        x = self.level

        if self._taper:
            y = self._taper(*ts_args)
            if self.ndim:
                y = y.reshape(-1, *np.ones(self.ndim, int))
            x = x * y
        else:
            x = super().__call__(
                n, n0, sample_edges, upsample_factor, force_time_axis=force_time_axis
            )

        return x

    def derivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        *,
        force_time_axis: bool | Literal["once", "tile"] = True,
        **kwargs,
    ) -> NDArray | float:
        """generate derivative samples

        Parameters
        ----------
        n
            number of samples to generate
        n0, optional
            starting time index, by default 0
        sample_edges, optional
            True to shift sample timing by half a sample and generate
            (n+1) samples, by default False
        upsample_factor, optional
            upsampling factor, by default 0. If `sample_edges=False`
            the output sample size is `n*upsample_factor`, or if
            `sample_edges=True`, `n*upsample_factor + upsample_factor-1`.
        nu, optional
            _description_, by default 1
        force_time_axis, optional
            _description_, by default True

        Returns
        -------
            _description_
        """

        # specialized implementation
        args = (n, n0, sample_edges, upsample_factor)

        if self._taper:
            dx = self._taper.derivative(*args, nu=nu)
            if self.ndim:
                dx = dx.reshape(-1, *np.ones(self.ndim, int))
            dx = dx * self.level
        else:
            dx = np.zeros((n, *self.shape) if force_time_axis else self.shape)
        return dx

    def antiderivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        args = (n, n0, sample_edges, upsample_factor)
        kwargs["nu"] = nu
        return self._taper_antiderivative(*args, **kwargs)

    # @property
    # def order_of_vanishing(self) -> int:
    #     return 1
