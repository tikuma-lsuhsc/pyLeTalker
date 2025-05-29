from __future__ import annotations

from typing import Literal, overload, TYPE_CHECKING, Sequence
from numpy.typing import NDArray

import logging

logger = logging.getLogger("letalker")

import numpy as np


from .abc import FunctionGenerator
from .mixins import TaperMixin, BiasMixin, ScaleMixin
from . import _functions as _f

if TYPE_CHECKING:
    from .StepGenerator import StepTypeLiteral


class BaseExponentialGenerator(FunctionGenerator):

    def __init__(
        self, exponent: FunctionGenerator, *, base: float | None = None, **kwargs
    ):

        super().__init__(**kwargs)

        self._base = base and float(base)
        self._base_log = base and np.log(self._base)
        self._exponent = exponent

    @property
    def shape(self) -> tuple[int, ...]:
        """shape of each sample (empty if scalar)"""
        return self._exponent.shape

    @property
    def ndim(self) -> int:
        """number of dimensions of each sample (0 if scalar)"""
        return self._exponent.ndim

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

        x = self._exponent(n, n0, sample_edges, upsample_factor, **kwargs)
        return np.exp(x) if self._base is None else self._base**x

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

        dx = self._exponent.derivative(*ts_args, nu, **kwargs)
        if self._base_log is not None:
            dx = self._base_log * dx

        return dx * self(*ts_args, **kwargs)

    def antiderivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        # numerical integration trapazoid approximation at 2x rate
        f = self(self, n, n0, True, upsample_factor + 1, **kwargs)
        return _f.trapezoid_rule(f, self.fs, 2)

    # @property
    # def order_of_vanishing(self) -> int:
    #     return None

    # @property
    # def has_closed_form(self) -> bool:
    #     return False


class ExponentialGenerator(TaperMixin, BiasMixin, ScaleMixin, BaseExponentialGenerator):
    """Generates exponential function generator wrapper"""

    @overload
    def __init__(
        self,
        exponent: FunctionGenerator,
        *,
        base: float | None = None,
        scale: FunctionGenerator | float | None = None,
        bias: FunctionGenerator | float | None = None,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.002,
        transition_initial_on: bool = False,
    ):
        """Exponential function generator wrapper


        Parameters
        ----------
        exponent
            a function generator to provide the exponent value
        base, optional
            base of the exponential function, by default None to use the Euler's number e
        scale, optional
            gain to be multiplied to the exponential function, by default None (=1.0)
        bias, optional
            offset to be added to the exponential function after scaling, by default None (=0.0)
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

    def __init__(self, exponent: FunctionGenerator, **kwargs):
        super().__init__(
            exponent=exponent, ScaleBaseClass=BaseExponentialGenerator, **kwargs
        )

        # if function is tapered, apply the taper generator to both scale and bias
        if self._taper:
            # add taper as additional scaling factor
            self._scale_add(self._taper)

            # adjust bias bias by multiplying it by taper
            self._bias_multiply(self._taper)

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

        args = (n, n0, sample_edges, upsample_factor)
        x = super().__call__(*args, **kwargs)

        return self._bias_call(self._scale_call(x, *args, **kwargs), *args, **kwargs)

    def derivative(
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

        if self._scale_dynamic:
            dx = self._scale_derivative(*args, **kwargs)
        else:
            dx = self._scale_const * super().derivative(*args, **kwargs)

        if self._bias_dynamic:
            dx = self._bias_derivative(dx, *args, **kwargs)

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

        if self._scale_set:
            ix = self._scale_antiderivative(*args, **kwargs)
        else:
            ix = super().antiderivative(*args, **kwargs)

        if self._bias_set:
            ix += self._bias_antiderivative(*args, **kwargs)

        return ix

    # @property
    # def order_of_vanishing(self) -> int | None:
    #     """order of the derivative to vanish for all time, None if not vanishing"""

    #     return None

    # @property
    # def has_closed_form(self) -> bool:
    #     return not self._taper
