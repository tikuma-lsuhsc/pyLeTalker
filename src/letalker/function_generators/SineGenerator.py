from __future__ import annotations

from typing import Literal, overload, Sequence
from numpy.typing import NDArray

from math import pi
import numpy as np

from .abc import AnalyticFunctionGenerator, FunctionGenerator

from .mixins import TaperMixin, BiasMixin, ScaleMixin, StepTypeLiteral
from . import _functions as _f
from .Constant import Constant

import logging

logger = logging.getLogger("letalker")

_2pi = 2 * np.pi


class BaseSineGenerator(AnalyticFunctionGenerator):
    phi0: FunctionGenerator

    def __init__(
        self,
        fo: float | NDArray | FunctionGenerator,
        phi0: float | NDArray | Literal["random"] | FunctionGenerator | None = None,
        **kwargs,
    ):
        """BaseSineGenerator

        :param fo: fundamental frequency, a float value in Hz or a function generator object
        :param phi0: initial phase, defaults to None (=0). Specify 'random' to randomly set the phase.
                     Random phase is drawn at the time of construction.
        """

        super().__init__(fo, **kwargs)

        is_fgen = isinstance(phi0, FunctionGenerator)
        if phi0 is None:
            phi0 = 0.0
        elif isinstance(phi0, str):
            if phi0 != "random":
                raise TypeError(f"unknown phi0 value: {phi0}")
            phi0 = _2pi * np.random.rand(*(self.shape))
        elif not is_fgen:
            phi0 = np.asarray(phi0)
            if phi0.ndim and self._fo.shape != phi0.shape:
                raise TypeError(
                    f"shape of phi0 {phi0.shape} does not match the fo {self._fo.shape}"
                )

        # set initial phase
        self.phi0 = phi0 if is_fgen else Constant(phi0)

    def phase(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        **kwargs,
    ) -> NDArray:
        """generate phase waveform"""

        args = (n, n0, sample_edges, upsample_factor)
        return _2pi * self._fo.antiderivative(*args, **kwargs) + self.phi0(
            *args, **kwargs
        )

    def __call__(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        *,
        analytic: bool = False,
        **kwargs,
    ) -> NDArray:
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

        args = (n, n0, sample_edges, upsample_factor)

        if self._fo.is_fixed and self.phi0.is_fixed:
            # sinusoid with fixed frequency & phase
            x = _f.sine(
                _2pi * self._fo.level,
                self.phi0.level,
                self.ts(*args),
                analytic=analytic,
            )
        else:
            # sinusoid with dynamic frequency
            phi = self.phase(*args, **kwargs)
            x = np.exp(1j * (pi / 2 - phi)) if analytic else np.sin(phi)

        return x

    def derivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        *,
        analytic: bool = False,
        **kwargs,
    ) -> NDArray:

        args = (n, n0, sample_edges, upsample_factor)
        kwargs["nu"] = nu

        if self._fo.is_fixed and self.phi0.is_fixed:
            # fixed frequency sinusoid
            dx = _f.sine_derivative(
                _2pi * self._fo.level,
                self.phi0.level,
                self.ts(*args),
                analytic=analytic,
                nu=nu,
            )
        else:
            # fixed frequency sinusoid
            phi = self.phase(*args)
            if analytic:
                phi -= (
                    pi / 2
                )  # so that real part corresponds to the non-analytic output
            if nu == 1:
                omega = _2pi * self._fo(*args)
                dx = 1j * omega * np.exp(1j * phi) if analytic else omega * np.cos(phi)
            else:
                raise NotImplementedError(
                    "higher-order derivative has not been implemented"
                )
                # create
                # phi_ = ProductGenerator(1j * _2pi, self._fo)
                # phi.derivative(*args, **kwargs)

        return dx

    @property
    def solvable_antiderivative(self) -> bool:
        return self.fo_is_fixed

    def antiderivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        *,
        analytic: bool = False,
        **kwargs,
    ) -> NDArray:

        args = (n, n0, sample_edges, upsample_factor)

        if self.fo_is_fixed and self.phi0.is_fixed:
            x = _f.sine_antiderivative(
                _2pi * self._fo.level, self.phi0.level, self.ts(*args), analytic, nu
            )
        else:
            if nu > 1:
                raise NotImplementedError(
                    "higher-order antiderivative has not been implemented"
                )

            # numerical integration trapazoid approximation at 2x rate
            f = BaseSineGenerator.__call__(
                self, n, n0, True, upsample_factor + 1, analytic=analytic
            )
            return _f.trapezoid_rule(f, self.fs, 2)

        return x

    # @property
    # def order_of_vanishing(self) -> int | None:
    #     return None


class SineGenerator(TaperMixin, BiasMixin, ScaleMixin, BaseSineGenerator):
    """Sine wave generator (analytic)"""

    @overload
    def __init__(
        self,
        fo: float | FunctionGenerator,
        A: float | FunctionGenerator | None = None,
        phi0: float | Literal["random"] | FunctionGenerator | None = None,
        *,
        bias: float | FunctionGenerator | None = None,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.004 / pi,
        transition_initial_on: bool = False,
        **kwargs,
    ):
        """SineGenerator

        fo
            fundamental frequency, a float value in Hz or a function generator object
        A
            amplitude, fixed (`float`) or time-varying (`FunctionGenerator`), defaults to None (=1)
        phi0
            initial phase, defaults to None (=0). Specify 'random' to randomly set the phase.
            Random phase is drawn at the time of construction.
        bias
            a constant offset value
        transition_time, optional
            transition time or times if multiple steps, by default 0.0
        transition_type, optional
            transition type or types if multiple steps with different types, by default 'raised_cos'
        transition_time_constant, optional
            transition time constant/duration or constants/durations if multiple steps with different settings, by default 0.002
        transition_initial_on, optional
            initial signal level, by default 0.0
        """

    def __init__(
        self,
        fo: float | FunctionGenerator,
        A: float | FunctionGenerator | None = None,
        phi0: float | Literal["random"] | FunctionGenerator | None = None,
        **kwargs,
    ):
        super().__init__(
            fo=fo, phi0=phi0, scale=A, ScaleBaseClass=BaseSineGenerator, **kwargs
        )

        # if function is tapered, apply the taper generator to both scale and bias
        if self._taper:
            # add taper as additional scaling factor
            self._scale_add(self._taper)

            # adjust bias bias by multiplying it by taper
            self._bias_multiply(self._taper)

    def __repr__(self):
        A = f"{self._scale_const}" if self._scale_const != 1.0 else ""
        fo = str(self._fo)
        phi0 = f"+ {self.phi0} " if self.phi0 else ""

        return f"SineGenerator[{A}sin( 2pi {fo} t {phi0})]"

    def __call__(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        *,
        analytic: bool = False,
        **kwargs,
    ) -> NDArray:
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

        args = (n, n0, sample_edges, upsample_factor)
        x = super().__call__(*args, **kwargs, analytic=analytic)
        x = self._scale_call(x, *args, **kwargs)
        return self._bias_call(x, *args, **kwargs, analytic=analytic)

    def derivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        *,
        analytic: bool = False,
        **kwargs,
    ) -> NDArray:

        args = (n, n0, sample_edges, upsample_factor)
        kwargs["nu"] = nu

        if self._scale_dynamic:
            logger.warning(
                "The derivative of sinusoid with dynamic scaling factor is not accurate. If this feature is needed, request via an issue on GitHub."
            )
            dx = self._scale_derivative(*args, **kwargs)
        else:
            dx = self._scale_const * super().derivative(
                *args, analytic=analytic, **kwargs
            )

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
        *,
        analytic: bool = False,
        **kwargs,
    ) -> NDArray:

        args = (n, n0, sample_edges, upsample_factor)
        kwargs["nu"] = nu

        if self._scale_set:
            ix = self._scale_antiderivative(*args, **kwargs)
        else:
            ix = super().antiderivative(*args, analytic=analytic, **kwargs)

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
