from __future__ import annotations

from typing import Literal, overload, Sequence
from numpy.typing import NDArray, ArrayLike

from math import pi
import numpy as np

from .abc import AnalyticFunctionGenerator, FunctionGenerator

from .mixins import TaperMixin, BiasMixin, ScaleMixin, StepTypeLiteral
from . import _functions as _f
from .Constant import Constant
from .Interpolator import ClampedInterpolator

_2pi = 2 * np.pi

RosenbergPulseType = Literal[
    "triangular",
    "polynomial",
    "trigonometric1",
    "trigonometric2",
    "trignometric3",
    "trapezoidal",
]


class BaseRosenbergGenerator(AnalyticFunctionGenerator):

    pulse_type: RosenbergPulseType = "polynomial"
    oq: float | FunctionGenerator = 0.5
    sq: float | FunctionGenerator = 2.0
    phase: float = 0.0

    def __init__(
        self,
        fo: float | FunctionGenerator,
        *,
        pulse_type: RosenbergPulseType | None = None,
        open_quotient: float | FunctionGenerator | None = None,
        speed_quotient: float | FunctionGenerator | None = None,
        phase: float = 0.0,
        **kwargs,
    ):
        """no-mixin implementation of Rosenberg's pulse trains

        Parameters
        ----------
        fo
            fundamental frequency, a float value in Hz or a function generator object
        pulse_type, optional
            pulse type, by default 'polynomial'
        open_quotient, optional
            open quotient (voicing open-time/period), a value between 0 and 1, by default 0.5
        speed_quotient, optional
            speed quotient (rise/fall time of open period), a positive value, by default 2.0
        phase, optional
            pulse phasing between -1 and 1, by default 0 to start the opening phase at time n0.
            A positive delay moves the pulse to the right while a negative
            delay moves the pulse to the left.
        """

        # TODO: CONVERT TO a FunctionGenerator class and MOVE TO function_generators package
        # TODO: implement other types

        if pulse_type is not None and pulse_type != "polynomial":
            raise NotImplementedError(
                'Currently, only "polynomial" pulse type implemented.'
            )

        super().__init__(fo, **kwargs)

        if open_quotient is not None:
            self.oq = open_quotient
        if speed_quotient is not None:
            self.sq = speed_quotient
        if phase is not None:
            self.phase = phase

    def wrapped_phase(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        **kwargs,
    ) -> NDArray:
        """generate wrapped normalized phase waveform"""

        return (
            self._fo.antiderivative(n, n0, sample_edges, upsample_factor)
            - self.phase / 2
        ) % 1.0

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

        if analytic:
            raise TypeError("analytic=True is not supported.")

        oq = self.oq
        if isinstance(oq, FunctionGenerator):
            oq = oq(n, n0, sample_edges, upsample_factor)

        sq = self.sq
        if isinstance(sq, FunctionGenerator):
            sq = sq(n, n0, sample_edges, upsample_factor)

        # time wrapped and normalized to period
        phi = self.wrapped_phase(n, n0, sample_edges, upsample_factor)

        f = np.empty_like(phi)

        tf2 = phi > oq  # closed
        f[tf2] = 0.0

        pk = sq * oq / (1 + sq)
        tf0 = phi <= pk  # opening
        tf1 = ~(tf0 | tf2)  # closing

        pktm = pk[tf0] if np.asarray(pk).ndim else pk
        x = phi[tf0] / pktm
        f[tf0] = 3 * x**2 - 2 * x**3

        pktm = pk[tf1] if np.asarray(pk).ndim else pk
        oqtm = oq[tf1] if np.asarray(oq).ndim else oq
        x = (phi[tf1] - pktm) / (oqtm - pktm)
        f[tf1] = 1 - x**2

        return f

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

        raise NotImplementedError("derivative has not been implemented")

    @property
    def solvable_antiderivative(self) -> bool:
        return False

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

        raise NotImplementedError("antiderivative has not been implemented")

    # @property
    # def order_of_vanishing(self) -> int | None:
    #     return None


class RosenbergGenerator(TaperMixin, BiasMixin, ScaleMixin, BaseRosenbergGenerator):
    """Rosenberg's pulse trains (Rosenberg, 1971)"""

    @overload
    def __init__(
        self,
        fo: float | FunctionGenerator,
        alpha: float | FunctionGenerator | None = None,
        *,
        open_quotient: float | FunctionGenerator | None = None,
        speed_quotient: float | FunctionGenerator | None = None,
        phase: float = 0.0,
        pulse_type: RosenbergPulseType | None = None,
        bias: float | FunctionGenerator | None = None,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.004 / pi,
        transition_initial_on: bool = False,
        **kwargs,
    ):
        """Rosenberg's pulse trains

        Parameters
        ----------
        fo
            fundamental frequency, a float value in Hz or a function generator object
        alpha
            amplitude, fixed (`float`) or time-varying (`FunctionGenerator`), defaults to None (=1)
        pulse_type, optional
            pulse type, by default 'polynomial'
        open_quotient, optional
            open quotient (voicing open-time/period), a value between 0 and 1, by default 0.5
        speed_quotient, optional
            speed quotient (rise/fall time of open period), a positive value, by default 2.0
        phase, optional
            pulse phasing between -1 and 1, by default 0 to start the opening phase at time n0.
            A positive delay moves the pulse to the right while a negative
            delay moves the pulse to the left.

        fo
            fundamental frequency, a float value in Hz or a function generator object
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
        alpha: float | FunctionGenerator | None = None,
        **kwargs,
    ):
        super().__init__(
            fo=fo, scale=alpha, ScaleBaseClass=BaseRosenbergGenerator, **kwargs
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
        kwargs["analytic"] = analytic
        x = super().__call__(*args, **kwargs)

        return self._bias_call(self._scale_call(x, *args, **kwargs), *args, **kwargs)

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

        raise NotImplementedError("derivative has not been implemented")

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

        raise NotImplementedError("antiderivative has not been implemented")

    # @property
    # def order_of_vanishing(self) -> int | None:
    #     """order of the derivative to vanish for all time, None if not vanishing"""
    #     return None

    # @property
    # def has_closed_form(self) -> bool:
    #     return False
