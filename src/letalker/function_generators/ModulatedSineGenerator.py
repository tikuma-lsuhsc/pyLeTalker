from __future__ import annotations

from typing import Literal, Sequence, overload
from numpy.typing import ArrayLike

import numpy as np

from .abc import FunctionGenerator
from .SineGenerator import SineGenerator
from .StepGenerator import StepTypeLiteral
from .ProductGenerator import ProductGenerator
from .Constant import Constant

pi = np.pi
_2pi = 2 * pi


class ModulatedSineGenerator(SineGenerator):
    """Modulated sine wave generator (analytic)"""
    _fm: bool = False  # true if FM enabled
    _am: bool = False  # true if AM enabled

    @overload
    def __init__(
        self,
        fo: float | ArrayLike | FunctionGenerator,
        m_freq: float | ArrayLike | FunctionGenerator,
        am_extent: float | ArrayLike | FunctionGenerator | None = None,
        fm_extent: float | ArrayLike | FunctionGenerator | None = None,
        am_phi0: (
            float | ArrayLike | Literal["random"] | FunctionGenerator | None
        ) = None,
        fm_phi0: (
            float | ArrayLike | Literal["random"] | FunctionGenerator | None
        ) = None,
        A: float | ArrayLike | FunctionGenerator | None = None,
        phi0: float | ArrayLike | Literal["random"] | None = None,
        *,
        bias: float | FunctionGenerator | None = None,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.004 / pi,
        transition_initial_on: bool = False,
    ):
        """ModulatedSineGenerator - configure SineGenerator to produce AM/FM modulated

        fo
            fundamental frequency, a float value in Hz or a function generator object
        m_freq
            modulation frequency, unit-less relative frequency to fo
        am_extent
            ratio (Amax-Amin)/(Amax+Amin) where Amax and Amin are the maximum and
            minimum instantaneous cycle peaks, respectively
        fm_extent
            ratio (Tmax-Tmin)/(Tmax+Tmin) where Tmax and Tmin are maximum and minimum
            instantaneous cycle durations, respectively
        am_phi0
            initial phase of amplitude modulation
        fm_phi0
            initial phase of frequency modulation
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
        fo: float | ArrayLike | FunctionGenerator,
        m_freq: float | ArrayLike | FunctionGenerator,
        am_extent: float | ArrayLike | FunctionGenerator | None = None,
        fm_extent: float | ArrayLike | FunctionGenerator | None = None,
        am_phi0: (
            float | ArrayLike | Literal["random"] | FunctionGenerator | None
        ) = None,
        fm_phi0: (
            float | ArrayLike | Literal["random"] | FunctionGenerator | None
        ) = None,
        A: float | ArrayLike | FunctionGenerator | None = None,
        phi0: float | ArrayLike | Literal["random"] | None = None,
        **kwargs,
    ):

        # define absolute modulation frequency
        if isinstance(fo, FunctionGenerator) or isinstance(m_freq, FunctionGenerator):
            fm = ProductGenerator(fo, m_freq)
        else:
            fm = Constant(np.asarray(fo) * np.asarray(m_freq))

        # amplitude modulation phase
        if isinstance(am_phi0, str):
            if am_phi0 != "random":
                raise ValueError(f"unknown am_phi0 value: {am_phi0}")
            am_phi0 = _2pi * np.random.rand(*fm.shape)

        # frequency modulation phase
        if isinstance(fm_phi0, str):
            if fm_phi0 != "random":
                raise ValueError(f"unknown fm_phi0 value: {fm_phi0}")
            fm_phi0 = _2pi * np.random.rand(*fm.shape)

        # redefine the amplitude function if AM enabled
        if am_extent is not None and np.any(am_extent != 0):
            self._am = True
            if A is None:
                A = 1.0
            A = ProductGenerator(A, SineGenerator(fm, am_extent, am_phi0, bias=1.0))

        # redefine the frequency function if FM enabled
        if fm_extent is not None and np.any(fm_extent != 0):
            self._fm = True
            # convert normalized FM extent to absolute FM extent in Hz
            if isinstance(fo, FunctionGenerator) or isinstance(
                fm_extent, FunctionGenerator
            ):
                fm_extent = ProductGenerator(fo, fm_extent)
            else:
                fm_extent = Constant(np.asarray(fo) * np.asarray(fm_extent))
            fo = SineGenerator(fm, fm_extent, fm_phi0, bias=fo)

        super().__init__(fo=fo, A=A, phi0=phi0, **kwargs)

    @property
    def fo(self) -> FunctionGenerator:
        """fundamental frequency (excluding the FM effect)"""
        fo = self._fo
        return (
            (fo._biases[0] if len(fo._biases) else Constant(fo._bias_const))
            if self._fm
            else fo
        )
