from __future__ import annotations

from typing import Literal, Sequence, cast
from numpy.typing import NDArray, ArrayLike
from collections.abc import Callable

from math import pi
import numpy as np
from types import MethodType

from . import _functions as _f
from .abc import FunctionGenerator
from .StepGenerator import StepGenerator, StepTypeLiteral
from .utils import align_signals

_2pi = 2 * pi


class BiasMixin:
    _bias_const: float = 0.0
    _biases: list[FunctionGenerator]

    def __init__(self, *, bias: FunctionGenerator | float | None = None, **kwargs):

        super().__init__(**kwargs)
        self._biases = []

        if bias is not None:
            self._bias_add(bias)

    def _bias_add(self, bias: FunctionGenerator | float):
        """add additional bias function (e.g., flutter function)

        Parameters
        ----------
        bias
            _description_
        """

        from .Constant import Constant

        if not isinstance(bias, FunctionGenerator):
            self._bias_const = self._bias_const + np.asarray(bias)
        elif isinstance(bias, Constant) and bias.is_fixed:
            self._bias_const = self._bias_const + self._bias_const
        else:
            self._biases.append(bias)

    def _bias_multiply(self, gain: FunctionGenerator | float):
        """multiply bias functions with another generator

        Parameters
        ----------
        gain
            multiply the bias generator by this generator
        """

        from .ProductGenerator import ProductGenerator

        if self._bias_dynamic:
            self._biases = [ProductGenerator(b, gain) for b in self._biases]

        if np.any(self._bias_const):
            if not isinstance(gain, FunctionGenerator):
                self._bias_const *= gain
            elif gain.is_fixed:
                self._bias_const *= gain.level
            else:
                self._biases.append(ProductGenerator(self._bias_const, gain))
                self._bias_const = 0.0

    @property
    def _bias_set(self) -> bool:
        """True if bias is set"""
        return np.any(self._bias_const != 0.0) or len(self._biases)

    @property
    def _bias_dynamic(self) -> bool:
        """True if the bias is a constant

        - no need to pass `x` to `_scale_derivative()`
        - pass unscaled antiderivative instead of function values for `_scale_antiderivative()`
        """

        return len(self._biases)

    def _bias_call(
        self,
        x: NDArray,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        **kwargs,
    ):
        y = None
        if self._bias_dynamic:
            y = _f.sum(
                self._biases,
                n,
                n0,
                sample_edges,
                upsample_factor,
                bias=self._bias_const if self._bias_const != 0.0 else None,
            )
            x, y = align_signals(x, y)
            x = x + y
        elif np.any(self._bias_const):
            x += self._bias_const

        return x

    def _bias_derivative(
        self,
        x: NDArray,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ):
        if self._bias_dynamic:
            x = x + _f.sum_derivative(
                self._biases, n, n0, sample_edges, upsample_factor, nu=nu, **kwargs
            )

        return x

    def _bias_antiderivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ):
        args = (n, n0, sample_edges, upsample_factor)
        if self._bias_dynamic:
            y = _f.sum_antiderivative(
                self._biases,
                *args,
                nu=nu,
                bias=self._bias_const if np.any(self._bias_const) else None,
                **kwargs,
            )
        else:
            t = self.ts(*args).ravel()
            y = _f.poly_antiderivative([self._bias_const], t, nu)
        return y


class ScaleMixin:
    _scale_BaseClass: FunctionGenerator
    _scale_const: float = 1.0
    _scalers: list[FunctionGenerator]

    def __init__(
        self,
        *,
        scale: FunctionGenerator | float | None = None,
        ScaleBaseClass: type,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._scalers = []
        self._scale_BaseClass = ScaleBaseClass

        if scale is not None:
            self._scale_add(scale)

    def _scale_add(self, scaler: FunctionGenerator | float):
        """add additional scaling function (e.g., taper function)

        Parameters
        ----------
        scaler
            _description_
        """
        from .Constant import Constant

        if not isinstance(scaler, FunctionGenerator):
            self._scale_const = self._scale_const * np.asarray(scaler)
        elif scaler.is_fixed:
            self._scale_const = self._scale_const * cast(Constant, scaler).level
        else:
            self._scalers.append(scaler)

    @property
    def _scale_set(self) -> bool:
        """True if bias is set"""
        return np.any(self._scale_const != 1.0) or len(self._scalers)

    @property
    def _scale_dynamic(self) -> bool:
        """True if the scaling factor is a constant

        - no need to pass `x` to `_scale_derivative()`
        - pass unscaled antiderivative instead of function values for `_scale_antiderivative()`
        """

        return len(self._scalers)

    def _scale_call(
        self,
        x: NDArray,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        **kwargs,
    ):

        if len(self._scalers):
            y = _f.product(
                self._scalers,
                n,
                n0,
                sample_edges,
                upsample_factor,
                gain=self._scale_const if np.any(self._scale_const != 1.0) else None,
                **kwargs,
            )
            x, y = align_signals(x, y)
        else:
            y = self._scale_const

        return x * y

    def _scale_derivative(
        self,
        n: int,
        n0: int,
        sample_edges: bool,
        upsample_factor: int,
        nu: int = 1,
        **kwargs,
    ):
        if self._scale_dynamic:
            gain = self._scale_const if np.any(self._scale_const != 1.0) else None
            this_class, self.__class__ = self.__class__, self._scale_BaseClass
            try:
                x = _f.product_derivative(
                    [self, *self._scalers],
                    n,
                    n0,
                    sample_edges,
                    upsample_factor,
                    nu=nu,
                    gain=gain,
                    **kwargs,
                )
            finally:
                self.__class__ = this_class

        else:
            x = self._scale_const * self._scale_BaseClass.derivative(
                n, n0, sample_edges, upsample_factor, nu=nu, **kwargs
            )

        return x

    def _scale_antiderivative(
        self,
        n: int,
        n0: int,
        sample_edges: bool,
        upsample_factor: int,
        nu: int = 1,
        **kwargs,
    ):
        args = (n, n0, sample_edges, upsample_factor)

        gain = self._scale_const if np.all(self._scale_const != 1.0) else None
        if self._scale_dynamic:
            this_class, self.__class__ = self.__class__, self._scale_BaseClass
            try:
                x = _f.product_antiderivative(
                    [self, *self._scalers], *args, nu=nu, gain=gain, **kwargs
                )
            finally:
                self.__class__ = this_class
        else:
            x = self._scale_BaseClass.antiderivative(self, *args, nu, **kwargs)
            if gain is not None:
                x *= gain

        return x


class TaperMixin:
    _taper_BaseClass: FunctionGenerator | None

    def __init__(
        self,
        *,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.004 / pi,
        transition_initial_on: bool = False,
        TaperBaseClass: type[FunctionGenerator] | None = None,
        **kwargs,
    ):
        """add a function generator an ability to introduce on/off transition effects

        Parameters
        ----------
        transition_time, optional
            time of the on/off transition or times if multiple transition points, by default None (no taper effect)
        transition_type, optional
            transition type or types if multiple transitions with different types, by default 'raised_cos'
        transition_time_constant, optional
            transition step_time constant/duration or constants/durations if multiple steps with different settings, by default 0.002
        transition_initial_on, optional
            True to turn off the waveform at the first transition time, by default False

        Note: If `transition_type` or `transition_time_constant` is shorter than `transition_time`, their values are cycled.
        """

        super().__init__(**kwargs)

        self._taper = (
            None
            if transition_time is None
            else StepGenerator(
                transition_time,
                (1.0, 0.0) if transition_initial_on else (0.0, 1.0),
                transition_type=transition_type,
                transition_time_constant=transition_time_constant,
            )
        )

        if TaperBaseClass is not None:
            self._taper_BaseClass = TaperBaseClass

    def _taper_call(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        *,
        bias: bool = False,
        **kwargs,
    ):
        args = (n, n0, sample_edges, upsample_factor)
        this_class, self.__class__ = self.__class__, self._taper_BaseClass
        try:
            x = self(*args, **kwargs)
            if self._taper:
                y = self._taper(*args, **kwargs)
                x, y = align_signals(x, y)
                x = x * y

        finally:
            self.__class__ = this_class

        if bias and (bias_mixin := cast(BiasMixin, self))._bias_set:
            if self._taper:
                if np.any(bias_mixin._bias_const):
                    y = self._taper(**args, **kwargs)
                    x, y = align_signals(x, y)
                    return x + y
                if bias_mixin._bias_dynamic:
                    x = sum(
                        (
                            _f.product([b, self._taper], *args, **kwargs)
                            for b in bias_mixin._biases
                        ),
                        start=x,
                    )
            else:
                x = bias_mixin._bias_call(x, *args, **kwargs)

        return x

    def _taper_derivative(
        self,
        n: int,
        n0: int,
        sample_edges: bool,
        upsample_factor: int,
        nu: int = 1,
        *,
        bias: bool = False,
        **kwargs,
    ):
        args = (n, n0, sample_edges, upsample_factor)
        kwargs["nu"] = nu
        this_class, self.__class__ = self.__class__, self._taper_BaseClass
        try:
            x = (
                _f.product_derivative([self, self._taper], *args, **kwargs)
                if self._taper
                else self.derivative(*args, **kwargs)
            )
        finally:
            self.__class__ = this_class

        if bias:
            bias_mixin = cast(BiasMixin, self)
            if not self._taper:
                x = bias_mixin._bias_derivative(x, *args, **kwargs)
            elif bias_mixin._bias_dynamic:
                x = sum(
                    (
                        _f.product_derivative([b, self._taper], *args, **kwargs)
                        for b in bias_mixin._biases
                    ),
                    start=x,
                )

        return x

    def _taper_antiderivative(
        self,
        n: int,
        n0: int,
        sample_edges: bool,
        upsample_factor: int,
        nu: int = 1,
        *,
        bias: bool = False,
        **kwargs,
    ):
        args = (n, n0, sample_edges, upsample_factor)
        this_class, self.__class__ = self.__class__, self._taper_BaseClass
        try:
            x = (
                _f.product_antiderivative([self, self._taper], *args, nu=nu, **kwargs)
                if self._taper
                else self.antiderivative(*args, nu=nu, **kwargs)
            )
        finally:
            self.__class__ = this_class

        if bias and (bias_mixin := cast(BiasMixin, self))._bias_set:
            if np.any(bias_mixin._bias_const):
                t = self.ts(*args).ravel()
                y = _f.poly_antiderivative([bias_mixin._bias_const], t, nu)
                x, y = align_signals(x, y)
                x = x + y
            if bias_mixin._bias_dynamic:
                if self._taper:
                    x = sum(
                        (
                            _f.product_antiderivative(
                                [b, self._taper], *args, nu=nu, **kwargs
                            )
                            for b in bias_mixin._biases
                        ),
                        start=x,
                    )
                else:
                    x += _f.sum_antiderivative(
                        bias_mixin._biases, *args, nu=nu, **kwargs
                    )
        return x
