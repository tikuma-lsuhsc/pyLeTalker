from __future__ import annotations

from typing import Sequence, Literal, Callable
from numpy.typing import NDArray

from .abc import FunctionGenerator

from math import pi, ceil
import numpy as np
from scipy.special import erf, expit, log_expit

from . import _functions as _f

StepTypeLiteral = Literal[
    "step",
    "linear",
    "raised_cos",
    "exp_decay",
    "exp_decay_rev",
    "logistic",
    "atan",
    "tanh",
    "erf",
]


class StepGenerator(FunctionGenerator):
    "Step function generator"
    _step_level0: float  # initial value
    _step_args: list[tuple]  # tuples of step function name and arguments

    def __init__(
        self,
        transition_times: Sequence[float] | float,
        levels: Sequence[float],
        *,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.002,
        **kwargs,
    ):
        """stepping function generator

        Parameters
        ----------
        transition_times
            transition times (must be monotonically increasing and has one less item than `levels`)
        levels
            a sequence of the signal levels
        transition_type, optional
            transition type or types if multiple steps with different types, by default 'raised_cos'
        transition_time_constant, optional
            transition time constant/duration or constants/durations if multiple steps with different settings, by default 0.002

        Note: If `type` or `time_constant` is shorter than `time`, their values are cycled.
        """

        transition_times = np.atleast_1d(transition_times)

        if not all(
            t0 < t1 for t0, t1 in zip(transition_times[:-1], transition_times[1:])
        ):
            raise ValueError(
                f"{transition_times=} must have monotonically increasing values."
            )

        levels = np.asarray(levels)

        if len(levels) < 2:
            raise TypeError(f"levels ({len(levels)=}) must have at least two items")

        super().__init__(**kwargs)

        step_level = levels[1:]
        step_time = transition_times
        step_type = transition_type
        step_time_constant = transition_time_constant
        step_initial_level = levels[0]

        if step_time is None:
            self._step_args = []
        else:

            def tile_to_n_trans(v, n_steps, v0=None):
                v = np.atleast_1d(v)
                if v0 is not None:
                    v = [v0, *v]
                    n_steps += 1
                return np.tile(v, ceil(n_steps / len(v)))

            times = np.atleast_1d(step_time)
            n_steps = len(times)  # number of transitions
            types = tile_to_n_trans(step_type, n_steps)
            taus = tile_to_n_trans(step_time_constant, n_steps)
            levels = tile_to_n_trans(step_level, n_steps, step_initial_level)

            self._step_level0 = levels[0]  # initial level
            self._step_args = [*zip(types, times, levels[1:] - levels[:-1], taus)]

    @property
    def shape(self) -> tuple[int]:
        """shape of each sample (empty if scalar)"""
        return self._step_level0.shape

    @property
    def ndim(self) -> int:
        """number of dimensions of each sample (0 if scalar)"""
        return self._step_level0.ndim

    @property
    def nb_steps(self) -> int:
        return len(self._step_args)

    def __call__(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        **kwargs,
    ) -> NDArray | float:
        """generate waveform

        Parameters
        ----------
        n
            number of samples to generate

        n0
            initial sample index, defaults to 0

        """

        ts_args = (n, n0, sample_edges, upsample_factor)
        t = self.ts(*ts_args)

        # composite stepping multiplier
        return sum(
            [self._step_methods[name](*args, t) for name, *args in self._step_args],
            start=self._step_level0,
        )

    def derivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        *,
        force_time_axis: bool = False,
        **kwargs,
    ) -> NDArray | float:

        ts_args = (n, n0, sample_edges, upsample_factor)
        t = self.ts(*ts_args)

        if len(self._step_args):

            # composite stepping multiplier
            x = sum(
                [
                    self._dstep_methods[name](*args, t, nu)
                    for name, *args in self._step_args
                ]
            )

        else:
            x = np.zeros(self.shape)

        if force_time_axis and np.ndim(x) == self.ndim:
            x = np.repeat(x[np.newaxis, ...], repeats=n, axis=0)

        return x

    def antiderivative(
        self, n, n0=0, sample_edges=False, upsample_factor=0, nu=1, **kwargs
    ) -> NDArray | float:

        t = self.ts(n, n0, sample_edges, upsample_factor)

        # composite stepping multiplier
        return sum(
            [
                self._istep_methods[name](*args, t, nu)
                for name, *args in self._step_args
            ],
            (
                _f.poly_antiderivative([self._step_level0], t.ravel(), nu)
                if np.any(self._step_level0 != 0)
                else np.zeros(self.shape)
            ),
        )

    # @property
    # def order_of_vanishing(self) -> int | None:
    #     if self._nb_steps:
    #         raise NotImplementedError(
    #             "derivative with transitions is not currently implemented"
    #         )

    #     oov = {
    #         "step": 1,
    #         "linear": 2,
    #         "raised_cos": None,
    #         "exp_decay": None,
    #         "logistic": None,
    #         "atan": None,
    #         "tanh": None,
    #         "erf": None,
    #     }

    #     oovs = [oov[name] for name, *_ in self._step_args]

    #     return None if any(o is None for o in oovs) else max(o is None for o in oovs)

    # STEP TRANSITION

    @staticmethod
    def _step_step(t0: float, x1: NDArray, _: float, t: NDArray):
        x = np.empty((len(t), *x1.shape))
        tf = t.ravel() <= t0
        x[tf, ...] = 0
        x[~tf, ...] = x1
        return x

    @staticmethod
    def _dstep_step(t0: float, x1: NDArray, _: float, t: NDArray, nu: int) -> float:
        return np.zeros_like(x1)

    @staticmethod
    def _istep_step(t0: float, x1: NDArray, _: float, t: NDArray, nu: int) -> float:
        return _f.pwpoly_antiderivative([[np.zeros_like(x1)], [x1]], [t0], t.ravel())

    # LINEAR TRANSITION

    @staticmethod
    def _linear_args(
        t0: float, x1: NDArray, tau: float, trigger_level: float = 0.5
    ) -> tuple[float, float, float]:
        m = x1 / tau
        b = x1 / 2 - t0 * m
        dt = tau / 2
        return [[np.zeros_like(x1)], np.stack([b, m], 0), [x1]], [t0 - dt, t0 + dt]

    @staticmethod
    def _step_linear(t0: float, x1: NDArray, tau: float, t: NDArray):
        return _f.pwpoly(*StepGenerator._linear_args(t0, x1, tau), t.ravel())

    @staticmethod
    def _dstep_linear(t0: float, x1: NDArray, tau: float, t: NDArray, nu: int):
        if nu > 1:
            return np.zeros_like(x1)
        return _f.pwpoly_derivative(*StepGenerator._linear_args(t0, x1, tau), t.ravel())

    @staticmethod
    def _istep_linear(t0: float, x1: NDArray, tau: float, t: NDArray, nu: int):
        return _f.pwpoly_antiderivative(
            *StepGenerator._linear_args(t0, x1, tau), t.ravel()
        )

    # RAISED COSINE TRANSITION

    @staticmethod
    def _pwfun(
        shape: tuple[int, ...], t: NDArray, tb: Sequence[float], fcn: Sequence[Callable]
    ):
        x = np.empty((t.shape[0], *shape))
        tr = t.ravel()  # flattened for range check
        tf0 = tr < tb[0]
        x[tf0, ...] = 0
        tf1 = tr >= tb[1] if len(tb) > 1 else ~tf0
        x[tf1, ...] = fcn[-1](t[tf1])
        if len(tb) > 1:
            tf = ~(tf0 | tf1)
            x[tf, ...] = fcn[0](t[tf])
        return x

    @staticmethod
    def _raised_cos_args(
        t0: float, tcos: float, trigger_level: float = 0.5
    ) -> tuple[float, float, float]:
        a = pi / 2 * tcos
        return pi / a, t0 - a / 2, t0 + a / 2

    @staticmethod
    def _step_raised_cos(t0: float, x1: NDArray, tcos: float, t: NDArray):

        omega, ta, tb = StepGenerator._raised_cos_args(t0, tcos)
        fcn = [
            lambda t: x1 / 2 * (1 + _f.sine(omega, 0.0, (t - t0), False)),
            lambda _: x1,
        ]
        return StepGenerator._pwfun(x1.shape, t, [ta, tb], fcn)

    @staticmethod
    def _dstep_raised_cos(t0: float, x1: NDArray, tcos: float, t: NDArray, nu: int):

        omega, ta, tb = StepGenerator._raised_cos_args(t0, tcos)
        fcn = [
            lambda t: x1 / 2 * _f.sine_derivative(omega, 0.0, (t - t0), False, nu),
            lambda _: np.zeros_like(x1),
        ]
        return StepGenerator._pwfun(x1.shape, t, [ta, tb], fcn)

    @staticmethod
    def _istep_raised_cos(t0: float, x1: NDArray, tcos: float, t: NDArray, nu: int):

        omega, ta, tb = StepGenerator._raised_cos_args(t0, tcos)
        g = x1 / 2
        fcn1 = lambda t: _f.poly_antiderivative(
            [g], (t.ravel() - ta), nu
        ) + g * _f.sine_antiderivative(omega, 0.0, (t - t0), False, nu)

        xb = fcn1(tb)

        fcn = [
            fcn1,
            lambda t: _f.poly_antiderivative([x1], t.ravel() - tb, nu) + xb,
        ]
        return StepGenerator._pwfun(x1.shape, t, [ta, tb], fcn)

    # EXPONENTIAL DECAY TRANSITION

    @staticmethod
    def _exp_decay_args(
        t0: float, tau: float, trigger_level: float = 0.5
    ) -> tuple[float, float]:

        # so that at t=0, dg/dt = 1
        tau = 2 / pi * tau

        # shift "on" time back so 0.5 point is at t0
        t0 = t0 + np.log(0.5) * tau

        return tau, t0

    @staticmethod
    def _step_exp_decay(t0: float, x1: NDArray, tau: float, t: NDArray):

        tau, t0 = StepGenerator._exp_decay_args(t0, tau)
        fcn = lambda t: x1 * (1 - np.exp(-(t - t0) / tau))
        return StepGenerator._pwfun(x1.shape, t, [t0], [fcn])

    @staticmethod
    def _dstep_exp_decay(t0: float, x1: NDArray, tau: float, t: NDArray, nu: int):

        tau, t0 = StepGenerator._exp_decay_args(t0, tau)
        beta = -1 / tau
        fcn = lambda t: -x1 * beta**nu * np.exp(beta * (t - t0))
        return StepGenerator._pwfun(x1.shape, t, [t0], [fcn])

    @staticmethod
    def _istep_exp_decay(t0: float, x1: NDArray, tau: float, t: NDArray, nu: int):

        tau, t0 = StepGenerator._exp_decay_args(t0, tau)
        a = -tau
        b = -1 / tau
        fcn0 = lambda t: x1 * (
            _f.poly_antiderivative([np.ones_like(x1)], (t.ravel() - t0), nu)
            - a**nu * np.exp(b * (t - t0))
        )

        xb = fcn0(t0)
        fcn = lambda t: fcn0(t) - xb

        return StepGenerator._pwfun(x1.shape, t, [t0], [fcn])

    # TIME-REVERSED EXPONENTIAL DECAY TRANSITION

    @staticmethod
    def _exp_decay_rev_args(
        t0: float, tau: float, trigger_level: float = 0.5
    ) -> tuple[float, float]:

        # so that at t=0, dg/dt = 1
        tau = 2 / pi * tau

        # shift "off" time forward so user t0 specifies the 0.5 point
        t0 = t0 - np.log(0.5) * tau

        return tau, t0

    @staticmethod
    def _step_exp_decay_rev(t0: float, x1: NDArray, tau: float, t: NDArray):

        x = StepGenerator._step_exp_decay(-t0, x1, tau, -t[::-1, ...])
        return x1 - x[::-1]

    @staticmethod
    def _dstep_exp_decay_rev(t0: float, x1: NDArray, tau: float, t: NDArray, nu: int):

        dx = StepGenerator._dstep_exp_decay(-t0, x1, tau, -t[::-1, ...], nu)
        return dx[::-1]

    @staticmethod
    def _istep_exp_decay_rev(t0: float, x1: NDArray, tau: float, t: NDArray, nu: int):

        tau, t0 = StepGenerator._exp_decay_rev_args(t0, tau)
        x = np.empty((t.shape[0], *x1.shape))
        tf = t[:, 0] < t0
        x[tf, :] = x1 * tau**nu * np.exp((t[tf, :] - t0) / tau)
        tf[:] = ~tf
        x[tf, :] = x1 * (t[tf, :] - t0) ** nu + x1 * tau**nu

        return x

    # LOGISTIC TRANSITION

    @staticmethod
    def _step_logistic(t0: float, x1: NDArray, a: float, t: NDArray):
        a /= 4
        nu = (t - t0) / a
        return x1 * expit(nu) # 1/(1+exp(-nu))

    @staticmethod
    def _dstep_logistic(t0: float, x1: NDArray, a: float, t: NDArray, nu: int):
        if nu > 1:
            raise NotImplemented(
                "higher-order derivatives of the logistics function is not currently supported"
            )
        t = t - t0
        a /= 4
        # e = np.exp(-t / a)
        nu = t/a
        return x1/a * expit(nu)*expit(-nu)
        # return x1 * e / (a * (1 + e) ** 2)
        # x1 / a * (1 / (e**(-1) + 1)) * 1 / (1 + e)

    @staticmethod
    def _istep_logistic(t0: float, x1: NDArray, a: float, t: NDArray, nu: int):
        if nu > 1:
            raise NotImplemented(
                "higher-order antiderivative of the logistics function is not currently supported"
            )
        t = t - t0
        a /= 4
        nu = t/a
        
        # x = x1 * (a * np.log(1 + np.exp(-nu)) + t)
        return x1 * (t-a * log_expit(nu)).reshape(-1,*np.ones(x1.ndim,int))

    # ATAN TRANSITION

    @staticmethod
    def _step_atan(t0: float, x1: NDArray, a: float, t: NDArray):
        a /= pi
        return x1 * (0.5 + np.atan((t - t0) / a) / pi)

    @staticmethod
    def _dstep_atan(t0: float, x1: NDArray, a: float, t: NDArray, nu: int):

        if nu > 1:
            raise NotImplemented(
                "higher-order derivatives of the arctan function is not currently supported"
            )
        t = t - t0
        a /= pi
        return x1 / (pi * a * (1 + t**2 / a**2))

    @staticmethod
    def _istep_atan(t0: float, x1: NDArray, a: float, t: NDArray, nu: int):
        if nu > 1:
            raise NotImplemented(
                "higher-order antiderivative of the arctan function is not currently supported"
            )
        t = t - t0
        a /= pi
        return x1 * (t / 2 + (-a * np.log(a**2 + t**2) / 2 + t * np.atan(t / a)) / pi)

    # TANH TRANSITION

    @staticmethod
    def _step_tanh(t0: float, x1: NDArray, a: float, t: NDArray):
        a /= 2
        return x1 / 2 * (1 + np.tanh((t - t0) / a))

    @staticmethod
    def _dstep_tanh(t0: float, x1: NDArray, a: float, t: NDArray, nu: int):
        if nu > 1:
            raise NotImplemented(
                "higher-order derivatives of the tanh function is not currently supported"
            )
        t = t - t0
        a /= 2
        return x1 * (1 - np.tanh(t / a) ** 2) / (2 * a)

    @staticmethod
    def _istep_tanh(t0: float, x1: NDArray, a: float, t: NDArray, nu: int):
        if nu > 1:
            raise NotImplemented(
                "higher-order antiderivative of the tanh function is not currently supported"
            )
        t = t - t0
        a /= 2
        return x1 * (-a * np.log(np.tanh(t / a) + 1) / 2 + t)

    # ERF TRANSITION
    @staticmethod
    def _step_erf(t0: float, x1: NDArray, a: float, t: NDArray):
        a = a / np.sqrt(pi)
        return x1 / 2 * (1 + erf((t - t0) / a))

    @staticmethod
    def _dstep_erf(t0: float, x1: NDArray, a: float, t: NDArray, nu: int):
        if nu > 1:
            raise NotImplemented(
                "higher-order derivatives of the erf function is not currently supported"
            )
        a = a / np.sqrt(pi)
        t = t - t0
        return x1 * np.exp(-(t**2) / a**2) / (np.sqrt(pi) * a)

    @staticmethod
    def _istep_erf(t0: float, x1: NDArray, a: float, t: NDArray, nu: int):
        if nu > 1:
            raise NotImplemented(
                "higher-order antiderivative of the erf function is not currently supported"
            )
        a = a / np.sqrt(pi)
        t = t - t0
        return x1 * (
            a * np.exp(-((t / a) ** 2)) / (2 * np.sqrt(pi)) + t * erf(t / a) / 2 + t / 2
        )

    _step_methods = {
        "step": _step_step,
        "linear": _step_linear,
        "raised_cos": _step_raised_cos,
        "exp_decay": _step_exp_decay,
        "exp_decay_rev": _step_exp_decay_rev,
        "logistic": _step_logistic,
        "atan": _step_atan,
        "tanh": _step_tanh,
        "erf": _step_erf,
    }

    _dstep_methods = {
        "step": _dstep_step,
        "linear": _dstep_linear,
        "raised_cos": _dstep_raised_cos,
        "exp_decay": _dstep_exp_decay,
        "exp_decay_rev": _dstep_exp_decay_rev,
        "logistic": _dstep_logistic,
        "atan": _dstep_atan,
        "tanh": _dstep_tanh,
        "erf": _dstep_erf,
    }

    _istep_methods = {
        "step": _istep_step,
        "linear": _istep_linear,
        "raised_cos": _istep_raised_cos,
        "exp_decay": _istep_exp_decay,
        "exp_decay_rev": _istep_exp_decay_rev,
        "logistic": _istep_logistic,
        "atan": _istep_atan,
        "tanh": _istep_tanh,
        "erf": _istep_erf,
    }
