from __future__ import annotations

from typing import Literal, overload, Sequence
from numpy.typing import NDArray, ArrayLike

import numpy as np

from .abc import FunctionGenerator
from .mixins import TaperMixin, BiasMixin
from . import _functions as _f
from .SineGenerator import BaseSineGenerator
from .StepGenerator import StepTypeLiteral
from .ProductGenerator import ProductGenerator
from .Constant import Constant

_2pi = 2 * np.pi


class BaseFlutterGenerator(FunctionGenerator):

    _sine_gen: BaseSineGenerator
    _ampl: FunctionGenerator

    def __init__(
        self,
        fl: float | ArrayLike | FunctionGenerator,
        f: ArrayLike | float,
        n: int | None = None,
        phi0: ArrayLike | Literal["random"] | None = None,
        shape: int | tuple[int, ...] | None = None,
        **kwargs,
    ) -> NDArray:
        """generates a sum of sinusoids

        Parameters
        ----------
        fl
            maximum amplitude. If array, specifies amplitudes of each signals.
        f
            frequencies of the flutter sinusoids in Hz. If float and n is also
            specified, it is treated as the maximum allowable frequency, and
            the actual frequencies are randomly chosen.

            For array of signals, the shape of f should be (*shape, n) with the last
            dimension specifying the number of sinusoids to sum per signal.
        n
            number of fluttering sinusoids, by default None.
            If f is scalar and n is not None, n fluttering sinusoids are randomly
            chosen with the f as their maximum frequency. The frequencies are
            drawn from a uniform distribution bounded by 0 and f. By default,
        phi0, optional
            initial phase(s) of the flutter sinusoids, or 'random' to
            pick the phases randomly, by default None (to use 'random'). For an array
            of signals with specific initial phases, the shape of phi0 is (*shape, n).
        shape, optional
            shape of the array of signals to generate, by default None (automatically
            determined). Alternately, the number of signals can also be set via specifying
            `fl`, `f`, or `phi0` with additional array dimensions. If f or phi0 does not
            specify the exact frequencies or phases, each signal gets different random set
            of frequencies or phases.

        Note: The randomness is resolved during the construction. Each object
        is guaranteed to produce the same output
        """

        super().__init__(**kwargs)

        if isinstance(shape,int):
            shape = (shape,)

        freqs = np.asarray(f)
        if freqs.ndim:
            if shape is None:
                shape = freqs.shape[:-1]
            if n is None:
                n = freqs.shape[-1]

        if phi0 is None:
            # default to random phase
            phi0 = "random"
        else:
            phi0 = np.asarray(phi0)
            if phi0.ndim:
                if shape is None:
                    shape = phi0.shape[:-1]
                if n is None:
                    n = phi0.shape[-1]

        if isinstance(fl, ProductGenerator):
            fl = fl * (1 / n)
        elif isinstance(fl, FunctionGenerator):
            fl = ProductGenerator(fl, 1 / n)
        else:
            fl = Constant(np.asarray(fl) / n)

        if fl.ndim and shape is None:
            shape = fl.shape
        elif fl.ndim and shape != fl.shape:
            raise ValueError(f"the shape of fl must be {shape}")

        if not shape:
            shape = ()
        elif isinstance(shape, int):
            shape = (shape,)

        if freqs.ndim == 0:
            if n is None:
                # single-sinusoid
                freqs = np.full(*(*shape, 1), freqs)
                n = freqs.shape[0]
            else:
                # frequencies randomly drawn from uniform distributions between (i*freq/n, (i+1)*freq/n)
                df = freqs / n
                flims = df * np.arange(n).reshape((*np.ones_like(shape, int), n))
                freqs = df * np.random.rand(*shape, n) + flims

        self._sine_gen = BaseSineGenerator(fo=freqs, phi0=phi0)
        self._ampl = fl

        self._shape = shape

    @property
    def shape(self) -> tuple[int]:
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

        """

        x = self._sine_gen(
            n, n0, sample_edges, upsample_factor, analytic=analytic, **kwargs
        )

        a = self._ampl(n, n0, sample_edges, upsample_factor)[..., np.newaxis]
        return np.einsum("...j,...j->...", x, a)

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

        x = self._sine_gen.derivative(
            n, n0, sample_edges, upsample_factor, nu=nu, analytic=analytic, **kwargs
        )
        return np.einsum("...j,...j->...", x, self._ampl)

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

        if self._ampl.is_fixed:
            x = self._sine_gen.antiderivative(
                n, n0, sample_edges, upsample_factor, nu=nu, analytic=analytic, **kwargs
            )
            return np.einsum("...j,...j->...", x, self._ampl.level[..., np.newaxis])
        else:
            x = self(
                n,
                n0,
                True,
                upsample_factor + 1,
                nu=nu,
                analytic=analytic,
                **kwargs,
            )
            return _f.trapezoid_rule(x, self.fs, 2)

    # @property
    # def order_of_vanishing(self) -> int:
    #     return None


class FlutterGenerator(TaperMixin, BiasMixin, BaseFlutterGenerator):
    """Generates quasi-random sum of sinusoids (generalization of Klatt & Klatt, 1990)"""

    @overload
    def __init__(
        self,
        fl: float,
        f: ArrayLike | float | None = None,
        n: int | None = None,
        phi0: ArrayLike | Literal["random"] | None = None,
        *,
        bias: float | FunctionGenerator = 0.0,
        transition_time: float | Sequence[float] | None = None,
        transition_type: StepTypeLiteral | Sequence[StepTypeLiteral] = "raised_cos",
        transition_time_constant: float | Sequence[float] = 0.002,
        transition_initial_on: bool = False,
        **kwargs,
    ) -> NDArray:
        """Generates quasi-random sum of sinusoids (generalization of Klatt & Klatt, 1990)

        Parameters
        ----------
        fl
            maximum fluctuation. To match the original flutter definition of Klatt Eq. (1), with its "FL"
            flutter control parameter (0-100) set fl = (FL/50) * (fo/100) * n, where fo is the fundamental
            frequency, and n is the number of sinusoids (see below)
        f
            frequencies of the flutter sinusoids in Hz, by default None. If None, the Klatt's original
            definition is used: f = [12.7, 7.1, 4.7]
        n
            number of fluttering sinusoids, by default None.
            If f is scalar and n is not None, n fluttering sinusoids are randomly
            chosen with the f as their maximum frequency. The frequencies are
            drawn from a uniform distribution bounded by 0 and f. By default, None
        phi0, optional
            initial phase(s) of the flutter sinusoids, or 'random' to
            pick the phases randomly, by default None (to use 'random')
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

        Note: The randomness is resolved during the construction. Each object
        is guaranteed to produce the same output
        """

    def __init__(
        self,
        fl: float,
        f: ArrayLike | float | None = None,
        n: int | None = None,
        phi0: ArrayLike | Literal["random"] | None = None,
        **kwargs,
    ):
        if f is None:
            f = [12.7, 7.1, 4.7]

        kwargs["TaperBaseClass"] = BaseFlutterGenerator
        super().__init__(fl=fl, f=f, n=n, phi0=phi0, **kwargs)

        if self._taper:
            if self._bias_set:
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

        """

        args = (n, n0, sample_edges, upsample_factor)

        return self._taper_call(*args, analytic=analytic, bias=True, **kwargs)

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

        return self._taper_derivative(
            n,
            n0,
            sample_edges,
            upsample_factor,
            nu=nu,
            analytic=analytic,
            bias=True,
            **kwargs,
        )

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
        return self._taper_antiderivative(
            *args, nu=nu, analytic=analytic, bias=True, **kwargs
        )

    # @property
    # def order_of_vanishing(self) -> int:
    #     if self._taper:
    #         raise NotImplementedError(
    #             "derivative with transitions is not currently implemented"
    #         )

    #     return None
