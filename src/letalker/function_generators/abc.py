from __future__ import annotations

from typing import TYPE_CHECKING
from numpy.typing import NDArray, ArrayLike
from collections.abc import Callable

import abc
import numpy as np

from ..core import TimeSampleHandler

__all__ = [
    "SampleGenerator",
    "FunctionGenerator",
    "AnalyticFunctionGenerator",
    "NoiseGenerator",
]


class SampleGenerator(TimeSampleHandler, Callable, metaclass=abc.ABCMeta):
    """Abstract baseclass of classes to generate"""

    # auto_repeat: bool = True

    def __init__(self, **kwargs):
        if len(kwargs):
            raise TypeError(f"Unresolved keyword arguments remain: {kwargs=}")

    def ts(
        self,
        nb_samples: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
    ):
        """generate time samples

        Parameters
        ----------
        nb_samples
            total number of samples to generate
        n0, optional
            index of the first sample to generate, by default 0
        sample_edges, optional
            True to generate times at the edges of the time bins, by default False
        upsample_factor, optional
            Specify >1 to generate denser sample set at the upsampled sampling rate, by default 0

        Returns
        -------
            nb_samples-element NumPy array containing the time sequence in second
        """

        t = TimeSampleHandler.ts(nb_samples, n0, sample_edges, upsample_factor)
        if self.ndim:
            t = t.reshape(-1, *np.ones(self.ndim, int))
        return t

    @abc.abstractmethod
    def __call__(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        **kwargs,
    ) -> NDArray:
        """generate and return waveform samples

        Parameters
        ----------
        n
            number of samples to generate (spanning only one of its dimensions)
        n0, optional
            first sample index, by default 0
        """

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int]:
        """shape of each sample (empty if scalar)"""

    def output_shape(self, n: int, force_time_axis: bool = False) -> tuple[int, ...]:
        """shape of generated sample array

        Parameters
        ----------
        n
            number of samples to generate
        force_time_axis, optional
            True to generate sample array with the time axis, by default False

        Returns
        -------
            shape of generated data array
        """

        shape = self.shape
        if force_time_axis or not self.is_fixed:
            shape = (n, *shape)
        return shape

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """number of dimensions of each sample (0 if scalar)"""

    @property
    def is_fixed(self) -> bool:
        """True if truly constant (i.e., without any transition)"""
        return False  # only True for Constant


class FunctionGenerator(SampleGenerator, metaclass=abc.ABCMeta):
    """Abstract base class of function generators

    Function generators are sample generators with concrete mathematical expression.

    """

    @abc.abstractmethod
    def derivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
    ) -> NDArray | float:
        """_summary_

        Parameters
        ----------
        n
            _description_
        n0, optional
            _description_, by default 0
        nu, optional
            _description_, by default 1

        Returns
        -------
            _description_
        """

    @abc.abstractmethod
    def antiderivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
    ) -> NDArray:
        """_summary_

        Parameters
        ----------
        n
            _description_
        n0, optional
            _description_, by default 0
        nu, optional
            _description_, by default 1

        Returns
        -------
            _description_

        Raises
        ------
        ValueError
            _description_
        """


class AnalyticFunctionGenerator(FunctionGenerator, metaclass=abc.ABCMeta):
    """Abstract base class of function generators, which can possibly generate analytic signal version of the function."""

    _fo: FunctionGenerator  # fundamental frequency

    def __init__(self, fo: float | FunctionGenerator, **kwargs):
        from .Constant import Constant

        super().__init__(**kwargs)

        self._fo: FunctionGenerator = (
            fo if isinstance(fo, FunctionGenerator) else Constant(fo)
        )

    @property
    def fo(self) -> FunctionGenerator:
        """function generator which generates a sequence of instantaneous fundamental frequency in Hz"""
        return self._fo

    @property
    def shape(self) -> tuple[int]:
        """shape of each sample (empty if scalar)"""
        return self._fo.shape

    @property
    def ndim(self) -> int:
        """number of dimensions of each sample (0 if scalar)"""
        return self._fo.ndim

    @property
    def fo_is_fixed(self) -> bool:
        """True if all parameters are fixed"""
        from .Constant import Constant

        return isinstance(self._fo, Constant) and self._fo.is_fixed

    @abc.abstractmethod
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
        """generate and return waveform samples

        Parameters
        ----------
        n
            number of samples to generate (spanning only one of its dimensions)
        n0, optional
            first sample index, by default 0
        analytic, optional
            True to return complex-valued analytic signal, by default False
        """


class NoiseGenerator(SampleGenerator, metaclass=abc.ABCMeta):
    """_summary_

    Parameters
    ----------
    SampleGenerator
        _description_
    metaclass, optional
        _description_, by default abc.ABCMeta
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self._buffer: NDArray
        self._buffer_n0: int
        self._buffer_z1: NDArray | None

        self.clear()

    def clear(self):
        """clear buffered noise samples"""
        self._buffer = np.empty((0, *self.shape))
        self._buffer_n0 = 0
        self._buffer_z1 = None

    @abc.abstractmethod
    def generate(
        self, n: int, z0: ArrayLike | None = None, **kwargs
    ) -> tuple[NDArray, NDArray | None]:
        """generate noise samples (no buffering)

        Parameters
        ----------
        n
            number of samples to generate
        z0, optional
            initial generator states, by default None

        Returns
        -------
            x
                generated samples
            z1
                final generator states, None if stateless generator
        """

    def __call__(
        self, n: int, n0: int = 0, regenerate: bool = False, **kwargs
    ) -> NDArray:
        """generate and return waveform samples

        Parameters
        ----------
        n
            number of samples to generate (spanning only one of its dimensions)
        n0, optional
            first sample index, by default 0
        regenerate, optional
            True to return complex-valued analytic signal, by default False
        """

        buffer_n1 = 0  # time index of the first unbuffered sample
        buffer_n = 0  # number of buffered samples
        n0_ = n1_ = 0  # starting and ending sample index relative to the
        if regenerate:
            self.clear()
            n0_ = 0
            n1_ = n
        elif self._buffer is not None:
            if n0 < self._buffer_n0:
                raise ValueError(
                    f"Cannot generate samples prior to the buffered samples (n>={self._buffer_n0}). Eitehr clear() first or call with `regenerate=True`."
                )
            buffer_n = self._buffer.shape[0]
            buffer_n1 = self._buffer_n0 + buffer_n
            n0_ = n0 - self._buffer_n0  # relative to the first sample in the buffer
            n1_ = n0_ + n
            if n1_ <= buffer_n1:
                # all samples already available
                return self._buffer[n0_:n1_]

        # generate missing samples
        x, self._buffer_z1 = self.generate(n1_ - buffer_n1, self._buffer_z1, **kwargs)

        if buffer_n:
            # append new samples to the existing
            self._buffer = np.concatenate([self._buffer, x], axis=0)
            n0 = n0 - self._buffer_n0
            x = self._buffer[n0_:n1_, ...]
        else:
            # buffer all the samples and return them all
            self._buffer = x
            self._buffer_n0 = n0

        return x

    # @property
    # def differentiable(self) -> bool:
    #     return False

    # @property
    # def integrable(self) -> bool:
    #     return False

    # @property
    # def order_of_vanishing(self) -> int | None:
    #     return None
