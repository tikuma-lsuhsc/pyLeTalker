from __future__ import annotations

from numpy.typing import NDArray

import numpy as np

from .abc import FunctionGenerator
from . import _functions as _f


class ProductGenerator(FunctionGenerator):
    """Combines function generator outputs by multiplication"""

    _generators: list[FunctionGenerator]
    _gain: float

    def __init__(
        self,
        *generators: FunctionGenerator | float,
        **kwargs,
    ):
        """ProductGenerator"""

        from .Constant import Constant

        super().__init__(**kwargs)
        self._generators = []
        self._gain = 1.0

        for g in generators:
            if isinstance(g, Constant):
                self._gain *= g.level
            elif isinstance(g, FunctionGenerator):
                self._generators.append(g)
            else:
                self._gain *= g

    @property
    def shape(self) -> tuple[int]:
        """shape of each sample (empty if scalar)"""
        return max(self._generators, key=lambda g: g.ndim).shape

    @property
    def ndim(self) -> int:
        """number of dimensions of each sample (0 if scalar)"""
        return max(g.ndim for g in self._generators)

    def __mul__(self, other: FunctionGenerator | float):
        if isinstance(other, ProductGenerator):
            return ProductGenerator(
                *self._generators, self._gain, *other._generators, other._gain
            )
        if isinstance(other, FunctionGenerator):
            return ProductGenerator(*self._generators, self._gain, other)
        try:
            return ProductGenerator(*self._generators, self._gain, float(other))
        except TypeError as e:
            return NotImplemented

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

        return self._gain * _f.product(
            self._generators, n, n0, sample_edges, upsample_factor, **kwargs
        )

    def derivative(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        nu: int = 1,
        **kwargs,
    ) -> NDArray:

        return _f.product_derivative(
            self._generators,
            n,
            n0,
            sample_edges,
            upsample_factor,
            nu=nu,
            gain=self._gain,
            **kwargs,
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

        return self._gain * _f.product_antiderivative(
            self._generators, n, n0, sample_edges, upsample_factor, nu=nu, **kwargs
        )

    # @property
    # def order_of_vanishing(self) -> int | None:

    #     raise NotImplementedError(
    #         "order_of_vanishing property is not currently implemented"
    #     )
