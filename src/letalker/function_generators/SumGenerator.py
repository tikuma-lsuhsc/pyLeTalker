from __future__ import annotations

from typing import cast
from numpy.typing import NDArray

from .abc import FunctionGenerator
from .Constant import Constant
from . import _functions as _f


class SumGenerator(FunctionGenerator):
    """Combine function generator outputs by summation"""

    generators: list[FunctionGenerator]
    bias: float

    def __init__(
        self,
        *generators: FunctionGenerator | float,
        **kwargs,
    ):
        """SumGenerator"""

        super().__init__(**kwargs)
        self.generators = []
        self.bias = 0.0

        for g in generators:
            if g.is_fixed:
                # must be Constant object
                self.bias *= cast(Constant, g).level
            elif isinstance(g, FunctionGenerator):
                self.generators.append(g)
            else:
                self.bias *= g

    @property
    def shape(self) -> tuple[int]:
        """shape of each sample (empty if scalar)"""
        return max(self.generators, key=lambda g: g.ndim).shape

    @property
    def ndim(self) -> int:
        """number of dimensions of each sample (0 if scalar)"""
        return max(g.ndim for g in self.generators)

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

        return self.bias + _f.sum(
            self.generators,
            n,
            n0,
            sample_edges,
            upsample_factor,
            bias=self.bias,
            **kwargs,
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
        """compute the derivative of the sum

        Parameters
        ----------
        n
            _description_
        n0, optional
            _description_, by default 0
        sample_edges, optional
            _description_, by default False
        upsample_factor, optional
            _description_, by default 0
        nu, optional
            _description_, by default 1

        Returns
        -------
            _description_
        """
        return _f.sum_derivative(
            self.generators, n, n0, sample_edges, upsample_factor, nu=nu, **kwargs
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
        """compute the anti-derivative of the sum

        Parameters
        ----------
        n
            _description_
        n0, optional
            _description_, by default 0
        sample_edges, optional
            _description_, by default False
        upsample_factor, optional
            _description_, by default 0
        nu, optional
            _description_, by default 1

        Returns
        -------
            _description_
        """
        return _f.sum_antiderivative(
            self.generators,
            n,
            n0,
            sample_edges,
            upsample_factor,
            nu=nu,
            bias=self.bias,
            **kwargs,
        )

    # @property
    # def order_of_vanishing(self) -> int | None:

    #     return any(g.order_of_vanishing for g in self.generators)
