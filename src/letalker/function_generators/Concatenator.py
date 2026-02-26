from __future__ import annotations

from numpy.typing import NDArray
from collections.abc import Sequence

import numpy as np

from .abc import FunctionGenerator
from . import _functions as _f


class Concatenator(FunctionGenerator):
    """Sequencing function generators"""

    _generators: list[FunctionGenerator]

    def __init__(
        self,
        *generators: FunctionGenerator | float,
        transition_times: Sequence[float] | float,
        **kwargs,
    ):
        """Chain multiple generator to execute in a chain

        g_0(t)
        g_1(t-t0)
        g_2(t-t1)
        ...
        g_k(t-t_{k-1})

        """

        super().__init__(**kwargs)

        ngens = len(generators)

        if not ngens:
            raise ValueError("At least one generator must be specified")

        shape0 = generators[0].shape
        if not all(g.shape == shape0 for g in generators):
            raise ValueError(
                "All generators must have the same number of dimensions and shape."
            )

        t_switch = np.atleast_1d(transition_times)
        dt_switch = np.diff(t_switch)
        if t_switch.shape != (ngens - 1,):
            raise ValueError(f"Transition_times must specify {ngen - 1} time points.")
        if np.any(dt_switch <= 0):
            raise ValueError(
                "Transition_times must be strictly monotonically increasing."
            )

        self._generators = tuple(generators)
        self._nswitch = np.round(t_switch * self.fs).astype(int)

        # evaluate the integral
        for g in zip(generators[:-1], (0, *t_switch), np.diff(t_switch)):
            g.antiderivative()

    @property
    def shape(self) -> tuple[int]:
        return max(self._generators, key=lambda g: g.ndim).shape

    @property
    def ndim(self) -> int:
        return max(g.ndim for g in self._generators)

    def __call__(
        self,
        n: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
        **kwargs,
    ) -> NDArray:

        return self.bias + _f.sum(
            self._generators,
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

        return _f.sum_derivative(
            self._generators, n, n0, sample_edges, upsample_factor, nu=nu, **kwargs
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

        return _f.sum_antiderivative(
            self._generators,
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

    #     return any(g.order_of_vanishing for g in self._generators)
