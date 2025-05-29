from __future__ import annotations

from typing import Literal
from numpy.typing import NDArray

from math import sqrt
import numpy as np

from .abc import NoiseGenerator

#: supported distributions of the innovation process
WhiteNoiseDistributionLiteral = Literal["gaussian", "two_point", "uniform"]


class WhiteNoiseGenerator(NoiseGenerator):
    """White noise generator"""

    #: peak psd level (i.e., variance of the innovation process)
    psd_level: float

    #: distribution of the innovation process
    innovation_distribution: WhiteNoiseDistributionLiteral

    def __init__(
        self,
        psd_level: float | None = None,
        innovation_distribution: WhiteNoiseDistributionLiteral = "gaussian",
    ):
        """White noise generator

        Parameters
        ----------

        psd_level, optional
            PSD peak level in 1/Hz (default: 1)
        innovation_distribution, optional
            distribution of innovation process with zero mean and unit variance:
                "gaussian"   Gaussian (normal) distribution
                "uniform"    Uniform distribution
                "two_point"  Two-point (±1) distribution
        """

        if psd_level is None:
            psd_level = 1.0

        super().__init__()

        self._sig = sqrt(psd_level) if psd_level > 0 and psd_level != 1.0 else None

        self._inno = {
            "gaussian": np.random.randn,
            "two-point": lambda n: np.array([-1, 1])[np.random.randint(0, 2, n)],
            "uniform": lambda n: np.sqrt(12) * np.random.rand(n),
        }[innovation_distribution]

        self._last = None

    @property
    def shape(self) -> tuple[int]:
        """shape of each sample (empty if scalar)"""
        return ()

    @property
    def ndim(self) -> int:
        """number of dimensions of each sample (0 if scalar)"""
        return 1

    def generate(self, n: int, z0: None = None, **_) -> tuple[NDArray, None]:
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

        x = self._inno(n)
        if self._sig:
            x *= self._sig

        return x, z0
