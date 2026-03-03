from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._backend import LeTalkerAspirationNoiseRunner
from ..constants import noise_bpass, noise_dist, noise_psd_level, noise_REc
from ..function_generators import ColoredNoiseGenerator
from ..function_generators.abc import NoiseGenerator
from .abc import AspirationNoise, Element


class LeTalkerAspirationNoise(AspirationNoise):
    REc: float = noise_REc

    _default_noise_source_args = noise_bpass
    noise_source: NoiseGenerator = ColoredNoiseGenerator(
        *_default_noise_source_args,
        psd_level=noise_psd_level,
        innovation_distribution=noise_dist,
        analog=False,
    )
    # psd_level = 1e-10 / default_fs
    # psd_level = (4e-6 / 18.826) ** 2 * 12 / default_fs * 1000 / 5
    # psd_level = 0.000004 # original

    @dataclass
    class Results(Element.Results):
        src_noise: NDArray
        re2b: NDArray
        re2: NDArray
        unoise: NDArray

        @property
        def re(self) -> NDArray:
            return self.re2**0.5

        @property
        def re_b(self) -> NDArray:
            return self.re2b**0.5

    # override result class
    _ResultsClass = Results

    Runner = LeTalkerAspirationNoiseRunner

    def __init__(
        self,
        noise_source: NoiseGenerator | float | dict | None = None,
        REc: float | None = None,
    ):
        """_summary_

        Parameters
        ----------
        noise_source, optional
            _description_, by default None
        REc, optional
            critical Reynolds number, by default None
        """
        super().__init__()

        if REc is not None:
            self.REc = float(REc)

        if noise_source is not None:
            self.noise_source = (
                noise_source
                if isinstance(noise_source, NoiseGenerator)
                else (
                    ColoredNoiseGenerator(**noise_source)
                    if isinstance(noise_source, dict)
                    else ColoredNoiseGenerator(
                        *self._default_noise_source_args, psd_level=float(noise_source)
                    )
                )
            )

    @property
    def need_length(self) -> bool:
        """True if vocal fold length is necessary to generate aspiration noise"""
        return True

    def generate_noise(self, nb_samples: int, n0: int = 0) -> NDArray:
        """Generate noise (new data)

        Parameters
        ----------
        nb_samples
            number of samples to generate
        n0, optional
            starting time index, by default 0

        Returns
        -------
            array of nb_sample-element noise sample array
        """
        return self.noise_source(nb_samples, n0)

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n", "nf", "re2b", "re2", "ug_noise"]

    def generate_sim_params(
        self, n: int, n0: int = 0, *, length: NDArray
    ) -> tuple[NDArray, ...]:
        """Generate element parameters (possibly time varying)"""

        nuL_inv = 1 / (self.nu * length)
        nf = self.generate_noise(n, n0)
        RE2b = self.REc**2  # pre-square to save computation

        return np.atleast_1d(nuL_inv), np.atleast_1d(nf), RE2b
