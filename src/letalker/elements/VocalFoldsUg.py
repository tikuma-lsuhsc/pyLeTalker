from __future__ import annotations

from typing import cast
from numpy.typing import NDArray, ArrayLike

from ..function_generators.abc import SampleGenerator

from functools import cached_property
from dataclasses import dataclass
import numpy as np

from .abc import Element, BlockRunner, AspirationNoise

from ..__util import format_parameter
from .VocalFoldsBase import VocalFoldsBase
from .abc import VocalTract
from ..core import has_numba

if has_numba:
    import numba as nb


class VocalFoldsUg(VocalFoldsBase):
    """Vocal source model with known glottal flow function"""

    RunnerSpec = (
        [
            ("n", nb.int64),
            ("ug", nb.float64[:]),
            ("rhoca_sg", nb.float64[:]),
            ("rhoca_eplx", nb.float64[:]),
            ("psg", nb.float64[:]),
            ("peplx", nb.float64[:]),
        ]
        if has_numba
        else []
    )

    class Runner:
        n: int
        _ug: np.ndarray
        rhoca_sg: np.ndarray
        rhoca_eplx: np.ndarray
        psg: np.ndarray
        peplx: np.ndarray

        def __init__(
            self,
            nb_steps: int,
            s_in: NDArray,
            anoise: BlockRunner,
            ug: NDArray,
            rhoca_sg: NDArray,
            rhoca_eplx: NDArray,
        ):
            self.n = nb_steps
            self._ug = ug
            self.rhoca_sg = rhoca_sg
            self.rhoca_eplx = rhoca_eplx
            self.psg = np.empty(nb_steps)
            self.peplx = np.empty(nb_steps)
            self._anoise = anoise

        def step(self, i: int, fsg: float, beplx: float) -> tuple[float, float]:

            ug = self._ug[i if i < self._ug.shape[0] else -1]
            rhoca_eplx = self.rhoca_eplx[i if i < self.rhoca_eplx.shape[0] else -1]
            rhoca_sg = self.rhoca_sg[i if i < self.rhoca_sg.shape[0] else -1]

            ug += self._anoise.step(i, ug)

            # compute the forward pressure in epilarnx
            feplx = beplx + ug * rhoca_eplx

            # compute the backward pressure in subglottis
            bsg = fsg - ug * rhoca_sg

            self.psg[i] = fsg + bsg
            self.peplx[i] = feplx + beplx

            return feplx, bsg

    def __init__(
        self,
        ug: ArrayLike | SampleGenerator,
        *,
        length: float | NDArray | SampleGenerator | None = None,
        upstream: VocalTract | float | None = None,
        downstream: VocalTract | float | None = None,
        aspiration_noise: bool | dict | AspirationNoise | None = None,
    ):

        super().__init__(upstream, downstream, aspiration_noise)

        self._ug = format_parameter(ug, 0)
        self._length = format_parameter(length, 0, optional=True)

    @property
    def known_length(self) -> bool:
        """True if vocal fold length is necessary to generate aspiration noise"""
        return self._length is not None

    def length(self, n: int | None = None, n0: int = 0) -> NDArray | None:
        return self._length and self._length(n, n0)

    @property
    def _nb_states(self) -> int:
        """number of states, excluding aspiration model"""
        return 0

    @property
    def runner_info(self) -> tuple[type, list[tuple[str, type]]]:

        return VocalFoldsUg.Runner, VocalFoldsUg.RunnerSpec

    def generate_sim_params(self, n: int, n0: int = 0) -> tuple[NDArray, ...]:
        """Generate parameter arrays"""

        ug = self._ug(n, n0)
        rhoca_eplx, rhoca_sg = self.configure_resistances(n, n0)

        return ug, rhoca_eplx, rhoca_sg

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n", "_ug", "psg", "peplx"]

    @dataclass
    class Results(Element.Results):
        _ug: NDArray
        psg: NDArray
        peplx: NDArray
        aspiration_noise: Element.Results | None

        @cached_property
        def ug(self) -> NDArray:
            ug = self._ug
            if self.aspiration_noise:
                ug = ug + self.aspiration_noise.unoise
            return ug

    _ResultsClass = Results  # override result class
