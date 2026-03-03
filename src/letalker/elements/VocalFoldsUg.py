from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from numpy.typing import ArrayLike, NDArray

from ..__util import format_parameter
from .._backend import VocalFoldsUgRunner
from ..function_generators.abc import SampleGenerator
from .abc import AspirationNoise, Element, VocalTract
from .VocalFoldsBase import VocalFoldsBase


class VocalFoldsUg(VocalFoldsBase):
    """Vocal source model with known glottal flow function"""

    Runner = VocalFoldsUgRunner

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
