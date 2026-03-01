from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..__util import format_parameter
from .._backend import PyRunnerBase
from ..constants import vt_atten as atten_default
from ..function_generators.abc import SampleGenerator
from .abc import Element, VocalTract


class LossyCylinderVocalTract(VocalTract):
    """Lossy Straight Tube Vocal Tract model"""

    _atten: float = atten_default  #: flow attenuation factor/half unit section
    _area: SampleGenerator  #:cross-sectional area of the tube
    _nb_sections: int  #: number of tubelets

    class Runner(PyRunnerBase):
        n: int  # number of samles
        m: int  # number of sections
        gain: NDArray  # total propagation gain
        s: np.ndarray

        def __init__(
            self, nb_steps: int, states0: NDArray, nb_sections: int, gain: NDArray
        ):
            super().__init__()

            self.n = nb_steps
            self.m = nb_sections
            self.gain = gain
            self.s = states0.reshape(-1, 2)

        def step(self, i: int, f1: float, bK: float) -> tuple[float, float]:
            """time update

            Parameters
            ----------
            f1
                forward pressure input
            bK
                backward pressure input

            Returns
            -------
                fK
                    forward pressure output
                b1
                    backward pressure output

            """

            s = self.s
            j = i % self.m
            fK, b1 = self.gain * s[j, :]
            s[j, :] = (f1, bK)
            return fK, b1

    def __init__(
        self,
        desired_length: float,
        area: float | ArrayLike | SampleGenerator,
        atten: float | None = None,
    ):
        """Ideal damped cylindrical vocal tract model

        Parameters
        ----------
        desired_length
            Desired total tube length in cm. The actual length is rounded to the nearest integer multiple of
            sound propagation distance per time sample interval.
        area
            cross-sectional area in cm²
        atten, optional
            loss factor per half unit length, by default None
        """

        self._nb_sections = round(desired_length / (2 * self.dz))
        self._area = format_parameter(area, 0)
        if atten is not None:
            self._atten = float(atten)

    @property
    def nb_sections(self) -> int:
        """number of tube sections"""
        return self._nb_sections

    @property
    def dz(self) -> float:
        """thickness of each cross-section in cm

        During the period of one time-sample, sound propagates over 2 cross-sections
        """
        return self.c / (2 * self.fs)

    @property
    def total_length(self) -> float:
        """total length of vocal tract"""
        return self.nb_sections * self.dz

    def generate_sim_params(self, n: int, n0: int = 0, **_) -> tuple[NDArray, ...]:

        gain = (1 - self._atten) ** (2 * self._nb_sections)

        return self._nb_sections, gain

    @property
    def nb_states(self) -> int:
        """number of states"""
        # final output unit-delays are not considered internal
        return 2 * self._nb_sections

    def get_area(self, n: int | None = None, n0: int = 0) -> NDArray:
        """get cross-sectional areas of the first and last tubes

        Parameters
        ----------
        n, optional
            number of samples, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            If vocal tract is not dynamic and n is None, 2-element 1D array containing
            [first, last] area measures in cm². If n is specified, 2D array with the first
            dimension being the time axis. Note that if the tract is not dynamic, the time
            axis will have only one element.
        """
        area = self._area

        if not area.is_fixed:
            if n is None:
                raise ValueError("Must specify n (and n0) for a dynamic lips")

        return area(n, n0)

    @property
    def input_area_is_fixed(self) -> bool:
        """True if input section is fixed"""
        return self._area.is_fixed

    @property
    def output_area_is_fixed(self) -> bool:
        """True if output section is fixed"""
        return self._area.is_fixed

    def input_area(self, n: int | None = None, n0: int = 0) -> NDArray:
        """get cross-sectional areas of the first tubes

        Parameters
        ----------
        n, optional
            number of samples, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            If vocal tract is not dynamic and n is None, 2-element 1D array containing
            first area measures in cm². If n is specified, 2D array with the first
            dimension being the time axis. Note that if the tract is not dynamic, the time
            axis will have only one element.
        """

        return self.get_area(n, n0)

    def output_area(self, n: int | None = None, n0: int = 0) -> NDArray:
        """get cross-sectional areas of the first tubes

        Parameters
        ----------
        n, optional
            number of samples, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            If vocal tract is not dynamic and n is None, 2-element 1D array containing
            last area measures in cm². If n is specified, 2D array with the first
            dimension being the time axis. Note that if the tract is not dynamic, the time
            axis will have only one element.
        """

        return self.get_area(n, n0)

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n"]

    @dataclass
    class Results(Element.Results):
        final_states: NDArray
        propagation_gain: NDArray

    # override result class
    _ResultsClass = Results

    def create_result(
        self,
        runner: LossyCylinderVocalTract.Runner,
        *extra_items,
        n0: int = 0,
    ) -> Element.Results:
        """Creates simulation result object"""

        final_states = runner.s.reshape(-1)
        gain = runner.gain

        return super().create_result(runner, final_states, gain, *extra_items, n0=n0)
