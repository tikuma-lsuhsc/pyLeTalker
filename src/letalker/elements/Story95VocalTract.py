from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..__util import format_parameter
from .._backend import Story95VocalTractRunner
from ..constants import (
    vt_atten as atten_default,
)
from ..core import classproperty
from ..function_generators.abc import SampleGenerator
from .abc import Element, VocalTract


class Story95VocalTract(VocalTract):
    """Wave-reflection vocal tract model (Liljencrants, 1985; Story, 1995)"""

    _atten: float = atten_default
    _areas: SampleGenerator
    log_sections: bool = False
    _nb_states: int

    Runner = Story95VocalTractRunner

    def __init__(
        self,
        areas: ArrayLike | SampleGenerator,
        M: float | None = None,
        K: float | None = None,
        B: float | None = None,
        omega_v: float | None = None,
        alpha: float | None = None,
        log_sections: bool = False,
    ):
        """Wave-reflection vocal tract model

        :param areas: cross-sectional areas of vocal tract segments
        :param atten: per-section attenuation factor, defaults to None
        :param log_sections: `True` to log the incidental pressure at every vocal tract segment, defaults to False
        :param min_areas: minimum allowable cross-sectional area to enforce positive area, defaults to None
        """
        # """Wave-reflection vocal tract model

        # The wave-reflection vocal tract model represents the vocal tract as a
        # series of short lossy tube segments. Each segment has a fixed length
        # :math:`c/fs/2` (by default, 0385 cm).

        # Parameters
        # ----------
        # areas
        #     cross-sectional areas of vocal tract segments.
        # atten, optional
        #     per-section attenuation factor, by default `None`
        # log_sections, optional
        #     `True` to log the incidental pressure at every vocal tract segment,
        #     by default `False`
        # min_areas, optional
        #     minimum allowable cross-sectional area to enforce positive area,
        #     by default `None` to use `1e-6`.

        # """

        self._areas = format_parameter(areas, 1)
        self._min_areas = min_areas or 1e-6

        if self._areas.shape[-1] % 2:
            raise ValueError("Tube must have an even number of cross-sectional areas.")

        if atten is not None:
            self._atten = atten

        if log_sections:
            self.log_sections = True

    @property
    def nb_sections(self) -> int:
        """number of tube sections"""
        return self._areas.shape[-1]

    @classproperty
    def dz(cls) -> float:
        """thickness of each cross-section in cm

        During the period of one time-sample, sound propagates over 2 cross-sections
        """
        return cls.c / (2 * cls.fs)

    @property
    def z(self) -> NDArray:
        """tube position vector in cm relative to glottis"""
        return np.arange(self.nb_sections) * self.dz

    @property
    def total_length(self) -> float:
        """total length of vocal tract"""
        return self.nb_sections * self.dz

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n", "sout"]

    def generate_sim_params(self, n: int, n0: int = 0, **_) -> tuple[NDArray, ...]:

        areas = self.areas(n, n0)

        alpha = 1 - self._atten / areas**0.5
        r = (areas[..., :-1] - areas[..., 1:]) / (areas[..., :-1] + areas[..., 1:])

        return (
            alpha[..., ::2],
            alpha[..., 1::2],
            r[..., ::2],
            r[..., 1::2],
            self.log_sections,
        )

    @property
    def nb_states(self) -> int:
        """number of states"""
        # final output unit-delays are not considered internal
        M = self.nb_sections // 2
        return 2 * (M - 1)

    @property
    def input_area_is_fixed(self) -> bool:
        """True if input section is fixed"""
        return self._areas.is_fixed

    @property
    def output_area_is_fixed(self) -> bool:
        """True if output section is fixed"""
        return self._areas.is_fixed

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

        areas = self.areas(n or 1, n0)
        return areas[0, 0] if n is None else areas[:, 0]

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

        areas = self.areas(n or 1, n0)
        return areas[0, -1] if n is None else areas[:, -1]

    def areas(self, n: int | None = None, n0: int = 0) -> NDArray:
        """get cross-sectional areas of the tube sections

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
        areas = self._areas
        min_aras = self._min_areas

        if not self._areas.is_fixed and n is None:
            raise ValueError("Must specify n (and n0) for a dynamic lips")

        return np.maximum(areas(n, n0), min_aras)  # guarantee non-negativity

    @dataclass
    class Results(Element.Results):
        final_states: NDArray
        propagation_gains: NDArray
        reflection_coefficients: NDArray
        pout_sections: NDArray
        uout_sections: NDArray

        @property
        def re_sections(self) -> NDArray | None:
            """Reynolds numbers of sections"""
            if self.uout_sections is None:
                return None

            element = cast(Story95VocalTract, self.element)
            areas = element.areas(self.n1 - self.n0, self.n0)[:, :-1]
            dia = 2 * (areas / np.pi) ** 0.5
            return dia * self.uout_sections / (element.nu * areas)
            # RE2 = (ug * nu_inv / L) ** 2

        @property
        def areas(self) -> NDArray:
            """cross-sectional areas of the tube sections"""

            element = cast(Story95VocalTract, self.element)
            return element.areas(self.n1 - self.n0, self.n0)

        @property
        def dx(self) -> float:
            """lengh of each tube section in cm"""
            return cast(Story95VocalTract, self.element).dz

    # override result class
    _ResultsClass = Results

    def create_result(
        self,
        runner: Story95VocalTractRunner,
        *extra_items,
        n0: int = 0,
    ) -> Element.Results:
        """Creates simulation result object"""

        alph_odd = runner.alph_odd
        alph_even = runner.alph_even
        r_odd = runner.r_odd
        r_even = runner.r_even
        shape = list(alph_odd.shape)
        shape[-1] += alph_even.shape[-1]
        alpha = np.empty(shape)
        alpha[..., ::2] = alph_odd
        alpha[..., 1::2] = alph_even
        if shape[0] == 1:
            alpha = alpha[0, :]

        shape = list(r_odd.shape)
        shape[-1] += r_even.shape[-1]
        r = np.empty(shape)
        r[..., ::2] = r_odd
        r[..., 1::2] = r_even
        if shape[0] == 1:
            r = r[0, :]

        if self.log_sections:
            outcomes = runner.p_sections
            pout_sections = outcomes.sum(axis=1)
            area = self.areas(runner.n, n0)[:, :-1]
            uout_sections = (
                area / (self.rho * self.c) * (outcomes[:, 0, :] - outcomes[:, 1, :])
            )
            items = alpha, r, pout_sections, uout_sections
        else:
            items = alpha, r, None, None

        return super().create_result(runner, *items, *extra_items, n0=n0)
