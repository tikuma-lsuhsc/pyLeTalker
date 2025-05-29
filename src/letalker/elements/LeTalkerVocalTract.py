from __future__ import annotations
from typing import cast
from numpy.typing import ArrayLike, NDArray

from dataclasses import dataclass
import numpy as np

from ..constants import (
    vt_atten as atten_default,
    vocaltract_areas,
    fs as default_fs,
    TwoLetterVowelLiteral,
)
from .abc import Element, VocalTract
from ..function_generators.abc import SampleGenerator
from ..__util import format_parameter

from ..core import has_numba

if has_numba:
    import numba as nb


class LeTalkerVocalTract(VocalTract):
    """Wave-reflection vocal tract model (Liljencrants, 1985; Story, 1995)"""

    _atten: float = atten_default
    _areas: SampleGenerator
    log_sections: bool = False
    _nb_states: int

    RunnerSpec = (
        [
            ("n", nb.int64),
            ("alph_odd", nb.float64[:, :]),
            ("alph_even", nb.float64[:, :]),
            ("r_odd", nb.float64[:, :]),
            ("r_even", nb.float64[:, :]),
            ("s", nb.float64[:]),
            ("p_sections", nb.float64[:, :, :]),
        ]
        if has_numba
        else []
    )

    class Runner:
        n: int
        alph_odd: NDArray  # odd-section propagation gains
        alph_even: NDArray  # even-section propagation gains
        r_odd: NDArray  # odd-junction reflection coefficients
        r_even: NDArray  # even-junction reflection coefficients
        s: np.ndarray

        def __init__(
            self,
            nb_steps: int,
            s: NDArray,
            alph_odd: NDArray,
            alph_even: NDArray,
            r_odd: NDArray,
            r_even: NDArray,
        ):
            self.n = nb_steps
            self.alph_odd = alph_odd
            self.alph_even = alph_even
            self.r_odd = r_odd
            self.r_even = r_even
            self.s = s

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

            alph_odd = self.alph_odd[i if i < self.alph_odd.shape[0] else -1]
            alph_even = self.alph_even[i if i < self.alph_even.shape[0] else -1]
            r_odd = self.r_odd[i if i < self.r_odd.shape[0] else -1]
            r_even = self.r_even[i if i < self.r_even.shape[0] else -1]
            s = self.s

            ns = r_even.shape[0]
            f_even = s[:ns]
            b_odd = s[ns:]

            f_odd = np.empty_like(alph_odd)
            b_even = np.empty_like(alph_even)

            # ---even junctions [(F2,B3),(F4,B5),...,(FK-2,BK-1)]->[(F3,B2),(F5,B4),...]--- */
            Psi = (f_even - b_odd) * r_even
            f_odd[0] = f1
            f_odd[1:] = f_even + Psi
            b_even[:-1] = b_odd + Psi
            b_even[-1] = bK

            f_odd *= alph_odd
            b_even *= alph_even

            # ---odd junctions [(F1,B2),(F3,B4),...]->[(F2,B1),(F4,B3),...]--- */
            Psi = (f_odd - b_even) * r_odd  # [F3,F5,...] - [B2, B4, ...]
            f_odd += Psi
            b_even += Psi

            # --- attenuate pressure to "pre-"propagate signal to the output of the section
            f_odd *= alph_even
            b_even *= alph_odd

            s[:ns] = f_odd[:-1]
            s[ns:] = b_even[1:]

            return f_odd[-1], b_even[0]

    class LoggedRunner:
        n: int
        alph_odd: NDArray  # odd-section propagation gains
        alph_even: NDArray  # even-section propagation gains
        r_odd: NDArray  # odd-junction reflection coefficients
        r_even: NDArray  # even-junction reflection coefficients
        s: np.ndarray
        p_sections: np.ndarray  # internal pressure (columns: forward/backward)

        def __init__(
            self,
            nb_steps: int,
            s: NDArray,
            alph_odd: NDArray,
            alph_even: NDArray,
            r_odd: NDArray,
            r_even: NDArray,
        ):
            self.n = nb_steps
            self.alph_odd = alph_odd
            self.alph_even = alph_even
            self.r_odd = r_odd
            self.r_even = r_even
            self.s = s

            n_sections = r_odd.shape[-1] + r_even.shape[-1]
            self.p_sections = np.empty((nb_steps, 2, n_sections))

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

            alph_odd = self.alph_odd[i if i < self.alph_odd.shape[0] else -1]
            alph_even = self.alph_even[i if i < self.alph_even.shape[0] else -1]
            r_odd = self.r_odd[i if i < self.r_odd.shape[0] else -1]
            r_even = self.r_even[i if i < self.r_even.shape[0] else -1]
            s = self.s

            ns = r_even.shape[0]
            f_even = s[:ns]
            b_odd = s[ns:]

            f_odd = np.empty_like(alph_odd)
            b_even = np.empty_like(alph_even)

            # ---even junctions [(F2,B3),(F4,B5),...,(FK-2,BK-1)]->[(F3,B2),(F5,B4),...]--- */
            Psi = (f_even - b_odd) * r_even
            f_odd[0] = f1
            f_odd[1:] = f_even + Psi
            b_even[:-1] = b_odd + Psi
            b_even[-1] = bK

            self.p_sections[i, 0, 1::2] = f_even
            self.p_sections[i, 1, 1::2] = b_even[:-1]

            f_odd *= alph_odd  # f_even input to odd junction
            b_even *= alph_even  # b_even input to odd junction

            # ---odd junctions [(F1,B2),(F3,B4),...]->[(F2,B1),(F4,B3),...]--- */
            Psi = (f_odd - b_even) * r_odd  # [F3,F5,...] - [B2, B4, ...]
            f_odd += Psi
            b_even += Psi

            # grab odd-stage output forward pressure
            self.p_sections[i, 0, ::2] = f_odd

            # --- attenuate pressure to "pre-"propagate signal to the output of the section
            f_odd *= alph_even  # misnamed, f_even next state
            b_even *= alph_odd  # misnamed, b_odd next state

            self.p_sections[i, 1, ::2] = b_even  # odd-stage output backward pressure

            s[:ns] = f_odd[:-1]
            s[ns:] = b_even[1:]

            return f_odd[-1], b_even[0]

    def __init__(
        self,
        areas: ArrayLike | TwoLetterVowelLiteral | SampleGenerator,
        atten: float | None = None,
        log_sections: bool = False,
        min_areas: float | None = None,
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
        if isinstance(areas, str):
            areas = vocaltract_areas[areas]
            if self.fs != default_fs:
                raise ValueError(
                    f"Incorrect {self.fs=} to use default vocal tract area."
                )

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

    @property
    def dz(self) -> float:
        """thickness of each cross-section in cm

        During the period of one time-sample, sound propagates over 2 cross-sections
        """
        return self.c / (2 * self.fs)

    @property
    def z(self) -> NDArray:
        """tube position vector in cm relative to glottis"""
        return np.arange(self.nb_sections) * self.dz

    @property
    def total_length(self) -> float:
        """total length of vocal tract"""
        return self.nb_sections * self.dz

    @property
    def runner_info(self) -> tuple[type, list[tuple[str, type]]]:
        Spec = LeTalkerVocalTract.RunnerSpec
        return (
            LeTalkerVocalTract.LoggedRunner
            if self.log_sections
            else LeTalkerVocalTract.Runner
        ), (Spec if self.log_sections else Spec[:-1])

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n", "s"]

    def generate_sim_params(self, n: int, n0: int = 0, **_) -> tuple[NDArray, ...]:

        areas = self.areas(n, n0)

        alpha = 1 - self._atten / areas**0.5
        r = (areas[..., :-1] - areas[..., 1:]) / (areas[..., :-1] + areas[..., 1:])

        return alpha[..., ::2], alpha[..., 1::2], r[..., ::2], r[..., 1::2]

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

            element = cast(LeTalkerVocalTract, self.element)
            areas = element.areas(self.n1 - self.n0, self.n0)[:, :-1]
            dia = 2 * (areas / np.pi) ** 0.5
            return dia * self.uout_sections / (element.nu() * areas)
            # RE2 = (ug * nu_inv / L) ** 2

        @property
        def areas(self) -> NDArray:
            """cross-sectional areas of the tube sections"""

            element = cast(LeTalkerVocalTract, self.element)
            return element.areas(self.n1 - self.n0, self.n0)

        @property
        def dx(self) -> float:
            """lengh of each tube section in cm"""
            return cast(LeTalkerVocalTract, self.element).dz

    # override result class
    _ResultsClass = Results

    def create_result(
        self,
        runner: LeTalkerVocalTract.Runner | LeTalkerVocalTract.LoggedRunner,
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
