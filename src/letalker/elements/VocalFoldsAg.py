from __future__ import annotations
from numpy.typing import NDArray, ArrayLike
from typing import cast
from collections.abc import Callable

import abc
from dataclasses import dataclass
from functools import cached_property
from math import pi
import numpy as np

from ..function_generators.abc import SampleGenerator

from ..__util import format_parameter
from .abc import Element, AspirationNoise, BlockRunner, VocalTract
from .VocalFoldsBase import VocalFoldsBase

from ..core import has_numba
from ..function_generators.utils import align_signals

if has_numba:
    import numba as nb


class VocalFoldsAgBase(VocalFoldsBase, metaclass=abc.ABCMeta):

    RunnerSpec = (
        [
            ("n", nb.int64),
            ("R", nb.float64[:]),
            ("Qa", nb.float64[:]),
            ("a", nb.float64[:]),
            ("rhoca_sg", nb.float64[:]),
            ("rhoca_eplx", nb.float64[:]),
            ("ug", nb.float64[:]),
            ("psg", nb.float64[:]),
            ("peplx", nb.float64[:]),
        ]
        if has_numba
        else []
    )

    class Runner:
        n: int
        R: np.ndarray
        Qa: np.ndarray
        a: np.ndarray
        rhoca_sg: np.ndarray
        rhoca_eplx: np.ndarray
        ug: np.ndarray
        psg: np.ndarray
        peplx: np.ndarray

        def __init__(
            self,
            nb_steps: int,
            s_in: NDArray,
            anoise: BlockRunner,
            R: NDArray,
            Qa: NDArray,
            a: NDArray,
            rhoca_sg: NDArray,
            rhoca_eplx: NDArray,
        ):
            self.n = nb_steps
            self.R = R
            self.Qa = Qa
            self.a = a
            self.rhoca_sg = rhoca_sg
            self.rhoca_eplx = rhoca_eplx
            self.ug = np.empty(nb_steps)
            self.psg = np.empty(nb_steps)
            self.peplx = np.empty(nb_steps)
            self._anoise = anoise

        def step(self, i: int, fsg: float, beplx: float) -> tuple[float, float]:

            R = self.R[i if i < self.R.shape[0] else -1]
            Qa = self.Qa[i if i < self.Qa.shape[0] else -1]
            a = self.a[i if i < self.a.shape[0] else -1]
            rhoca_eplx = self.rhoca_eplx[i if i < self.rhoca_eplx.shape[0] else -1]
            rhoca_sg = self.rhoca_sg[i if i < self.rhoca_sg.shape[0] else -1]

            # compute the glottal flow
            Q = Qa * (fsg - beplx)
            if fsg < beplx:
                # depends on whether ug is positive or negative
                a = -a
                Q = -Q
            ug = a * ((R**2 + Q) ** 0.5 - R)

            ug += self._anoise.step(i, ug)

            self.ug[i] = ug

            # compute the forward pressure in epilarynx
            feplx = beplx + ug * rhoca_eplx

            # compute the backward pressure in subglottis
            bsg = fsg - ug * rhoca_sg

            self.psg[i] = fsg + bsg
            self.peplx[i] = feplx + beplx

            return feplx, bsg

    @property
    @abc.abstractmethod
    def known_length(self) -> bool:
        """True if model specifies the vibrating vocal fold length"""

    @abc.abstractmethod
    def length(
        self, nb_samples: int | None = None, n0: int = 0
    ) -> NDArray | float | None:
        """glottal length

        Parameters
        ----------
        nb_samples, optional
            number of samples, required if dynamic, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            glottal length in cm if specified. 1D array if fo or L0 is dynamic otherwise a scalar value.
        """

    @abc.abstractmethod
    def glottal_area(self, nb_samples: int, n0: int = 0) -> NDArray:
        """glottal area (minimum opening along the depth)

        :param nb_samples: _description_
        :param n0: starting sample time, defaults to 0
        :return: glottal area samples
        """

    @property
    def runner_info(self) -> tuple[type, list[tuple[str, type]]]:

        return VocalFoldsAgBase.Runner, VocalFoldsAgBase.RunnerSpec

    def generate_sim_params(self, n: int, n0: int = 0) -> tuple[NDArray, ...]:
        """Generate parameter arrays"""

        c = self.c
        rho = self.rho
        ga = self.glottal_area(n, n0)

        rhoc2 = rho * c**2

        subglottal_area, epiglottal_area = self.areas_of_connected_tracts(n, n0)

        # align (potentially) time-varying parameters array dimensions
        ga, epiglottal_area, subglottal_area = align_signals(
            ga, epiglottal_area, subglottal_area
        )

        # Ishizaka-Flanagan pressure recovery coefficient
        ke = 2 * ga / epiglottal_area * (1 - ga / epiglottal_area)

        # kc = 1 # kinetic pressure loss coefficient

        # transglottal pressure coefficient (kc-ke)
        kt = 1 - ke

        R = ga * (1 / subglottal_area + 1 / epiglottal_area)

        Qa = 4.0 * kt / rhoc2
        a = ga * c / kt

        rhoca_eplx, rhoca_sg = self.configure_resistances(n, n0)

        return R, Qa, a, rhoca_eplx, rhoca_sg

    @property
    def _nb_states(self) -> int:
        """number of states"""
        return 0

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n", "ug", "psg", "peplx"]

    @dataclass
    class Results(Element.Results):
        ug: NDArray
        psg: NDArray
        peplx: NDArray
        aspiration_noise: Element.Results | None

        @cached_property
        def length(self) -> NDArray:
            element = cast(VocalFoldsAg, self.element)
            return element.length(self.nb_samples, self.n0)

        @cached_property
        def glottal_area(self) -> NDArray:
            element = cast(VocalFoldsAg, self.element)
            return element.glottal_area(self.nb_samples, self.n0)

    _ResultsClass = Results  # override result class


class VocalFoldsAg(VocalFoldsAgBase):
    """Vocal fold model with known glottal area waveform"""

    _ag_gen: SampleGenerator  # glottal area waveform generator
    _l_gen: SampleGenerator | None  # vocal fold length

    def __init__(
        self,
        ag: ArrayLike | Callable | SampleGenerator,
        *,
        upstream: VocalTract | None = None,
        downstream: VocalTract | None = None,
        length: float | ArrayLike | SampleGenerator | None = None,
        aspiration_noise: bool | dict | AspirationNoise | None = None,
    ):
        """Vocal fold model with known glottal area waveform

        Parameters
        ----------
        ag
            glottal area samples/generator. If an array, it will be converted to a
            `ClampedInterpolator` object at the simulation sampling rate. If a callable,
            it is expected to take the form:

                ag(n:int, n0:int) -> NDArray

            where n is the number of samples to generate at the simulation sampling rate
            and n0 is the time index of the first sample.
        As, optional
            subglottal tract cross-sectional area, by default inf
        Ae, optional
            epiglottal tract cross-sectional area, by default inf
        length, optional
            vocal fold length, only required to use an aspiration noise model which needs
            the vocal fold length
        aspiration_noise, optional
            True or dict value to enable the default bandpass noise model (dict input
            sets the aspiration noise model keyword arguments) or an `AspirationNoise`
            object.

        References
        ----------
        [1] I. R. Titze, “Parameterization of the glottal area, glottal flow, and vocal fold contact area,”
            J. Acoust. Soc. Am., vol. 75, no. 2, pp. 570–580, 1984, doi: 10.1121/1.390530.

        """

        super().__init__(upstream, downstream, aspiration_noise)

        self._ag_gen = format_parameter(ag)
        if length is not None:
            self._l_gen = format_parameter(length)

    @property
    def known_length(self) -> bool:
        """True if model specifies the vibrating vocal fold length"""
        return self._l_gen is not None

    def length(
        self, nb_samples: int | None = None, n0: int = 0
    ) -> NDArray | float | None:
        """vibrating glottal length

        Parameters
        ----------
        nb_samples, optional
            number of samples, required if dynamic, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            glottal length in cm if specified. 1D array if fo or L0 is dynamic otherwise a scalar value.
        """

        return self._l_gen and self._l_gen(nb_samples, n0)

    def glottal_area(self, nb_samples: int, n0: int = 0) -> NDArray:
        """glottal area (minimum opening along the depth)

        :param nb_samples: _description_
        :param n0: starting sample time, defaults to 0
        :return: glottal area samples
        """

        return self._ag_gen(nb_samples, n0)
