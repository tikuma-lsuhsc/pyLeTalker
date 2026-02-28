from __future__ import annotations

import abc

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .abc import AspirationNoise, BlockRunner, Element, VocalFolds, VocalTract
from .LeTalkerAspirationNoise import LeTalkerAspirationNoise


class NoAspirationNoiseRunner:
    def __init__(self): ...
    def step(self, i: int, ug: float) -> float:
        return 0.0


def glottal_flow_params(
    ga: float, As: float, Ae: float, c: float, rhoc2: float
) -> tuple[float, float, float]:
    """compute glottal flow parameter

    :param ga: glottal area
    :param As: subglottal tube cross-sectional area
    :param Ae: epiglottal tube cross-sectional area
    :param c: speed of sound
    :param rhoc2: air density (rho) times the square of speed of sound (c)
    :return R: 4.0 * kt / rhoc2 (kt=transglottal pressure coefficient)
    :return Qa: ga / Astar (Astar = 1/(1/As+1/Ae))
    :return a: -ga * c / kt
    """

    # Ishizaka-Flanagan pressure recovery coefficient
    ke = 2 * ga / Ae * (1 - ga / Ae)

    # kc = 1 # kinetic pressure loss coefficient

    # transglottal pressure coefficient (kc-ke)
    kt = 1 - ke

    Astar = (As * Ae) / (As + Ae)

    Qa = 4.0 * kt / rhoc2
    R = ga / Astar
    a = -ga * c / kt
    return Qa, R, a


def calculate_glottal_flow(
    i: int,
    Qa: float,
    R: float,
    a: float,
    fsg: float,
    beplx: float,
    rhoca_eplx: float,
    rhoca_sg: float,
    anoise: BlockRunner,
) -> tuple[float, float, float, float]:
    """2-port glottis update function given current glottal flow

    Parameters
    ----------
    fsg
        forward pressure input from subglottis
    beplx
        backward pressure input from epilarynx
    ug
        glottal flow
    rhoca_eplx
        epilaryngeal "resistance" [rho/(c*epilarynx_area)]
    rhoca_sg
        subglottal "resistance" [rho/(c*subglottis_area)]

    Returns
    -------
        feplx
            forward pressure output to epilarynx
        bsg
            backward pressure output to subglottis

    NOTE: This function is mostly for a reference-only and not recommended to
          be called by concrete vocal fold update functions.
    """

    # compute the glottal flow
    Q = Qa * (fsg - beplx)
    if fsg < beplx:
        # depends on whether ug is positive or negative
        R = -R
        Q = -Q
    ug = a * (R - (R**2 + Q) ** 0.5)

    # add turbulent flow noise
    ug += anoise.step(i, ug)

    # compute the output forward and backward pressures
    feplx = beplx + ug * rhoca_eplx
    bsg = fsg - ug * rhoca_sg

    # compute the total pressures
    psg = fsg + bsg
    peplx = feplx + beplx

    return feplx, bsg, ug, psg, peplx


class VocalFoldsBase(VocalFolds):
    """Vocal folds as a pure glottal flow source"""

    noise_model: AspirationNoise | None = None

    # OUTCOMES
    # psg: NDArray  # subglottal pressure
    # peplx: NDArray  # epilarngeal pressure
    # unoise: NDArray  # aspiration noise flow component

    def __init__(
        self,
        upstream: VocalTract | float | None = None,
        downstream: VocalTract | float | None = None,
        aspiration_noise: bool | dict | AspirationNoise | None = None,
    ):
        """_summary_

        Parameters
        ----------
        upstream, optional
            subglottal tract object or subglottal cross-sectional area, by default None (open space with inf area)
        downstream, optional
            epiglottal tract object or epiglottal cross-sectional area, by default None (open space with inf area)
        aspiration_noise, optional
            True to use the default aspiration noise model (`LeTalkerAspirationNoise`), a dict to use
            the default model with specific parameters, False or None to disable noise injection, or
            specify an existing AspirationNoise object, by default None (no noise injection)
        """

        super().__init__(upstream, downstream)

        if aspiration_noise:
            self.add_aspiration_noise(aspiration_noise)

    def add_aspiration_noise(
        self, noise_model: AspirationNoise | bool | dict | None
    ) -> AspirationNoise | None:
        """add aspiration noise model

        Parameters
        ----------
        noise_model
            True to use the default aspiration noise model, a dict to use the default model
            with the specified keyword arguments, False or None to disable noise injection, or
            specify an existing AspirationNoise object.

        Returns
        -------
            Assigned AspirationNoise object or None if disabled
        """

        if noise_model is False:
            noise_model = None
        elif noise_model is not None:
            if noise_model is True or isinstance(noise_model, dict):
                noise_model_kws = {}
                if noise_model is not True:
                    noise_model_kws = {**noise_model, **noise_model_kws}
                noise_model = LeTalkerAspirationNoise(**noise_model_kws)
        self.noise_model = noise_model
        return noise_model

    def areas_of_connected_tracts(self, n: int, n0: int = 0) -> tuple[NDArray, NDArray]:
        """get cross-sectional area of connected vocal tract tube sections

        Parameters
        ----------
        n
            number of simulation samples
        n0, optional
            starting time index, by default 0

        Returns
        -------
        subglottal_area
            fixed or dynamic cross-sectional area of connecting subglottal tube
        epiglottal_area
            fixed or dynamic cross-sectional area of connecting epiglottal tube
        """

        subglottal_area = (
            self.upstream.output_area(n, n0)
            if isinstance(self.upstream, VocalTract)
            else np.array([self.upstream or np.inf])
        )

        epiglottal_area = (
            self.downstream.input_area(n, n0)
            if isinstance(self.downstream, VocalTract)
            else np.array([self.downstream or np.inf])
        )

        return subglottal_area, epiglottal_area

    def configure_resistances(self, n: int, n0: int = 0) -> tuple[NDArray, NDArray]:
        """Generate epilaryngeal and subglottal resistances

        Parameters
        ----------
        n
            number of simulation samples
        n0, optional
            starting time index, by default 0

        Returns
        -------
            _description_
        """

        rhoc = self.rho * self.c

        subglottal_area, epiglottal_area = self.areas_of_connected_tracts(n, n0)
        rhoca_eplx = rhoc / epiglottal_area
        rhoca_sg = rhoc / subglottal_area

        return rhoca_eplx, rhoca_sg

    @property
    def nb_states(self) -> int:
        nmodel = self.noise_model
        return self._nb_states + (nmodel.nb_states if nmodel else 0)

    @property
    @abc.abstractmethod
    def _nb_states(self) -> int:
        """number of states, excluding aspiration model"""

    def create_runner(
        self,
        n: int,
        n0: int = 0,
        s_in: ArrayLike | None = None,
        noise_free: bool = False,
    ) -> BlockRunner:
        """instantiate a vocal folds element runner

        (overloading abc.Element to account for the aspiration noise sub-element)

        Parameters
        ----------
        n
            number of samples to synthesize
        n0, optional
            starting time index, by default 0
        s_in, optional
            initial states, by default None
        noise_free, optional
            True to force runner to exclude aspiration noise

        Returns
        -------
            simulation runner object
        """

        if s_in is None:
            s_in = np.zeros(self.nb_states)
        elif isinstance(s_in, np.ndarray) and s_in.shape != (self.nb_states,):
            raise TypeError(f"s_in must be a 1D numpy array of size {self.nb_states}")

        params = self.generate_sim_params(n, n0)
        ns = self._nb_states

        an_runner = (
            NoAspirationNoiseRunner()
            if noise_free
            else self.create_noise_runner(n, n0, s_in[ns:])
        )

        return self.Runner(n, s_in[:ns], an_runner, *params)

    def create_noise_runner(
        self,
        n: int,
        n0: int = 0,
        s_in: ArrayLike | None = None,
        length: NDArray | None = None,
    ) -> BlockRunner:

        noise_model = self.noise_model

        if not noise_model:
            return NoAspirationNoiseRunner()

        if not self.known_length and noise_model.need_length:
            raise TypeError(
                "Aspiration noise generator requires vocal fold length information but vocal fold model cannot provide it pre-simulation."
            )

        noise_kws = (
            {"length": self.length(n, n0) if length is None else length}
            if self.known_length
            else {}
        )

        return noise_model.create_runner(n, n0, s_in, **noise_kws)

    def create_result(
        self, runner: BlockRunner, *extra_items, n0: int = 0
    ) -> Element.Results:
        """Returns simulation result object"""

        anoise = (
            self.noise_model.create_result(runner._anoise, n0=n0)
            if self.noise_model
            and hasattr(
                runner._anoise, "n"
            )  # 'n' is not present in NoAspirationNoiseRunner
            else None
        )

        return super().create_result(runner, anoise, n0=n0)
