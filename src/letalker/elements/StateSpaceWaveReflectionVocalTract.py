from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, cast

import control as ct
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..__util import format_parameter
from .._backend import WaveReflectionVocalTractRunner
from ..constants import B, K, M
from ..constants import c as c_default
from ..constants import rho_air as rho_air_default
from ..core import classproperty
from ..function_generators.abc import SampleGenerator
from .abc import Element, VocalTract

rhoc_default = rho_air_default * c_default


def join_fb_ss(ss1: ct.StateSpace, ss2: ct.StateSpace) -> ct.StateSpace:
    ...
    # nstates = ss1.
    # A =


class LosslessJunctionBlock:
    # Lossless junction VT block with possible independent pressure/flow sources

    def ss(
        self,
        areas: np.ndarray,
        has_pressure_source: bool,
        has_flow_source: bool,
        *,
        treat_sources_as_states: bool = True,
        rhoc: float = rhoc_default,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        den = (areas[..., :-1] + areas[..., 1:]).reshape(..., 1, 1)
        nstates = has_pressure_source + has_flow_source
        m1 = [areas[..., :-1], rhoc] / den
        m2 = [areas[..., 1:], rhoc] / den
        d = m1 - m2
        D = np.empty((*areas.shape[:-1], areas.shape[-1] - 1, 2, 2))
        D[..., 0, 0] = 2 * m1
        D[..., 1, 1] = 2 * m2
        D[..., 0, 1] = -d
        D[..., 1, 0] = d
        if nstates == 0:
            C = np.empty(0)
        else:
            C = np.empty((*areas.shape[:-1], areas.shape[-1] - 1, 2, nstates))
            if has_pressure_source:
                C[..., 0, 0] = -m1
                C[..., 1, 0] = m2
            if has_flow_source:
                C[..., -min(nstates, 1)] = rhoc / den
        return np.empty((0, 0)), np.empty((0, C.shape[-1])), C, D


class ShuntBlocks:
    def ss(
        self,
        areas: float | np.ndarray,
        include_heat_loss: bool,
        *,
        fs: float | None = None,
        length: float | np.ndarray | None = None,
        yielding_wall_tf_iter: Callable | None = None,
        heat_loss_tf_iter: Callable | None = None,
        sample_kws: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        tfs = self.p2u_tf(
            areas,
            include_heat_loss,
            fs=fs,
            length=length,
            yielding_wall_tf_iter=yielding_wall_tf_iter,
            heat_loss_tf_iter=heat_loss_tf_iter,
            sample_kws=sample_kws,
        )

        if fs is not None and fs > 0:
            kws = {"method": "bilinear"} | (sample_kws or {})
            sys = [s.sample(1 / fs, **kws) for s in sys]

        return

    def p2u_to_fbss(
        self,
        area: float,
        p2u: ct.TransferFunction,
        rhoc: float = rhoc_default,
    ):
        ssw = ct.tf2ss(p2u)
        two_a = 2 * area
        rhocd = rhoc * p2u.D
        den = two_a + rhocd
        A = ssw.A - (ssw.B @ ssw.C) / den
        B = np.tile(ssw.B.reshape(-1, 1) * (two_a / den), (1, 2))
        C = np.tile(-rhoc / den * ssw.C.reshape(1, -1), (2, 1))
        d11 = two_a / den
        d12 = -rhocd / den
        D = np.array([[d11, d12], [d12, d11]])
        return A, B, C, D

    def p2u_tf(
        self,
        areas: float | np.ndarray,
        include_heat_loss: bool,
        *,
        length: float | np.ndarray | None = None,
        yielding_wall_tf_iter: Callable | None = None,
        heat_loss_tf_iter: Callable | None = None,
    ) -> ct.TransferFunction | np.ndarray[ct.TransferFunction]:
        """total pressure to wall flow transfer functions

        Args:
            areas: _description_
            include_heat_loss: _description_
            fs: _description_. Defaults to fs_default.
            length: _description_. Defaults to None.
            yielding_wall_tf_iter: _description_. Defaults to None.
            heat_loss_tf_iter: _description_. Defaults to None.
            sample_kws: _description_. Defaults to None.

        Returns:
            _description_
        """

        def default_yielding_wall_tf_iter(A, length):
            x = np.atleast_1d(2 * length * np.sqrt(np.pi * areas))
            Lw = M / x
            Cw = x / K
            Rw = B / x
            for l, c, r in zip(Lw, Cw, Rw):
                yield [c, 0], [l * c, r * c, 1]

        def default_heat_loss_tf_iter(A, length):
            rho = rho_air_default  # 1.14e-3  # gm/cm^3 - air density
            c = c_default  # 3.5e4  # cm/sec - speed of sound
            lamb = 0.055e-3  # cal/cm-sec-deg - coefficient of heat conduction
            cp = 0.24  # cal/gm-degree - specific heat of air
            eta = 1.4  # adiabatic constant
            xi = 1.08  # shape factor
            f = 1000  # evaluated at 1 kHz
            c = 2 * np.pi * (eta - 1) / (rho * c**2) * np.sqrt(lamb / (cp * rho))
            Ga = c * xi * np.sqrt(A * f) * length
            for g in Ga:
                yield [g], [1]

        areas_ = np.atleast_1d(areas).reshape(-1)

        if yielding_wall_tf_iter is None:
            yielding_wall_tf_iter = default_yielding_wall_tf_iter

        sys = [ct.tf(num, den) for num, den in yielding_wall_tf_iter(areas_, length)]

        if include_heat_loss:
            if heat_loss_tf_iter is None:
                heat_loss_tf_iter = default_heat_loss_tf_iter

            sys = [
                yw + ct.tf(num, den)
                for yw, (num, den) in zip(sys, heat_loss_tf_iter(areas_, length))
            ]

        return (
            sys[0] if isinstance(areas, float) else np.array(sys).reshape(areas_.shape)
        )


class StateSpaceWaveReflectionVocalTract(VocalTract):
    """Wave-reflection vocal tract model (Liljencrants, 1985; Story, 1995)"""

    _A: np.ndarray
    _B: np.ndarray
    _C: np.ndarray
    _D: np.ndarray
    _E: np.ndarray
    _nb_states: int
    _nb_sections: int

    approximate_visc_loss: bool = False
    kinetic_drop: bool = False
    fricative_noise: dict[int, AspirationNoise] | None = None
    section_structure: Literal["lattice", "general", "auto"] = "auto"
    log_sections: bool = False

    Runner = WaveReflectionVocalTractRunner

    def __init__(
        self,
        D: ArrayLike | SampleGenerator | None = None,
        A: ArrayLike | SampleGenerator | None = None,
        B: ArrayLike | SampleGenerator | None = None,
        C: ArrayLike | SampleGenerator | None = None,
        E: ArrayLike | SampleGenerator | None = None,
        approximate_visc_loss: bool = False,
        kinetic_drop: bool | None = None,
        fricative_noise: dict[int, AspirationNoise] | None = None,
        section_structure: Literal["lattice", "general", "auto"] | None = None,
        log_sections: bool = False,
    ):
        """Linear state-space representation of wave-reflection vocal tract model

        Args:
            D: feedthrough matrix. It's shape is ``(nsection,2,2)`` where the
                number of tube sections ``nsection`` must be a positive even number.
            A: state matrix. Defaults to None.
            B: input matrix. Defaults to None.
            C: output matrix. Defaults to None.
            E: auxiliary feedthrough matrix to allow additional inputs like
                viscous loss or fricative noise. Defaults to None.
            approximate_visc_loss: True to inject the previous stage's viscous
                loss of the previous section per (Story 1995). Defaults to False.
            kinetic_drop_at: _description_. Defaults to None.
            fricative_noise: _description_. Defaults to None.
            section_structure: _description_. Defaults to False.
            log_sections: _description_. Defaults to False.

        Raises:
            ValueError: _description_
        """

        self._D = format_parameter(D, shape=(-1, 2, 2))

        self._nb_sections = nseg = self._D.shape[0]
        if nseg % 2:
            raise ValueError("Tube must have an even number of sections.")

        self._A = format_parameter(A, shape=(nseg, -1, -1), optional=True)
        if A is None:
            self._nb_states = nst = 0
        else:
            nseg_, self._nb_states, nst = self._A.shape
            if nseg_ != nseg:
                raise ValueError(
                    "Number of tube sections does not match between D and A matrices"
                )
            if nst != self._nb_states:
                raise ValueError(
                    "The last two dimensions of the state matrix A must have the same size."
                )
        self._B = format_parameter(B, shape=(nseg, nst, 2), optional=True)
        self._C = format_parameter(C, shape=(nseg, 2, nst), optional=True)
        self._E = format_parameter(E, shape=(nseg, 2, -1), optional=True)
        naux = self._E.shape[-1]

        self.approximate_visc_loss = bool(approximate_visc_loss)
        self.kinetic_drop = bool(kinetic_drop)
        self.fricative_noise = fricative_noise

        if naux != (self.approximate_visc_loss or self.kinetic_drop) + (
            self.fricative_noise is not None
        ):
            raise ValueError(
                "Dimension of matrix E does not match the number of auxiliary inputs."
            )

        self.section_structure = section_structure or "auto"

        if log_sections:
            self.log_sections = True

    def _detect_lattice(self): ...

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

            element = cast(WaveReflectionVocalTract, self.element)
            areas = element.areas(self.n1 - self.n0, self.n0)[:, :-1]
            dia = 2 * (areas / np.pi) ** 0.5
            return dia * self.uout_sections / (element.nu * areas)
            # RE2 = (ug * nu_inv / L) ** 2

        @property
        def areas(self) -> NDArray:
            """cross-sectional areas of the tube sections"""

            element = cast(WaveReflectionVocalTract, self.element)
            return element.areas(self.n1 - self.n0, self.n0)

        @property
        def dx(self) -> float:
            """lengh of each tube section in cm"""
            return cast(WaveReflectionVocalTract, self.element).dz

    # override result class
    _ResultsClass = Results

    def create_result(
        self,
        runner: WaveReflectionVocalTractRunner,
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
