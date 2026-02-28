from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

from numpy.typing import ArrayLike, NDArray

from ..constants import PL as default_lung_pressure
from ..function_generators import ClampedInterpolator, Constant
from ..function_generators.abc import FunctionGenerator
from .abc import Element, Lungs
from ..core import has_numba

if has_numba:
    import numba as nb


class LungsBase(Lungs):
    """Lungs Model Concrete Base Class

    sim_func signature:
        source signature: [f2, out_states, res] = sim(b2, in_states, *params)
    """

    @dataclass
    class Results(Element.Results):
        plung: NDArray

    _ResultsClass = Results  # override result class

    RunnerSpec = [("n", nb.int64), ("plung", nb.float64[:])] if has_numba else []

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n", "plung"]

    def __init__(
        self,
        lung_pressure: float | ArrayLike | FunctionGenerator | None = None,
    ):
        """Lungs

        Parameters
        ----------
        lung_pressure, optional
            respiratory driving pressure from lungs (dyn/cm²), by default None to use the
            default value `lt.constants.PL`
        """
        # use soft step-function and load transition samlung_pressurees
        # if array, use it as is
        self._lung_pressure = (
            Constant(
                float(lung_pressure or default_lung_pressure), transition_time=0.002
            )
            if isinstance(lung_pressure, (Number, type(None)))
            else (
                lung_pressure
                if isinstance(lung_pressure, FunctionGenerator)
                else ClampedInterpolator(self.fs, lung_pressure, transition_time=0.002)
            )
        )

    @property
    def lung_pressure(self) -> FunctionGenerator:
        return self._lung_pressure

    def generate_sim_params(self, n: int, n0: int = 0, **_) -> tuple[NDArray, ...]:
        """parameters:
        - lung_pressure: lung pressure
        """

        # get the iterator factories for the 2 parameters
        return (self._lung_pressure(n, n0),)

    @property
    def nb_states(self) -> int:
        """number of states"""
        return 0

    ####################################################


class LeTalkerLungs(LungsBase):
    """Constant-pressure lung model with impedance matching"""

    class Runner:
        n: int
        plung: np.ndarray

        def __init__(self, nb_steps: int, s: NDArray, plung: NDArray):
            self.n = nb_steps
            self.plung = plung

        def step(self, i: int, blung: float) -> float:
            """perform one time-step update"""

            # ------------- lung_pressure pressure to forward pressure -----------  */
            lung_pressure = self.plung[i if i < self.plung.shape[0] else -1]
            return 0.9 * lung_pressure - 0.8 * blung

    @property
    def runner_info(self) -> tuple[type, list[tuple[str, type]]]:
        return LeTalkerLungs.Runner, LungsBase.RunnerSpec


#################################################################


class OpenLungs(LungsBase):
    """Constant-pressure lung model with zero-reflection"""

    class Runner:
        n: int
        plung: np.ndarray

        def __init__(self, nb_steps: int, s: NDArray, plung: NDArray):
            self.n = nb_steps
            self.plung = plung

        def step(self, i: int, blung: float) -> float:
            """perform one time-step update"""
            return self.plung[i if i < self.plung.shape[0] else -1]

    @property
    def runner_info(self) -> tuple[type, list[tuple[str, type]]]:
        return OpenLungs.Runner, LungsBase.RunnerSpec


#################################################################


class NullLungs(Lungs):
    """Zero-pressure lung model, only reflecting 80% of the backward pressure"""

    RunnerSpec = [("n", nb.int64)] if has_numba else []

    class Runner:
        n: int

        def __init__(self, nb_steps: int, s_in: NDArray):
            self.n = nb_steps

        def step(self, i: int, blung: float) -> float:
            """perform one time-step update"""
            return -0.8 * blung

    @property
    def runner_info(self) -> tuple[type, list[tuple[str, type]]]:
        return NullLungs.Runner, LungsBase.RunnerSpec

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n"]

    @property
    def is_dynamic(self) -> bool:
        """True if element has a time-varying parameter"""
        return False

    def generate_sim_params(self, n: int, n0: int = 0) -> tuple[NDArray, ...]:
        """parameters: None"""

        # get the iterator factories for the 2 parameters
        return ()

    @property
    def nb_states(self) -> int:
        """number of states"""
        return 0
