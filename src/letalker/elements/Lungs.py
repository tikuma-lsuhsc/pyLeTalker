from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

from numpy.typing import ArrayLike, NDArray

from .._backend import LeTalkerLungsRunner, NullLungsRunner, OpenLungsRunner
from ..constants import PL as default_lung_pressure
from ..function_generators import ClampedInterpolator, Constant
from ..function_generators.abc import FunctionGenerator
from .abc import Element, Lungs


class LungsBase(Lungs):
    """Lungs Model Concrete Base Class

    sim_func signature:
        source signature: [f2, out_states, res] = sim(b2, in_states, *params)
    """

    @dataclass
    class Results(Element.Results):
        plung: NDArray

    _ResultsClass = Results  # override result class

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

    Runner = LeTalkerLungsRunner


#################################################################


class OpenLungs(LungsBase):
    """Constant-pressure lung model with zero-reflection"""

    Runner = OpenLungsRunner


#################################################################


class NullLungs(Lungs):
    """Zero-pressure lung model, only reflecting 80% of the backward pressure"""

    Runner = NullLungsRunner

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
