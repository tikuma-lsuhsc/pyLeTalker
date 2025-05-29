from __future__ import annotations

from typing import Any
from numpy.typing import ArrayLike, NDArray
from collections.abc import Sequence
from typing import Protocol

import abc
from dataclasses import dataclass
import numpy as np

from ..constants import (
    c as c_default,
    rho_air as rho_default,
    mu as mu_default,
)

from ..core import TimeSampleHandler, compile_jitclass_if_numba

from ..function_generators import Constant

__all__ = ["Element", "Lungs", "Lips", "VocalFolds", "VocalTract", "AspirationNoise"]


class BlockRunner(Protocol):
    """Protocol for Element classes' Runner member class

    To run the synthesis, every Element object instantiates a runner object which
    obeys this protocol.
    """

    n: int  #: number of samples

    def step(self, i: int, *args, **kwargs) -> float | tuple[float, float]:
        """perform the operation for one simulation step at time i"""


class Element(TimeSampleHandler, metaclass=abc.ABCMeta):

    c: float = c_default  #: speed of sound in cm/s
    rho: float = rho_default  #: air density in g/cm³ (= kg/mm³)
    mu: float = mu_default  #: air viscosity in dyne-s/cm²  (= Pa·s = kg/(m·s))

    @staticmethod
    def modify_global_constants(
        fs: int | None = None,
        c: float | None = None,
        rho: float | None = None,
        mu: float | None = None,
    ):
        """Modify the system constants (use None to keep existing)

        Parameters
        ----------
        fs
            time sampling rate in S/s (samples/second)
        c, optional
            speed of sound in cm/s, by default None
        rho, optional
            air density in g/cm³ (= kg/mm³), by default None
        mu, optional
            air viscosity in dyne-s/cm²  (= Pa·s = kg/(m·s)), by default None
        """
        if fs is not None:
            TimeSampleHandler.set_fs(fs)
        if c is not None:
            Element.c = c
        if rho is not None:
            Element.rho = rho
        if mu is not None:
            Element.mu = mu

    @staticmethod
    def nu() -> float:
        """kinematic viscosity in dyne·s·cm/g (= 10⁻³ mm²/s)"""
        return Element.mu / Element.rho

    @property
    @abc.abstractmethod
    def is_source(self) -> bool:
        """True if takes no input"""

    @property
    @abc.abstractmethod
    def is_sink(self) -> bool:
        """True if produces no output"""

    @dataclass
    class Results:
        element: Element
        """who produced this results"""
        n0: int
        """starting sample index"""
        nb_samples: int
        """number of samples"""

        @property
        def n1(self) -> int:
            """ending sample index (exclusive)"""
            return self.nb_samples + self.n0

        @property
        def fs(self) -> int:
            """sampling rate"""
            return self.element.fs

        @property
        def ts(self) -> NDArray:
            """time vector"""
            return self.element.ts(self.nb_samples, self.n0)

    def create_runner(
        self, n: int, n0: int = 0, s_in: ArrayLike | None = None, **kwargs
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

        Returns
        -------
            simulation runner object
        """

        if s_in is None:
            s_in = np.zeros(self.nb_states)
        elif isinstance(s_in, np.ndarray) and s_in.shape != (self.nb_states,):
            raise TypeError(f"s_in must be a 1D numpy array of size {self.nb_states}")
        params = self.generate_sim_params(n, n0, **kwargs)
        CLS = compile_jitclass_if_numba(*self.runner_info)
        return CLS(n, s_in, *params)

    @property
    @abc.abstractmethod
    def runner_info(self) -> tuple[type, list[tuple[str, type]]]:
        """element's runner class and its numba jitclass spec"""

    @abc.abstractmethod
    def generate_sim_params(
        self, n: int, n0: int = 0, **kwargs
    ) -> tuple[NDArray | Sequence[Any], ...]:
        """generate parameters, which may be time dependent

        Parameters
        ----------
        n
            number of samples
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            list of _sim_func arguments. Each tuple item must be iterable. Each of its elements
            is used for a sample. Once exhausted, the last element is repeatedly used till the
            simulation ends.
        """

    @property
    @abc.abstractmethod
    def nb_states(self) -> int:
        """number of states"""

    @property
    @abc.abstractmethod
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""

    _ResultsClass: Element.Results

    def create_result(
        self, runner: BlockRunner, *extra_items, n0: int = 0
    ) -> Element.Results:
        """Returns simulation result object"""

        fieldnames = self._runner_fields_to_results

        return self._ResultsClass(
            self, n0, *(getattr(runner, pn) for pn in fieldnames), *extra_items
        )


class VocalTract(Element, metaclass=abc.ABCMeta):
    """Vocal Tract Model Base

    Vocal tract elements serve to connect all the other elements of the voice production simulation.
    """

    @property
    def is_source(self) -> bool:
        """True if takes no input"""
        return False

    @property
    def is_sink(self) -> bool:
        """True if produces no output"""
        return False

    @property
    @abc.abstractmethod
    def input_area_is_fixed(self) -> bool:
        """True if input section is fixed"""

    @property
    @abc.abstractmethod
    def output_area_is_fixed(self) -> bool:
        """True if output section is fixed"""

    @abc.abstractmethod
    def input_area(self, n: int, n0: int = 0) -> float | NDArray:
        """calculate input cross-sectional area

        Parameters
        ----------
        n
            number of simulation cycles
        n0, optional
            starting time index, by default 0

        Returns
        -------
            cross-sectional area of the first tube. Returns a float
            number of fixed or an 1D array if dynamic.
        """

    @abc.abstractmethod
    def output_area(self, n: int, n0: int = 0) -> float | NDArray:
        """calculate input cross-sectional area

        Parameters
        ----------
        n
            number of simulation cycles
        n0, optional
            starting time index, by default 0

        Returns
        -------
            cross-sectional area of the last tube. Returns a float
            number of fixed or an 1D array if dynamic.
        """


class Lungs(Element, metaclass=abc.ABCMeta):
    """Lungs Model Base Class"""

    downstream: VocalTract | float = np.inf

    def __init__(self, downstream: VocalTract | None = None):
        if downstream is not None:
            self.downstream = downstream

    @property
    def is_source(self) -> bool:
        """True if takes no input"""
        return True

    @property
    def is_sink(self) -> bool:
        """True if produces no output"""
        return False

    @property
    def is_dynamic(self) -> bool:
        if isinstance(self.downstream, VocalTract):
            return not self.downstream.input_area_is_fixed
        else:
            return np.asarray(self.downstream).ndim > 0

    def link(self, dst: VocalTract):
        """set the upstream vocal tract object

        Parameters
        ----------
        dst
            downstream vocal tract
        """

        self.downstream = dst

    def unlink(self):
        """disconnect the upstream vocal tract object"""

        self.downstream = None


class Lips(Element, metaclass=abc.ABCMeta):
    """Lips Model Base"""

    upstream: VocalTract | float = np.inf

    def __init__(self, upstream: VocalTract | float | None = None):
        if upstream is not None:
            self.upstream = upstream

    @property
    def is_source(self) -> bool:
        """True if takes no input"""
        return False

    @property
    def is_sink(self) -> bool:
        """True if produces no output"""
        return True

    @property
    def is_dynamic(self) -> bool:
        if isinstance(self.upstream, VocalTract):
            return not self.upstream.output_area_is_fixed
        else:
            return np.asarray(self.upstream).ndim > 0

    def link(self, src: VocalTract):
        """set the downstream vocal tract object

        Parameters
        ----------
        src
            upstream vocal tract
        """

        self.upstream = src

    def unlink(self):
        """disconnect the downstream vocal tract object"""

        self.upstream = Lips.upstream

    @property
    @abc.abstractmethod
    def pout_index(self) -> int:
        """column index of the resultant NDArray with radiated pressure output"""


class AspirationNoise(Element, metaclass=abc.ABCMeta):
    """Aspiration Noise Generator Base Class

    sim_func signature:

        filter signature: [f2, b1, out_states, res] = sim(f1, b2, in_states, *params)
        source signature: [f2, out_states, res] = sim(b2, in_states, *params)
        sink signature: [b1, out_states, res] = sim(f1, in_states, *params)
    """

    @property
    def is_source(self) -> bool:
        """True if takes no input"""
        return True

    @property
    def is_sink(self) -> bool:
        """True if produces no output"""
        return True

    @abc.abstractmethod
    def generate_noise(self, N: int, n0: int = 0) -> NDArray: ...

    @property
    def nb_states(self) -> int:
        """number of states"""
        return 0

    @property
    def nb_results(self) -> int:
        """no output signals"""
        return 0

    @property
    @abc.abstractmethod
    def need_length(self) -> bool:
        """True if vocal fold length is necessary to generate aspiration noise"""

    @abc.abstractmethod
    def generate_sim_params(self, n: int, n0: int = 0, **kwargs) -> tuple[NDArray, ...]:
        """Generate element parameters (possibly time varying)"""


class VocalFolds(Element, metaclass=abc.ABCMeta):
    """Vocal Folds Base Class"""

    downstream: VocalTract | float | None = None
    upstream: VocalTract | float | None = None

    def __init__(
        self,
        upstream: VocalTract | float | None = None,
        downstream: VocalTract | float | None = None,
    ):
        if downstream is not None:
            self.downstream = downstream
        if upstream is not None:
            self.upstream = upstream

    @property
    def is_source(self) -> bool:
        """True if takes no input"""
        return False

    @property
    def is_sink(self) -> bool:
        """True if produces no output"""
        return False

    def link(self, src: VocalTract | None = None, dst: VocalTract | None = None):
        """set downstream and upstream vocal tract objects

        Parameters
        ----------
        src
            upstream vocal tract object. If None, no changes will be made
        dst
            downstream vocal tract object. If None, no changes will be made
        """

        if src is not None:
            self.upstream = src
        if dst is not None:
            self.downstream = dst

    def unlink(self):
        """disconnect both the downstream and upstream vocal tract objects"""

        self.upstream = self.downstream = None

    @abc.abstractmethod
    def add_aspiration_noise(
        self, noise_model: AspirationNoise | bool | dict | None
    ) -> AspirationNoise:
        """add noise source to the model

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

    @property
    @abc.abstractmethod
    def known_length(self) -> bool:
        """True if vocal fold length is known pre-simulation"""

    def length(self, n: int | None = None, n0: int = 0) -> float | NDArray:
        """Generate vocal fold length samples"""
        raise NotImplementedError(
            "Model cannot compute vocal fold length pre-simulation"
        )

    def create_noise_runner(
        self,
        n: int,
        n0: int = 0,
        s_in: ArrayLike | None = None,
        length: NDArray | None = None,
    ) -> BlockRunner:
        """Generate aspiration noise generation function and its parameters

        Parameters
        ----------
        n
            number of samples to generate
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            fixed_params
                tuple of fixed parameters
            dynamic_params
                tuple of possibly time-varying parameters
        """
