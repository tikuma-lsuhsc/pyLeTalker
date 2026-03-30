from __future__ import annotations

import abc
from typing import Any, Iterator, Protocol

import control as ct
import numpy as np

from ..core import TimeSampleHandler

__all__ = ["TwoPortSystem"]


class LTIFactory(Protocol):
    def __call__(self, area: float, length: float) -> ct.LTI | float: ...

    @property
    def nb_states(self) -> int: ...


class TwoPortSystem(TimeSampleHandler, metaclass=abc.ABCMeta):
    @property
    def dz(self) -> float:
        """length (in cm) of each tube section"""

    @abc.abstractmethod
    def nb_states(self) -> int:
        """Number of internal states"""

    @abc.abstractmethod
    def nb_sections(self) -> int:
        """Number of tube sections"""

    @abc.abstractmethod
    def nb_inputs(self) -> int:
        """Number of inputs: 2 + number of auxilary sources"""

    @abc.abstractmethod
    def nb_outputs(self) -> int:
        """Number of outputs (typically 2)"""

    @abc.abstractmethod
    def is_lattice(self) -> bool:
        """``True`` if system can be implemented by Story95 lattice structure with 2 reflection coefficients plus a gain"""

    @abc.abstractmethod
    def has_aux_pressure_source(self) -> bool:
        """``True`` if system has auxiliary pressure source at junction"""

    @abc.abstractmethod
    def has_aux_flow_source(self) -> bool:
        """``True`` if system has auxiliary flow source at junction"""

    @abc.abstractmethod
    def ss(
        self,
        n: int,
        *,
        n0: int = 0,
        sample: bool = True,
        sample_kws: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """generate state-space matrices of all sections as a sample generator

        Parameters
        ----------
        n
            number of samples to generate (spanning only one of its dimensions)
        n0, optional
            first sample index, by default 0
        sample, optional
            ``True`` (default) to convert the system to discrete-time system, sampled at
            the ``self.fs`` sampling rate. ``False`` to return a continuous-time
            system. Control Systems Library package's ``sample()`` method is used
            to make the conversion.
        sample_kws, optional
            Specify the continuous-to-discrete time conversion options. See below
            for a brief summary of the available options or see Control Systems
            Library documentation.

        Returns
        -------
        A
            State matrix, 4D array of shape ``(nb_samples,nb_sections,nb_states,nb_states)``.
        B
            Input matrix, 4D array of shape ``(nb_samples,nb_sections,nb_states,nb_inputs)``.
            If auxilirary pressure source is enabled, the 3rd input is always pressure source,
            and the last input is always auxilirary flow source if enabled.
        C
            Output matrix, 4D array of shape ``(nb_samples,nb_sections,2,nb_states)``
        D
            Feed-through matrix, 4D array of shape ``(nb_samples,nb_sections,2,nb_inputs)``


        Sample Options
        --------------

        ``'method'``: Method to use for sampling:

            * ``'gbt'``: generalized bilinear transformation
            * ``'backward_diff'``: Backwards difference (``'gbt'`` with ``alpha=1.0``)
            * ``'bilinear'`` (or ``'tustin'``): Tustin's approximation (``'gbt'`` with ``alpha=0.5``) (default)
            * ``'euler'``: Euler (or forward difference) method (``'gbt'`` with ``alpha=0``)
            * ``'zoh'``: zero-order hold

        ``'alpha'``: ``float`` within [0, 1]
            The generalized bilinear transformation weighting parameter,
            which should only be specified with ``method='gbt'``, and is
            ignored otherwise.

        ``'prewarp_frequency'``: ``float`` within [0, infinity)
            The frequency [rad/s] at which to match with the input continuous-time
            system's magnitude and phase (the gain = 1 crossover frequency, for
            example). Should only be specified with ``method = 'bilinear'`` or
            ``'gbt'`` with ``alpha = 0.5`` and ignored otherwise.

        """

    @abc.abstractmethod
    def iter_ss(
        self,
        n: int,
        *,
        n0: int = 0,
        sample: bool = True,
        sample_kws: dict[str, Any] | None = None,
    ) -> Iterator[tuple[int, int, ct.StateSpace]]:
        """generate state-space models and iterate over n samples

        Args
        ----
        n
            number of samples to generate (spanning only one of its dimensions)
        n0, optional
            first sample index, by default 0
        sample, optional
            ``True`` (default) to convert the system to discrete-time system, sampled at
            the ``self.fs`` sampling rate. ``False`` to return a continuous-time
            system. Control Systems Library package's ``sample()`` method is used
            to make the conversion.
        sample_kws, optional
            Specify the continuous-to-discrete time conversion options. See below
            for a brief summary of the available options or see Control Systems
            Library documentation.

        Yields
        ------
            list of (Control Systems Library) state-space model objects of all
            tube sections at each time instance

        Sample Options
        --------------

        ``'method'``: Method to use for sampling:

            * ``'gbt'``: generalized bilinear transformation
            * ``'backward_diff'``: Backwards difference (``'gbt'`` with ``alpha=1.0``)
            * ``'bilinear'`` (or ``'tustin'``): Tustin's approximation (``'gbt'`` with ``alpha=0.5``) (default)
            * ``'euler'``: Euler (or forward difference) method (``'gbt'`` with ``alpha=0``)
            * ``'zoh'``: zero-order hold

        ``'alpha'``: ``float`` within [0, 1]
            The generalized bilinear transformation weighting parameter,
            which should only be specified with ``method='gbt'``, and is
            ignored otherwise.

        ``'prewarp_frequency'``: ``float`` within [0, infinity)
            The frequency [rad/s] at which to match with the input continuous-time
            system's magnitude and phase (the gain = 1 crossover frequency, for
            example). Should only be specified with ``method = 'bilinear'`` or
            ``'gbt'`` with ``alpha = 0.5`` and ignored otherwise.
        """

    # @abc.abstractmethod
    # def __call__(
    #     self,
    #     n: int,
    #     n0: int = 0,
    #     sample_edges: bool = False,
    #     upsample_factor: int = 0,
    #     **kwargs,
    # ) -> NDArray:
    #     """generate and return waveform samples

    #     Parameters
    #     ----------
    #     n
    #         number of samples to generate (spanning only one of its dimensions)
    #     n0, optional
    #         first sample index, by default 0
    #     """

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int]:
        """shape of each sample (empty if scalar)"""
