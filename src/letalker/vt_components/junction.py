from typing import Any, Iterator

import control as ct
import numpy as np

from ..__util import format_parameter
from ..constants import c as c_default
from ..constants import rho_air as rho_air_default
from ..function_generators import SampleGenerator
from .abc import TwoPortSystem

rhoc_default = rho_air_default * c_default


class LosslessJunction(TwoPortSystem):
    # Lossless junction VT block with possible independent pressure/flow sources

    areas: SampleGenerator
    _aux_psrc: bool
    _aux_fsrc: bool
    rhoc: float = rhoc_default

    def __init__(
        self,
        areas: np.ndarray,
        has_pressure_source: bool,
        has_flow_source: bool,
        *,
        rhoc: float | None = None,
    ):
        """Tube section junction

        Parameters
        ----------
        areas
            cross-sectional areas of tube sections
        has_pressure_source
            ``True`` if there is any pressure source at junction, i.e., kinetic
            pressure drop or approximated viscous loss of the previous section
        has_flow_source
            ``True`` if there is a turbulent flow source at junction
        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """
        super().__init__()

        self.areas = format_parameter(areas, 1)
        self._aux_psrc = bool(has_pressure_source)
        self._aux_fsrc = bool(has_flow_source)
        if rhoc is not None:
            self.rhoc = rhoc

    def nb_states(self) -> int:
        """Number of internal states"""
        return 0

    def nb_sections(self) -> int:
        """Number of tube junctions (1 less than tube sections)"""
        return self.areas.shape[0] - 1

    def nb_inputs(self) -> int:
        """Number of inputs: 2 + number of auxilary sources"""
        return 2 + self._aux_psrc + self._aux_psrc

    def nb_outputs(self) -> int:
        """Number of outputs (always 2)"""
        return 2

    def is_lattice(self) -> bool:
        """``True`` if system can be implemented by Story95 lattice structure with 2 reflection coefficients plus a gain"""
        return True

    def has_aux_pressure_source(self) -> bool:
        """``True`` if system has auxiliary pressure source at junction"""
        return self._aux_psrc

    def has_aux_flow_source(self) -> bool:
        """``True`` if system has auxiliary flow source at junction"""
        return self._aux_fsrc

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

        areas = self.areas(n, n0)
        D = self._ss_d(areas)
        n = D.shape[0]
        nsec = self.nb_sections
        nin = self.nb_inputs

        return (
            np.empty((n, nsec, 0, 0)),
            np.empty((n, nsec, 0, nin)),
            np.empty((n, nsec, 2, 0)),
            D,
        )

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
        i
            time sample index
        j
            junction index
        sys
            Control Systems Library state-space model object

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

        A = np.empty((0, 0))
        B = np.empty((0, self.nb_inputs))
        C = np.empty((2, 0))

        for i in range(n0, n0 + n):
            D = self._ss_d(self.areas(1, i))
            for j, Dj in enumerate(D[0]):
                yield i, j, ct.StateSpace(A, B, C, Dj)

    def _ss_d(self, areas: np.ndarray) -> np.ndarray:
        """compute the feedthrough matrices"""

        rhoc = self.rhoc
        den = (areas[..., :-1] + areas[..., 1:]).reshape(..., 1, 1)
        m1 = [areas[..., :-1], rhoc] / den
        m2 = [areas[..., 1:], rhoc] / den
        d = m1 - m2
        D = np.zeros((*areas.shape[:-1], self.nb_sections, 2, self.nb_inputs))
        D[..., 0, 0] = 2 * m1
        D[..., 1, 1] = 2 * m2
        D[..., 0, 1] = -d
        D[..., 1, 0] = d

        if self.has_aux_pressure_source:
            D[..., 0, 2] = -m1
            D[..., 1, 2] = m2
        if self.has_aux_flow_source:
            D[..., :, -1] = rhoc / den
        return D
