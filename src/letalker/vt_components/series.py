from typing import Any, Iterator, overload

import control as ct
import numpy as np

from ..__util import format_parameter
from ..constants import c as c_default
from ..constants import mu as mu_default
from ..constants import rho_air as rho_air_default
from ..function_generators import SampleGenerator
from .abc import LTIFactory, TwoPortSystem

rhoc_default = rho_air_default * c_default


class DefaultViscousLossTF(LTIFactory):
    """Viscous loss model by Story 1995

    Parameters
    ----------
    LTIFactory
        _description_

    Returns
    -------
        _description_
    """

    omega: float = np.pi * 2000
    rhomu: float = rho_air_default * mu_default

    def __init__(self, omega: float | None = None, rhomu: float | None = None):
        super().__init__()

        if omega is not None:
            self.omega = omega
        if rhomu is not None:
            self.rhomu = rhomu

    def __call__(self, area: float, length: float) -> ct.LTI | float:

        c = 2 * (area * np.pi) ** 0.5 * length * (self.rhomu / 2) ** 0.5 / area**2
        k = self.omega**0.5
        Rvsc = c * k
        Lvsc = c / k

        return ct.tf([Lvsc, Rvsc], [1])

    @property
    def nb_states(self) -> int:
        return 1


class DefaultLaminarResistance(LTIFactory):
    """Fixed laminar resistance model

    Parameters
    ----------
    LTIFactory
        _description_

    Returns
    -------
        _description_
    """

    mu: float = mu_default

    def __init__(self, mu: float | None = None):
        super().__init__()

        if mu is not None:
            self.mu = mu

    def __call__(self, area: float, length: float) -> ct.LTI | float:

        return 8 * np.pi * self.mu * area**-2 * length

    @property
    def nb_states(self) -> int:
        return 0


class SeriesNetwork(TwoPortSystem):
    # Series networks representing pressure losses

    areas: SampleGenerator
    length: float
    _lti_factories: tuple[LTIFactory]
    rhoc: float = rhoc_default

    @overload
    def __init__(
        self,
        areas: np.ndarray,
        length: float,
        viscous_loss: LTIFactory,
        /,
        rhoc: float | None = None,
    ):
        """series system with a viscous loss transfer function

        Parameters
        ----------
        areas
            cross-sectional areas of tube sections
        length
            length (in cm) of each tube section
        viscous_loss
            factory to create a continuous-time transfer function of viscous loss
            (input: flow, output: pressure drop) given a cross-sectional
            area and length of a tube section
        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """

    @overload
    def __init__(
        self,
        areas: np.ndarray,
        length: float,
        viscous_loss: LTIFactory,
        laminar_resistance: LTIFactory,
        /,
        rhoc: float | None = None,
    ):
        """shunt system with a viscous loss and laminar resistance transfer functions

        Parameters
        ----------
        areas
            cross-sectional areas of tube sections
        length
            length (in cm) of each tube section
        viscous_loss
            factory to create a continuous-time transfer function of viscous loss
            (input: flow, output: pressure drop) given a cross-sectional
            area and length of a tube section
        laminar_resistance
            factory to create a continuous-time transfer function of laminar
            resistance (input: flow, output: pressure drop) given a cross-sectional
            area and length of a tube section
        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """

    def __init__(
        self,
        areas: np.ndarray,
        length: float,
        /,
        *u_to_p_lti_systems: tuple[LTIFactory],
        rhoc: float | None = None,
    ):
        """generic series system with multiple parallel flow-to-pressure-drop subsystems

        Parameters
        ----------
        areas
            cross-sectional areas of tube sections
        u_to_p_lti_systems
            factories to create continuous-time transfer functions from flow to
            pressure drop that are present in series (or parallel in systems sense).
        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """
        super().__init__()

        if len(u_to_p_lti_systems) == 0:
            raise ValueError(
                "At least one pressure-to-wall-flow subsystem must be given."
            )

        self.areas = format_parameter(areas, 1)
        self.length = length
        self._lti_factories = u_to_p_lti_systems
        if rhoc is not None:
            self.rhoc = rhoc

    def nb_states(self) -> int:
        """Number of internal states"""
        return sum(f.nb_states for f in self._lti_factories)

    def nb_sections(self) -> int:
        """Number of tube junctions (1 less than tube sections)"""
        return self.areas.shape[0] - 1

    def nb_inputs(self) -> int:
        """Number of inputs: always 2"""
        return 2

    def nb_outputs(self) -> int:
        """Number of outputs: always 2"""
        return 2

    def is_lattice(self) -> bool:
        """``True`` if system can be implemented by Story95 lattice structure with 2 reflection coefficients plus a gain"""
        return False

    def has_aux_pressure_source(self) -> bool:
        """Always ``False``, no auxiliary pressure source"""
        return False

    def has_aux_flow_source(self) -> bool:
        """Always ``False``, no auxiliary flow source"""
        return False

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

        nsec = self.nb_sections
        nst = self.nb_states

        if self.areas.is_fixed():
            n = 1

        A = np.empty((n, nsec, nst, nst))
        B = np.empty((n, nsec, nst, 2))
        C = np.empty((n, nsec, 2, nst))
        D = np.empty((n, nsec, 2, 2))

        for i, j, Aij, Bij, Cij, Dij in self._iter_ss(
            n, n0=n0, sample=sample, sample_kws=sample_kws
        ):
            A[i, j, :, :] = Aij
            B[i, j, :, :] = Bij
            C[i, j, :, :] = Cij
            D[i, j, :, :] = Dij

        return A, B, C, D

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

        for i, j, A, B, C, D in self._iter_ss(
            n, n0=n0, sample=sample, sample_kws=sample_kws
        ):
            yield i, j, ct.ss(A, B, C, D, sample and self.dt)

    def _iter_ss(
        self,
        n: int,
        *,
        n0: int = 0,
        sample: bool = True,
        sample_kws: dict[str, Any] | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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
        A,B,C,D
            state-space matrices

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

        default_sample_kws = {"method": "bilinear"}
        sample_kws = (
            default_sample_kws
            if sample_kws is None
            else {**default_sample_kws, **sample_kws}
        )
        rhoc = self.rhoc
        length = self.length

        for i, areas in enumerate(self.areas(n, n0=n0)):
            for j, area in enumerate(areas):
                Hv = ct.parallel(f(area, length) for f in self._lti_factories).ss()
                if isinstance(Hv, ct.TransferFunction):
                    Hv = ct.tf2ss(Hv)
                if sample:
                    Hv = Hv.sample(self.dt, **sample_kws)

                two_rhoc = 2 * rhoc
                ad = area * Hv.D
                den = 1 / (ad + two_rhoc)
                A = Hv.A - (Hv.B * den * area / rhoc) @ Hv.C
                b = Hv.B * (2 * area * den)
                B = np.stack([b, -b], 1)
                c = den * Hv.C
                C = np.stack([c, -c, 0])
                k1 = two_rhoc * den
                k2 = ad * den
                D = np.array([k1, k2], [k2, k1])

                yield i, j, A, B, C, D
