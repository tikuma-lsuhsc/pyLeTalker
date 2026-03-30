from typing import Any, Iterator, overload

import control as ct
import numpy as np

from ..__util import format_parameter
from ..constants import c as c_default
from ..constants import rho_air as rho_air_default
from ..function_generators import SampleGenerator
from .abc import LTIFactory, TwoPortSystem

rhoc_default = rho_air_default * c_default


class DefaultYieldingWallTF(LTIFactory):
    """Yielding wall model by Milenkvic1988

    Parameters
    ----------
    LTIFactory
        _description_

    Returns
    -------
        _description_
    """

    M: float = 1.5  # g/cm^2
    K: float = 33000  # dyne/cm^3
    B: float = 1060  # dyne/cm^3

    def __init__(
        self, Gt: float | None = None, K: float | None = None, B: float | None = None
    ):
        super().__init__()

        if M is not None:
            self.M = M
        if K is not None:
            self.K = K
        if B is not None:
            self.B = B

    def __call__(self, area: float, length: float) -> ct.LTI:

        c = 2 * length * (np.pi * area) ** 0.5
        Lw = self.M / c
        Cw = c / self.K
        Rw = self.B / c

        return ct.tf([Cw, 0], [Lw * Cw, Rw * Cw, 1])

    @property
    def nb_states(self) -> int:
        return 2


class DefaultHeatLossGain(LTIFactory):
    """Fixed heat loss model

    Parameters
    ----------
    LTIFactory
        _description_

    Returns
    -------
        _description_
    """

    Gt: float = 8.07e-7 * 1.08 * 1000 * 0.5  # g/cm^2

    def __init__(self, Gt: float | None = None):
        super().__init__()

        if Gt is not None:
            self.Gt = Gt

    def __call__(self, area: float, length: float) -> float:

        return self.Gt * length * area**-0.5

    @property
    def nb_states(self) -> int:
        return 0


class ShuntNetwork(TwoPortSystem):
    # Shunt networks representing flows into wall

    areas: SampleGenerator
    _lti_factories: tuple[LTIFactory]
    rhoc: float = rhoc_default

    @overload
    def __init__(
        self,
        areas: np.ndarray,
        yielding_wall: LTIFactory,
        /,
        rhoc: float | None = None,
    ):
        """shunt system with a yielding wall transfer function

        Parameters
        ----------
        areas
            cross-sectional areas of tube sections
        yielding_wall
            factory to craete a continuous-time transfer function of yielding wall
            (input: pressure, output: wall volume flow) given a cross-sectional
            area and length of a tube section
        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """

    @overload
    def __init__(
        self,
        areas: np.ndarray,
        yielding_wall: LTIFactory,
        heat_loss: LTIFactory,
        /,
        rhoc: float | None = None,
    ):
        """shunt system with a yielding wall and heat loss transfer functions

        Parameters
        ----------
        areas
            cross-sectional areas of tube sections
        yielding_wall
            factory to create a continuous-time transfer function of yielding wall
            (input: pressure, output: wall volume flow) given a cross-sectional
            area and length of a tube section
        heat_loss
            factory to create a continuous-time transfer function of heat loss
            (input: pressure, output: wall volume flow) given a cross-sectional
            area and length of a tube section
        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """

    def __init__(
        self,
        areas: np.ndarray,
        /,
        *p_to_uw_lti_systems: tuple[LTIFactory],
        rhoc: float | None = None,
    ):
        """generic shunt system with multiple parallel pressure-to-wall-flow subsystems

        Parameters
        ----------
        areas
            cross-sectional areas of tube sections
        p_to_uw_lti_systems
            factories to create continuous-time transfer functions from pressure
            to wall-flow that are present in parallel.
        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """
        super().__init__()

        if len(p_to_uw_lti_systems) == 0:
            raise ValueError(
                "At least one pressure-to-wall-flow subsystem must be given."
            )

        self.areas = format_parameter(areas, 1)
        self._lti_factories = p_to_uw_lti_systems
        if rhoc is not None:
            self.rhoc = rhoc

    def nb_states(self) -> int:
        """Number of internal states"""
        return sum(f.nb_states for f in self._lti_factories)

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
            else {**default_sample_kws, **self._sample_kws}
        )
        rhoc = self.rhoc
        length = self.dz

        for i, areas in enumerate(self.areas(n, n0=n0)):
            for j, area in enumerate(areas):
                Hw = ct.parallel(f(area) for f in self._lti_factories).ss()
                if isinstance(Hw, ct.TransferFunction):
                    Hw = ct.tf2ss(Hw)
                if sample:
                    Hw = Hw.sample(self.dt, **sample_kws)

                two_area = 2 * area
                rhoc_d = rhoc * Hw.D
                den = 1 / (two_area + rhoc_d)
                k1 = two_area / den
                k2 = -rhoc / den
                A = Hw.A - (Hw.B * den) @ Hw.C
                B = np.tile(Hw.B * k1, (1, 2))
                C = np.tile(k2 * Hw.C, (2, 1))
                k3 = k2 * Hw.D
                D = np.array([k1, k3], [k3, k1])

                yield i, j, A, B, C, D
