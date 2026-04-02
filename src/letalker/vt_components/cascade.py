from typing import Any, Iterator

import control as ct
import numpy as np

from ..constants import c as c_default
from ..constants import rho_air as rho_air_default
from .abc import TwoPortSystem

rhoc_default = rho_air_default * c_default


class CascadeNetwork(TwoPortSystem):
    # Interconnection of two-port networks

    rhoc: float
    _subnetworks: tuple[TwoPortSystem]

    def __init__(self, *networks: tuple[TwoPortSystem], rhoc: float | None = None):
        """generic series system with multiple series flow-to-pressure-drop subsystems

        Parameters
        ----------
        areas
            cross-sectional areas of tube sections
        networks
            Two-port networks to be cascaded in the order given.
        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """
        super().__init__()

        if len(networks) == 0:
            raise ValueError(
                "At least one pressure-to-wall-flow subsystem must be given."
            )

        self._subnetworks = networks
        if rhoc is not None:
            self.rhoc = rhoc

    def nb_states(self) -> int:
        """Number of internal states"""
        return sum(net.nb_states() for net in self._subnetworks)

    def nb_sections(self) -> int:
        """Number of tube junctions (1 less than tube sections)"""
        return self._subnetworks[0].shape[0]

    def nb_inputs(self) -> int:
        """Number of inputs: 2 to 4"""
        return max(net.nb_inputs() for net in self._subnetworks)

    def nb_outputs(self) -> int:
        """Number of outputs: always 2"""
        return max(net.nb_outputs() for net in self._subnetworks)

    def is_lattice(self) -> bool:
        """``True`` if system can be implemented by Story95 lattice structure with 2 reflection coefficients plus a gain"""
        return all(net.is_lattice() for net in self._subnetworks)

    def has_aux_pressure_source(self) -> bool:
        """``True`` if a junction therein expects auxiliary pressure source(s)"""
        return any(net.has_aux_pressure_source() for net in self._subnetworks)

    def has_aux_flow_source(self) -> bool:
        """``True`` if a junction therein expects an auxiliary flow source"""
        return any(net.has_aux_flow_source() for net in self._subnetworks)

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
                Hv = ct.parallel(f(area, length) for f in self._subnetworks).ss()
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

    def _join(self, sys1: ct.LTI, sys2: ct.LTI) -> ct.StateSpace:

        ss1 = ct.ss(sys1)
        ss2 = ct.ss(sys2)

        nst1, nst2 = ss1.nstates, ss2.nstates
        nin1, nin2 = ss1.ninputs, ss2.ninputs
        nout1, nout2 = ss1.noutputs, ss2.noutputs

        assert nout1 == 2 and nout2 == 2 and nin1 >= 2 and nin2 >= 2

        naux1 = nin1 - nout1
        naux2 = nin2 - nout2
        nout = 2

        naux = naux1 + naux2
        nin = nout + naux
        nst = nst1 + nst2

        b12 = ss1.B[:, 1]
        b21 = ss2.B[:, 0]
        c11 = ss1.C[0, :]
        c22 = ss2.C[1, :]
        (d111, d112), (d121, d122) = ss1.D[:, :nout]
        (d211, d212), (d221, d222) = ss2.D[:, :nout]

        gamma = 1 - d112 * d221

        A = np.zeros((nst, nst))
        A[:nst1, :nst1] = ss1.A + (b12 * d221 / gamma) @ c11
        A[:nst1, nst1:] = (b12 / gamma) @ c22
        A[nst1:, :nst1] = (b21 / gamma) @ c11
        A[:nst1, :nst1] = ss2.A + (b21 * d112) @ c22

        B = np.zeros((nst, nin))
        B[:nst1, :nout] = ss1.B[:, :nout] @ np.array(
            [[1, 0], [d111 * d221 / gamma, d222 / gamma]]
        )
        B[nst1:, :nout] = ss2.B[:, :nout] @ np.array(
            [[d111 / gamma, d112 * d222 / gamma], [0, 1]]
        )
        if naux:
            n1 = nout + naux1
            B[:nst1, nout:n1] = ss1.B[:, nout:]
            B[nst1:, n1:] = ss2.B[:, nout:]

        K1 = np.array([[d211 / gamma, 0], [d122 * d221 / gamma, 1]])
        K2 = np.array([[1, d112 * d211 / gamma], [0, d122 / gamma]])
        C = np.zeros((nout, nst))
        C[:, :nst1] = K1 @ ss1.C
        C[:, :nst2] = K2 @ ss2.C

        D = np.array(
            [
                [d111 * d211 / gamma, d112 * d211 * d222 / gamma + d212],
                [d111 * d122 * d221 / gamma + d121, d122 * d222 / gamma],
            ]
        )
        if naux:
            D = K1 @ ss1.D[:, nout:]
            D = K2 @ ss2.D[:, nout:]

            n1 = nout + naux1
            B[:nst1, nout:n1] = ss1.B[:, nout:]
            B[nst1:, n1:] = ss2.B[:, nout:]

        return ct.ss(A, B, C, D)
