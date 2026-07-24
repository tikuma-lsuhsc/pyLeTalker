from typing import Any

import control as ct
import numpy as np

from ..constants import c as c_default
from ..constants import mu as mu_default
from ..constants import rho_air as rho_air_default
from ..constants import vt_atten as atten_default
from .abc import LTIImpedanceFactory, LTISegmentFactory
from .cascade import cascade

rhoc_default = rho_air_default * c_default


class DefaultLosslessPropagationTF(LTIImpedanceFactory):
    """Flanagan's Acoustic L (without viscous loss)

    .. math::

      P/U = s L_a

    where

    .. math::

      L_a = \rho/A

    is the acoustic inertance per unit length.

    """

    rho: float = rho_air_default

    def __init__(
        self,
        rho: float | None = None,
    ):
        super().__init__()

        if rho is not None:
            self.rho = rho

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.TransferFunction:

        L = self.rho / area

        tf = ct.tf([L * length, 0], [1])
        assert isinstance(tf, ct.TransferFunction)

        if fs is not None:
            tf = 1 / (1 / tf).sample(1 / fs, **(sample_kws if sample_kws else {}))

        return tf


class DefaultYieldingWall(LTIImpedanceFactory):
    """Yielding wall impedance by Milenkvic1988"""

    M: float = 1.5  # g/cm^2
    K: float = 33000  # dyne/cm^3
    B: float = 1060  # dyne s/cm^3

    def __init__(
        self, M: float | None = None, K: float | None = None, B: float | None = None
    ):
        """Create a factory to create a default yielding wall transfer function model

        The transfer function is evaluated on the fly as the object is called
        with area and length of the vocal tract segment.

        Parameters
        ----------
        M, optional
            mass per unit surface area in g/cm², by default 1.5
        K, optional
            stiffness per unit surface area in dyne/cm³, by default 33000
        B, optional
            resistance per unit surface area in dyne×s/cm³, by default 1060
        """
        super().__init__()

        if M is not None:
            self.M = M
        if K is not None:
            self.K = K
        if B is not None:
            self.B = B

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.LTI:
        """Create a tf model of one vocal tract segment

        Parameters
        ----------
        area
            vocal tract segment cross-sectional area in cm²
        length
            vocal tract segment length in cm

        Returns
        -------
            Transfer function object.
        """
        c: float = 2 * length * (np.pi * area) ** 0.5
        Lw = self.M / c
        Cw = c / self.K
        Rw = self.B / c

        tf = ct.tf([Lw * Cw, Rw * Cw, 1], [Cw, 0])
        assert tf is not None
        return (
            tf
            if fs is None
            else tf.sample(1 / fs, **(sample_kws if sample_kws else {}))
        )


class DefaultHeatLossGain(LTIImpedanceFactory):
    """Fixed heat loss impedance model"""

    Gt: float = 8.07e-7 * 1.08 * 1000 * 0.5  # g/cm^2

    def __init__(self, Gt: float | None = None):
        """_summary_

        Parameters
        ----------
        Gt, optional
            _description_, by default None
        """
        super().__init__()

        if Gt is not None:
            self.Gt = Gt

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.LTI:

        return ct.tf([1.0], [self.Gt * length * area**-0.5], dt=fs and 1 / fs)


class DefaultViscousLossTF(LTIImpedanceFactory):
    """Flangan's Acoustic R, viscous loss model"""

    resistive_only: bool = False
    omega: float = np.pi * 2000  # default: 1000 Hz (Story, 1995)
    rho: float = rho_air_default
    mu: float = mu_default

    _r_const: float = (rho_air_default * mu_default / 2 * omega) ** 0.5
    _l_const: float = (rho_air_default * mu_default / (2 * omega)) ** 0.5

    def __init__(
        self,
        resistive_only: bool = False,
        omega: float | None = None,
        rho: float | None = None,
        mu: float | None = None,
    ):
        super().__init__()

        if resistive_only:
            self.resistive_only = True

        if omega is not None or rho is not None or mu is not None:
            if rho is None:
                rho = self.rho
            else:
                self.rho = rho
            if mu is None:
                mu = self.mu
            else:
                self.mu = mu
            if omega is None:
                omega = self.omega
            else:
                self.omega = omega

            self.sqrt_rhomu_half = ((rho * (mu or mu_default)) / 2) ** 0.5

            c = rho * mu / 2
            self._r_const = (c * omega) ** 0.5
            self._l_const = (c / omega) ** 0.5

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.LTI:

        # r = (area / np.pi) ** 0.5
        # S = 2 * np.pi * r
        a = 2 * (np.pi / area) ** 0.5 / area

        Rvsc = a * self._r_const * length
        if self.resistive_only:
            tf = ct.tf([Rvsc], [1])
        else:
            Lvsc = a * self._l_const * length
            tf = ct.tf([Lvsc, Rvsc], [1])

        assert isinstance(tf, ct.LTI)

        return (
            tf
            if fs is None
            else tf.sample(1 / fs, **(sample_kws if sample_kws else {}))
        )


class DefaultLaminarResistance(LTIImpedanceFactory):
    """Fixed laminar resistance model"""

    mu: float = mu_default

    def __init__(self, mu: float | None = None):
        super().__init__()

        if mu is not None:
            self.mu = mu

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.LTI:

        return ct.tf([8 * np.pi * self.mu * area**-2 * length], [1.0], dt=fs and 1 / fs)


#######################################


class ShuntNetwork(LTISegmentFactory):
    # Shunt networks representing flows into wall

    _z_factories: tuple[LTISegmentFactory]
    rhoc: float = rhoc_default

    def __init__(
        self,
        *impedance_factories: tuple[LTISegmentFactory],
        rhoc: float | None = None,
    ):
        """Factory of a 2-in/2-out wave-reflection two-port subsystem, which
        models yielding wall and heat loss

        Parameters
        ----------
        impedance_factories
            factories to create continuous-time transfer functions from to-wall
            flow rate to the pressure drop. All impedances are present in parallel.

            These yielding wall and heat loss functions (or gains). If
            non-assigned,

        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """

        self._z_factories = impedance_factories
        if rhoc is not None:
            self.rhoc = rhoc

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
        _use_improper: bool = False,
    ) -> ct.LTI:
        """generate state-space models and iterate over n samples

        Args
        ----
        area
            cross-sectional areas of tube sections
        length
            length (in cm) of each tube section
        """

        kws = {} if sample_last else {"fs": fs, "sample_kws": sample_kws}

        # calculate and combine all impedances into a single system
        itY = (1 / f(area, length, **kws) for f in self._z_factories)
        try:
            Yz0 = next(itY)
        except StopIteration:
            Yz = ct.tf([0], [1], dt=fs and 1 / fs)
        else:
            Yz = sum(itY, start=Yz0)

        assert Yz.issiso()

        Y = area / self.rhoc

        # conditioned on the properness of Z
        nznum, nzden = [len(n[0][0]) for n in (Yz.num, Yz.den)]
        if nznum < nzden or (nznum == nzden and not _use_improper):
            # use admittance
            Hw: ct.StateSpace = Yz.to_ss()

            Q = np.array([[1, -1, 0], [1, 0, -1], [Y, Y, Hw.D[0, 0]]])
            P = np.zeros((3, Hw.nstates))
            P[2] = -Hw.C[0]
            R = np.array([[1, -1], [0, -1], [Y, Y]])

        else:
            # use impedance
            Hw = (1 / Yz).to_ss()

            Q = np.array([[1, -1, 0], [1, 0, -Hw.D[0, 0]], [Y, Y, 1]])
            P = np.zeros((3, Hw.nstates))
            P[1] = Hw.C[0]
            R = np.array([[1, -1], [0, Y], [Y, Y]])

        C, cu = np.vsplit(np.linalg.lstsq(Q, P)[0], [2])
        D, du = np.vsplit(np.linalg.lstsq(Q, R)[0], [2])
        A = Hw.A + Hw.B @ cu
        B = Hw.B @ du

        sys = ct.StateSpace(
            A, B, C, D, dt=Hw.dt, inputs=["F1", "B2"], outputs=["F2", "B1"]
        )

        if sample_last and fs is not None:
            sys = sys.sample(1 / fs, **kws)

        return sys


class SeriesNetwork(LTISegmentFactory):
    _z_factories: tuple[LTIImpedanceFactory]
    rhoc: float = rhoc_default

    def __init__(
        self,
        /,
        *series_impedances: tuple[LTIImpedanceFactory],
        rhoc: float | None = None,
    ):
        """Factory of a 2-in/2-out wave-reflection two-port subsystem, which
        models viscous and laminar losses of a vocal tract segment

        Parameters
        ----------
        series_impedances
            factories to create continuous-time transfer functions from flow to
            pressure drop that are present in series (or parallel in systems sense).

            These include viscous and laminar loss functions (or gains)

        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """

        self._z_factories = series_impedances

        if rhoc is not None:
            self.rhoc = rhoc

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
        _use_improper: bool = False,
    ) -> ct.StateSpace:

        kws = {} if sample_last else {"fs": fs, "sample_kws": sample_kws}

        # calculate and combine all impedances into a single system
        itZ = (f(area, length, **kws) for f in self._z_factories)
        try:
            Z0 = next(itZ)
        except StopIteration:
            Z = ct.tf([0], [1], dt=fs and 1 / fs)
        else:
            Z = sum(itZ, start=Z0)

        assert Z.issiso()

        Y = area / self.rhoc

        # conditioned on the properness of Z
        nznum, nzden = [len(n[0][0]) for n in (Z.num, Z.den)]
        if nznum < nzden or (nznum == nzden and not _use_improper):
            # use impedance
            Hv: ct.StateSpace = Z.to_ss()

            Q = np.array([[1, -1, Hv.D[0, 0]], [Y, Y, 0], [Y, 0, -1]])
            P = np.zeros((3, Hv.nstates))
            P[0] = -Hv.C[0]
            R = np.array([[1, -1], [Y, Y], [0, Y]])

        else:
            # use admittance
            Hv = (1 / Z).to_ss()

            Q = np.array([[1, -1, 1], [Y, Y, 0], [Y, 0, -Hv.D[0, 0]]])
            P = np.zeros((3, Hv.nstates))
            P[-1] = Hv.C[0]
            R = np.array([[1, -1], [Y, Y], [0, Y]])

        C, cu = np.vsplit(np.linalg.lstsq(Q, P)[0], [2])
        D, du = np.vsplit(np.linalg.lstsq(Q, R)[0], [2])
        A = Hv.A + Hv.B @ cu
        B = Hv.B @ du

        C[np.isclose(C, [[0]])] = 0
        D[np.isclose(D, [[0]])] = 0

        sys = ct.StateSpace(
            A, B, C, D, dt=Hv.dt, inputs=["F1", "B2"], outputs=["F2", "B1"]
        )

        if sample_last and fs is not None:
            sys = sys.sample(1 / fs, **kws)

        return sys


######################


class SegmentBase(LTISegmentFactory):
    series: SeriesNetwork
    shunt: ShuntNetwork

    def __init__(
        self, series: SeriesNetwork | None = None, shunt: ShuntNetwork | None = None
    ):
        self.series = series or SeriesNetwork(
            DefaultLaminarResistance(), DefaultViscousLossTF()
        )
        self.shunt = shunt or ShuntNetwork(DefaultHeatLossGain(), DefaultYieldingWall())


class TSegment(SegmentBase):
    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.LTI:

        series = self.series(area, length / 2)
        shunt = self.shunt(area, length)

        sys = cascade(cascade(series, shunt), series)

        if fs is None:
            return sys

        return sys.sample(Ts=1 / fs, **(sample_kws or {}))


class PiSegment(SegmentBase):
    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.LTI:

        shunt = self.shunt(area, length / 2)
        series = self.series(area, length)

        sys = cascade(cascade(shunt, series), shunt)
        if fs is None:
            return sys
        return sys.sample(Ts=1 / fs, **(sample_kws or {}))


class DTDelaySegmentBase(LTISegmentFactory):
    atten: float = atten_default

    def __init__(self, atten: float | None = None):
        if atten is not None:
            self.atten = atten


class DTForwardDelay(DTDelaySegmentBase):
    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
        input_id: int = 1,
        output_id: int = 2,
    ) -> ct.LTI:

        assert fs is not None, "DTForwardDelay is discrete-time only"

        alpha = 1 - self.atten / area**0.5

        A = np.zeros((1, 1))
        B = np.array([[alpha, 0]])
        C = np.array([[1], [0]])
        D = np.array([[0, 0], [0, alpha]])

        return ct.ss(
            A,
            B,
            C,
            D,
            1 / fs,
            inputs=["F1", "B2"],
            outputs=["F2", "B1"],
        )


class DTBackwardDelay(DTDelaySegmentBase):
    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.LTI:

        assert fs is not None, "DTBackwardDelay is discrete-time only"

        alpha = 1 - self.atten / area**0.5

        A = np.zeros((1, 1))
        B = np.array([[0, alpha]])
        C = np.array([[0], [1]])
        D = np.array([[alpha, 0], [0, 0]])

        return ct.ss(
            A,
            B,
            C,
            D,
            1 / fs,
            inputs=["F1", "B2"],
            outputs=["F2", "B1"],
        )
