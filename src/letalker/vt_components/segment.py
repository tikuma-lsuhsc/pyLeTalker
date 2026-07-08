from typing import Any

import control as ct
import numpy as np

from ..constants import c as c_default
from ..constants import mu as mu_default
from ..constants import rho_air as rho_air_default
from .abc import LTISegmentFactory
from .cascade import cascade

rhoc_default = rho_air_default * c_default


class DefaultYieldingWall(LTISegmentFactory):
    """Yielding wall model by Milenkvic1988"""

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

        tf = ct.tf([Cw, 0], [Lw * Cw, Rw * Cw, 1])
        assert tf is not None
        return (
            tf
            if fs is None
            else tf.sample(1 / fs, **(sample_kws if sample_kws else {}))
        )


class DefaultHeatLossGain(LTISegmentFactory):
    """Fixed heat loss model"""

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

        return ct.tf([self.Gt * length * area**-0.5], [1.0], dt=fs and 1 / fs)


class DefaultViscousLossTF(LTISegmentFactory):
    """Viscous loss model by Story 1995

    Parameters
    ----------
    LTISegmentFactory
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

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.LTI:

        c = 2 * (area * np.pi) ** 0.5 * length * (self.rhomu / 2) ** 0.5 / area**2
        k = self.omega**0.5
        Rvsc = c * k
        Lvsc = c / k

        tf = ct.tf([Lvsc, Rvsc], [1])
        assert isinstance(tf, ct.LTI)

        return (
            tf
            if fs is None
            else ct.sample(tf, 1 / fs, **(sample_kws if sample_kws else {}))
        )


class DefaultLaminarResistance(LTISegmentFactory):
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

    _lti_factories: tuple[LTISegmentFactory]
    rhoc: float = rhoc_default

    def __init__(
        self,
        *p_to_u_lti_systems: tuple[LTISegmentFactory],
        rhoc: float | None = None,
    ):
        """Factory of a 2-in/2-out wave-reflection two-port subsystem, which
        models yielding wall and heat loss

        Parameters
        ----------
        p_to_u_lti_systems
            factories to create continuous-time transfer functions from pressure
            drop to to-wall flow rate that are present in parallel.

            These yielding wall and heat loss functions (or gains). If
            non-assigned,

        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """

        self._lti_factories = p_to_u_lti_systems
        if rhoc is not None:
            self.rhoc = rhoc

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.LTI:
        """generate state-space models and iterate over n samples

        Args
        ----
        area
            cross-sectional areas of tube sections
        length
            length (in cm) of each tube section
        """

        kws = {"fs": fs, "sample_kws": sample_kws}

        nsys = len(self._lti_factories)
        Hw: ct.StateSpace = (
            ct.parallel(*(f(area, length, **kws) for f in self._lti_factories))
            if nsys
            else ct.tf([0.0], [1.0], fs and 1 / fs)  # no shunt loss
        ).to_ss()

        two_area = 2 * area
        rhoc_d = self.rhoc * Hw.D[0, 0]
        den = 1 / (two_area + rhoc_d)
        k1 = two_area / den
        k2 = -self.rhoc / den
        A = Hw.A - (Hw.B * den) @ Hw.C
        B = np.tile(Hw.B * k1, (1, 2))
        C = np.tile(k2 * Hw.C, (2, 1))
        k3 = k2 * Hw.D[0, 0]
        D = np.array([[k1, k3], [k3, k1]])

        return ct.StateSpace(A, B, C, D, dt=Hw.dt)


class SeriesNetwork(LTISegmentFactory):
    _lti_factories: tuple[LTISegmentFactory]
    rhoc: float = rhoc_default

    def __init__(
        self,
        /,
        *u_to_p_lti_systems: tuple[LTISegmentFactory],
        rhoc: float | None = None,
    ):
        """Factory of a 2-in/2-out wave-reflection two-port subsystem, which
        models viscous and laminar losses of a vocal tract segment

        Parameters
        ----------
        u_to_p_lti_systems
            factories to create continuous-time transfer functions from flow to
            pressure drop that are present in series (or parallel in systems sense).

            These include viscous and laminar loss functions (or gains)

        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """

        self._lti_factories = u_to_p_lti_systems

        if rhoc is not None:
            self.rhoc = rhoc

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.LTI:
        """generate a 2-in/2-out state-space model of from (F1,B2)->(F2,B1)

        Args
        ----
        area
            cross-sectional areas of tube sections
        length
            length (in cm) of each tube section
        """

        rhoc = self.rhoc
        kws = {"fs": fs, "sample_kws": sample_kws}

        nsys = len(self._lti_factories)
        Hv: ct.TransferFunction = (
            ct.parallel(*(f(area, length, **kws) for f in self._lti_factories))
            if nsys
            else ct.tf(
                [rhoc / area], [1.0], dt=fs and 1 / fs
            )  # no loss flow-pressure conversion
        ).to_tf()

        if len(Hv.den[0][0]) == 1 and len(Hv.num[0][0]) == 2:
            L, R = Hv.num[0][0] / Hv.den[0][0]
            Z = rhoc / area
            A = -(R + 2 * Z) / L
            b = 2 / L
            B = np.array([[b, -b]])
            C = np.array([[Z], [-Z]])
            D = np.array([[0, 1], [1, 0]])
        else:
            Hv: ct.StateSpace = Hv.to_ss()
            two_rhoc = 2 * rhoc
            ad = area * Hv.D[0, 0]
            den = 1 / (ad + two_rhoc)
            A = Hv.A - (Hv.B * den * area / rhoc) @ Hv.C
            b = Hv.B * (2 * area * den)
            B = np.stack([b, -b], 1)
            c = den * Hv.C
            C = np.stack([c, -c, 0])
            k1 = two_rhoc * den
            k2 = ad * den
            D = np.array([k1, k2], [k2, k1])

        return ct.StateSpace(A, B, C, D, dt=Hv.dt)


######################


class TSegment(LTISegmentFactory):
    series: SeriesNetwork
    shunt: ShuntNetwork

    def __init__(self, series: SeriesNetwork, shunt: ShuntNetwork):
        self.series = series
        self.shunt = shunt

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.LTI:

        series = self.series(area, length / 2)
        shunt = self.shunt(area, length)

        return cascade(cascade(series, shunt), series)


class PiSegment(LTISegmentFactory):
    series: SeriesNetwork
    shunt: ShuntNetwork

    def __init__(self, series: SeriesNetwork, shunt: ShuntNetwork):
        self.series = series
        self.shunt = shunt

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.LTI:

        shunt = self.shunt(area, length / 2)
        series = self.series(area, length)

        return cascade(cascade(shunt, series), shunt)
