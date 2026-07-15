from typing import Any

import control as ct
import numpy as np

from ..constants import c as c_default
from ..constants import mu as mu_default
from ..constants import rho_air as rho_air_default
from ..constants import vt_atten as atten_default
from .abc import LTISegmentFactory, LTISinkFactory, LTISourceFactory
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
        sample_last: bool = False,
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
        sample_last: bool = False,
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

    omega: float = np.pi * 2000  # default: 1000 Hz (Story, 1995)
    rho: float = rho_air_default
    mu: float = mu_default

    _r_const: float = (rho_air_default * mu_default / 2 * omega) ** 0.5
    _l_const: float = (rho_air_default * mu_default / (2 * omega)) ** 0.5

    def __init__(
        self,
        omega: float | None = None,
        rho: float | None = None,
        mu: float | None = None,
    ):
        super().__init__()

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
        sample_last: bool = False,
    ) -> ct.LTI:

        # r = (area / np.pi) ** 0.5
        # S = 2 * np.pi * r
        a = 2 * (np.pi / area) ** 0.5 / area

        Rvsc = a * self._r_const * length
        Lvsc = (self.rho / area + a * self._l_const) * length

        tf = ct.tf([Lvsc, Rvsc], [1])
        assert isinstance(tf, ct.LTI)

        return (
            tf
            if fs is None
            else tf.sample(1 / fs, **(sample_kws if sample_kws else {}))
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
        sample_last: bool = False,
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
        sample_last: bool = False,
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
            else ct.tf([0.0], [1.0], fs and 1 / fs)  # no flow loss
        ).to_ss()

        Y = area / self.rhoc
        dw = Hw.D[0, 0]
        den = 2 * Y + dw

        C = np.tile(-Hw.C / den, (2, 1))
        D = np.array([[2 * Y, -dw], [-dw, 2 * Y]]) / den
        A = Hw.A + Hw.B @ C[1:]
        B = Hw.B @ (D[1:] + np.array([[1, 0]]))

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
        sample_last: bool = False,
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
            else ct.tf([0], [1], dt=fs and 1 / fs)  # no pressure loss
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
            B = np.concatenate([b, -b], 1)
            c = den * Hv.C
            C = np.concatenate([c, -c])
            k1 = two_rhoc * den
            k2 = ad * den
            D = np.array([[k1, k2], [k2, k1]])

        return ct.StateSpace(A, B, C, D, dt=Hv.dt)


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
            name="forward_delay",
            inputs=[f"F{input_id}", f"B{output_id}"],
            outputs=[f"F{output_id}", f"B{input_id}"],
            states=[f"next_F{output_id}"]
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
        input_id: int = 1,
        output_id: int = 2,
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
            name="forward_delay",
            inputs=[f"F{input_id}", f"B{output_id}"],
            outputs=[f"F{output_id}", f"B{input_id}"],
            states=[f"next_B{input_id}"]
        )
