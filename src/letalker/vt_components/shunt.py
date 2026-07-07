from typing import overload

import control as ct
import numpy as np

from ..constants import c as c_default
from ..constants import rho_air as rho_air_default
from .abc import LTIFactory

rhoc_default = rho_air_default * c_default


class DefaultYieldingWall(LTIFactory):
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

    def __call__(self, area: float, length: float) -> ct.LTI | float:
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
        return tf

    @property
    def nb_states(self) -> int:
        return 2


class DefaultHeatLossGain(LTIFactory):
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

    def __call__(self, area: float, length: float) -> float:

        return self.Gt * length * area**-0.5

    @property
    def nb_states(self) -> int:
        return 0


class ShuntNetwork(LTIFactory):
    # Shunt networks representing flows into wall

    _yielding_wall_factory: LTIFactory
    _heat_loss_factory: LTIFactory
    rhoc: float = rhoc_default

    @overload
    def __init__(
        self,
        yielding_wall: LTIFactory,
        heat_loss: LTIFactory,
        /,
        rhoc: float | None = None,
    ):
        """Shunt two-port system modeling the yielding wall of a vocal tract segment

        Parameters
        ----------
        yielding_wall
            factory to create a continuous-time transfer function of yielding wall
            (input: pressure, output: wall volume flow) given a cross-sectional
            area and length of a tube section
        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """

        self._yielding_wall_factory = yielding_wall
        self._heat_loss_factory = heat_loss

    @property
    def nb_states(self) -> int:
        """Number of internal states"""
        return self._yielding_wall_factory.nb_states + (
            self._heat_loss_factory.nb_states
        )

    def __call__(self, area: float, length: float) -> ct.LTI:
        """generate state-space models and iterate over n samples

        Args
        ----
        area
            cross-sectional areas of tube sections
        length
            length (in cm) of each tube section
        """

        Hw: ct.StateSpace = ct.parallel(
            self._yielding_wall_factory(area, length),
            self._heat_loss_factory(area, length),
        ).ss()

        two_area = 2 * area
        rhoc_d = self.rhoc * Hw.D
        den = 1 / (two_area + rhoc_d)
        k1 = two_area / den
        k2 = -self.rhoc / den
        A = Hw.A - (Hw.B * den) @ Hw.C
        B = np.tile(Hw.B * k1, (1, 2))
        C = np.tile(k2 * Hw.C, (2, 1))
        k3 = k2 * Hw.D
        D = np.array([k1, k3], [k3, k1])

        return ct.StateSpace(A, B, C, D)
