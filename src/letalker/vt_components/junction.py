from typing import Any, Protocol

import control as ct
import numpy as np

from ..constants import c as c_default
from ..constants import rho_air as rho_air_default

rhoc_default = rho_air_default * c_default


class LTIJunctionFactory(Protocol):
    def __call__(
        self,
        area1: float,
        area2: float,
        *,
        has_pressure_source: bool,
        has_flow_source: bool,
    ) -> ct.StateSpace:
        """create a two-port system modeling a junction of two vocal tract segments

        Parameters
        ----------
        area1
            cross-sectional area in cm² of the upstream section
        area2
            cross-sectional area in cm² of the downstream section
        areas
            cross-sectional areas of tube sections
        has_pressure_source
            ``True`` if there is any pressure source at junction, i.e., kinetic
            pressure drop or approximated viscous loss of the previous section
        has_flow_source
            ``True`` if there is a turbulent flow source at junction

        Returns
        -------
            feed-through only state-space model
        """


class LosslessJunction(LTIJunctionFactory):
    # Lossless junction VT block with possible independent pressure/flow sources

    rhoc: float = rhoc_default

    def __init__(
        self,
        *,
        rhoc: float | None = None,
    ):
        """factory to generate a tube section junction

        Parameters
        ----------
        rhoc, optional
            physical constant: air density times speed of sound, by default uses
            the system constant
        """

        if rhoc is not None:
            self.rhoc = rhoc

    def __call__(
        self,
        area1: float,
        area2: float,
        *,
        has_pressure_source: bool = False,
        has_flow_source: bool = False,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.StateSpace:
        """create a feed-through only two-port junction system

        Parameters
        ----------
        area1
            cross-sectional area in cm² of the upstream section
        area2
            cross-sectional area in cm² of the downstream section
        has_pressure_source, optional
            ``True`` if there is any pressure source at junction, i.e., kinetic
            pressure drop or approximated viscous loss of the previous section, by default False
        has_flow_source, optional
            ``True`` if there is a turbulent flow source at junction, by default False

        Returns
        -------
            feed-through only state-space model
        """
        rhoc = self.rhoc

        nin = 2 + has_flow_source + has_pressure_source
        den = area1 + area2
        m1 = area1 / den
        m2 = area2 / den
        d = m1 - m2

        D = np.empty((2, nin))
        D[0, 0] = 2 * area1 / den
        D[1, 1] = 2 * area2 / den
        D[0, 1] = -d
        D[1, 0] = d

        if has_pressure_source:
            D[0, 2] = -m1
            D[1, 2] = m2
        if has_flow_source:
            D[:, -1] = rhoc / den

        return ct.ss(
            np.empty((0, 0)), np.empty((0, nin)), np.empty((2, 0)), D, dt=fs and 1 / fs
        )
