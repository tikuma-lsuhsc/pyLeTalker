from __future__ import annotations

from typing_extensions import Self, Literal

import numpy as np

from .WaveReflectionVocalTract import WaveReflectionVocalTract
from ._util import smb_vt_area_fun
from ..constants import smb_vt_area_data, SMB2018VocalTractSound


class SixPointVocalTract(WaveReflectionVocalTract):
    """Wave-reflection vocal tract model (Liljencrants, 1985; Story, 1995)"""

    def __init__(
        self,
        Ap: float,
        npc: float,
        xc: float,
        Ac: float,
        nca: float,
        Aa: float,
        nain: float,
        xin: float,
        Ain: float,
        xlip: float,
        Alip: float,
        xlar: float | None = None,
        Alar: float | None = None,
        nlarp: float | None = None,
        xp: float | None = None,
        xa: float | None = None,
        atten: float | None = None,
        log_sections: bool = False,
        min_areas: float | None = None,
    ):
        """Stone-Marxen-Birkholz 1D vocal tract model area function

        Args:
            Ap: Cross-sectional area in cm2 at the posterior control point
            npc: Warping exponent of segment between the posterior control point and
                constriction
            xc: Position of the constriction in cm
            Ac: Cross-sectional area in cm2 at the constriction
            nca: Warping exponent of segment between the constriction and the anterior
                control point.
            Aa: Cross-sectional area in cm2 at the anterior control point
            nain: Warping exponent of segment between the anterior control point and
                the incisors
            xin: Position of the incisors in cm
            Ain: Cross-sectional area in cm2 at the incisors
            xlip: Vocal tract length in cm
            Alip: Cross-sectional area in cm2 at the lips
            xlar: Length of the larynx tube in cm. Defaults to None.
            Alar: Cross-sectional area in cm2 of the larynx tube. Defaults to None.
            nlarp: Warping exponent of segment between the larynx and the posterior
                control point. Defaults to None.
            xp: Position of the posterior control point in cm. Defaults to None.
            xa: Position of the anterior control point in cm. Defaults to None.
        :param areas: cross-sectional areas of vocal tract segments
        :param atten: per-section attenuation factor, defaults to None
        :param log_sections: `True` to log the incidental pressure at every vocal tract segment, defaults to False
        :param min_areas: minimum allowable cross-sectional area to enforce positive area, defaults to None

        Returns:
            function to generate cross-sectional area profile as a function of given
            a vector of position along the center of vocal tract in cm

        Reference
        ---------

        [1] S. Stone, M. Marxen, and P. Birkholz, “Construction and evaluation of a
            parametric one-dimensional vocal tract model,” IEEE/ACM Trans. Audio, Speech,
            Language Process., vol. 26, no. 8, pp. 1381–1392, Aug. 2018,
            doi: 10.1109/TASLP.2018.2825601.


        """
        # """Wave-reflection vocal tract model

        # The wave-reflection vocal tract model represents the vocal tract as a
        # series of short lossy tube segments. Each segment has a fixed length
        # :math:`c/fs/2` (by default, 0385 cm).

        # Parameters
        # ----------
        # areas
        #     cross-sectional areas of vocal tract segments.
        # atten, optional
        #     per-section attenuation factor, by default `None`
        # log_sections, optional
        #     `True` to log the incidental pressure at every vocal tract segment,
        #     by default `False`
        # min_areas, optional
        #     minimum allowable cross-sectional area to enforce positive area,
        #     by default `None` to use `1e-6`.

        # """

        dx = SixPointVocalTract.dz

        k = round(xlip / dx / 2) * 2  # number of segments (must be even)
        x = np.arange(k) * dx

        # scale so that the length is nearest integer multiple of 2dx
        s = k * dx / xlip
        xc, xin, xlip = np.array([xc, xin, xlip]) * s
        if xlar is not None:
            xlar = xlar * s
        if xp is not None:
            xp = xp * s
        if xa is not None:
            xa = xa * s

        areas = smb_vt_area_fun(
            # fmt:off
            Ap,
            npc,
            xc,
            Ac,
            nca,
            Aa,
            nain,
            xin,
            Ain,
            xlip,
            Alip,
            xlar,
            Alar,
            nlarp,
            xp,
            xa,
            # fmt:on
        )(x)

        super().__init__(areas, atten, log_sections, min_areas)

    VocalTractSound = SMB2018VocalTractSound

    @staticmethod
    def from_preset(
        sound: SMB2018VocalTractSound,
        data_type: Literal["geom", "optim"] = "optim",
        reduced: bool = True,
        atten: float | None = None,
        log_sections: bool = False,
        min_areas: float | None = None,
    ) -> SixPointVocalTract:

        return SixPointVocalTract(
            *smb_vt_area_data(data_type, reduced)[sound],
            atten=atten,
            log_sections=log_sections,
            min_areas=min_areas,
        )
